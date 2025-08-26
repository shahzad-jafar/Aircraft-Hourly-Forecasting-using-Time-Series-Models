import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import haversine_distances

# Import configs from your config.py
from config import PREPROCESSED_DATA_DIR, WINDOW, HORIZON

# --- CONFIG ---
RAW_DATA_DIR = "./"
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)


# --- PART 1: LOAD RAW DATA ---
def load_raw_data():
    """Load 2024 and 2025 aircraft hourly data."""
    print("--- Loading Raw Aircraft Data ---")
    df_2024 = pd.read_parquet(os.path.join(RAW_DATA_DIR, 'aircraft_nyc_hourly_r6_2024.parquet'))
    print("✅ Loaded aircraft_nyc_hourly_r6_2024.parquet")
    
    df_2025 = pd.read_parquet(os.path.join(RAW_DATA_DIR, 'aircraft_nyc_hourly_r6_2025.parquet'))
    print("✅ Loaded aircraft_nyc_hourly_r6_2025.parquet")
    
    return df_2024, df_2025


# --- PART 2: CLEAN AND PREPARE DATA WITH INTERPOLATION ---
def prepare_dataframe(df, name, full_range=None):
    """Clean, prepare, and interpolate hourly gaps for the aircraft dataframe."""
    print(f"--- Preparing {name} ---")
    df_copy = df.copy()
    
    # Rename columns
    df_copy.rename(columns={'num_of_aircrafts': 'target', 'h3': 'item_id'}, inplace=True)
    
    # Ensure datetime
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    
    # Fill hourly gaps per item
    interpolated_list = []
    items = df_copy['item_id'].unique()
    if full_range is None:
        full_range = pd.date_range(df_copy['timestamp'].min(), df_copy['timestamp'].max(), freq='H')
    
    for item in items:
        group = df_copy[df_copy['item_id'] == item].set_index('timestamp').sort_index()
        group = group[~group.index.duplicated(keep='first')]
        group = group.reindex(full_range)
        group['item_id'] = item
        
        # Fill lat/lon and target
        group[['lat', 'lon']] = group[['lat', 'lon']].ffill()
        group['target'] = group['target'].interpolate(method='linear').fillna(0)
        
        interpolated_list.append(group.reset_index().rename(columns={'index': 'timestamp'}))
    
    df_filled = pd.concat(interpolated_list).reset_index(drop=True)
    
    # Remove timezone info
    if pd.api.types.is_datetime64_any_dtype(df_filled['timestamp']) and df_filled['timestamp'].dt.tz is not None:
        df_filled['timestamp'] = df_filled['timestamp'].dt.tz_convert(None)
    
    print(f"✅ Finished preparing {name}")
    return df_filled


# --- PART 3: SAVE FOR AUTOGUON ---
def save_for_autogluon(df_train, df_eval):
    print("--- Saving for AutoGluon ---")
    df_train[['timestamp', 'item_id', 'target']].to_csv(os.path.join(PREPROCESSED_DATA_DIR, 'autogluon_train.csv'), index=False)
    df_eval[['timestamp', 'item_id', 'target']].to_csv(os.path.join(PREPROCESSED_DATA_DIR, 'autogluon_eval.csv'), index=False)
    print("✅ AutoGluon CSV files saved")


# --- PART 4: CREATE AND SAVE GRAPH TENSORS FOR GRAPHWAVENET ---
def create_and_save_tensors(df_train, df_eval):
    print("--- Creating Graph Tensors for GraphWaveNet ---")
    
    # Get all unique nodes
    all_nodes = sorted(pd.concat([df_train['item_id'], df_eval['item_id']]).unique())
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Node features (lat/lon)
    node_features_df = pd.concat([df_train, df_eval]).groupby('item_id')[['lat', 'lon']].mean().reindex(all_nodes)
    node_features = torch.tensor(node_features_df.values, dtype=torch.float32)
    
    # Graph adjacency based on haversine distances
    coords_rad = np.radians(node_features_df.values)
    distance_matrix = haversine_distances(coords_rad)
    threshold = 0.02  # ~2 km
    adj_matrix = (distance_matrix < threshold).astype(np.float32)
    np.fill_diagonal(adj_matrix, 0)
    
    # Normalize adjacency matrix (for GraphWaveNet)
    adj_matrix_with_self = adj_matrix + np.eye(len(all_nodes))
    degree = np.sum(adj_matrix_with_self, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj_matrix_with_self @ d_mat_inv_sqrt
    
    # Full hourly range for both datasets
    full_range = pd.date_range(
        start=min(df_train['timestamp'].min(), df_eval['timestamp'].min()),
        end=max(df_train['timestamp'].max(), df_eval['timestamp'].max()),
        freq='H'
    )
    
    # Prepare data with full interpolation
    df_train_full = prepare_dataframe(df_train, "Train Tensor Prep", full_range=full_range)
    df_eval_full = prepare_dataframe(df_eval, "Eval Tensor Prep", full_range=full_range)
    
    # Create data matrices for GraphWaveNet
    def create_data_matrix(df):
        feature_matrix = df.pivot_table(
            index='timestamp', 
            columns='item_id', 
            values='target'
        ).reindex(columns=all_nodes).fillna(0)
        
        data = feature_matrix.values
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)  # [time, nodes, features]
        return data_tensor, feature_matrix.index
    
    train_data, train_timestamps = create_data_matrix(df_train_full)
    eval_data, eval_timestamps = create_data_matrix(df_eval_full)
    
    # --- DEBUG PRINTS ---
    print("\n--- Debug Info ---")
    print(f"Train tensor shape: {train_data.shape}")
    print(f"Eval tensor shape: {eval_data.shape}")
    print(f"Adjacency shape: {adj_normalized.shape}")
    print(f"Nodes count: {len(all_nodes)}")
    print(f"Edges count (original adj): {int(adj_matrix.sum())}")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Node features shape (lat/lon): {node_features.shape}")
    
    # Save training data
    torch.save({
        'x': train_data,
        'timestamps': train_timestamps.to_list(),
        'num_timesteps': train_data.shape[0],
        'num_nodes': len(all_nodes),
        'num_features': 1,
        'window': WINDOW,
        'horizon': HORIZON
    }, os.path.join(PREPROCESSED_DATA_DIR, 'train.pt'))
    
    # Save evaluation data
    torch.save({
        'x': eval_data,
        'timestamps': eval_timestamps.to_list(),
        'num_timesteps': eval_data.shape[0],
        'num_nodes': len(all_nodes),
        'num_features': 1,
        'window': WINDOW,
        'horizon': HORIZON
    }, os.path.join(PREPROCESSED_DATA_DIR, 'test.pt'))
    
    # Save adjacency matrix
    adj_mx = torch.tensor(adj_normalized, dtype=torch.float32)
    torch.save(adj_mx, os.path.join(PREPROCESSED_DATA_DIR, 'adj_mx.pt'))
    
    # Save additional graph info
    torch.save({
        'node_ids': all_nodes,
        'node_to_idx': node_to_idx,
        'node_features': node_features,
        'distance_matrix': torch.tensor(distance_matrix, dtype=torch.float32),
        'original_adj_matrix': torch.tensor(adj_matrix, dtype=torch.float32)
    }, os.path.join(PREPROCESSED_DATA_DIR, 'graph_info.pt'))
    
    print("\n✅ GraphWaveNet data files saved successfully")
    print(f"   - train.pt: Training data")
    print(f"   - test.pt: Evaluation data")
    print(f"   - adj_mx.pt: Normalized adjacency matrix")
    print(f"   - graph_info.pt: Additional graph information")


# --- MAIN ---
if __name__ == "__main__":
    df_2024, df_2025 = load_raw_data()
    
    df_2024_clean = prepare_dataframe(df_2024, "2024 Training")
    df_2025_clean = prepare_dataframe(df_2025, "2025 Evaluation")
    
    save_for_autogluon(df_2024_clean, df_2025_clean)
    create_and_save_tensors(df_2024_clean, df_2025_clean)
    
    print("\n🎉 All preprocessing completed successfully!")
    print(f"\n📊 Configuration:")
    print(f"   - Window size: {WINDOW} hours")
    print(f"   - Prediction horizon: {HORIZON} hours")
    print(f"   - Output directory: {PREPROCESSED_DATA_DIR}/")


