import os
import torch
import matplotlib
matplotlib.use("Agg")   # Safe for servers (headless mode)
import matplotlib.pyplot as plt
import pandas as pd

import config
import utils


def plot_full_series(train_target, test_target, predictor, datamodule,
                     node_id=0, feature_id=0,
                     freq="5min", start_time="2023-03-17 00:00:00",
                     save_path="plots/full_series.png"):

    # --- Ensure save folder exists ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- Build datetime index ---
    total_len = len(train_target) + len(test_target)
    date_index = pd.date_range(start=pd.Timestamp(start_time),
                               periods=total_len, freq=freq)

    # Ground truth series
    train_series = train_target[:, node_id, feature_id].cpu().numpy()
    test_series = test_target[:, node_id, feature_id].cpu().numpy()

    # --- Predictions from model ---
    predictor.eval()
    preds = []
    test_loader = datamodule.test_dataloader()
    with torch.no_grad():
        for batch in test_loader:
            y_hat = predictor.predict_batch(batch)  # [B, horizon, N, F]
            preds.extend(y_hat[:, :, node_id, feature_id].cpu().numpy().flatten())
    pred_series = preds[:len(test_series)]

    # --- Plotting ---
    plt.figure(figsize=(16, 6))

    # Train True
    plt.plot(date_index[:len(train_series)], train_series,
             color="black", label="Train True")

    # Validation/Test True
    plt.plot(date_index[len(train_series):len(train_series) + len(test_series)],
             test_series, color="green", label="Validation/Test True")

    # Predictions
    plt.plot(date_index[len(train_series):len(train_series) + len(pred_series)],
             pred_series, "--", color="blue", label="Predictions")

    plt.title(f"Time Series Train/Val/Test vs Predictions (Node {node_id}, Feature {feature_id})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- Save & show ---
    plt.savefig(save_path, dpi=200)
    print(f"[✔] Graph saved at: {os.path.abspath(save_path)}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()


def run_visualization(model_type="gwn",
                      checkpoint_path="checkpoints/best_model.pt",
                      node_id=0, feature_id=0,
                      freq="5min", start_time="2023-03-17 00:00:00"):

    print(f"\n--- Visualization for {model_type.upper()} ---")

    # 1. Load data
    train_target, train_mask, test_target, test_mask, edge_index, edge_weight, node_features = utils.load_preprocessed_data()

    dataset = utils.create_spatiotemporal_dataset(train_target, train_mask,
                                                  edge_index, edge_weight, node_features)
    dm = utils.create_datamodule(dataset)
    dm.setup()   # ✅ Important: prepare dataloaders

    # 2. Build model
    if model_type == "gwn":
        model = utils.build_graphwavenet_model(dataset, train_target.shape[1])
    elif model_type == "dcrnn":
        model = utils.build_dcrnn_model(dataset, train_target.shape[1])
    else:
        raise ValueError("Unknown model type")

    wrapped_model = utils.FixedTSLWrapper(model)
    predictor = utils.create_predictor(wrapped_model)

    # 3. Load weights safely
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)

    if "state_dict" in state_dict:   # Lightning checkpoint
        predictor.load_state_dict(state_dict["state_dict"], strict=False)
    else:   # Pure PyTorch state_dict
        predictor.load_state_dict(state_dict, strict=False)

    # 4. Plot Graph
    plot_full_series(train_target, test_target, predictor, dm,
                     node_id=node_id, feature_id=feature_id,
                     freq=freq, start_time=start_time,
                     save_path="plots/full_series.png")


if __name__ == "__main__":
    run_visualization(
        model_type="gwn",                          # "gwn" or "dcrnn"
        checkpoint_path="checkpoints/best_model.pt",  # path to your saved model
        node_id=0,
        feature_id=0,
        freq="5min",                              # adjust as per dataset
        start_time="2023-03-17 00:00:00"          # actual dataset start timestamp
    )