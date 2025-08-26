import torch
import torch.nn as nn
from tsl.data import SpatioTemporalDataset
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.nn.models.stgn import GraphWaveNetModel, DCRNNModel
from tsl.metrics.torch import MaskedMAE, MaskedMAPE
from tsl.engines import Predictor
import config



# import torch
# import torch.nn as nn
# from tsl.data import SpatioTemporalDataset
# from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
# from tsl.nn.models.stgn import GraphWaveNetModel, DCRNNModel
# from tsl.metrics.torch import MaskedMAE, MaskedMAPE
# from tsl.engines import Predictor
# import config


# === Adjacency pruning helper ===
def prune_adj_topk(adj, k=20):
    """Keep only top-k strongest edges per node."""
    new_adj = torch.zeros_like(adj)
    for i in range(adj.shape[0]):
        row = adj[i]
        if torch.count_nonzero(row) > 0:
            topk = torch.topk(row, min(k, row.numel())).indices
            new_adj[i, topk] = row[topk]
    return new_adj


# ==========================================================
# Load data (with pruning)
# ==========================================================
def load_preprocessed_data():
    data_path = config.PREPROCESSED_DATA_DIR

    train_data = torch.load(f"{data_path}/train.pt")
    test_data = torch.load(f"{data_path}/test.pt")
    adj_mx = torch.load(f"{data_path}/adj_mx.pt")
    graph_info = torch.load(f"{data_path}/graph_info.pt")

    train_target = train_data['x']  # [time, nodes, features]
    test_target = test_data['x']

    train_mask = torch.ones_like(train_target, dtype=torch.bool)
    test_mask = torch.ones_like(test_target, dtype=torch.bool)

    print(f"Original data shapes:")
    print(f"  Train: {train_target.shape}, Test: {test_target.shape}")
    print(f"  Adj:   {adj_mx.shape}, nnz={adj_mx.nonzero().shape[0]}")

    # ✅ prune adjacency to top-K neighbors
    adj_mx = prune_adj_topk(adj_mx, k=config.TOP_K)
    print(f"Pruned adjacency: nnz={adj_mx.nonzero().shape[0]}")

    expected_nodes = train_target.shape[1]
    if adj_mx.shape[0] != expected_nodes or adj_mx.shape[1] != expected_nodes:
        adj_mx = torch.eye(expected_nodes, dtype=adj_mx.dtype, device=adj_mx.device)

    nz = adj_mx.nonzero()
    edge_index = nz.t().contiguous()
    edge_weight = adj_mx[nz[:, 0], nz[:, 1]] if nz.numel() > 0 else torch.tensor([])

    node_features = graph_info.get("node_features", None)
    if node_features is None:
        node_features = torch.eye(expected_nodes)
    elif node_features.shape[0] != expected_nodes:
        if node_features.shape[0] > expected_nodes:
            node_features = node_features[:expected_nodes]
        else:
            pad = torch.zeros(expected_nodes - node_features.shape[0],
                              node_features.shape[1])
            node_features = torch.cat([node_features, pad], dim=0)

    return (train_target, train_mask,
            test_target, test_mask,
            edge_index, edge_weight, node_features)


# ==========================================================
# Dataset, DataModule, Models & Predictor (same as before)
# ==========================================================
class FixedSpatioTemporalDataset(SpatioTemporalDataset):
    def __init__(self, target, mask, connectivity, horizon, window, stride, covariates=None):
        self._n_nodes = target.shape[1]
        super().__init__(target=target, mask=mask,
                         connectivity=connectivity,
                         horizon=horizon, window=window,
                         stride=stride, covariates=covariates)

    @property
    def n_nodes(self):
        return self._n_nodes


def create_spatiotemporal_dataset(target, mask, edge_index, edge_weight, covariates):
    return FixedSpatioTemporalDataset(
        target=target, mask=mask,
        connectivity=(edge_index, edge_weight),
        horizon=config.HORIZON, window=config.WINDOW, stride=config.STRIDE,
        covariates={'u': covariates} if covariates is not None else None
    )


def create_datamodule(dataset):
    splitter = TemporalSplitter(val_len=config.VAL_LEN, test_len=config.TEST_LEN)
    return SpatioTemporalDataModule(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        workers=4,
        pin_memory=True,
        splitter=splitter
    )


class FixedTSLWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, edge_index=None, edge_weight=None, **kwargs):
        return self.model(x, edge_index=edge_index, edge_weight=edge_weight)


def build_graphwavenet_model(dataset, actual_nodes):
    params = config.GWN_PARAMS.copy()
    params.update(dict(
        input_size=dataset.n_channels,
        output_size=dataset.n_channels,
        horizon=dataset.horizon,
        n_nodes=actual_nodes,
        exog_size=0
    ))
    return GraphWaveNetModel(**params)


def build_dcrnn_model(dataset, actual_nodes):
    params = config.DCRNN_PARAMS.copy()
    params.update(dict(
        input_size=dataset.n_channels,
        output_size=dataset.n_channels,
        horizon=dataset.horizon,
        n_nodes=actual_nodes,
        exog_size=0
    ))
    return DCRNNModel(**params)


def create_predictor(model):
    return Predictor(
        model=model,
        optim_class=torch.optim.Adam,
        optim_kwargs=config.OPTIMIZER_PARAMS,
        loss_fn=MaskedMAE(),
        metrics={
            "mae": MaskedMAE(),
            "mape": MaskedMAPE(),
            "mae_at_15min": MaskedMAE(at=2),
            "mae_at_30min": MaskedMAE(at=5),
            "mae_at_60min": MaskedMAE(at=11),
        }
    )
# # ==========================================================
# # Load data
# # ==========================================================
# def load_preprocessed_data():
#     data_path = config.PREPROCESSED_DATA_DIR

#     train_data = torch.load(f"{data_path}/train.pt")
#     test_data = torch.load(f"{data_path}/test.pt")
#     adj_mx = torch.load(f"{data_path}/adj_mx.pt")
#     graph_info = torch.load(f"{data_path}/graph_info.pt")

#     train_target = train_data['x']  # [time, nodes, features]
#     test_target = test_data['x']

#     train_mask = torch.ones_like(train_target, dtype=torch.bool)
#     test_mask = torch.ones_like(test_target, dtype=torch.bool)

#     print(f"Original data shapes:")
#     print(f"  Train: {train_target.shape}, Test: {test_target.shape}")
#     print(f"  Adj:   {adj_mx.shape}")

#     expected_nodes = train_target.shape[1]
#     if adj_mx.shape[0] != expected_nodes or adj_mx.shape[1] != expected_nodes:
#         print("Adjacency mismatch, creating identity.")
#         adj_mx = torch.eye(expected_nodes, dtype=adj_mx.dtype, device=adj_mx.device)

#     nz = adj_mx.nonzero()
#     edge_index = nz.t().contiguous()
#     edge_weight = adj_mx[nz[:, 0], nz[:, 1]] if nz.numel() > 0 else torch.tensor([])

#     node_features = graph_info.get("node_features", None)
#     if node_features is None:
#         node_features = torch.eye(expected_nodes)
#     elif node_features.shape[0] != expected_nodes:
#         print("Node features mismatch → fixing")
#         if node_features.shape[0] > expected_nodes:
#             node_features = node_features[:expected_nodes]
#         else:
#             pad = torch.zeros(expected_nodes - node_features.shape[0],
#                               node_features.shape[1])
#             node_features = torch.cat([node_features, pad], dim=0)

#     print(f"Final data (time, nodes, feat): {train_target.shape}")

#     return (train_target, train_mask,
#             test_target, test_mask,
#             edge_index, edge_weight, node_features)


# # ==========================================================
# # Dataset
# # ==========================================================
# class FixedSpatioTemporalDataset(SpatioTemporalDataset):
#     """Force correct node interpretation: [time, nodes, features]."""
#     def __init__(self, target, mask, connectivity, horizon, window, stride, covariates=None):
#         self._n_nodes = target.shape[1]
#         super().__init__(target=target, mask=mask,
#                          connectivity=connectivity,
#                          horizon=horizon, window=window,
#                          stride=stride, covariates=covariates)

#     @property
#     def n_nodes(self):
#         return self._n_nodes


# def create_spatiotemporal_dataset(target, mask, edge_index, edge_weight, covariates):
#     available_length = target.shape[0] - config.WINDOW - config.HORIZON + 1
#     if available_length <= 0:
#         print("ERROR: not enough timesteps.")
#         return None

#     ds = FixedSpatioTemporalDataset(
#         target=target, mask=mask,
#         connectivity=(edge_index, edge_weight),
#         horizon=config.HORIZON, window=config.WINDOW, stride=config.STRIDE,
#         covariates={'u': covariates} if covariates is not None else None
#     )
#     print(f"✅ Dataset created: {len(ds)} samples, {ds.n_nodes} nodes")
#     if len(ds) > 0:
#         s = ds[0]
#         print(f"Sample x → {s.x.shape}, y → {s.y.shape}")
#     return ds


# # ==========================================================
# # DataModule (with default splitter)
# # ==========================================================
# def create_datamodule(dataset):
#     splitter = TemporalSplitter(val_len=config.VAL_LEN, test_len=config.TEST_LEN)
#     return SpatioTemporalDataModule(
#         dataset=dataset,
#         batch_size=config.BATCH_SIZE,
#         workers=0,
#         splitter=splitter
#     )


# # ==========================================================
# # Models
# # ==========================================================
# class FixedTSLWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, x, edge_index=None, edge_weight=None, **kwargs):
#         return self.model(x, edge_index=edge_index, edge_weight=edge_weight)


# def build_graphwavenet_model(dataset, actual_nodes):
#     params = config.GWN_PARAMS.copy()
#     params.update(dict(
#         input_size=dataset.n_channels,
#         output_size=dataset.n_channels,
#         horizon=dataset.horizon,
#         n_nodes=actual_nodes,
#         exog_size=0
#     ))
#     print("GraphWaveNet params:", params)
#     return GraphWaveNetModel(**params)


# def build_dcrnn_model(dataset, actual_nodes):
#     params = config.DCRNN_PARAMS.copy()
#     params.update(dict(
#         input_size=dataset.n_channels,
#         output_size=dataset.n_channels,
#         horizon=dataset.horizon,
#         n_nodes=actual_nodes,
#         exog_size=0
#     ))
#     print("DCRNN params:", params)
#     return DCRNNModel(**params)


# # ==========================================================
# # Predictor
# # ==========================================================
# def create_predictor(model):
#     return Predictor(
#         model=model,
#         optim_class=torch.optim.Adam,
#         optim_kwargs=config.OPTIMIZER_PARAMS,
#         loss_fn=MaskedMAE(),
#         metrics={
#             "mae": MaskedMAE(),
#             "mape": MaskedMAPE(),
#             "mae_at_15min": MaskedMAE(at=2),
#             "mae_at_30min": MaskedMAE(at=5),
#             "mae_at_60min": MaskedMAE(at=11),
#         }
#     )


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model