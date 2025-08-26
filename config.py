import torch

# Paths
PREPROCESSED_DATA_DIR = "./aircraft_preprocessed_data"
LOGS_DIR = "logs"
MODEL_CHECKPOINT_DIR = "checkpoints"

# Data
BATCH_SIZE = 64        # now can be bigger due to adjacency pruning
WINDOW = 12
HORIZON = 12
STRIDE = 1
VAL_LEN = 0.2
TEST_LEN = 0.2

# Top-K adjacency pruning (controls sparsity)
TOP_K = 20   # keep 20 strongest neighbors per node

# Training
EPOCHS = 20
PATIENCE = 5
OPTIMIZER_PARAMS = {"lr": 0.001, "weight_decay": 1e-5}

TRAINER_PARAMS = {
    "max_epochs": EPOCHS,
    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    "devices": 1,
    "log_every_n_steps": 20,
    "precision": 16,                # ✅ FP16 speeds up + saves memory
    "accumulate_grad_batches": 2,   # ✅ effective batch = BATCH_SIZE*2
}

# GraphWaveNet Params
GWN_PARAMS = {
    "hidden_size": 32,
    "ff_size": 256,
    "n_layers": 8,
    "temporal_kernel_size": 2,
    "spatial_kernel_size": 2,
    "learned_adjacency": True,
    "emb_size": 10,
    "dilation": 2,
    "dilation_mod": 2,
    "norm": "batch",
    "dropout": 0.3,
}

# DCRNN Params
DCRNN_PARAMS = {
    "hidden_size": 64,
    "n_layers": 2,
    "k_hops": 2,
}

# OPTIMIZER_PARAMS = {
#     'lr': 0.01
# }
