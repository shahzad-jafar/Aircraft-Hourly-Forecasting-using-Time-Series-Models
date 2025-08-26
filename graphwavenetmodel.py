import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import utils, config
from visualization import plot_full_series


def run(model_type="gwn"):
    print(f"\n--- Running {model_type.upper()} Experiment ---")

    # --- GPU / CPU info ---
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    # 1. Load preprocessed data
    train_target, train_mask, test_target, test_mask, edge_index, edge_weight, node_features = utils.load_preprocessed_data()
    actual_nodes = train_target.shape[1]

    # 2. Create dataset & datamodule
    dataset = utils.create_spatiotemporal_dataset(train_target, train_mask,
                                                  edge_index, edge_weight, node_features)
    dm = utils.create_datamodule(dataset)

    # 3. Build model
    if model_type == "gwn":
        model = utils.build_graphwavenet_model(dataset, actual_nodes)
    elif model_type == "dcrnn":
        model = utils.build_dcrnn_model(dataset, actual_nodes)
    else:
        raise ValueError("Unknown model_type: use 'gwn' or 'dcrnn'")

    wrapped_model = utils.FixedTSLWrapper(model)
    predictor = utils.create_predictor(wrapped_model)

    # 4. Logging & callbacks
    logger = TensorBoardLogger(save_dir=config.LOGS_DIR, name=model_type.upper())
    ckpt_cb = ModelCheckpoint(
        dirpath=config.MODEL_CHECKPOINT_DIR,
        filename="{epoch}-{val_mae:.4f}",   # filename with epoch & val_mae
        save_top_k=1,
        monitor="val_mae",
        mode="min"
    )
    es_cb = EarlyStopping(
        monitor="val_mae",
        patience=config.PATIENCE,
        mode="min"
    )

    # 5. Trainer
    trainer = pl.Trainer(
        **config.TRAINER_PARAMS,
        logger=logger,
        callbacks=[ckpt_cb, es_cb]
    )

    # 6. Train + Test
    trainer.fit(predictor, datamodule=dm)
    print("Best model checkpoint:", ckpt_cb.best_model_path)

    trainer.test(predictor, datamodule=dm)

    # ✅ Save a stable copy of best model as .pt
    best_path = os.path.join(config.MODEL_CHECKPOINT_DIR, "best_model.pt")
    torch.save(wrapped_model.state_dict(), best_path)
    print(f"[✔] Best model also saved to {best_path}")

    # 7. Visualization
    os.makedirs("plots", exist_ok=True)
    plot_full_series(
        train_target, test_target, predictor, dm,
        node_id=0, feature_id=0,
        freq="5min", start_time="2023-03-17 00:00:00",
        save_path="plots/full_series.png"
    )

    print("--- Experiment Finished ---")


if __name__ == "__main__":
    run("gwn")  # or run("dcrnn")