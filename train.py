import argparse
import yaml
import logging
import sys
import torch
import numpy as np
import random
import os

from src.data.dataset import VolatilityDataset
from src.models.volformer import VolFormer
from src.training.trainer import Trainer

def setup_logging(save_dir, exp_name):
    """Configures logging to console and file."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(save_dir, f"{exp_name}.log"))
        ]
    )

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train VolFormer for Stock Volatility Prediction")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)
    
    # Setup Dirs & Logging
    save_dir = config["training"]["save_dir"]
    exp_name = config["training"]["experiment_name"]
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(save_dir, exp_name)
    logger = logging.getLogger("Main")
    
    logger.info(f"Loaded config from {args.config}")
    
    # Set Seed
    set_seed(42)

    # 1. Data Loading
    logger.info("Loading and preprocessing data...")
    dataset = VolatilityDataset(config)
    X_train, y_train, X_val, y_val = dataset.load_and_prepare()
    
    train_loader, val_loader = dataset.get_dataloaders(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"]
    )
    
    # 2. Model Initialization
    logger.info("Initializing Model...")
    # Infer input dim from data (will be 1 for log_RV)
    d_in = X_train.shape[-1]
    
    model = VolFormer(
        d_in=d_in,
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        p_drop=config["model"]["p_drop"],
        use_cls=config["model"]["use_cls"],
        ff_mult=config["model"]["ff_mult"],
        max_len=config["model"]["max_len"]
    )
    
    # 3. Training
    logger.info("Starting Training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    trainer.train()
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
