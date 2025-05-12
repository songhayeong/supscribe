# Entry point for TabTransformer + VAE-based IVF strategy modeling

import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from model.tab_vae import TabVAEModel
from data.dataset import IVFStrategyDataset
from train.trainer import train_epoch
from train.evaluate import evaluate
from utils.logger import ExperimentLogger

# --------------------------
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml')
args = parser.parse_args()

# --------------------------
# Load config
with open('config/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# --------------------------
# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# --------------------------
# Logger init
logger = ExperimentLogger()
logger.log_config(cfg)

# --------------------------
# Dataset & DataLoader
train_dataset = IVFStrategyDataset(cfg['data']['train_csv'])
test_dataset = IVFStrategyDataset(cfg['data']['test_csv'])

train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

# --------------------------
# Model
model = TabVAEModel(cfg).to(device)
logger.log_model_architecture(model, "TabVAE")

# --------------------------
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])

# --------------------------
# Training loop
for epoch in range(1, cfg['training']['num_epochs'] + 1):
    print(f"\nEpoch {epoch}")
    train_metrics = train_epoch(model, train_loader, optimizer, device)
    test_metrics = evaluate(model, test_loader, device)
    print("Train:", train_metrics)
    print("Test :", test_metrics)

    # Log metrics
    logger.log_metrics(epoch, train_metrics, prefix="train")
    logger.log_metrics(epoch, test_metrics, prefix="test")


# --------------------------
# Save summary
logger.save_summary()
print(f"\n[Experiment logged to]: {logger.log_dir}")
