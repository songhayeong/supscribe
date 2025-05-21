# Entry point for TabTransformer + VAE-based IVF strategy modeling

import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from data.dataset import IVFStrategyDataset
from utils.logger import ExperimentLogger

# --------------------------
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml')
parser.add_argument('--model_type', type=str, choices=['attention', 'tabvae'], default='attention')
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
# Model Selection
if args.model_type == 'attention':
    from attn_analysis_model.model.attention_classifier import AttentionClassifier

    cat_dims = [len(cat) for cat in train_dataset.enc.categories_]

    model = AttentionClassifier(
        cat_dims=cat_dims,
        embed_dim=cfg['model']['embed_dim'],
        num_heads=cfg['model']['num_heads'],
        num_layers=cfg['model']['num_layers'],
        num_numeric=cfg['model']['num_numeric']
    )
    feature_names = train_dataset.cat_cols

elif args.model_type == 'tabvae':
    from tabvae_model.tab_vae import TabVAEModel
    model = TabVAEModel(cfg)
    feature_names = None

model = model.to(device)

logger.log_model_architecture(model, "TabVAE")

# --------------------------
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'])

# --------------------------
# Training loop
if args.model_type == 'tabvae':
    from train.trainer import train_epoch

    for epoch in range(1, cfg['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}")
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print("Train:", train_metrics)

        # Log metrics
        logger.log_metrics(epoch, train_metrics, prefix="train")

elif args.model_type == 'attention':
    from attn_analysis_model.train.trainer import train_epoch

    for epoch in range(1, cfg['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}")
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print("Train:", train_metrics)

        # Log metrics
        logger.log_metrics(epoch, train_metrics, prefix="train")


# --------------------------
if args.model_type == 'tabvae':
    from train.evaluate import evaluate
    # Evaluate once at the end
    print("\nEvaluating final tabvae_model on test set...")
    test_metrics = evaluate(model, test_loader, device)
    print("Test:", test_metrics)
    logger.log_metrics(cfg['training']['num_epochs'], test_metrics, prefix="test")

elif args.model_type == 'attention':
    from attn_analysis_model.train.evaulate import evaluate
    feature_names = train_dataset.cat_cols
    test_metrics = evaluate(model, test_loader, device)
    print("Test:", test_metrics)

# --------------------------
# Save summary
logger.save_summary()
print(f"\n[Experiment logged to]: {logger.log_dir}")
