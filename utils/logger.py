# utils/logger.py

import os
import yaml
import json
from datetime import datetime


class ExperimentLogger:
    def __init__(self, base_dir="logs"):
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(base_dir, now)
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics = []

    def log_config(self, cfg):
        with open(os.path.join(self.log_dir, "config.yaml"), 'w') as f:
            yaml.dump(cfg, f)

    def log_model_architecture(self, model, name="model"):
        with open(os.path.join(self.log_dir, f"{name}_summary.txt"), 'w') as f:
            f.write(str(model))

    def log_metrics(self, epoch, metrics, prefix="train"):
        record = {"epoch": epoch, "type": prefix, **metrics}
        self.metrics.append(record)

    def save_summary(self):
        with open(os.path.join(self.log_dir, "metrics.json"), 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def save_checkpoint(self, model, name="model.pt"):
        import torch
        torch.save(model.state_dict(), os.path.join(self.log_dir, name))
