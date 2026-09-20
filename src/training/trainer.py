"""Training loop with mixed precision, early stopping, and checkpointing."""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Unified training loop for all models.
    
    Handles the boilerplate: mixed precision for speed, early stopping so we don't overfit,
    and automatically saving the best model weights based on validation loss.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler=None,
        use_amp: bool = True,
        output_dir: str = "outputs",
        model_name: str = "model",
        patience: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.model_name = model_name
        self.patience = patience
        
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        
        self.output_dir = Path(output_dir)
        self.model_dir = self.output_dir / "models"
        self.metrics_dir = self.output_dir / "metrics"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [], "epoch_time": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run a single pass through the training data."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Training", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix(loss=loss.item(), acc=correct / total)
        
        return {
            "loss": running_loss / total,
            "accuracy": correct / total,
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            "loss": running_loss / total,
            "accuracy": correct / total,
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
    ) -> Dict:
        """Full training loop with early stopping."""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"Device: {self.device} | AMP: {self.use_amp} | Patience: {self.patience}")
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {trainable:,} trainable / {total_params:,} total")
        print(f"{'='*60}")
        
        total_train_time = 0.0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)
            
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
                print(f"  [DONE] New best model saved (val_loss={val_metrics['loss']:.4f})")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break
        
        self.history["total_train_time"] = total_train_time
        self.history["best_epoch"] = self.best_epoch
        self.history["best_val_loss"] = self.best_val_loss
        self.history["trainable_params"] = trainable
        self.history["total_params"] = total_params
        
        history_path = self.metrics_dir / f"{self.model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete: {total_train_time:.1f}s total | Best epoch: {self.best_epoch}")
        
        self._load_best_checkpoint()
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        path = self.model_dir / f"{self.model_name}_best.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "model_name": self.model_name,
        }, path)
    
    def _load_best_checkpoint(self):
        """Load the best checkpoint."""
        path = self.model_dir / f"{self.model_name}_best.pth"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        else:
            print("WARNING: No checkpoint found, using final epoch weights")
