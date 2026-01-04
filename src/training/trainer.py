import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from tqdm import tqdm
from typing import Optional

class Trainer:
    """
    Handles the training loop, validation, and checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: dict,
        device: str = "cuda"
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Hyperparams
        self.lr = float(config['training']['lr'])
        self.weight_decay = float(config['training']['weight_decay'])
        self.max_epochs = int(config['training']['max_epochs'])
        self.patience = int(config['training']['patience'])
        self.save_dir = config['training']['save_dir']
        
        # Optimizer & Scheduler
        self.opt = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=3
        )
        self.scaler = torch.amp.GradScaler(self.device.type)

        # Logging
        self.logger = logging.getLogger("Trainer")
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self):
        """Full training loop with early stopping"""
        self.logger.info(f"Starting training on {self.device}...")
        
        best_val = float("inf")
        best_state = None
        wait = 0
        
        for epoch in range(1, self.max_epochs + 1):
            start_time = time.time()
            
            # --- Train ---
            self.model.train()
            tr_loss, n_tr = 0.0, 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epochs}", leave=False)
            for xb, yb in pbar:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                
                self.opt.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=self.device.type):
                    pred = self.model(xb)
                    loss = F.mse_loss(pred, yb)
                
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
                
                tr_loss += loss.item() * xb.size(0)
                n_tr += xb.size(0)
                
                pbar.set_postfix({'loss': loss.item()})
                
            tr_loss /= max(1, n_tr)
            
            # --- Validation ---
            self.model.eval()
            va_loss, n_va = 0.0, 0
            with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
                for xb, yb in self.val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    pred = self.model(xb)
                    loss = F.mse_loss(pred, yb)
                    va_loss += loss.item() * xb.size(0)
                    n_va += xb.size(0)
            va_loss /= max(1, n_va)
            
            # --- Scheduler & Logging ---
            before_lr = self.opt.param_groups[0]["lr"]
            self.scheduler.step(va_loss)
            after_lr = self.opt.param_groups[0]["lr"]
            
            lr_msg = ""
            if after_lr < before_lr:
                lr_msg = f" | LR dropped: {before_lr:.1e} -> {after_lr:.1e}"
            
            improved = va_loss < best_val
            marker = "*" if improved else ""
            
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | "
                f"Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}{lr_msg} {marker}"
            )
            
            # --- Early Stopping ---
            if improved:
                best_val = va_loss
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                wait = 0
                
                # Save checkpoint
                ckpt_path = os.path.join(self.save_dir, "best_model.pt")
                torch.save(best_state, ckpt_path)
            else:
                wait += 1
                if wait >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.logger.info(f"Restored best model with Val Loss: {best_val:.6f}")
            
        return self.model
