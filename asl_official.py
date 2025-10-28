import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    from torch.amp import GradScaler
    from torch.amp import autocast
import sys
import os

# Add ASL path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ASL', 'src'))
from loss_functions.losses import AsymmetricLoss

class ASLOfficialClassifier(BaseEstimator, ClassifierMixin):
    """Official ASL training method adapted for binary classification"""
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, lr=1e-4, epochs=80, batch_size=128):
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
    def fit(self, X, y):
        # Convert to tensors
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        X_tensor = torch.FloatTensor(X_array.astype(np.float32))
        y_tensor = torch.FloatTensor(y_array.astype(np.float32)).unsqueeze(1)  # Binary classification
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )
        
        # Model architecture (similar to official)
        self.model = nn.Sequential(
            nn.Linear(X_tensor.shape[1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Binary output
        )
        
        # Official ASL loss and optimizer setup
        criterion = AsymmetricLoss(
            gamma_neg=self.gamma_neg, 
            gamma_pos=self.gamma_pos, 
            clip=self.clip, 
            disable_torch_grad_focal_loss=True
        )
        
        # Weight decay like official implementation
        weight_decay = 1e-4
        parameters = self._add_weight_decay(self.model, weight_decay)
        optimizer = torch.optim.Adam(params=parameters, lr=self.lr, weight_decay=0)
        
        # OneCycleLR scheduler like official
        steps_per_epoch = len(dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, steps_per_epoch=steps_per_epoch, 
            epochs=self.epochs, pct_start=0.2
        )
        
        # Mixed precision training like official
        try:
            scaler = GradScaler()
        except:
            scaler = GradScaler('cpu')
        
        # Training loop (official style)
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i, (batch_X, batch_y) in enumerate(dataloader):
                try:
                    with autocast():  # Mixed precision
                        output = self.model(batch_X)
                        loss = criterion(output, batch_y)
                except:
                    output = self.model(batch_X)
                    loss = criterion(output, batch_y)
                
                self.model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if i % 50 == 0 and epoch % 10 == 0:
                    print(f'Epoch [{epoch}/{self.epochs}], Step [{i}/{len(dataloader)}], '
                          f'LR {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.4f}')
            
            if epoch % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f'[ASL Official] Epoch {epoch}, Avg Loss: {avg_loss:.4f}')
        
        return self
    
    def _add_weight_decay(self, model, weight_decay):
        """Add weight decay like official implementation"""
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}
        ]
    
    def predict_proba(self, X):
        X_array = X.values if hasattr(X, 'values') else X
        X_tensor = torch.FloatTensor(X_array.astype(np.float32))
        
        self.model.eval()
        with torch.no_grad():
            try:
                with autocast():
                    logits = self.model(X_tensor).squeeze()
                    probs = torch.sigmoid(logits).numpy()
            except:
                logits = self.model(X_tensor).squeeze()
                probs = torch.sigmoid(logits).numpy()
        
        return np.column_stack([1-probs, probs])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

def compute_asl_weights(y, gamma_neg=4, gamma_pos=1):
    """Compute sample weights based on ASL principles"""
    pos_weight = len(y) / (2 * np.sum(y)) if np.sum(y) > 0 else 1
    neg_weight = len(y) / (2 * (len(y) - np.sum(y))) if len(y) - np.sum(y) > 0 else 1
    
    weights = np.where(y == 1, pos_weight * gamma_pos, neg_weight / gamma_neg)
    return weights