import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

class AsymmetricLoss(nn.Module):
    """Official ASL implementation adapted for binary classification"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: input logits
        y: targets (binary labels)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class ASLClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, lr=0.001, epochs=300, batch_size=256):
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
    def fit(self, X, y):
        X_array = X.values if hasattr(X, 'values') else X
        X_tensor = torch.FloatTensor(X_array)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
        
        # Enhanced architecture for imbalanced data
        self.model = nn.Sequential(
            nn.Linear(X_tensor.shape[1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        criterion = AsymmetricLoss(self.gamma_neg, self.gamma_pos, self.clip)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        # Training with mini-batches
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if epoch % 50 == 0:
                print(f"[ASL] Epoch {epoch}, Loss: {avg_loss:.4f}")
                
        return self
    
    def predict_proba(self, X):
        X_array = X.values if hasattr(X, 'values') else X
        X_tensor = torch.FloatTensor(X_array)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(logits).numpy()
        return np.column_stack([1-probs, probs])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

def compute_asl_weights(y, gamma_neg=4, gamma_pos=1):
    """Compute sample weights based on ASL principles"""
    pos_weight = len(y) / (2 * np.sum(y))  # Inverse frequency for positive class
    neg_weight = len(y) / (2 * (len(y) - np.sum(y)))  # Inverse frequency for negative class
    
    weights = np.where(y == 1, pos_weight * gamma_pos, neg_weight / gamma_neg)
    return weights