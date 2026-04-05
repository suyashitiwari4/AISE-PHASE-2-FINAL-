import torch
import torch.nn as nn
import numpy as np

# ── CONFIG ──
BATCH_SIZE = 4
CHANNELS = 8
LOOKBACK = 10
FORECAST = 16
H, W = 64, 64
EPOCHS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── DUMMY DATA ──
# Random data (simulates your pollution data)
X = torch.randn(50, CHANNELS, LOOKBACK, H, W)
y = torch.randn(50, FORECAST, H, W)

# ── SIMPLE MODEL ──
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Conv2d(16, FORECAST, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L, H, W)
        x = self.conv(x)              # (B, 16, L, H, W)
        x = x.mean(dim=2)             # collapse time dimension
        x = self.fc(x)                # (B, FORECAST, H, W)
        return x

model = SimpleModel().to(DEVICE)

# ── LOSS + OPTIMIZER ──
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── TRAINING LOOP ──
print("Training started...\n")

for epoch in range(EPOCHS):
    total_loss = 0

    for i in range(0, len(X), BATCH_SIZE):
        xb = X[i:i+BATCH_SIZE].to(DEVICE)
        yb = y[i:i+BATCH_SIZE].to(DEVICE)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

print("\nTraining completed ✅")
