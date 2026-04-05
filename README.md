#  Pollution Forecasting

 Team: Femme Coders
 
Competition: ANRF - AISEHack - Phase 2 - Theme 2 - Pollution Forecasting (IIT Delhi)

Platform: Kaggle

# Problem Statement

Forecast PM2.5 concentrations (fine particulate matter, µg/m³) over a spatial grid of 140 × 124 cells for the next 16 timesteps, given the past 10 timesteps of meteorological and emission features.

# Dataset Structure

├── raw/

│   ├── APRIL_16/        # Training — April 2016

│   ├── JULY_16/         # Training — July 2016

│   ├── OCT_16/          # Training — October 2016

│   └── DEC_16/          # Training — December 2016

├── test_in/             # 218 test samples × 10 

└── stats/

   └── feat_min_max.mat

# Feature Used
<img width="748" height="491" alt="image" src="https://github.com/user-attachments/assets/b7af0036-15a0-4ebf-95b3-6e36dfca7a6f" />

# Model Architecture — FastUNet

Input: (B, C=8, L=10, H=140, W=124)
         │
         ▼
LightTemporalEncoder
  ├── Reshape: (B, C×L, H, W)       ← flatten all timesteps into channels
  
  ├── Conv2d projection → hidden=96  ← parallel over all timesteps (no sequential loop)
  
  ├── Depthwise temporal mixing      ← spatial context fusion
  
  └── ECA channel attention          ← efficient channel reweighting
  
         │
         ▼  (B, 96, H, W)
UNet Encoder
  ├── e1: DWSepConv block → (B,  96, H,   W  )
  
  ├── e2: DWSepConv block → (B, 192, H/2, W/2)
  
  └── e3: DWSepConv block → (B, 384, H/4, W/4)
         │
         ▼
Bottleneck

  └── conv_block + ResidualBlock + ECA → (B, 384, H/8, W/8)
  
         │
         ▼
UNet Decoder  (bilinear upsample + skip connections)

  ├── d3: cat(up(bn), e3) → (B, 384, H/4, W/4)
  
  ├── d2: cat(up(d3), e2) → (B, 192, H/2, W/2)
  
  └── d1: cat(up(d2), e1) → (B,  96, H,   W  )
         │
         ▼
Head: Conv2d(96 → 48 → 16)

Output: (B, 16, 140, 124) → transposed to (B, 140, 124, 16)

# Training Pipeline

<img width="774" height="638" alt="image" src="https://github.com/user-attachments/assets/4bd4d0fb-f631-4e5e-aa8e-192f20c9b689" />

# Learning Rate Schedule 

Epochs 1–3:   linear warmup   (LR: 0 → 3e-4)

Epochs 4–13:  cosine annealing (LR: 3e-4 → 1.5e-5)

Epochs 14–25: SWA phase        (constant SWA_LR = 5e-5)

SWA (Stochastic Weight Averaging):
From epoch 14 onward, model weights are averaged across epochs. Batch norm statistics are recomputed on the full training set at the end. This consistently improves generalization over a single best-checkpoint approach.

# Loss Function

loss = weighted_MSE
     + 0.40 × weighted_SMAPE
     + 0.20 × Huber(delta=1.5)
     + 0.15 × spatial_gradient_loss
     + 2.00 × episode_SMAPE

Per-step weights — near-term forecasts are weighted more heavily:

Steps  1–2 : 2.5×    Steps  3–4 : 2.0×    Steps  5–6 : 1.5×

Steps  7–8 : 1.0×    Steps  9–10: 0.8×    Steps 11–12: 0.6×

Steps 13–14: 0.4×    Steps 15–16: 0.3×

Spatial gradient loss — penalises differences in spatial finite differences (x and y directions) between prediction and target. This preserves sharp pollution plume boundaries rather than producing over-smoothed spatial maps.

Episode loss — a rolling-window percentile mask (window=48 timesteps, 85th percentile) flags local pollution extremes. Windows containing flagged timesteps are upweighted 3.5× by the sampler and receive extra SMAPE loss pressure during training.

# Data Augmentation:

Applied during training only:

Horizontal flip (p=0.5)

Vertical flip (p=0.5)

Random timestep dropout (p=0.3) — zeros one random input timestep, forcing the model to be robust to missing or noisy observations

Weighted sampling — the WeightedRandomSampler oversamples high-pollution-episode windows (weight 3.5×) so the model encounters rare extreme events proportionally more often.

# Inference - 4 way TTA
<img width="616" height="274" alt="image" src="https://github.com/user-attachments/assets/42b68eed-f252-43ef-bc77-c3f0eddf2805" />

Final prediction = mean across all 4 augmented forward passes.

   



