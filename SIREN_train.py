from INR import SIREN
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

# ==================================================
# Image
# ==================================================

image_path = r"C:\image_path"

im = cv2.imread(image_path)

sigma_noise = 80

np.random.seed(42)

noise = np.random.normal(
    0,
    sigma_noise,
    im.shape
)

im_noisy = np.clip(
    im + noise,
    0,
    255
).astype(np.uint8)

H, W, C = im_noisy.shape

im_float = im_noisy.astype(np.float32) / 255.0


# ==================================================
# Coordinates
# ==================================================

yy, xx = np.meshgrid(
    np.linspace(-1, 1, H),
    np.linspace(-1, 1, W),
    indexing="ij"
)

coords = np.stack(
    [yy, xx],
    axis=-1
)

coords = coords.reshape(
    -1,
    2,
    order="F"
)

pixels = im_float.reshape(
    -1,
    C,
    order="F"
)


# ==================================================
# Torch
# ==================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

coords_t = torch.tensor(
    coords,
    dtype=torch.float32,
    device=device
)

pixels_t = torch.tensor(
    pixels,
    dtype=torch.float32,
    device=device
)


# ==================================================
# Model
# ==================================================

model = SIREN(
    in_dim=2,
    hidden_dim=128,
    out_dim=C,
    depth=4,
    omega_0=5
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

loss_fn = nn.MSELoss()


# ==================================================
# Training
# ==================================================

epochs = 3000
pbar = tqdm(range(epochs))

for epoch in pbar:

    optimizer.zero_grad()

    pred = model(coords_t)
    loss = loss_fn(pred, pixels_t)

    loss.backward()
    optimizer.step()

    pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.6f}")

# ==================================================
# Restore best model
# ==================================================

checkpoint = {
    "model_state": model.state_dict(),
    "hidden_dim": 128,
    "depth": 4,
    "omega_0": 5,
    "out_dim": C,
    "height": H,
    "width": W,
}

torch.save(
    checkpoint,
    r"C:\Users\lugte\anaconda_projects\ImageFiltering\inr_images\sitting_woman_siren_sigma80.pt"
)

print("Saved SIREN checkpoint.")


# ==================================================
# Reconstruct image
# ==================================================

with torch.no_grad():

    recon = model(coords_t)

recon = (
    recon
    .cpu()
    .numpy()
    .reshape(H, W, C, order="F")
)

recon_uint8 = np.clip(
    recon * 255,
    0,
    255
).astype(np.uint8)

cv2.imwrite("sitting_woman_siren_sigma80", recon_uint8)
print("Saved denoised image.")