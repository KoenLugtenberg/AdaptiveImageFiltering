import torch
import numpy as np
import cv2
from INR import SIREN
import matplotlib.pyplot as plt
from knn_templates import PSNR

image_path = r"C:\Users\lugte\anaconda_projects\ImageFiltering\images\donald_duck_comic.png"

im = cv2.imread(image_path)

sigma_noise = 20

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

# Load checkpoint

checkpoint = torch.load(
    r"C:\Users\lugte\anaconda_projects\ImageFiltering\inr_images\donald_duck_comic_siren.pt",
    map_location="cpu"
)

H = checkpoint["height"]
W = checkpoint["width"]

model = SIREN(
    in_dim=2,
    hidden_dim=checkpoint["hidden_dim"],
    out_dim=checkpoint["out_dim"],
    depth=checkpoint["depth"],
    omega_0=checkpoint.get("omega_0", 30)
)

model.load_state_dict(checkpoint["model_state"])
model.eval()

print("INR loaded.")


# Recreate coordinates

yy, xx = np.meshgrid(
    np.linspace(-1, 1, H),
    np.linspace(-1, 1, W),
    indexing="ij"
)

coords = np.stack([yy, xx], axis=-1)
coords = coords.reshape(-1, 2, order="F")

coords_t = torch.tensor(
    coords,
    dtype=torch.float32
)


# Reconstruct image

with torch.no_grad():
    recon = model(coords_t).numpy()

recon = recon.reshape(H, W, 3, order="F")
recon = np.clip(recon, 0, 1)

# Convert to uint8 for OpenCV
recon_uint8 = (255 * recon).astype(np.uint8)


# Display

cv2.imshow("INR Reconstruction", recon_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r"Images\donald_duck_comic_Reconstruction.png", recon_uint8)
print(f"PSNR of reconstruction = {PSNR(recon_uint8, im_noisy, 255)}")

with torch.no_grad():
    _, Q = model(coords_t, return_features=True)

Q = Q.numpy()
stds = np.std(Q, axis=0)

sorted_idx = np.argsort(stds)[::-1]

Q = Q[:, sorted_idx]

fig, axes = plt.subplots(5, 10, figsize=(10, 8))
k = 0

for i in range(5):
    for j in range(10):
        feature = Q[:, k].reshape(H, W, order="F")
        feature -= feature.min()
        feature /= feature.max() + 1e-8
        axes[i][j].imshow(feature)
        axes[i][j].axis('off')
        k += 1

plt.show()
