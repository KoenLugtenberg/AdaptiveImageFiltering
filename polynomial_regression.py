import cv2
from math import log10, sqrt
import matplotlib.pyplot as plt
import numpy as np


def PSNR(original, compressed, max_pixel=255):
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return 100
    return 20 * log10(max_pixel / sqrt(mse))


def rgb_to_intensity(img_rgb):
    return (
        0.299 * img_rgb[:, :, 0]
        + 0.587 * img_rgb[:, :, 1]
        + 0.114 * img_rgb[:, :, 2]
    )


def polynomial_design_matrix(X, Y, degree):
    """
    Creates polynomial feature matrix with all terms x^i y^j
    such that i + j <= degree.
    """

    x_flat = X.ravel()
    y_flat = Y.ravel()

    features = []
    powers = []

    for total_degree in range(degree + 1):
        for i in range(total_degree + 1):
            j = total_degree - i
            features.append((x_flat ** i) * (y_flat ** j))
            powers.append((i, j))

    Phi = np.column_stack(features)

    return Phi, powers


def fit_polynomial_channel(channel, X, Y, degree):
    Phi, powers = polynomial_design_matrix(X, Y, degree)

    z_flat = channel.ravel().astype(float)

    beta, *_ = np.linalg.lstsq(Phi, z_flat, rcond=None)

    Z_flat = Phi @ beta
    Z = Z_flat.reshape(channel.shape)

    return Z, beta, powers


def fit_polynomial_rgb(img_noise, X, Y, degree):
    img_filtered_float = np.zeros_like(img_noise, dtype=float)

    for channel in range(3):
        Z_channel, beta, powers = fit_polynomial_channel(
            img_noise[:, :, channel],
            X,
            Y,
            degree
        )

        img_filtered_float[:, :, channel] = Z_channel

    img_filtered = np.clip(img_filtered_float, 0, 255).astype(np.uint8)

    return img_filtered


def plot_surface(X, Y, Z, title, cmap="viridis"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        edgecolor="none",
        alpha=0.9
    )

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("Intensity")
    ax.set_zlim(0, 255)
    ax.view_init(elev=30, azim=45)

    fig.colorbar(surf, shrink=0.6)
    plt.show()


def plot_combined(X, Y, Z_true, Z_filtered, degree):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        Z_true,
        cmap="gray",
        edgecolor="none",
        alpha=0.35
    )

    ax.plot_surface(
        X,
        Y,
        Z_filtered,
        cmap="viridis",
        edgecolor="none",
        alpha=0.95
    )

    ax.set_title(f"True intensity and polynomial fit, degree {degree}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("Intensity")
    ax.set_zlim(0, 255)
    ax.view_init(elev=30, azim=45)

    plt.show()


img_bgr = cv2.imread(r"images\venice_houses.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

sigma_noise = 20
np.random.seed(42)

noise = np.random.normal(
    loc=0,
    scale=sigma_noise,
    size=img_rgb.shape
)

img_noise = np.clip(img_rgb + noise, 0, 255).astype(np.uint8)
n_rows, n_columns, _ = img_rgb.shape

x = np.arange(n_columns)
y = np.arange(n_rows)

X, Y = np.meshgrid(x, y)

X_norm = X / (n_columns - 1)
Y_norm = Y / (n_rows - 1)

intensity_true = rgb_to_intensity(img_rgb)
intensity_noisy = rgb_to_intensity(img_noise)

degrees = [3, 8, 20, 50]

filtered_images = []
psnr_values = []

for degree in degrees:
    print(f"\nFitting polynomial regression of degree {degree}")

    img_filtered = fit_polynomial_rgb(
        img_noise,
        X_norm,
        Y_norm,
        degree
    )

    psnr_filtered = PSNR(img_rgb, img_filtered)

    filtered_images.append(img_filtered)
    psnr_values.append(psnr_filtered)

    print(f"Degree {degree} PSNR: {psnr_filtered:.2f} dB")




# Show true, noisy, and polynomial filtered images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Convert 2x3 array of axes into a flat array
axes = axes.ravel()
axes[0].imshow(img_noise)
axes[0].set_title(f"Noisy image\nPSNR = {psnr_noisy:.2f} dB")
axes[0].axis("off")

for idx, degree in enumerate(degrees):
    axes[idx + 1].imshow(filtered_images[idx])
    axes[idx + 1].set_title(
        f"Degree {degree}\nPSNR = {psnr_values[idx]:.2f} dB"
    )
    axes[idx + 1].axis("off")

axes[5].imshow(img_rgb)
axes[5].set_title("True image\nPSNR = 100")
axes[5].axis("off")

plt.tight_layout()
plt.show()


# Plot PSNR versus polynomial degree
plt.figure(figsize=(7, 5))
plt.plot(degrees, psnr_values, marker="o")
plt.axhline(psnr_noisy, linestyle="--", label="Noisy image PSNR")
plt.xlabel("Polynomial degree")
plt.ylabel("PSNR (dB)")
plt.title("PSNR for polynomial regression filtering")
plt.legend()
plt.grid(True)
plt.show()


# 3D intensity plots for selected degrees
selected_degrees_for_3d = [3, 8, 50]

plot_surface(
    X,
    Y,
    intensity_true,
    "True image intensity",
    cmap="gray"
)

plot_surface(
    X,
    Y,
    intensity_noisy,
    "Noisy image intensity",
    cmap="gray"
)

for degree in selected_degrees_for_3d:
    idx = degrees.index(degree)

    intensity_filtered = rgb_to_intensity(filtered_images[idx])

    plot_surface(
        X,
        Y,
        intensity_filtered,
        f"Filtered intensity, polynomial degree {degree}",
        cmap="viridis"
    )

    plot_combined(
        X,
        Y,
        intensity_true,
        intensity_filtered,
        degree
    )


# Show true and noisy image
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("True image")
axes[0].axis("off")

axes[1].imshow(img_noise)
axes[1].set_title(
    f"Noisy image\nPSNR = {PSNR(img_rgb, img_noise):.2f} dB"
)
axes[1].axis("off")

plt.tight_layout()
plt.show()

# Show true and noisy intensity surfaces
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
plot_surface(
    ax1,
    intensity_true,
    "True image intensity"
)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
plot_surface(
    ax2,
    intensity_noisy,
    "Noisy image intensity"
)

plt.tight_layout()
plt.show()