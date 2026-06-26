from scipy.sparse.linalg import eigs, svds, norm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from knn_templates import gaussianW, bilateralW, nlmW, larkW, inrW, sirenW, PSNR, applyFilter, applyFilterImg, createGIF

# Data Choice and Prep

img_path = r"donald_duck_comic"
img = cv2.imread(r"images/" + img_path + r".png")
VMIN = 0; VMAX = 255

sigma_noise = 20
np.random.seed(42)

noise = np.random.normal(
    loc=0,
    scale=sigma_noise,
    size=img.shape
)

img_noise = np.clip(img + noise, VMIN, VMAX).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_noise = cv2.cvtColor(img_noise, cv2.COLOR_BGR2RGB)
size = img.shape[:2]

# Filter Creation
W_dict = {"Gaussian Blur": gaussianW(img_noise),
          "Bilateral": bilateralW(img_noise),
          "NLM": nlmW(img_noise),
          "LARK": larkW(img_noise),
          "INFK": sirenW(r"inr_images/" + img_path + r"_siren.pt")}

def leading_eigenvalues(W, k=100):
    vals = eigs(
        W,
        k=k,
        which='LM',
        return_eigenvectors=False
    )

    vals = np.sort(vals)[::-1]
    return vals


def show_eigenvalues(W_dict):
    plt.figure(figsize=(8, 5))

    for name, W in W_dict.items():

        vals = leading_eigenvalues(W, k=100)

        # For non-symmetric matrices from eigs()
        vals = np.sort(np.abs(vals))[::-1]

        plt.plot(
            np.arange(1, len(vals) + 1),
            vals,
            label=name,
            linewidth=2
        )

    plt.xlabel("Eigenvalue Index")
    plt.ylabel(r"$|\lambda|$")
    plt.title("Leading Eigenvalues of Denoising Operators")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_traces(W_dict):
    for name, W in W_dict.items():
        trace = W.diagonal().sum()
        print(f"Average eigenvalue for {name} = {trace / 150**2}")


def show_eigenvectors(W, image_shape, k=4):
    vals, vecs = eigs(W, k=k, which='LM')

    idx = np.argsort(np.abs(vals))[::-1]

    vals = vals[idx]
    vecs = vecs[:, idx]

    fig, axes = plt.subplots(1, k, figsize=(4*k, 4))

    if k == 1:
        axes = [axes]

    for i in range(k):

        v = np.real(vecs[:, i])

        # normalize for visualization
        v = (v - v.min()) / (v.max() - v.min() + 1e-12)

        axes[i].imshow(
            v.reshape(image_shape, order="F"),
            cmap='gray'
        )

        axes[i].set_title(
            f"$\\lambda$={vals[i].real:.4f}"
        )

        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


for name, W in W_dict.items():
    show_eigenvectors(
        W,
        image_shape=img.shape[:2],
        k=4
    )