from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import diags, csr_array
from scipy.ndimage import convolve
import seaborn as sns
from math import log10, sqrt
import cv2
import imageio.v2 as imageio
import torch.nn as nn
import torch
import math
import os
import json
import time
import itertools
import traceback
import pandas as pd
from tqdm.auto import tqdm


class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.linear.in_features
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(
        self,
        in_dim=2,
        hidden_dim=128,
        out_dim=3,
        depth=6,
        omega_0=30
    ):
        super().__init__()

        layers = [
            SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0)
        ]

        for _ in range(depth - 1):
            layers.append(
                SineLayer(hidden_dim, hidden_dim, omega_0=omega_0)
            )

        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, out_dim)

        with torch.no_grad():
            bound = math.sqrt(6 / hidden_dim) / omega_0
            self.head.weight.uniform_(-bound, bound)

    def forward(self, x, return_features=False):
        q = self.features(x)
        y = self.head(q)

        if return_features:
            return y, q
        return y


class INR(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, out_dim=3, depth=4):
        super().__init__()

        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim

        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, return_features=False):
        q = self.features(x)
        y = self.head(q)

        if return_features:
            return y, q
        return y


def PSNR(original, compressed, max_pixel):
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)

    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def cartesian(i, shape, row_stack=False):
    r = i % shape[0]
    c = i // shape[0]

    if row_stack:
        r = i % shape[0]
        c = i // shape[0]

    return r, c


def ind(coordinates, shape, row_stack=False):
    if not (0 <= coordinates[0] < shape[0] and 0 <= coordinates[1] < shape[1]):
        coordinates = (max(0, coordinates[0]), max(0, coordinates[1]))
        coordinates = (min(shape[0] - 1, coordinates[0]), min(shape[1] - 1, coordinates[1]))

    i = shape[0] * coordinates[1] + coordinates[0]

    if row_stack:
        i = shape[1] * coordinates[0] + coordinates[1]
    return i


def larkMetric(x_i, x_j):
    xi_arr = x_i[:2]
    xj_arr = x_j[:2]

    C = np.array([[x_i[2], x_i[3]],
                  [x_i[3], x_i[4]]])

    d = xi_arr - xj_arr
    return  d.T @ C @ d


def gaussianW(
    im,
    h_x=0.75,
    n_neighours=10
):
    Y = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:,:,0]
    n_rows, n_columns = Y.shape
    N = n_rows * n_columns

    row_grid, column_grid = np.meshgrid(
        np.arange(n_rows),
        np.arange(n_columns),
        indexing='ij'
    )

    X = np.empty((N, 2), dtype=np.float32)

    X[:, 0] = row_grid.ravel(order='F') / h_x
    X[:, 1] = column_grid.ravel(order='F') / h_x

    nbrs = NearestNeighbors(n_neighbors=n_neighours, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    kernels = np.exp(-distances ** 2)
    kernels /= kernels.sum(axis=1, keepdims=True)

    rows = np.repeat(np.arange(N), n_neighours)
    cols = indices.ravel()
    vals = kernels.ravel()

    W = csr_array((vals, (rows, cols)), shape=(N, N))

    return W


def bilateralW(
    im,
    h_x=3,
    h_y=35,
    n_neighours=20
):
    Y = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:,:,0]
    n_rows, n_columns = Y.shape
    N = n_rows * n_columns
    X = np.empty((N, 3), dtype=np.float32)

    row_grid, column_grid = np.meshgrid(
        np.arange(n_rows),
        np.arange(n_columns),
        indexing='ij'
    )

    X[:, 0] = row_grid.ravel(order='F') / h_x
    X[:, 1] = column_grid.ravel(order='F') / h_x
    X[:, 2] = Y.ravel(order='F') / h_y

    nbrs = NearestNeighbors(n_neighbors=n_neighours, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    kernels = np.exp(-distances ** 2)
    kernels /= kernels.sum(axis=1, keepdims=True)

    rows = np.repeat(np.arange(N), n_neighours)
    cols = indices.ravel()
    vals = kernels.ravel()

    W = csr_array((vals, (rows, cols)), shape=(N, N))

    return W


def nlmW(
    im,
    patch_size=1,
    h_y=35,
    n_neighours=30
):
    n_rows, n_cols, n_channels = im.shape
    N = n_rows * n_cols
    patch_len = 1 + patch_size * 2
    h_y = h_y * np.sqrt(patch_len ** 2)

    padded = np.pad(
        im,
        pad_width=((patch_size, patch_size),
                   (patch_size, patch_size),
                   (0, 0)),
        mode="edge"
    )

    patches = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=(patch_len, patch_len),
        axis=(0, 1)
    )

    patches = np.moveaxis(patches, 2, -1)
    X = patches.reshape(N, patch_len * patch_len * n_channels, order="F") / h_y
    nbrs = NearestNeighbors(n_neighbors=n_neighours, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    kernels = np.exp(-distances ** 2)
    kernels /= kernels.sum(axis=1, keepdims=True)

    rows = np.repeat(np.arange(N), n_neighours)
    cols = indices.ravel()
    vals = kernels.ravel()

    W = csr_array((vals, (rows, cols)), shape=(N, N))

    return W


def larkW(
    im,
    search_radius=2,
    h=100
):
    Y = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:,:,0]
    Y = Y / h
    n_rows, n_columns = Y.shape
    N = n_rows * n_columns
    X = np.empty((N, 5), dtype=np.float32)

    Ix = convolve(Y, [[-1, 0, 1]], mode="reflect")
    Iy = convolve(Y, [[-1], [0], [1]], mode="reflect")

    tensor_kernel = np.ones((3, 3), dtype=np.float32)

    c00 = convolve(Ix * Ix, tensor_kernel, mode="reflect")
    c01 = convolve(Ix * Iy, tensor_kernel, mode="reflect")
    c11 = convolve(Iy * Iy, tensor_kernel, mode="reflect")

    row_grid, column_grid = np.meshgrid(np.arange(n_rows), np.arange(n_columns), indexing="ij")

    X[:, 0] = row_grid.ravel(order='F')
    X[:, 1] = column_grid.ravel(order='F')
    X[:, 2] = c00.ravel(order='F')
    X[:, 3] = c01.ravel(order='F')
    X[:, 4] = c11.ravel(order='F')

    window_area = (2 * search_radius + 1) ** 2

    indices = np.empty((N, window_area), dtype=np.int32)
    distances = np.empty((N, window_area), dtype=np.float32)

    for i in range(N):
        ri = int(X[i, 0])
        ci = int(X[i, 1])

        r0 = ri - search_radius
        r1 = ri + search_radius + 1
        c0 = ci - search_radius
        c1 = ci + search_radius + 1
        k = 0

        for r in range(r0, r1):
            for c in range(c0, c1):
                j = ind((r, c), (n_rows, n_columns))
                indices[i, k] = j
                distances[i, k] = larkMetric(X[i, :], X[j, :])
                k += 1

    kernels = np.exp(-distances)
    kernels /= kernels.sum(axis=1, keepdims=True)

    rows = np.repeat(np.arange(N), window_area)
    cols = indices.ravel()
    vals = kernels.ravel()

    W = csr_array((vals, (rows, cols)), shape=(N, N))

    return W


def inrW(
    checkpoint_path,
    n_neighbours=20,
    temperature=0.5,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    H = checkpoint["height"]
    W_img = checkpoint["width"]
    C = checkpoint["out_dim"]

    model = INR(
        in_dim=2,
        hidden_dim=checkpoint["hidden_dim"],
        out_dim=C,
        depth=checkpoint["depth"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    N = H * W_img

    yy, xx = np.meshgrid(
        np.linspace(-1, 1, H),
        np.linspace(-1, 1, W_img),
        indexing="ij"
    )

    coords = np.stack([yy, xx], axis=-1)
    coords = coords.reshape(N, 2, order="F")

    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    with torch.no_grad():
        _, Q = model(coords_t, return_features=True)

    Q = Q.detach().cpu().numpy().astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbours,
        metric="euclidean",
        algorithm="auto"
    ).fit(Q)

    distances, indices = nbrs.kneighbors(Q)

    Qi = Q[:, None, :]
    Qj = Q[indices]

    logits = np.sum(Qi * Qj, axis=-1) / temperature
    logits -= logits.max(axis=1, keepdims=True)

    kernels = np.exp(logits)
    kernels /= kernels.sum(axis=1, keepdims=True)

    rows = np.repeat(np.arange(N), n_neighbours)
    cols = indices.ravel()
    vals = kernels.ravel()

    W = csr_array(
        (vals, (rows, cols)),
        shape=(N, N)
    )

    return W


def sirenW(
    checkpoint_path,
    n_neighbours=5,
    temperature = 0.01,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    H = checkpoint["height"]
    W_img = checkpoint["width"]
    C = checkpoint["out_dim"]

    model = SIREN(
        in_dim=2,
        hidden_dim=checkpoint["hidden_dim"],
        out_dim=C,
        depth=checkpoint["depth"],
        omega_0=checkpoint["omega_0"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    N = H * W_img

    yy, xx = np.meshgrid(
        np.linspace(-1, 1, H),
        np.linspace(-1, 1, W_img),
        indexing="ij"
    )

    coords = np.stack(
        [yy, xx],
        axis=-1
    )

    coords = coords.reshape(
        N,
        2,
        order="F"
    )

    coords_t = torch.tensor(
        coords,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        _, Q = model(
            coords_t,
            return_features=True
        )

    Q = Q.detach().cpu().numpy().astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbours,
        metric="cosine",
        algorithm="auto"
    ).fit(Q)

    distances, indices = nbrs.kneighbors(Q)

    Qi = Q[:, None, :]
    Qj = Q[indices]

    logits = np.sum(Qi * Qj, axis=-1) / temperature
    logits -= logits.max(axis=1, keepdims=True)

    kernels = np.exp(logits)
    kernels /= kernels.sum(axis=1, keepdims=True)

    rows = np.repeat(np.arange(N), n_neighbours)
    cols = indices.ravel()
    vals = kernels.ravel()

    W = csr_array(
        (vals, (rows, cols)),
        shape=(N, N)
    )

    return W


def applyFilter(
    im,
    W,
    n_times=1
):
    im_vec = im.reshape(-1, order='F')

    for _ in range(n_times):
        im_vec = W @ im_vec

    im_filtered = im_vec.reshape(im.shape, order='F')
    return np.clip(im_filtered, 0, 255).astype(np.uint8)


def applyFilterImg(im, W, n_times=1):
    components = cv2.split(im)

    filtered_components = []

    for component in components:
        filtered = applyFilter(component, W, n_times)
        filtered_components.append(filtered)

    return cv2.merge(filtered_components)


def createGIF(frames, output_path="animation.gif", fps=20):
    with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def grid_search_filter(
    noisy_img,
    true_img,
    W_func,
    param_grid,
    output_dir="grid_results",
    save_images=True,
    resume=True,
    max_pixel=255
):
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "results.csv")
    jsonl_path = os.path.join(output_dir, "results.jsonl")

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    done = set()
    rows = []

    if resume and os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        rows = old.to_dict("records")

        for _, r in old.iterrows():
            sig = tuple(r[k] for k in keys)
            done.add(sig)

        print(f"Resuming: {len(done)} completed runs found.")

    best = None

    pbar = tqdm(combos, desc="Grid search")

    for values in pbar:
        params = dict(zip(keys, values))
        sig = tuple(params[k] for k in keys)

        if resume and sig in done:
            continue

        start = time.time()

        result = {
            **params,
            "status": "failed",
            "psnr": None,
            "runtime_sec": None,
            "image_path": None,
            "error": None,
        }

        try:
            n_times = params["n_times"]

            W_params = {
                k: v
                for k, v in params.items()
                if k != "n_times" and k != "checkpoint_path"
            }

            W = W_func(params["checkpoint_path"], **W_params)
            filtered = applyFilterImg(noisy_img, W, n_times=n_times)

            score = PSNR(true_img, filtered, max_pixel)

            img_path = None
            if save_images:
                name = "_".join([f"{k}-{v}" for k, v in {**params, "n_times": n_times}.items()])
                img_path = os.path.join(img_dir, f"{name}_psnr-{score:.3f}.png")
                cv2.imwrite(img_path, filtered)

            result.update({
                **params,
                "n_times": n_times,
                "status": "ok",
                "psnr": score,
                "runtime_sec": time.time() - start,
                "image_path": img_path,
            })

            if best is None or score > best["psnr"]:
                best = result

            pbar.set_postfix({
                "psnr": f"{score:.3f}",
                "best": f"{best['psnr']:.3f}" if best else None
            })

        except Exception:
            result["runtime_sec"] = time.time() - start
            result["error"] = traceback.format_exc()

        rows.append(result)

        # Save after every run
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        with open(jsonl_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    results = pd.DataFrame(rows)

    # Sort entire dataframe by PSNR
    results = results.sort_values(
        "psnr",
        ascending=False,
        na_position="last"
    )

    # Save final sorted version
    results.to_csv(csv_path, index=False)

    # Print best result
    best_row = results.iloc[0]
    print("\nBest result:")
    print(best_row)

    results_ok = results[results["status"] == "ok"].copy()

    return results_ok


if __name__ == "__main__":
    img_path = r"donald_duck_comic"
    img = cv2.imread(r"images/" + img_path + r".png")

    sigma_noise = 20
    np.random.seed(42)

    noise = np.random.normal(
        loc=0,
        scale=sigma_noise,
        size=img.shape
    )

    img_noise = np.clip(img + noise, 0, 255).astype(np.uint8)
    img_noise = cv2.cvtColor(img_noise, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    param_grid = {
    "checkpoint_path": [
        r"inr_images/donald_duck_comic_siren_epoch_4900.pt"
    ],

    "n_neighbours": [
        5,
        6
    ],

    "temperature": [
        0.001
    ],

    "n_times": [
        1,
        2
    ]
}

    results = grid_search_filter(
        noisy_img=img_noise,
        true_img=img,
        W_func=sirenW,
        param_grid=param_grid,
        output_dir="siren_grid_full_4000_5000"
    )

    print(results.head(10))