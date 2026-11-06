"""
Exercise "VAE" — Artificial Neural Networks & Deep Learning (Insper, 2026.2)

Generates every figure and number used in docs/exercises/vae/index.md.

    python docs/exercises/vae/code/vae.py

Outputs:
    docs/exercises/vae/figures/fig*.png      figures shown in the report
    docs/exercises/vae/figures/results.json  every number quoted in the report
    data/                                    MNIST download (git-ignored)

The VAE-specific parts — reparameterization, the BCE reconstruction term summed over pixels,
the closed-form KL term and sampling from the prior — are written by hand. PyTorch is used for
the layers, autograd and Adam; torchvision only to load MNIST; scipy for norm.ppf; scikit-learn
only for a k-NN used in the analysis.

The sections delimited by "--8<-- [start:...]" / "--8<-- [end:...]" are the snippets
embedded, item by item, in the report.
"""
# --8<-- [start:setup]
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                 # repository root
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)  # same numbers on every run (CPU)

BATCH, EPOCHS, LR, BETA = 128, 20, 1e-3, 1.0
results = {}  # every number reported in the text is stored here


def save(fig, name):
    """Save a figure into figures/ and close it."""
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def r3(a):
    """Round to 3 decimals for the JSON dump (floats or arrays)."""
    return np.round(np.asarray(a, dtype=float), 3).tolist()
# --8<-- [end:setup]


# =============================================================================
# Exercise 1.A — data
# =============================================================================
# --8<-- [start:data]
tf = transforms.ToTensor()  # pixels in [0, 1]
full_train = datasets.MNIST(ROOT / "data", train=True, download=True, transform=tf)
test_set = datasets.MNIST(ROOT / "data", train=False, download=True, transform=tf)

train_set, val_set = random_split(
    full_train, [50_000, 10_000], generator=torch.Generator().manual_seed(42)
)


def to_tensors(ds):
    """Materialize a dataset as (N, 784) float images and (N,) labels (goes through ToTensor)."""
    x, y = next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False)))
    return x.view(len(ds), 784), y


X_train, y_train = to_tensors(train_set)   # 50 000 x 784
X_val, y_val = to_tensors(val_set)         # 10 000 x 784
X_test, y_test = to_tensors(test_set)      # 10 000 x 784 — touched only for final numbers
# --8<-- [end:data]
results["data"] = {
    "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    "pixel_range": [float(X_train.min()), float(X_train.max())],
    "frac_pixels_near_0_or_1": float(((X_train < 0.1) | (X_train > 0.9)).float().mean()),
}


# =============================================================================
# Exercise 1.B — model and loss
# =============================================================================
# --8<-- [start:model]
class VAE(nn.Module):
    """784 -> 512 -> 256 -> (mu, log sigma^2) in R^d ;  d -> 256 -> 512 -> 784 (sigmoid)."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, d)       # mean of q(z|x)
        self.logvar_head = nn.Linear(256, d)   # log sigma^2 of q(z|x): unconstrained, so sigma > 0 always
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),  # x_hat_j = Bernoulli probability of pixel j
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu, logvar, generator=None):
        """z = mu + sigma * eps, eps ~ N(0, I).

        The randomness lives in eps, which does not depend on the parameters, so gradients
        flow through mu and sigma = exp(logvar / 2) by ordinary backprop.
        """
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.shape, generator=generator)
        return mu + sigma * eps

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator)
        return self.decode(z), mu, logvar


def vae_loss_terms(x, x_hat, mu, logvar):
    """Both terms of the negative ELBO, in nats per image (summed over pixels / latent dims,
    averaged over the images of the batch)."""
    B = x.shape[0]
    # Reconstruction: -sum_j [x_j log x_hat_j + (1 - x_j) log(1 - x_hat_j)]  (Bernoulli NLL).
    # reduction="sum" over the 784 pixels, then / B. ("mean" would divide by 784 too = beta ~ 784.)
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / B
    # KL(N(mu, sigma^2) || N(0, I)) in closed form: 1/2 sum_i (mu_i^2 + sigma_i^2 - 1 - log sigma_i^2)
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar) / B
    return recon, kl
# --8<-- [end:model]


# =============================================================================
# Exercise 1.C — training (the same function trains the d = 16 and the d = 2 models)
# =============================================================================
# --8<-- [start:train]
@torch.no_grad()
def evaluate(model, X, batch=1000):
    """Average reconstruction and KL (nats/image) on a fixed set, with z ~ q(z|x).

    A dedicated generator seeded at 0 makes every evaluation use the same eps, and keeps the
    global RNG (used by training) untouched.
    """
    model.eval()
    g = torch.Generator().manual_seed(0)
    rec_sum = kl_sum = 0.0
    for i in range(0, len(X), batch):
        x = X[i:i + batch]
        x_hat, mu, logvar = model(x, generator=g)
        rec, kl = vae_loss_terms(x, x_hat, mu, logvar)
        rec_sum += rec.item() * len(x)
        kl_sum += kl.item() * len(x)
    return rec_sum / len(X), kl_sum / len(X)


def train_vae(d):
    torch.manual_seed(42)                       # fixed seed before building each model
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X_train), batch_size=BATCH, shuffle=True)

    hist = {k: [] for k in ["train_rec", "train_kl", "val_rec", "val_kl"]}
    hist["val_rec_epoch0"], hist["val_kl_epoch0"] = evaluate(model, X_val)  # untrained model
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rec_sum = kl_sum = 0.0
        for (x,) in loader:
            x_hat, mu, logvar = model(x)
            rec, kl = vae_loss_terms(x, x_hat, mu, logvar)
            loss = rec + BETA * kl              # negative ELBO, averaged over the batch
            opt.zero_grad()
            loss.backward()
            opt.step()
            rec_sum += rec.item() * len(x)      # accumulate per-image sums for the epoch average
            kl_sum += kl.item() * len(x)
        hist["train_rec"].append(rec_sum / len(X_train))
        hist["train_kl"].append(kl_sum / len(X_train))
        v_rec, v_kl = evaluate(model, X_val)    # model selection / monitoring: validation only
        hist["val_rec"].append(v_rec)
        hist["val_kl"].append(v_kl)
        print(f"d={d:2d} epoch {epoch:2d}  train rec {hist['train_rec'][-1]:7.2f} "
              f"kl {hist['train_kl'][-1]:6.2f} | val rec {v_rec:7.2f} kl {v_kl:6.2f}")
    return model, hist
# --8<-- [end:train]


t0 = time.time()
vae16, hist16 = train_vae(16)
time16 = time.time() - t0

# Final numbers: the first and only time the test set goes through the model.
# --8<-- [start:test16]
test_rec16, test_kl16 = evaluate(vae16, X_test)
print(f"d=16 TEST  rec {test_rec16:.2f}  kl {test_kl16:.2f}  -ELBO {test_rec16 + test_kl16:.2f}")
# --8<-- [end:test16]
results["ex1"] = {
    "train_seconds": round(time16, 1),
    "test_rec": round(test_rec16, 2), "test_kl": round(test_kl16, 2),
    "test_neg_elbo": round(test_rec16 + test_kl16, 2),
    "val_kl_first_epoch": round(hist16["val_kl"][0], 2),
    "val_kl_last_epoch": round(hist16["val_kl"][-1], 2),
    "val_rec_epoch0": round(hist16["val_rec_epoch0"], 2),
    "val_kl_epoch0": round(hist16["val_kl_epoch0"], 2),
    "history": {k: r3(v) for k, v in hist16.items()},
}

# --- Figure 1: the two loss terms, train and validation ---------------------
epochs = np.arange(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, key, title in [(axes[0], "rec", "Reconstruction term (BCE)"),
                       (axes[1], "kl", "KL term")]:
    ax.plot(epochs, hist16[f"train_{key}"], "o-", ms=3, label="train")
    ax.plot(epochs, hist16[f"val_{key}"], "s-", ms=3, label="validation")
    ax.set_title(f"{title}, d = 16")
    ax.set_xlabel("epoch")
    ax.set_ylabel("nats per image")
    ax.set_xticks(range(0, EPOCHS + 1, 2))
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle("Figure 1 — the two terms of the loss during training")
fig.tight_layout()
save(fig, "fig1_loss_terms.png")


# =============================================================================
# Exercise 1.D — reconstructions and samples
# =============================================================================
# --8<-- [start:fig23]
first_idx = [int((y_test == k).nonzero()[0]) for k in range(10)]  # first test image of each digit
with torch.no_grad():
    vae16.eval()
    mu, _ = vae16.encode(X_test[first_idx])
    recon = vae16.decode(mu)                   # decode the mean, no sampling

    torch.manual_seed(0)
    z_prior = torch.randn(64, 16)              # z ~ N(0, I)
    samples = vae16.decode(z_prior)
# --8<-- [end:fig23]
results["ex1"]["first_test_index_per_digit"] = first_idx

fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
for k in range(10):
    axes[0, k].imshow(X_test[first_idx[k]].view(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[1, k].imshow(recon[k].view(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[0, k].set_title(f"{k} (#{first_idx[k]})", fontsize=9)
    for r in range(2):
        axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("decode(μ)")
fig.supxlabel("digit (test index)")
fig.suptitle("Figure 2 — first test image of each digit and its reconstruction, d = 16")
fig.tight_layout()
save(fig, "fig2_reconstructions.png")

grid = samples.view(8, 8, 28, 28).permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28)
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
ax.set_xticks(np.arange(8) * 28 + 13.5, range(1, 9))
ax.set_yticks(np.arange(8) * 28 + 13.5, range(1, 9))
ax.set_xlabel("sample column")
ax.set_ylabel("sample row")
ax.set_title("Figure 3 — 64 images decoded from z ~ N(0, I), d = 16 (seed 0)")
save(fig, "fig3_prior_samples.png")

# Evidence for 1.E.2(a): do prior samples fall where the encoder puts training data?
with torch.no_grad():
    mu_tr16, logvar_tr16 = vae16.encode(X_train)
# per-dimension variance of the encoder means, and average posterior std
results["ex1"]["mu_var_per_dim"] = r3(mu_tr16.var(0))
results["ex1"]["mean_posterior_std_per_dim"] = r3(torch.exp(0.5 * logvar_tr16).mean(0))
# distance from each prior sample to the nearest training mu, vs. a test mu to the nearest training mu
with torch.no_grad():
    mu_te16, _ = vae16.encode(X_test)
nn_prior = torch.cdist(z_prior, mu_tr16).min(1).values
nn_test = torch.cdist(mu_te16[:1000], mu_tr16).min(1).values
results["ex1"]["nn_dist_prior_to_train_mu_median"] = round(float(nn_prior.median()), 2)
results["ex1"]["nn_dist_test_to_train_mu_median"] = round(float(nn_test.median()), 2)
# evidence for 1.E.2(b): how "grey" are decoded images vs. real ones
real_mid = float(((X_test > 0.1) & (X_test < 0.9)).float().mean())
with torch.no_grad():
    rec_all = vae16.decode(mu_te16)
rec_mid = float(((rec_all > 0.1) & (rec_all < 0.9)).float().mean())
smp_mid = float(((samples > 0.1) & (samples < 0.9)).float().mean())
results["ex1"]["frac_grey_pixels"] = {"real_test": round(real_mid, 4),
                                      "reconstructions": round(rec_mid, 4),
                                      "prior_samples": round(smp_mid, 4)}


# =============================================================================
# Exercise 2.A — the same VAE with d = 2
# =============================================================================
# --8<-- [start:train2]
t0 = time.time()
vae2, hist2 = train_vae(2)                     # same data, seed, beta, optimizer and budget
time2 = time.time() - t0
test_rec2, test_kl2 = evaluate(vae2, X_test)
print(f"d= 2 TEST  rec {test_rec2:.2f}  kl {test_kl2:.2f}  -ELBO {test_rec2 + test_kl2:.2f}")
# --8<-- [end:train2]
results["ex2"] = {
    "train_seconds": round(time2, 1),
    "test_rec": round(test_rec2, 2), "test_kl": round(test_kl2, 2),
    "test_neg_elbo": round(test_rec2 + test_kl2, 2),
    "history": {k: r3(v) for k, v in hist2.items()},
}


# Per-digit test reconstruction (decode(mu), deterministic) for both models — used in 2.E.2
@torch.no_grad()
def per_digit_rec(model):
    mu, _ = model.encode(X_test)
    x_hat = model.decode(mu)
    bce = F.binary_cross_entropy(x_hat, X_test, reduction="none").sum(1)
    return [float(bce[y_test == k].mean()) for k in range(10)]


results["ex2"]["per_digit_rec_decode_mu"] = {"d16": r3(per_digit_rec(vae16)),
                                            "d2": r3(per_digit_rec(vae2))}

# =============================================================================
# Exercise 2.B — the encoded test set
# =============================================================================
with torch.no_grad():
    mu_te2, _ = vae2.encode(X_test)
    mu_tr2, _ = vae2.encode(X_train)
mu_te2_np, y_te_np = mu_te2.numpy(), y_test.numpy()

cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(8, 7))
for k in range(10):
    m = y_te_np == k
    ax.scatter(mu_te2_np[m, 0], mu_te2_np[m, 1], s=3, alpha=0.6, color=cmap(k), label=str(k))
lim = norm.ppf(0.95)
ax.add_patch(plt.Rectangle((-lim, -lim), 2 * lim, 2 * lim, fill=False, ls="--", lw=1.2,
                           color="black", label="Fig. 5 grid range"))
ax.set_xlabel("μ₁")
ax.set_ylabel("μ₂")
ax.set_title("Figure 4 — encoder means μ of the 10 000 test images, d = 2")
ax.legend(title="digit", markerscale=4, loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.set_aspect("equal")
ax.grid(alpha=0.3)
save(fig, "fig4_latent_scatter.png")

# Which digits overlap? k-NN (k = 15) fit on the *training* means, evaluated on the test means.
knn = KNeighborsClassifier(n_neighbors=15).fit(mu_tr2.numpy(), y_train.numpy())
pred = knn.predict(mu_te2_np)
cm = confusion_matrix(y_te_np, pred)
pairs = []
for i in range(10):
    for j in range(i + 1, 10):
        pairs.append((int(cm[i, j] + cm[j, i]), i, j))
pairs.sort(reverse=True)
results["ex2"]["knn15_test_acc_in_2d_latent"] = round(float((pred == y_te_np).mean()), 4)
results["ex2"]["most_confused_pairs"] = [{"pair": [i, j], "errors": n} for n, i, j in pairs[:6]]
results["ex2"]["class_centroids"] = {str(k): r3(mu_te2_np[y_te_np == k].mean(0)) for k in range(10)}

# =============================================================================
# Exercise 2.C — the decoded plane, on a grid of equal prior mass
# =============================================================================
# --8<-- [start:fig5]
q = np.linspace(0.05, 0.95, 15)                # 15 evenly spaced quantiles
zs = norm.ppf(q)                               # z = Phi^{-1}(q): equal prior mass per cell
Z1, Z2 = np.meshgrid(zs, zs[::-1])             # rows: z2 from high (top) to low; cols: z1 low -> high
z_grid = torch.tensor(np.stack([Z1.ravel(), Z2.ravel()], 1), dtype=torch.float32)
with torch.no_grad():
    vae2.eval()
    plane = vae2.decode(z_grid).view(15, 15, 28, 28)
# --8<-- [end:fig5]
results["ex2"]["grid_z_values"] = r3(zs)
tiles = plane.permute(0, 2, 1, 3).reshape(15 * 28, 15 * 28)
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(tiles, cmap="gray", vmin=0, vmax=1)
ax.set_xticks(np.arange(15) * 28 + 13.5, [f"{v:.2f}" for v in zs], rotation=90, fontsize=8)
ax.set_yticks(np.arange(15) * 28 + 13.5, [f"{v:.2f}" for v in zs[::-1]], fontsize=8)
ax.set_xlabel("z₁ = Φ⁻¹(q),  q = 0.05 … 0.95")
ax.set_ylabel("z₂ = Φ⁻¹(q),  q = 0.05 … 0.95")
ax.set_title("Figure 5 — decoded 15 × 15 grid of the latent plane, d = 2")
save(fig, "fig5_latent_plane.png")

# =============================================================================
# Exercise 2.D — interpolation: pixel space vs. latent space (d = 16 model)
# =============================================================================
# --8<-- [start:fig6]
idx_a, idx_b = first_idx[1], first_idx[0]      # first test "1" and first test "0"
x_a, x_b = X_test[idx_a], X_test[idx_b]
t = torch.linspace(0, 1, 10).unsqueeze(1)
pixel_row = (1 - t) * x_a + t * x_b            # straight line in pixel space
with torch.no_grad():
    vae16.eval()
    mu_ab, _ = vae16.encode(torch.stack([x_a, x_b]))
    z_path = (1 - t) * mu_ab[0] + t * mu_ab[1] # straight line in the d = 16 latent space
    latent_row = vae16.decode(z_path)
# --8<-- [end:fig6]
results["ex2"]["interp_indices"] = {"a": idx_a, "label_a": int(y_test[idx_a]),
                                    "b": idx_b, "label_b": int(y_test[idx_b])}

fig, axes = plt.subplots(2, 10, figsize=(12, 2.9))
for c in range(10):
    axes[0, c].imshow(pixel_row[c].view(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[1, c].imshow(latent_row[c].view(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[1, c].set_xlabel(f"t={t[c].item():.2f}", fontsize=8)
    for r in range(2):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
axes[0, 0].set_ylabel("pixel")
axes[1, 0].set_ylabel("latent\n(d = 16)")
fig.suptitle(f"Figure 6 — interpolating test #{idx_a} (digit {int(y_test[idx_a])}) → "
             f"test #{idx_b} (digit {int(y_test[idx_b])})")
fig.supxlabel("interpolation weight t")
fig.tight_layout()
save(fig, "fig6_interpolation.png")


# Evidence for 2.E.3: how far is each interpolated image from the nearest real training image?
# (RMS pixel distance; the two endpoints are references for what a "digit-like" value is.)
def nn_rms(imgs):
    return (torch.cdist(imgs, X_train).min(1).values / np.sqrt(784)).numpy()


results["ex2"]["interp_nn_rms_dist"] = {"pixel": r3(nn_rms(pixel_row)), "latent": r3(nn_rms(latent_row))}
results["ex2"]["interp_ink"] = {"pixel": r3(pixel_row.sum(1)), "latent": r3(latent_row.sum(1))}

with open(FIG_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in results.items()},
                 indent=1))
