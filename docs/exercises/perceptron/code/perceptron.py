"""
Exercise "Perceptron" — Artificial Neural Networks & Deep Learning (Insper, 2026.2)

Generates every dataset, figure and number used in docs/exercises/perceptron/index.md.

    python docs/exercises/perceptron/code/perceptron.py

Outputs:
    docs/exercises/perceptron/figures/fig*.png      figures shown in the report
    docs/exercises/perceptron/figures/results.json  every number quoted in the report

The perceptron (activation, prediction, update rule and training loop) is written
from scratch with NumPy only; no third-party model is used anywhere in this file.

The sections delimited by "--8<-- [start:...]" / "--8<-- [end:...]" are the
snippets embedded, item by item, in the report.
"""
# --8<-- [start:setup]
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)  # ONE generator, shared by every exercise below

COLORS = {0: "tab:blue", 1: "tab:orange"}
results = {}  # every number reported in the text is stored here


def save(fig, name):
    """Save a figure into figures/ and close it."""
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_classes(mean0, mean1, cov, n=1000):
    """Two Gaussian classes of `n` 2D points each, stacked as class 0 then class 1."""
    X0 = rng.multivariate_normal(mean0, cov, size=n)
    X1 = rng.multivariate_normal(mean1, cov, size=n)
    X = np.vstack([X0, X1])
    y = np.repeat([0, 1], n)
    return X, y
# --8<-- [end:setup]


# =============================================================================
# The perceptron — written once here, reused unchanged by both exercises
# =============================================================================

# --8<-- [start:perceptron]
def step(z):
    """Binary step activation: 1 if z >= 0, else 0."""
    return np.where(z >= 0, 1, 0)


def predict(X, w, b):
    """Forward pass for one point or for a whole matrix of points."""
    return step(X @ w + b)


def accuracy(X, y, w, b):
    """Fraction of the full dataset classified correctly by (w, b)."""
    return float(np.mean(predict(X, w, b) == y))


def train(X, y, eta, max_epochs=100, pocket=False, w0=None):
    """Train a single-layer perceptron by the error-driven rule, from scratch.

    w <- w + eta * (y - yhat) * x        b <- b + eta * (y - yhat)

    With 0/1 labels the error (y - yhat) is 0 on a correct prediction (no update)
    and +1 / -1 on the two kinds of mistake.

    Initialization: w ~ N(0, 0.01^2), b = 0 (never the all-zero start; see item D).
    `w0` reuses an initialization already drawn, so that two runs can differ in
    eta and in nothing else.
    Stopping: a full pass with no update, or `max_epochs` epochs.

    With pocket=True the best-so-far (w, b) — the one with the highest accuracy on
    the full dataset ever observed right after an update — is copied and kept.
    That copy is the only thing the pocket adds to the loop.
    """
    w = rng.normal(0, 0.01, size=2) if w0 is None else w0.copy()  # non-zero start
    b = 0.0
    w0 = w.copy()

    acc_hist, upd_hist, pocket_hist = [], [], []
    best_acc, best_w, best_b, best_epoch = accuracy(X, y, w, b), w.copy(), b, 0

    for epoch in range(1, max_epochs + 1):
        updates = 0
        for i in range(len(X)):
            yhat = step(X[i] @ w + b)      # prediction for this single sample
            error = y[i] - yhat            # 0, +1 or -1
            if error != 0:
                w = w + eta * error * X[i]  # weight update
                b = b + eta * error         # bias update
                updates += 1
                if pocket:                  # best-so-far bookkeeping
                    a = accuracy(X, y, w, b)
                    if a > best_acc:
                        best_acc, best_w, best_b, best_epoch = a, w.copy(), b, epoch

        acc_hist.append(accuracy(X, y, w, b))  # accuracy after every epoch
        upd_hist.append(updates)
        pocket_hist.append(best_acc)
        if updates == 0:                       # a clean pass: converged
            break

    return {
        "w0": w0, "w": w, "b": b, "epochs": len(acc_hist),
        "acc": acc_hist[-1], "acc_hist": acc_hist, "upd_hist": upd_hist,
        "pocket_w": best_w, "pocket_b": best_b, "pocket_acc": best_acc,
        "pocket_epoch": best_epoch, "pocket_hist": pocket_hist,
    }
# --8<-- [end:perceptron]


# --8<-- [start:plothelpers]
def scatter_classes(ax, X, y, alpha=0.6):
    """Scatter the two classes, one colour each."""
    for k in (0, 1):
        ax.scatter(*X[y == k].T, s=10, alpha=alpha, color=COLORS[k], label=f"Class {k}")


def draw_boundary(ax, w, b, xlim, **kwargs):
    """Draw the line w . x + b = 0 across the current x range."""
    xs = np.linspace(*xlim, 200)
    if abs(w[1]) < 1e-12:                      # vertical boundary
        ax.axvline(-b / w[0], **kwargs)
    else:
        ax.plot(xs, -(w[0] * xs + b) / w[1], **kwargs)


def mark_errors(ax, X, y, w, b, label="Misclassified"):
    """Circle the points the boundary gets wrong; returns how many there are."""
    bad = predict(X, w, b) != y
    ax.scatter(*X[bad].T, s=14, marker="x", color="black", alpha=0.55,
               linewidths=0.7, label=f"{label} ({bad.sum()})")
    return int(bad.sum())
# --8<-- [end:plothelpers]


# =============================================================================
# Exercise 1 — Separable data
# =============================================================================

# --8<-- [start:ex1a]
COV_SEP = [[0.5, 0.0], [0.0, 0.5]]
X1, y1 = make_classes(mean0=[1.5, 1.5], mean1=[5.0, 5.0], cov=COV_SEP, n=1000)

fig, ax = plt.subplots(figsize=(6, 5.5))
scatter_classes(ax, X1, y1)
ax.set(title="Figure 1 — Separable data: 1000 points per class",
       xlabel="$x_1$", ylabel="$x_2$")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
save(fig, "fig1_separable.png")
# --8<-- [end:ex1a]

# --8<-- [start:ex1c]
run1 = train(X1, y1, eta=0.01)          # the eta = 0.01 run asked for in item B
# --8<-- [end:ex1c]

XLIM1 = (X1[:, 0].min() - 0.5, X1[:, 0].max() + 0.5)
YLIM1 = (X1[:, 1].min() - 0.5, X1[:, 1].max() + 0.5)

fig, ax = plt.subplots(figsize=(6, 5.5))
scatter_classes(ax, X1, y1)
draw_boundary(ax, run1["w"], run1["b"], XLIM1, color="black", lw=2,
              label=r"$\mathbf{w}\cdot\mathbf{x}+b=0$")
n_bad1 = mark_errors(ax, X1, y1, run1["w"], run1["b"])
ax.set(title=f"Figure 2 — Decision boundary after {run1['epochs']} epochs "
             f"(accuracy {run1['acc']:.4f})",
       xlabel="$x_1$", ylabel="$x_2$", xlim=XLIM1, ylim=YLIM1)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
save(fig, "fig2_boundary.png")

fig, ax = plt.subplots(figsize=(6.5, 4))
epochs1 = np.arange(1, run1["epochs"] + 1)
ax.plot(epochs1, run1["acc_hist"], marker="o", ms=4, color="tab:blue",
        label=r"Accuracy ($\eta = 0.01$)")
ax.axhline(1.0, ls="--", lw=1, color="grey", label="100%")
ax.set(title="Figure 3 — Accuracy per epoch, separable data",
       xlabel="Epoch", ylabel="Accuracy on the full dataset", ylim=(0.45, 1.02))
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
save(fig, "fig3_accuracy.png")

# --8<-- [start:ex1d]
# Same data, same code, same initial weights: only eta changes.
run1_fast = train(X1, y1, eta=1.0, w0=run1["w0"])

def unit(v):
    """Direction of a weight vector: w / ||w||."""
    return v / np.linalg.norm(v)

u_slow, u_fast = unit(run1["w"]), unit(run1_fast["w"])
cos = float(np.clip(u_slow @ u_fast, -1, 1))
angle_deg = float(np.degrees(np.arccos(cos)))

# The perceptron convergence theorem bounds the total number of mistakes by
# (R / gamma)^2, with R = max ||x|| and gamma the margin of a separating
# hyperplane. Both exist here; gamma is measured on the boundary actually found.
R1 = float(np.max(np.linalg.norm(X1, axis=1)))
# A margin that does not depend on the boundary the run happened to stop at:
# project the data on the direction joining the means and halve the empty gap.
direction = unit(np.array([5.0, 5.0]) - np.array([1.5, 1.5]))
proj = X1 @ direction
gap = float(proj[y1 == 1].min() - proj[y1 == 0].max())   # > 0 iff separable
gamma1 = gap / 2
mistake_bound = (R1 / gamma1) ** 2
# --8<-- [end:ex1d]

results["ex1"] = {
    "w0": run1["w0"].tolist(),
    "w": run1["w"].tolist(), "b": run1["b"], "epochs": run1["epochs"],
    "acc": run1["acc"], "n_misclassified": n_bad1,
    "acc_hist": run1["acc_hist"], "upd_hist": run1["upd_hist"],
    "eta1": {
        "w0": run1_fast["w0"].tolist(),
        "w": run1_fast["w"].tolist(), "b": run1_fast["b"],
        "epochs": run1_fast["epochs"], "acc": run1_fast["acc"],
        "upd_hist": run1_fast["upd_hist"],
    },
    "unit_w_eta001": u_slow.tolist(), "unit_w_eta1": u_fast.tolist(),
    "cos_between": cos, "angle_deg": angle_deg,
    "total_updates": int(sum(run1["upd_hist"])),
    "total_updates_eta1": int(sum(run1_fast["upd_hist"])),
    "R": R1, "gamma": gamma1, "mistake_bound": mistake_bound,
    "norm_w_eta001": float(np.linalg.norm(run1["w"])),
    "norm_w_eta1": float(np.linalg.norm(run1_fast["w"])),
    "offset_eta001": float(-run1["b"] / np.linalg.norm(run1["w"])),
    "offset_eta1": float(-run1_fast["b"] / np.linalg.norm(run1_fast["w"])),
}


# =============================================================================
# Exercise 2 — Overlapping data
# =============================================================================

# --8<-- [start:ex2a]
COV_OVL = [[1.5, 0.0], [0.0, 1.5]]
X2, y2 = make_classes(mean0=[3.0, 3.0], mean1=[4.0, 4.0], cov=COV_OVL, n=1000)

fig, ax = plt.subplots(figsize=(6, 5.5))
scatter_classes(ax, X2, y2, alpha=0.5)
ax.set(title="Figure 4 — Overlapping data: 1000 points per class",
       xlabel="$x_1$", ylabel="$x_2$")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
save(fig, "fig4_overlap.png")
# --8<-- [end:ex2a]

# --8<-- [start:ex2b]
run2 = train(X2, y2, eta=0.01, pocket=True)   # same implementation, pocket on
# --8<-- [end:ex2b]

XLIM2 = (X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5)
YLIM2 = (X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5)

# Two panels so the 997 misclassified points of the final boundary stay readable;
# both boundaries are drawn in both panels, the panel's own one highlighted.
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
panels = [(axes[0], run2["w"], run2["b"], "Final", "black"),
          (axes[1], run2["pocket_w"], run2["pocket_b"], "Pocket", "tab:red")]
n_bad = {}
for ax, w, b, name, colour in panels:
    scatter_classes(ax, X2, y2, alpha=0.3)
    n_bad[name] = mark_errors(ax, X2, y2, w, b, label=f"{name} misses")
    for ax2, w2_, b2_, name2, colour2 in panels:   # both lines, in both panels
        draw_boundary(ax, w2_, b2_, XLIM2, color=colour2, lw=2.2 if name2 == name else 1.2,
                      ls="-" if name2 == name else ":",
                      label=f"{name2} boundary", zorder=4)
    ax.set(title=f"{name} weights — accuracy "
                 f"{(run2['acc'] if name == 'Final' else run2['pocket_acc']):.4f}",
           xlabel="$x_1$", xlim=XLIM2, ylim=YLIM2)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("$x_2$")
fig.suptitle("Figure 5 — Final vs pocket decision boundary, overlapping data")
n_bad_final = n_bad["Final"]
save(fig, "fig5_boundaries.png")

fig, ax = plt.subplots(figsize=(6.5, 4))
epochs2 = np.arange(1, run2["epochs"] + 1)
ax.plot(epochs2, run2["acc_hist"], color="tab:blue", lw=1.2, label="Current weights")
ax.plot(epochs2, run2["pocket_hist"], color="tab:red", lw=2, label="Pocket (best so far)")
ax.axhline(0.5, ls=":", lw=1, color="grey", label="Chance (50%)")
ax.set(title="Figure 6 — Current vs pocket accuracy per epoch, overlapping data",
       xlabel="Epoch", ylabel="Accuracy on the full dataset", ylim=(0.3, 0.85))
ax.legend(loc="center right")
ax.grid(alpha=0.3)
save(fig, "fig6_pocket_curves.png")

# --8<-- [start:ex2d]
# How far the final boundary sits from the data, in the units of the plot:
#   distance from a point p to the line  w . x + b = 0  is  |w . p + b| / ||w||.
norm_w2 = float(np.linalg.norm(run2["w"]))
dist_origin = float(-run2["b"] / norm_w2)               # signed offset of the line
dist_cloud = float((run2["w"] @ np.array([3.5, 3.5]) + run2["b"]) / norm_w2)
mean_norm_x = float(np.mean(np.linalg.norm(X2, axis=1)))  # ||x|| ~ 5 for this data

# Reference: the best line for two Gaussians with equal covariance is the
# perpendicular bisector of the means, x1 + x2 = 7 (computed, not fitted).
bayes_w, bayes_b = np.array([1.0, 1.0]), -7.0
bayes_acc = accuracy(X2, y2, bayes_w, bayes_b)

# Per mistake the bias moves by eta and the weights by eta*||x||, so the offset
# |b| / ||w|| a run of aligned mistakes can build is at most about 1 / ||x||.
max_offset_aligned = 1.0 / mean_norm_x
# --8<-- [end:ex2d]

results["ex2"] = {
    "w": run2["w"].tolist(), "b": run2["b"], "acc": run2["acc"],
    "n_misclassified_final": n_bad_final,
    "pocket_w": run2["pocket_w"].tolist(), "pocket_b": run2["pocket_b"],
    "pocket_acc": run2["pocket_acc"], "pocket_epoch": run2["pocket_epoch"],
    "epochs": run2["epochs"], "acc_hist": run2["acc_hist"],
    "pocket_hist": run2["pocket_hist"], "upd_hist": run2["upd_hist"],
    "total_updates": int(sum(run2["upd_hist"])),
    "norm_pocket_w": float(np.linalg.norm(run2["pocket_w"])),
    "pocket_offset": float(-run2["pocket_b"] / np.linalg.norm(run2["pocket_w"])),
    "max_offset_aligned": max_offset_aligned,
    "norm_w": norm_w2, "line_offset_from_origin": dist_origin,
    "signed_dist_cloud_centre": dist_cloud, "mean_norm_x": mean_norm_x,
    "frac_predicted_class1": float(np.mean(predict(X2, run2["w"], run2["b"]) == 1)),
    "bayes_acc": bayes_acc,
}

(FIG_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "hist" not in kk}
                      for k, v in results.items()}, indent=2))
