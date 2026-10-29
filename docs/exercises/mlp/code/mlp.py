"""
Exercise "MLP" — Artificial Neural Networks & Deep Learning (Insper, 2026.2)

Generates every dataset, figure and number used in docs/exercises/mlp/index.md.

    python docs/exercises/mlp/code/mlp.py

Outputs:
    docs/exercises/mlp/figures/fig*.png      figures shown in the report
    docs/exercises/mlp/figures/results.json  every number quoted in the report

The MLP (activations, losses, forward pass, backward pass, gradient-descent update and
training loop) is written from scratch with NumPy. scikit-learn is used only where the
statement allows it: make_blobs, make_classification, train_test_split, confusion_matrix
and the LogisticRegression baselines.

The sections delimited by "--8<-- [start:...]" / "--8<-- [end:...]" are the snippets
embedded, item by item, in the report.
"""
# --8<-- [start:setup]
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)  # ONE generator, shared by every exercise below

COLORS = ["tab:blue", "tab:orange", "tab:green"]
results = {}  # every number reported in the text is stored here


def save(fig, name):
    """Save a figure into figures/ and close it."""
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def r4(a):
    """Round to 4 decimals for the JSON dump (floats or arrays)."""
    return np.round(np.asarray(a, dtype=float), 4).tolist()
# --8<-- [end:setup]


# =============================================================================
# Exercise 1 — the hand computation, checked with plain NumPy
# =============================================================================

# --8<-- [start:ex1]
x = np.array([0.5, -0.2])
y = 1.0
W1 = np.array([[0.3, -0.1],
               [0.2, 0.4]])       # row i = weights into hidden neuron i
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3


def forward_ex1(W1, b1, W2, b2):
    z1 = W1 @ x + b1              # hidden pre-activation
    h1 = np.tanh(z1)              # hidden activation
    u2 = W2 @ h1 + b2             # output pre-activation
    yhat = np.tanh(u2)            # output
    L = (y - yhat) ** 2           # squared error, N = 1
    return z1, h1, u2, yhat, L


z1, h1, u2, yhat, L = forward_ex1(W1, b1, W2, b2)

# backward pass — chain rule, one factor at a time
dL_dyhat = -2 * (y - yhat)
dL_du2 = dL_dyhat * (1 - yhat ** 2)          # tanh'(u) = 1 - tanh(u)^2
dL_dW2 = dL_du2 * h1
dL_db2 = dL_du2
dL_dh1 = dL_du2 * W2
dL_dz1 = dL_dh1 * (1 - h1 ** 2)
dL_dW1 = np.outer(dL_dz1, x)                 # dL/dW1[i, j] = dL/dz1[i] * x[j]
dL_db1 = dL_dz1

# gradient-descent update of all eight parameters
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2
*_, yhat_new, L_new = forward_ex1(W1_new, b1_new, W2_new, b2_new)
# --8<-- [end:ex1]

results["ex1"] = {
    "z1": r4(z1), "h1": r4(h1), "u2": r4(u2), "yhat": r4(yhat), "L": r4(L),
    "dL_dyhat": r4(dL_dyhat), "tanh_prime_u2": r4(1 - yhat ** 2), "dL_du2": r4(dL_du2),
    "dL_dW2": r4(dL_dW2), "dL_db2": r4(dL_db2), "dL_dh1": r4(dL_dh1),
    "tanh_prime_z1": r4(1 - h1 ** 2), "dL_dz1": r4(dL_dz1),
    "dL_dW1": r4(dL_dW1), "dL_db1": r4(dL_db1),
    "W1_new": r4(W1_new), "b1_new": r4(b1_new), "W2_new": r4(W2_new), "b2_new": r4(b2_new),
    "yhat_new": r4(yhat_new), "L_new": r4(L_new),
}


# =============================================================================
# The MLP — written once here, reused verbatim by Exercises 2, 3 and 4
# =============================================================================

# --8<-- [start:mlp]
class MLP:
    """Fully connected network, from scratch.

    layer_sizes : [n_inputs, hidden_1, ..., hidden_k, n_outputs] — any depth.
    output      : "sigmoid" (binary, 1 output unit, binary cross-entropy) or
                  "softmax" (K output units, categorical cross-entropy).
    rng         : generator used for the weight initialization.

    Hidden layers use tanh. Weights ~ N(0, 1/fan_in) (Xavier/LeCun scale for tanh),
    biases start at 0.
    """

    def __init__(self, layer_sizes, output, rng):
        self.sizes = list(layer_sizes)
        self.output = output
        self.W = [rng.normal(0, np.sqrt(1 / n_in), size=(n_in, n_out))
                  for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in self.sizes[1:]]

    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    # ---- activations of the output layer ---------------------------------
    @staticmethod
    def sigmoid(u):
        return 1 / (1 + np.exp(-u))

    @staticmethod
    def softmax(u):
        e = np.exp(u - u.max(axis=1, keepdims=True))  # shift for numerical stability
        return e / e.sum(axis=1, keepdims=True)

    # ---- forward pass ----------------------------------------------------
    def forward(self, X):
        """X: (N, n_inputs) -> output probabilities (N, n_outputs).
        Keeps every layer's activation in self.a for the backward pass."""
        self.a = [X]
        last = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            u = self.a[-1] @ W + b                      # pre-activation of layer l
            if l < last:
                self.a.append(np.tanh(u))               # hidden layer
            elif self.output == "sigmoid":
                self.a.append(self.sigmoid(u))
            else:
                self.a.append(self.softmax(u))
        return self.a[-1]

    # ---- loss ------------------------------------------------------------
    def loss(self, P, Y):
        """Mean cross-entropy. Y has the same shape as P (0/1 column or one-hot)."""
        P = np.clip(P, 1e-12, 1 - 1e-12)
        if self.output == "sigmoid":
            return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))
        return float(-np.mean(np.sum(Y * np.log(P), axis=1)))

    # ---- backward pass ---------------------------------------------------
    def backward(self, Y):
        """Gradients of the mean loss w.r.t. every W and b (after a forward pass).

        Output layer: for sigmoid + BCE and for softmax + categorical CE the
        derivative w.r.t. the pre-activation is the same expression,
            dL/du_out = (P - Y) / N,
        so one backward pass serves both tasks.
        Hidden layer l (tanh):  delta_l = (delta_{l+1} @ W_{l+1}^T) * (1 - a_l^2).
        """
        N = len(Y)
        delta = (self.a[-1] - Y) / N                       # dL/du at the output
        self.dW = [None] * len(self.W)
        self.db = [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            self.dW[l] = self.a[l].T @ delta               # dL/dW_l
            self.db[l] = delta.sum(axis=0)                 # dL/db_l
            if l > 0:                                      # propagate to layer below
                delta = (delta @ self.W[l].T) * (1 - self.a[l] ** 2)

    # ---- update ----------------------------------------------------------
    def step(self, lr):
        """Plain gradient descent on every parameter."""
        for l in range(len(self.W)):
            self.W[l] -= lr * self.dW[l]
            self.b[l] -= lr * self.db[l]

    # ---- training loop and prediction -------------------------------------
    def fit(self, X, Y, lr, epochs):
        """Full-batch gradient descent; returns the training loss of every epoch."""
        history = []
        for _ in range(epochs):
            P = self.forward(X)
            history.append(self.loss(P, Y))
            self.backward(Y)
            self.step(lr)
        return history

    def predict(self, X):
        P = self.forward(X)
        if self.output == "sigmoid":
            return (P[:, 0] >= 0.5).astype(int)
        return P.argmax(axis=1)


def one_hot(y, k):
    return np.eye(k)[y]


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))
# --8<-- [end:mlp]


# =============================================================================
# Exercise 2 — binary classification a line cannot do
# =============================================================================

# --8<-- [start:ex2a]
centers = [[0, 0], [3, 3], [-3, -3]]
X2, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                   cluster_std=1.2, random_state=42)
y2 = (c > 0).astype(int)      # class 0 = centre blob; class 1 = the two outer blobs
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2,
                                              random_state=42, stratify=y2)
# --8<-- [end:ex2a]

fig, ax = plt.subplots(figsize=(6, 5.5))
for k in (0, 1):
    ax.scatter(*X2[y2 == k].T, s=10, alpha=0.6, color=COLORS[k], label=f"Class {k}")
ax.set(title="Figure 1 — Exercise 2 data (1000 points)", xlabel="$x_1$", ylabel="$x_2$")
ax.set_aspect("equal")
ax.legend()
save(fig, "fig1_data_ex2.png")

# --8<-- [start:ex2b]
baseline2 = LogisticRegression().fit(X2_tr, y2_tr)
acc_lr2 = accuracy(y2_te, baseline2.predict(X2_te))
# --8<-- [end:ex2b]

# --8<-- [start:ex2c]
ARCH2, LR2, EPOCHS2 = [2, 8, 1], 0.5, 1000
mlp2 = MLP(ARCH2, output="sigmoid", rng=rng)
hist2 = mlp2.fit(X2_tr, y2_tr[:, None], lr=LR2, epochs=EPOCHS2)
acc_mlp2 = accuracy(y2_te, mlp2.predict(X2_te))
acc_mlp2_train = accuracy(y2_tr, mlp2.predict(X2_tr))
# --8<-- [end:ex2c]

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(range(1, EPOCHS2 + 1), hist2, color="tab:purple", label=f"MLP {ARCH2}")
ax.set(title="Figure 2 — Exercise 2 training loss", xlabel="Epoch",
       ylabel="Binary cross-entropy (train)")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig2_loss_ex2.png")

# decision regions on a grid covering the test points
pad = 1.0
gx, gy = np.meshgrid(np.linspace(X2_te[:, 0].min() - pad, X2_te[:, 0].max() + pad, 400),
                     np.linspace(X2_te[:, 1].min() - pad, X2_te[:, 1].max() + pad, 400))
grid_pred = mlp2.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
frac_grid_class1 = float(grid_pred.mean())

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.contourf(gx, gy, grid_pred, levels=[-0.5, 0.5, 1.5], colors=COLORS[:2], alpha=0.2)
ax.contour(gx, gy, grid_pred, levels=[0.5], colors="k", linewidths=1)
for k in (0, 1):
    ax.scatter(*X2_te[y2_te == k].T, s=14, color=COLORS[k], edgecolor="k", linewidth=0.3,
               label=f"Class {k} (test)")
wrong = mlp2.predict(X2_te) != y2_te
ax.scatter(*X2_te[wrong].T, s=60, facecolor="none", edgecolor="red", linewidth=1,
           label="misclassified")
ax.plot([], [], color="k", linewidth=1, label="decision boundary")
ax.set(title=f"Figure 3 — MLP decision regions (test acc. {acc_mlp2:.3f})",
       xlabel="$x_1$", ylabel="$x_2$")
ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=8)
save(fig, "fig3_boundary_ex2.png")

# --8<-- [start:ex2d]
# How the logistic baseline splits the test set: fraction predicted as class 1,
# and its accuracy on each class separately.
pred_lr2 = baseline2.predict(X2_te)
frac_lr2_class1 = float(pred_lr2.mean())
lr2_recall = [accuracy(y2_te[y2_te == k], pred_lr2[y2_te == k]) for k in (0, 1)]
# --8<-- [end:ex2d]

results["ex2"] = {
    "n_train": int(len(X2_tr)), "n_test": int(len(X2_te)),
    "class_counts": np.bincount(y2).tolist(),
    "baseline_acc": acc_lr2, "baseline_coef": r4(baseline2.coef_[0]),
    "baseline_intercept": r4(baseline2.intercept_[0]),
    "baseline_frac_pred_class1": frac_lr2_class1, "baseline_recall": lr2_recall,
    "arch": ARCH2, "lr": LR2, "epochs": EPOCHS2, "n_params": mlp2.n_params(),
    "loss_first": hist2[0], "loss_last": hist2[-1],
    "acc_test": acc_mlp2, "acc_train": acc_mlp2_train,
    "n_test_misclassified": int(wrong.sum()), "frac_grid_class1": frac_grid_class1,
}


# =============================================================================
# Exercise 3 — multi-class, same network
# =============================================================================

# --8<-- [start:ex3a]
X3, y3 = make_classification(n_samples=1500, n_features=4, n_informative=4,
                             n_redundant=0, n_repeated=0, n_classes=3,
                             n_clusters_per_class=2, class_sep=1.0, random_state=42)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2,
                                              random_state=42, stratify=y3)
Y3_tr = one_hot(y3_tr, 3)
# --8<-- [end:ex3a]

# --8<-- [start:ex3c]
ARCH3, LR3, EPOCHS3 = [4, 16, 3], 0.5, 2000
mlp3 = MLP(ARCH3, output="softmax", rng=rng)        # same class, softmax output
hist3 = mlp3.fit(X3_tr, Y3_tr, lr=LR3, epochs=EPOCHS3)
pred3 = mlp3.predict(X3_te)
acc_mlp3 = accuracy(y3_te, pred3)
cm3 = confusion_matrix(y3_te, pred3)
# --8<-- [end:ex3c]

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(range(1, EPOCHS3 + 1), hist3, color="tab:purple", label=f"MLP {ARCH3}")
ax.set(title="Figure 4 — Exercise 3 training loss", xlabel="Epoch",
       ylabel="Categorical cross-entropy (train)")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig4_loss_ex3.png")


def plot_confusion(cm, title, name):
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), [f"Class {k}" for k in range(3)])
    ax.set_yticks(range(3), [f"Class {k}" for k in range(3)])
    ax.set(title=title, xlabel="Predicted class", ylabel="True class")
    fig.colorbar(im, ax=ax, label="Number of test points")
    save(fig, name)


plot_confusion(cm3, f"Figure 5 — Confusion matrix, test set (acc. {acc_mlp3:.3f})",
               "fig5_confusion_ex3.png")

# --8<-- [start:ex3d]
# Most-confused pair: largest sum of the two off-diagonal cells (i, j) + (j, i).
pairs = {(i, j): int(cm3[i, j] + cm3[j, i]) for i in range(3) for j in range(i + 1, 3)}
worst_pair = max(pairs, key=pairs.get)

# Logistic-regression baseline on the same split (one linear boundary per class).
baseline3 = LogisticRegression(max_iter=1000).fit(X3_tr, y3_tr)
pred_lr3 = baseline3.predict(X3_te)
acc_lr3 = accuracy(y3_te, pred_lr3)
cm_lr3 = confusion_matrix(y3_te, pred_lr3)
# --8<-- [end:ex3d]

# Structure of the data behind the confusion: each class is 2 Gaussian clusters.
# Without cluster labels, we measure how mixed the classes are locally — for each
# test point, the share of its 15 nearest training points that belong to each class.
d = ((X3_te[:, None, :] - X3_tr[None, :, :]) ** 2).sum(-1)
nn = np.argsort(d, axis=1)[:, :15]
nn_mix = np.zeros((3, 3))
for k in range(3):
    nn_mix[k] = np.bincount(y3_tr[nn[y3_te == k]].ravel(), minlength=3) / nn[y3_te == k].size
centroid_dist = [[float(np.linalg.norm(X3[y3 == i].mean(0) - X3[y3 == j].mean(0)))
                  for j in range(3)] for i in range(3)]

results["ex3"] = {
    "n_train": int(len(X3_tr)), "n_test": int(len(X3_te)),
    "class_counts_test": np.bincount(y3_te).tolist(),
    "arch": ARCH3, "lr": LR3, "epochs": EPOCHS3, "n_params": mlp3.n_params(),
    "loss_first": hist3[0], "loss_last": hist3[-1],
    "acc_test": acc_mlp3, "acc_train": accuracy(y3_tr, mlp3.predict(X3_tr)),
    "confusion": cm3.tolist(), "pair_errors": {f"{i}-{j}": v for (i, j), v in pairs.items()},
    "worst_pair": list(worst_pair),
    "baseline_acc": acc_lr3, "baseline_confusion": cm_lr3.tolist(),
    "nn15_class_mix": r4(nn_mix), "centroid_dist": r4(centroid_dist),
}


# =============================================================================
# Exercise 4 — a second hidden layer, everything else unchanged
# =============================================================================

# --8<-- [start:ex4a]
ARCH4 = [4, 16, 16, 3]                               # only the depth changes
mlp4 = MLP(ARCH4, output="softmax", rng=rng)
hist4 = mlp4.fit(X3_tr, Y3_tr, lr=LR3, epochs=EPOCHS3)  # same data, lr and budget
acc_mlp4 = accuracy(y3_te, mlp4.predict(X3_te))
# --8<-- [end:ex4a]

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(range(1, EPOCHS3 + 1), hist3, color="tab:blue",
        label=f"Ex. 3 — 1 hidden layer {ARCH3}")
ax.plot(range(1, EPOCHS3 + 1), hist4, color="tab:red",
        label=f"Ex. 4 — 2 hidden layers {ARCH4}")
ax.set(title="Figure 6 — Training loss, 1 vs 2 hidden layers", xlabel="Epoch",
       ylabel="Categorical cross-entropy (train)")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig6_loss_compare.png")

# --8<-- [start:ex4b]
# Same comparison repeated with three initialization seeds. Each seed gets its own
# generator so that both architectures see "seed s" independently of run order.
SEEDS = [42, 43, 44]
runs = {"shallow": [], "deep": []}
for s in SEEDS:
    for key, arch in (("shallow", ARCH3), ("deep", ARCH4)):
        m = MLP(arch, output="softmax", rng=np.random.default_rng(s))
        h = m.fit(X3_tr, Y3_tr, lr=LR3, epochs=EPOCHS3)
        runs[key].append({"seed": s, "loss_last": h[-1],
                          "acc_train": accuracy(y3_tr, m.predict(X3_tr)),
                          "acc_test": accuracy(y3_te, m.predict(X3_te))})
summary4 = {key: {f: float(np.mean([r[f] for r in rs]))
                  for f in ("loss_last", "acc_train", "acc_test")}
            for key, rs in runs.items()}
summary4["diff_acc_test"] = summary4["deep"]["acc_test"] - summary4["shallow"]["acc_test"]
# --8<-- [end:ex4b]

results["ex4"] = {
    "arch": ARCH4, "lr": LR3, "epochs": EPOCHS3, "n_params": mlp4.n_params(),
    "n_params_shallow": mlp3.n_params(),
    "loss_last": hist4[-1], "acc_test": acc_mlp4,
    "acc_train": accuracy(y3_tr, mlp4.predict(X3_tr)),
    "seeds": SEEDS, "runs": runs, "mean": summary4,
}

(FIG_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    print(json.dumps(results, indent=2))
