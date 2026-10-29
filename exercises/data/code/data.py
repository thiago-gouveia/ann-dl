"""
Exercise "Data" — Artificial Neural Networks & Deep Learning (Insper, 2026.2)

Generates every dataset, figure and number used in docs/exercises/data/index.md.

    python docs/exercises/data/code/data.py

Outputs:
    docs/exercises/data/figures/fig*.png      figures shown in the report
    docs/exercises/data/figures/results.json  every number quoted in the report

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
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
DATA_CSV = HERE.parent / "data" / "train.csv"  # Kaggle "Spaceship Titanic" train.csv
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)  # ONE generator, shared by every exercise below

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
results = {}  # every number reported in the text is stored here


def save(fig, name):
    """Save a figure into figures/ and close it."""
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
# --8<-- [end:setup]


# =============================================================================
# Exercise 1 — Point clouds: geometry and spread in 2D
# =============================================================================

# --8<-- [start:ex1a]
MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100


def make_clouds(scale=1.0):
    """4 Gaussian clouds, 100 points each; every std is multiplied by `scale`.

    Each coordinate is drawn independently (diagonal covariance), so the std
    vector [sx, sy] of a class is exactly the spread along x and along y.
    """
    X = np.vstack([rng.normal(MEANS[k], STDS[k] * scale, size=(N_PER_CLASS, 2))
                   for k in range(4)])
    y = np.repeat(np.arange(4), N_PER_CLASS)
    return X, y


X1, y1 = make_clouds(1.0)


def plot_clouds(ax, X, y, title):
    """Scatter one colour per class and mark each class mean with an X."""
    for k in range(4):
        ax.scatter(*X[y == k].T, s=12, alpha=0.6, color=COLORS[k], label=f"Class {k}")
        ax.scatter(*MEANS[k], marker="X", s=180, color=COLORS[k],
                   edgecolor="black", linewidth=1.2, zorder=5)
    ax.scatter([], [], marker="X", s=100, color="white", edgecolor="black", label="Class mean")
    ax.set(title=title, xlabel="$x_1$", ylabel="$x_2$")


fig, ax = plt.subplots(figsize=(8, 5.5))
plot_clouds(ax, X1, y1, "Figure 1 — Four Gaussian clouds (s = 1, 100 points per class)")
ax.legend(loc="upper left", fontsize=8)
save(fig, "fig1_clouds.png")
# --8<-- [end:ex1a]

# --8<-- [start:ex1b_generate]
SCALES = [0.5, 1.0, 2.0, 4.0]
# 4 datasets x 4 classes: same means, stds multiplied by s
spread = {s: make_clouds(s) for s in SCALES}

# shared limits over all four datasets -> honest visual comparison
all_pts = np.vstack([X for X, _ in spread.values()])
pad = 1.0
xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), SCALES):
    X, y = spread[s]
    plot_clouds(ax, X, y, f"s = {s}")
    ax.set(xlim=xlim, ylim=ylim)
axes[0, 0].legend(loc="upper right", fontsize=8)
fig.suptitle("Figure 2 — Same four classes, standard deviations multiplied by s", fontsize=13)
save(fig, "fig2_spread.png")
# --8<-- [end:ex1b_generate]

# --8<-- [start:ex1b_ratio]
# separation ratio r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j), at s = 1
sigma_bar = STDS.mean(axis=1)  # (sigma_x + sigma_y) / 2 per class
pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
ratios = {}
for i, j in pairs:
    dist = np.linalg.norm(MEANS[i] - MEANS[j])
    ratios[(i, j)] = dist / (sigma_bar[i] + sigma_bar[j])
    print(f"r_{i}{j}: distance = {dist:.3f}  sigma_bar sum = {sigma_bar[i] + sigma_bar[j]:.2f}"
          f"  r = {ratios[(i, j)]:.3f}")

min_pair = min(ratios, key=ratios.get)
r_min = ratios[min_pair]
# means are fixed, every sigma scales with s  ->  r_ij(s) = r_ij(1) / s
print(f"smallest: r_{min_pair[0]}{min_pair[1]} = {r_min:.3f} at s=1 -> {r_min / 2:.3f} at s=2")
# --8<-- [end:ex1b_ratio]

# --8<-- [start:ex1b_mixing]
def mixing_rate(X, y):
    """Fraction of points whose nearest class mean is not their own class mean."""
    d = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2)  # (n_points, 4)
    return float(np.mean(d.argmin(axis=1) != y))


mixing = {s: mixing_rate(*spread[s]) for s in SCALES}
for s in SCALES:
    print(f"s = {s}: mixing rate = {mixing[s]:.4f}  smallest r = {r_min / s:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(SCALES, [100 * mixing[s] for s in SCALES], "o-", color="tab:purple",
        label="Mixing rate (nearest mean is wrong)")
for s in SCALES:
    ax.annotate(f"{100 * mixing[s]:.2f}%", (s, 100 * mixing[s]),
                textcoords="offset points", xytext=(0, 8), ha="center")
ax.set_xscale("log", base=2)
ax.set_xticks(SCALES, [str(s) for s in SCALES])
ax.set(xlabel="scale factor s (log scale)", ylabel="mixing rate (%)",
       title="Figure 3 — Mixing rate vs. spread", ylim=(-3, 50))
ax2 = ax.twinx()  # the smallest separation ratio on the same s axis
ax2.plot(SCALES, [r_min / s for s in SCALES], "s--", color="gray",
         label=f"smallest $r_{{ij}}$ (pair {min_pair[0]}-{min_pair[1]}) = {r_min:.2f}/s")
ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
ax2.set_ylabel("smallest separation ratio $r_{ij}$")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
save(fig, "fig3_mixing.png")

# pair-wise view of where the mixing happens (true class -> nearest mean)
for s in SCALES:
    X, y = spread[s]
    nearest = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2).argmin(axis=1)
    confusion = pd.crosstab(pd.Series(y, name="true"), pd.Series(nearest, name="nearest mean"))
    print(f"\ns = {s}\n{confusion}")


def linearly_separable(A, B, n_angles=36000):
    """Geometric test in 2D (nothing is trained): A and B can be split by a
    straight line iff, along some direction w, every projection of A lies
    below every projection of B. Directions are scanned every 0.01 degree."""
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    W = np.stack([np.cos(theta), np.sin(theta)])  # (2, n_angles)
    return bool(np.any((B @ W).min(axis=0) > (A @ W).max(axis=0)))


separable = {}
for s in SCALES:
    X, y = spread[s]
    separable[s] = {f"{i}-{j}": linearly_separable(X[y == i], X[y == j]) for i, j in pairs}
    print(f"s = {s}: pairs NOT separable by a line:",
          [p for p, ok in separable[s].items() if not ok] or "none")
# --8<-- [end:ex1b_mixing]

# --8<-- [start:ex1c]
# "Sketch" of the boundaries a well-trained network would approach: the
# Bayes-optimal regions computed from the TRUE generating Gaussians (equal
# priors). Nothing is fitted - it only uses the parameters of item A.
def log_density(grid, k, scale=1.0):
    mu, sd = MEANS[k], STDS[k] * scale
    return -np.sum(np.log(sd)) - 0.5 * np.sum(((grid - mu) / sd) ** 2, axis=-1)


gx, gy = np.meshgrid(np.linspace(-4, 19, 700), np.linspace(-6, 13, 700))
grid = np.stack([gx, gy], axis=-1)
bayes = np.argmax(np.stack([log_density(grid, k) for k in range(4)]), axis=0)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.contourf(gx, gy, bayes, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=COLORS, alpha=0.12)
ax.contour(gx, gy, bayes, levels=[0.5, 1.5, 2.5], colors="black", linewidths=1.4)
plot_clouds(ax, X1, y1, "Figure 1 (annotated) — sketched decision boundaries")
ax.plot([], [], color="black", label="Sketched boundary")
ax.set(xlim=(-1, 18), ylim=(-3, 12))
ax.legend(loc="upper left", fontsize=8)
save(fig, "fig1_boundaries.png")

# Share of points that fall on the wrong side even of these optimal boundaries
bayes_err = {}
for s in SCALES:
    X, y = spread[s]
    pred = np.argmax(np.stack([log_density(X, k, s) for k in range(4)]), axis=0)
    bayes_err[s] = float(np.mean(pred != y))
    print(f"s = {s}: points on the wrong side of the optimal boundaries = {bayes_err[s]:.4f}")
# --8<-- [end:ex1c]

results["ex1"] = {
    "sigma_bar": sigma_bar.round(4).tolist(),
    "ratios_s1": {f"{i}-{j}": round(float(r), 4) for (i, j), r in ratios.items()},
    "smallest_pair": f"{min_pair[0]}-{min_pair[1]}",
    "smallest_r_s1": round(float(r_min), 4),
    "smallest_r_s2": round(float(r_min / 2), 4),
    "mixing_rate": {str(s): mixing[s] for s in SCALES},
    "bayes_error": {str(s): bayes_err[s] for s in SCALES},
}


# =============================================================================
# Exercise 2 — Non-linearity in higher dimensions
# =============================================================================

# --8<-- [start:ex2a]
N_5D = 500
MU_A = np.zeros(5)
SIGMA_A = np.array([[1.0, 0.8, 0.1, 0.0, 0.0],
                    [0.8, 1.0, 0.3, 0.0, 0.0],
                    [0.1, 0.3, 1.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5, 1.0, 0.2],
                    [0.0, 0.0, 0.0, 0.2, 1.0]])
MU_B = np.full(5, 1.5)
SIGMA_B = np.array([[1.5, -0.7, 0.2, 0.0, 0.0],
                    [-0.7, 1.5, 0.4, 0.0, 0.0],
                    [0.2, 0.4, 1.5, 0.6, 0.0],
                    [0.0, 0.0, 0.6, 1.5, 0.3],
                    [0.0, 0.0, 0.0, 0.3, 1.5]])

# a covariance matrix must be symmetric positive-definite: check before sampling
for name, S in [("Sigma_A", SIGMA_A), ("Sigma_B", SIGMA_B)]:
    assert np.allclose(S, S.T) and np.all(np.linalg.eigvalsh(S) > 0), name

XA = rng.multivariate_normal(MU_A, SIGMA_A, size=N_5D)
XB = rng.multivariate_normal(MU_B, SIGMA_B, size=N_5D)
X_I = np.vstack([XA, XB])
y_I = np.repeat([0, 1], N_5D)  # 0 = class A, 1 = class B
# --8<-- [end:ex2a]

# --8<-- [start:ex2b]
def shell(n, rho_mean, rho_std):
    """n points x = rho * u, u uniform on the unit sphere of R^5, rho ~ N(mean, std)."""
    v = rng.normal(size=(n, 5))                      # isotropic Gaussian ...
    u = v / np.linalg.norm(v, axis=1, keepdims=True)  # ... normalised -> uniform direction
    rho = rng.normal(rho_mean, rho_std, size=n)
    return rho[:, None] * u


XC = shell(N_5D, 2.0, 0.4)  # core
XD = shell(N_5D, 5.0, 0.4)  # shell
X_II = np.vstack([XC, XD])
y_II = np.repeat([0, 1], N_5D)  # 0 = class C, 1 = class D
# --8<-- [end:ex2b]

# --8<-- [start:ex2c]
datasets = {
    "I": (X_I, y_I, ["Class A", "Class B"]),
    "II": (X_II, y_II, ["Class C (core)", "Class D (shell)"]),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
pca_var, center_dist, radii = {}, {}, {}
for ax, (name, (X, y, labels)) in zip(axes, datasets.items()):
    pca = PCA(n_components=2).fit(X)       # linear projection onto 2 directions
    Z = pca.transform(X)
    pca_var[name] = pca.explained_variance_ratio_
    for c in (0, 1):
        ax.scatter(*Z[y == c].T, s=10, alpha=0.55, color=COLORS[c], label=labels[c])
    ax.set(title=f"Dataset {name} — PCA (PC1+PC2 = {pca_var[name].sum():.1%} of variance)",
           xlabel=f"PC1 ({pca_var[name][0]:.1%})", ylabel=f"PC2 ({pca_var[name][1]:.1%})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=9)

    # geometric measures, computed in the ORIGINAL 5D space
    center_dist[name] = float(np.linalg.norm(X[y == 0].mean(axis=0) - X[y == 1].mean(axis=0)))
    radii[name] = np.linalg.norm(X, axis=1)
    print(f"Dataset {name}: explained variance PC1={pca_var[name][0]:.4f} "
          f"PC2={pca_var[name][1]:.4f} sum={pca_var[name].sum():.4f}; "
          f"center distance (5D) = {center_dist[name]:.4f}")
fig.suptitle("Figure 4 — 2D PCA projections of the two 5D datasets", fontsize=13)
save(fig, "fig4_pca.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, (name, (X, y, labels)) in zip(axes, datasets.items()):
    bins = np.linspace(0, radii[name].max() * 1.05, 45)
    for c in (0, 1):
        r = radii[name][y == c]
        ax.hist(r, bins=bins, alpha=0.55, color=COLORS[c],
                label=f"{labels[c]} (mean {r.mean():.2f}, std {r.std():.2f})")
    ax.set(title=f"Dataset {name}", xlabel=r"radius $\|x\|$ (5D)", ylabel="count")
    ax.legend(fontsize=9)
fig.suptitle(r"Figure 5 — Histogram of the radius $\|x\|$ per class", fontsize=13)
save(fig, "fig5_radius.png")

radius_stats = {name: {labels[c]: [round(float(radii[name][y == c].mean()), 4),
                                   round(float(radii[name][y == c].std()), 4),
                                   round(float(radii[name][y == c].min()), 4),
                                   round(float(radii[name][y == c].max()), 4)]
                       for c in (0, 1)}
                for name, (_, y, labels) in datasets.items()}
print("radius [mean, std, min, max]:", radius_stats)
# --8<-- [end:ex2c]

# --8<-- [start:ex2d]
# 1) A LINEAR rule: project on the direction joining the two class means and
#    threshold at the midpoint (a fixed hyperplane, nothing is trained).
def mean_direction_accuracy(X, y):
    m0, m1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)
    w = m1 - m0
    b = -w @ (m0 + m1) / 2
    return float(np.mean(((X @ w + b) > 0).astype(int) == y))


# 2) The NON-LINEAR rule for Dataset II: f(x) = ||x||^2 - R^2, with R between
#    the two shells (2.0 and 5.0). f > 0 -> shell (D), f < 0 -> core (C).
R = 3.5


def f_shell(X):
    return np.sum(X ** 2, axis=1) - R ** 2


acc_linear = {name: mean_direction_accuracy(X, y) for name, (X, y, _) in datasets.items()}
acc_norm = float(np.mean((f_shell(X_II) > 0).astype(int) == y_II))
print(f"hyperplane through the mean difference: Dataset I acc = {acc_linear['I']:.4f}, "
      f"Dataset II acc = {acc_linear['II']:.4f}")
print(f"f(x) = ||x||^2 - {R}^2 on Dataset II: accuracy = {acc_norm:.4f}")
# --8<-- [end:ex2d]

results["ex2"] = {
    "explained_variance": {k: [round(float(v), 4) for v in pca_var[k]] for k in pca_var},
    "explained_variance_sum": {k: round(float(pca_var[k].sum()), 4) for k in pca_var},
    "center_distance": {k: round(v, 4) for k, v in center_dist.items()},
    "center_distance_theoretical": {"I": round(float(np.linalg.norm(MU_B - MU_A)), 4), "II": 0.0},
    "radius_mean_std_min_max": radius_stats,
    "accuracy_mean_direction_hyperplane": {k: round(v, 4) for k, v in acc_linear.items()},
    "accuracy_norm_rule_II": round(acc_norm, 4),
}


# =============================================================================
# Exercise 3 — Preparing real-world data for a neural network (tanh)
# =============================================================================

# --8<-- [start:ex3a]
df = pd.read_csv(DATA_CSV)
TARGET = "Transported"
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC = ["Age"] + SPEND
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP = ["PassengerId", "Cabin", "Name"]  # identifiers / free text

print("shape:", df.shape)
print(df.dtypes)

balance = df[TARGET].value_counts()
balance_pct = df[TARGET].value_counts(normalize=True)
print(pd.DataFrame({"count": balance, "share": balance_pct.round(4)}))

missing = pd.DataFrame({"missing": df.isna().sum(),
                        "percent": (100 * df.isna().mean()).round(2)})
print(missing)

spend_stats = df[SPEND].agg(["mean", "median", "max"]).T
spend_stats["share of zeros (%)"] = 100 * (df[SPEND] == 0).mean()
spend_stats["skewness"] = df[SPEND].skew()
print(spend_stats.round(2))
# --8<-- [end:ex3a]

# --8<-- [start:ex3b]
X_raw = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)  # True -> 1 (transported), False -> 0

# split FIRST: every statistic below is learned from X_train only
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=SEED)
print("train:", X_train.shape, "test:", X_test.shape)
print("positive share  train = %.4f  test = %.4f" % (y_train.mean(), y_test.mean()))
foodcourt_train = X_train["FoodCourt"].agg(["mean", "median"])
print("FoodCourt (train, raw): mean = %.2f  median = %.2f" % tuple(foodcourt_train))
# --8<-- [end:ex3b]

# --8<-- [start:ex3c]
LOG_COLS = SPEND + ["TotalSpend"]
SCALED_COLS = ["Age"] + LOG_COLS

# 1) Missing data - statistics learned on the training set only
num_imputer = SimpleImputer(strategy="median").fit(X_train[NUMERIC])
cat_imputer = SimpleImputer(strategy="most_frequent").fit(X_train[CATEGORICAL].astype(object))
print("numeric medians (train):", dict(zip(NUMERIC, num_imputer.statistics_)))
print("categorical modes (train):", dict(zip(CATEGORICAL, cat_imputer.statistics_)))


def impute_and_engineer(X):
    """Impute, drop identifiers, create TotalSpend and compress the heavy tails."""
    num = pd.DataFrame(num_imputer.transform(X[NUMERIC]), columns=NUMERIC, index=X.index)
    cat = pd.DataFrame(cat_imputer.transform(X[CATEGORICAL].astype(object)),
                       columns=CATEGORICAL, index=X.index).astype(str)
    num["TotalSpend"] = num[SPEND].sum(axis=1)   # feature engineering
    num[LOG_COLS] = np.log1p(num[LOG_COLS])       # log(1 + x): heavy tails -> compact
    return num, cat                               # PassengerId, Cabin, Name are not kept


num_train, cat_train = impute_and_engineer(X_train)
num_test, cat_test = impute_and_engineer(X_test)

# 2) Categorical -> one-hot; categories are learned on the training set.
#    handle_unknown="ignore": a category never seen in training becomes an
#    all-zeros row for that feature instead of raising an error.
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cat_train)

# 3) Scaling to [-1, 1] (the output range of tanh), min/max learned on train only
scaler = MinMaxScaler(feature_range=(-1, 1)).fit(num_train[SCALED_COLS])


def to_matrix(num, cat):
    scaled = pd.DataFrame(scaler.transform(num[SCALED_COLS]), columns=SCALED_COLS, index=num.index)
    onehot = pd.DataFrame(encoder.transform(cat), columns=encoder.get_feature_names_out(),
                          index=cat.index)
    return pd.concat([scaled, onehot], axis=1)


Xtr = to_matrix(num_train, cat_train)
Xte = to_matrix(num_test, cat_test)
print("features:", list(Xtr.columns))

# unknown-category check: a test row with a category the encoder never saw
probe = cat_test.iloc[[0]].copy()
probe["HomePlanet"] = "Pluto"
print("unseen 'Pluto' ->", encoder.transform(probe)[0][:3], "(HomePlanet block is all zeros)")
# --8<-- [end:ex3c]

# --8<-- [start:ex3d]
col = "FoodCourt"
raw = X_train[col].dropna()
logged = np.log1p(num_imputer.transform(X_train[NUMERIC])[:, NUMERIC.index(col)])
final = Xtr[col]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
axes[0].hist(raw, bins=60, color="tab:blue", label="raw (non-missing)")
axes[0].set(title="Before: raw FoodCourt", xlabel="FoodCourt spending", ylabel="passengers")
axes[1].hist(logged, bins=60, color="tab:orange", label="imputed + log(1+x)")
axes[1].set(title="After log(1 + x)", xlabel="log(1 + FoodCourt)", ylabel="passengers")
axes[2].hist(final, bins=60, color="tab:green", label="imputed + log(1+x) + min-max")
axes[2].set(title="After full preprocessing", xlabel="scaled value (tanh range)",
            ylabel="passengers")
axes[2].axvline(-1, color="gray", linestyle=":")
axes[2].axvline(1, color="gray", linestyle=":")
for ax in axes:
    ax.legend(fontsize=9)
fig.suptitle("Figure 6 — FoodCourt on the training set, before and after preprocessing",
             fontsize=13)
save(fig, "fig6_foodcourt.png")

# Final checks
print("NaN left: train =", int(Xtr.isna().sum().sum()), " test =", int(Xte.isna().sum().sum()))
print("final shapes: train =", Xtr.shape, " test =", Xte.shape)
print("train range: [%.4f, %.4f]  test range: [%.4f, %.4f]"
      % (Xtr.values.min(), Xtr.values.max(), Xte.values.min(), Xte.values.max()))
print("test values outside [-1, 1]:", int(((Xte.values < -1) | (Xte.values > 1)).sum()))
# --8<-- [end:ex3d]

results["ex3"] = {
    "shape_raw": list(df.shape),
    "balance_count": {str(k): int(v) for k, v in balance.items()},
    "balance_share": {str(k): round(float(v), 4) for k, v in balance_pct.items()},
    "missing": {c: [int(m), float(p)] for c, m, p in
                zip(missing.index, missing["missing"], missing["percent"])},
    "spend_stats": {c: {k: round(float(v), 4) for k, v in row.items()}
                    for c, row in spend_stats.iterrows()},
    "train_shape_raw": list(X_train.shape),
    "test_shape_raw": list(X_test.shape),
    "positive_share_train": round(float(y_train.mean()), 4),
    "positive_share_test": round(float(y_test.mean()), 4),
    "foodcourt_train_mean": round(float(foodcourt_train["mean"]), 4),
    "foodcourt_train_median": round(float(foodcourt_train["median"]), 4),
    "numeric_medians_train": {c: float(v) for c, v in zip(NUMERIC, num_imputer.statistics_)},
    "categorical_modes_train": {c: str(v) for c, v in zip(CATEGORICAL, cat_imputer.statistics_)},
    "features": list(Xtr.columns),
    "final_shape_train": list(Xtr.shape),
    "final_shape_test": list(Xte.shape),
    "nan_train": int(Xtr.isna().sum().sum()),
    "nan_test": int(Xte.isna().sum().sum()),
    "range_train": [round(float(Xtr.values.min()), 4), round(float(Xtr.values.max()), 4)],
    "range_test": [round(float(Xte.values.min()), 4), round(float(Xte.values.max()), 4)],
    "test_values_outside_range": int(((Xte.values < -1) | (Xte.values > 1)).sum()),
}

with open(FIG_DIR / "results.json", "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nresults written to {FIG_DIR / 'results.json'}")
