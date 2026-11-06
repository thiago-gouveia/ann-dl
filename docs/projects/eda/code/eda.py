"""
Project 1 (EDA) — Hotel Booking Demand. Artificial Neural Networks & Deep Learning (Insper, 2026.2)

Generates every table, figure and number used in docs/projects/eda/index.md.

    python docs/projects/eda/code/eda.py

Outputs:
    docs/projects/eda/figures/fig*.png       figures shown in the report
    docs/projects/eda/figures/results.json   every number quoted in the report
    docs/projects/eda/figures/tables.txt     the long tables (embedded in the report as snippets)

Data loading, cleaning, split and the preprocessing pipeline live in pipeline.py (importable).
No model is trained here. Seed: random_state = 42 everywhere.
"""
# --8<-- [start:setup]
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline as P  # noqa: E402  (our own module: load / clean / split / preprocessor)

FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
SEED = P.SEED
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
TGT_COLORS = {0: "tab:blue", 1: "tab:red"}
TGT_NAMES = {0: "not canceled (0)", 1: "canceled (1)"}
sns.set_theme(style="whitegrid", context="notebook")
results, tables = {}, []


def save(fig, name):
    """Save a figure into figures/ and close it."""
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def table(title, df, floatfmt=".2f"):
    """Collect a markdown table into figures/tables.txt, wrapped in snippet markers (t1, t2, ...)
    so the report embeds the generated table instead of a hand-copied one."""
    key = "t" + title.split()[1]
    tables.append(f"--8<-- [start:{key}]\n{df.to_markdown(floatfmt=floatfmt)}\n--8<-- [end:{key}]\n")


def r(x, n=4):
    return round(float(x), n)
# --8<-- [end:setup]


# =============================================================================
# 1. Initial inspection (raw file: descriptive only, no statistic is used for preprocessing)
# =============================================================================
# --8<-- [start:quality]
raw = P.load_raw()
n_rows, n_cols = raw.shape

# 1.A — type and number of distinct values of every column (for the data dictionary)
dictionary = pd.DataFrame({"dtype": raw.dtypes.astype(str), "distinct": raw.nunique(),
                           "example": raw.iloc[0]})

# 1.B — missing values, duplicates, impossible values, leakage
missing = raw.isna().sum()
missing = pd.DataFrame({"missing": missing, "pct": 100 * missing / n_rows})
missing = missing[missing["missing"] > 0].sort_values("missing", ascending=False)

n_dup = int(raw.duplicated().sum())
guests = raw["adults"] + raw["children"].fillna(0) + raw["babies"]
impossible = {
    "0 guests (adults + children + babies = 0)": int((guests == 0).sum()),
    "adults = 0 but children > 0 (child without adult)": int(((raw.adults == 0) & (guests > 0)).sum()),
    "adr < 0 (negative price)": int((raw.adr < 0).sum()),
    "adr > 1000": int((raw.adr > 1000).sum()),
    "adr = 0 (free stay)": int((raw.adr == 0).sum()),
    "0 nights (weekend + week nights = 0)": int(((raw.stays_in_weekend_nights
                                                 + raw.stays_in_week_nights) == 0).sum()),
    "meal = 'Undefined' (same as SC per the source)": int((raw.meal == "Undefined").sum()),
}
constant_cols = [c for c in raw.columns if raw[c].nunique(dropna=False) == 1]

# leakage: the cancellation rate inside each value of the suspicious columns
leak_status = pd.crosstab(raw["reservation_status"], raw[P.TARGET])
room_changed = raw["reserved_room_type"] != raw["assigned_room_type"]
leak_room = raw.groupby(room_changed)[P.TARGET].agg(["mean", "size"])
status_date = pd.to_datetime(raw["reservation_status_date"])
arrival = pd.to_datetime(raw["arrival_date_year"].astype(str) + "-" + raw["arrival_date_month"]
                         + "-" + raw["arrival_date_day_of_month"].astype(str), format="%Y-%B-%d")
status_before_arrival = status_date < arrival
leak_date = raw.groupby(status_before_arrival)[P.TARGET].agg(["mean", "size"])
leak_parking = raw.groupby(raw["required_car_parking_spaces"] > 0)[P.TARGET].agg(["mean", "size"])
# --8<-- [end:quality]

results["raw"] = {
    "rows": n_rows, "cols": n_cols, "features": n_cols - 1,
    "missing": {c: {"n": int(m), "pct": r(p, 2)} for c, (m, p) in missing.iterrows()},
    "duplicates": n_dup, "duplicates_pct": r(100 * n_dup / n_rows, 2),
    "dup_cancel_rate": r(raw[raw.duplicated(keep=False)][P.TARGET].mean()),
    "dup_removed_cancel_rate": r(raw[raw.duplicated()][P.TARGET].mean()),
    "dup_removed_nonrefund_pct": r(100 * (raw[raw.duplicated()]["deposit_type"] == "Non Refund").mean(), 2),
    "nonrefund_raw": int((raw["deposit_type"] == "Non Refund").sum()),
    "nonrefund_after_dedup": int((raw.drop_duplicates()["deposit_type"] == "Non Refund").sum()),
    "nonrefund_raw_cancel_rate": r(raw.loc[raw["deposit_type"] == "Non Refund", P.TARGET].mean()),
    "impossible": impossible, "constant_cols": constant_cols,
    "leak_status": leak_status.to_dict(),
    "leak_room_changed": {str(k): {"rate": r(v["mean"]), "n": int(v["size"])}
                          for k, v in leak_room.iterrows()},
    "leak_status_date_before_arrival": {str(k): {"rate": r(v["mean"]), "n": int(v["size"])}
                                        for k, v in leak_date.iterrows()},
    "leak_parking": {str(k): {"rate": r(v["mean"]), "n": int(v["size"])}
                     for k, v in leak_parking.iterrows()},
    "target_rate_raw": r(raw[P.TARGET].mean()),
    "year_rates": {int(k): r(v) for k, v in raw.groupby("arrival_date_year")[P.TARGET].mean().items()},
    "date_range": [str(arrival.min().date()), str(arrival.max().date())],
}
table("Table 1 — column types and distinct values (raw file)", dictionary.drop(columns="example"))
table("Table 2 — missing values (raw file)", missing)

# =============================================================================
# 1.C — target   and   1.D — cleaning + split
# =============================================================================
# --8<-- [start:split]
df = P.clean(raw)                                 # duplicates + 0-guest rows removed
X_train_all, X_test_all, y_train, y_test = P.split(df)   # stratified 80/20, random_state=42
train = X_train_all.assign(**{P.TARGET: y_train})  # all analysis below: TRAINING SET ONLY
# --8<-- [end:split]

results["clean"] = {"rows": len(df), "removed_dup": n_dup,
                    "removed_zero_guests": int(len(raw) - n_dup - len(df)),
                    "target_rate": r(df[P.TARGET].mean())}
results["split"] = {"train": len(X_train_all), "test": len(X_test_all),
                    "train_rate": r(y_train.mean()), "test_rate": r(y_test.mean())}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (name, y) in zip(axes, [("raw file", raw[P.TARGET]), ("after cleaning", df[P.TARGET])]):
    counts = y.value_counts().sort_index()
    bars = ax.bar([TGT_NAMES[k] for k in counts.index], counts.values,
                  color=[TGT_COLORS[k] for k in counts.index], label=[TGT_NAMES[k] for k in counts.index])
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({100 * c / len(y):.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{name} — {len(y):,} bookings")
    ax.set_xlabel("is_canceled")
    ax.set_ylabel("number of bookings")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Figure 1 — target distribution: cancellations are the minority class")
fig.tight_layout()
save(fig, "fig01_target.png")

# =============================================================================
# 2.A — numerical features (train)
# =============================================================================
NUM_ALL = P.NUM_LOG + P.NUM_LIN + ["arrival_date_week_number", "arrival_date_year",
                                   "required_car_parking_spaces"]
NUM_ALL = sorted(NUM_ALL, key=list(raw.columns).index)
desc = train[NUM_ALL].describe().T[["mean", "50%", "std", "min", "25%", "75%", "max"]]
desc["skew"] = train[NUM_ALL].skew()
desc["% zeros"] = 100 * (train[NUM_ALL] == 0).mean()
desc = desc.rename(columns={"50%": "median", "25%": "Q1", "75%": "Q3"})
table("Table 3 — numerical features, training set", desc)
results["num_desc"] = desc.round(3).to_dict(orient="index")

fig, axes = plt.subplots(4, 4, figsize=(15, 12))
for ax, c in zip(axes.ravel(), NUM_ALL):
    x = train[c]
    discrete = x.nunique() <= 25
    if discrete:
        vc = x.value_counts().sort_index()
        ax.bar(vc.index, vc.values, width=0.8, color="tab:gray", label="count")
    else:
        ax.hist(x, bins=60, color="tab:gray", label="count")
    ax.axvline(x.median(), color="tab:orange", ls="--", lw=1.5, label=f"median {x.median():g}")
    ax.axvline(x.mean(), color="tab:green", ls=":", lw=1.5, label=f"mean {x.mean():.2f}")
    if desc.loc[c, "skew"] > 3:
        ax.set_yscale("log")
    ax.set_title(f"{c}\nskew {desc.loc[c, 'skew']:.1f}, zeros {desc.loc[c, '% zeros']:.0f}%", fontsize=9)
    ax.set_xlabel(c, fontsize=8)
    ax.set_ylabel("bookings" + (" (log)" if desc.loc[c, "skew"] > 3 else ""), fontsize=8)
    ax.legend(fontsize=6.5)
fig.suptitle("Figure 2 — distribution of the 16 numerical features (training set; log y-axis when skew > 3)",
             y=1.0)
fig.tight_layout()
save(fig, "fig02_numeric_hist.png")

# --8<-- [start:skew_rule]
# Skewness on train: raw, after clipping at the q99.9 (UpperClipper), and after clip + log1p
num_model = P.NUM_LOG + P.NUM_LIN
upper = train[num_model].quantile(P.CLIP_Q)
clipped = train[num_model].clip(lower=0, upper=upper, axis=1)
skew = pd.DataFrame({"raw": train[num_model].skew(), "clipped": clipped.skew(),
                     "clipped + log1p": np.log1p(clipped).skew()})
rule = (skew["clipped"] > 1) & (skew["clipped + log1p"].abs() < skew["clipped"].abs())
assert set(skew.index[rule]) == set(P.NUM_LOG), "NUM_LOG in pipeline.py must follow the rule"
rows_clipped = int((train[num_model].gt(upper, axis=1) | train[num_model].lt(0)).any(axis=1).sum())
# --8<-- [end:skew_rule]
skew["q99.9 (clip)"] = upper
skew["max"] = train[num_model].max()
skew["log1p?"] = np.where(rule, "yes", "no")
skew = skew.sort_values("clipped", ascending=False)
table("Table 4 — skewness rule for log1p (training set)", skew)
results["skew"] = skew.drop(columns="log1p?").round(3).to_dict(orient="index")
results["rows_clipped"] = {"n": rows_clipped, "pct": r(100 * rows_clipped / len(train), 2),
                           "per_feature": {c: int((train[c] > upper[c]).sum() + (train[c] < 0).sum())
                                           for c in num_model}}

fig, ax = plt.subplots(figsize=(11, 4.5))
xs = np.arange(len(skew))
for i, (col, color) in enumerate([("raw", "tab:gray"), ("clipped", "tab:blue"),
                                  ("clipped + log1p", "tab:green")]):
    ax.bar(xs + (i - 1) * 0.27, skew[col].clip(-5, 40), width=0.27, color=color, label=col)
ax.axhline(1, color="k", ls="--", lw=1, label="|skew| = 1")
ax.axhline(-1, color="k", ls="--", lw=1)
ax.set_xticks(xs, [f"{c}{' *' if f == 'yes' else ''}" for c, f in zip(skew.index, skew['log1p?'])],
              rotation=40, ha="right")
ax.set_ylabel("skewness (clipped to [-5, 40] for display)")
ax.set_xlabel("numerical feature (* = log1p applied)")
ax.set_title("Figure 3 — skewness before and after clipping and log1p (training set)")
ax.legend()
save(fig, "fig03_skewness.png")

# =============================================================================
# 2.B — categorical features (train)
# =============================================================================
CAT_ALL = ["hotel", "arrival_date_month", "meal", "country", "market_segment",
           "distribution_channel", "is_repeated_guest", "reserved_room_type",
           "assigned_room_type", "deposit_type", "agent", "customer_type", "has_company"]
cat_rows = []
for c in CAT_ALL:
    vc = train[c].value_counts(dropna=False)
    rare = vc[vc < P.MIN_FREQ]
    cat_rows.append({"feature": c, "cardinality": train[c].nunique(), "top": str(vc.index[0]),
                     "top %": 100 * vc.iloc[0] / len(train), "missing %": 100 * train[c].isna().mean(),
                     f"categories < {P.MIN_FREQ}": len(rare),
                     "rows in those %": 100 * rare.sum() / len(train)})
cat_tab = pd.DataFrame(cat_rows).set_index("feature")
table("Table 5 — categorical features, training set", cat_tab)
results["cat"] = cat_tab.round(3).to_dict(orient="index")
cat_freq = {c: (100 * train[c].value_counts(normalize=True)).round(2).head(12).to_dict()
            for c in CAT_ALL}
results["cat_freq_top12_pct"] = {c: {str(k): v for k, v in d.items()} for c, d in cat_freq.items()}

LOW = ["hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type",
       "deposit_type", "customer_type", "arrival_date_month"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
for ax, c in zip(axes.ravel(), LOW):
    vc = train[c].value_counts()
    if c == "arrival_date_month":
        vc = vc.reindex(MONTHS)
    pct = 100 * vc / len(train)
    colors = ["tab:red" if v < P.MIN_FREQ else "tab:blue" for v in vc.values]
    ax.barh([str(i) for i in vc.index][::-1], pct.values[::-1], color=colors[::-1])
    for i, (n, p) in enumerate(zip(vc.values[::-1], pct.values[::-1])):
        if p < 2:  # bars too small to see: write the count
            ax.text(p + pct.max() * 0.01, i, f"n={n}", va="center", fontsize=7,
                    color="tab:red" if n < P.MIN_FREQ else "black")
    ax.set_title(f"{c} ({len(vc)} categories)", fontsize=10)
    ax.set_xlabel("% of training bookings")
    ax.set_ylabel("category")
from matplotlib.patches import Patch  # noqa: E402
fig.legend(handles=[Patch(color="tab:blue", label=f"≥ {P.MIN_FREQ} bookings"),
                    Patch(color="tab:red", label=f"< {P.MIN_FREQ} bookings (rare)")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Figure 4 — frequency of the low-cardinality categorical features (training set)")
fig.tight_layout(rect=(0, 0.03, 1, 1))
save(fig, "fig04_categorical_bars.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, c in zip(axes, ["country", "agent"]):
    vc = train[c].value_counts()
    rank = np.arange(1, len(vc) + 1)
    ax.plot(rank, vc.values, "o-", ms=2.5, color="tab:blue", label="bookings per category")
    ax.axhline(P.MIN_FREQ, color="tab:red", ls="--", label=f"min_frequency = {P.MIN_FREQ}")
    ax.set_yscale("log")
    ax.set_xlabel(f"{c} category rank (most frequent first)")
    ax.set_ylabel("bookings (log)")
    ax2 = ax.twinx()
    ax2.plot(rank, 100 * vc.cumsum() / train[c].notna().sum(), color="tab:green",
             label="cumulative % of rows")
    ax2.set_ylabel("cumulative % of non-missing rows")
    ax2.grid(False)
    kept = int((vc >= P.MIN_FREQ).sum())
    ax.set_title(f"{c}: {len(vc)} categories, {kept} with ≥ {P.MIN_FREQ} bookings")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
fig.suptitle("Figure 5 — the two high-cardinality features: a few categories hold most rows")
fig.tight_layout()
save(fig, "fig05_high_cardinality.png")
results["high_card"] = {c: {"categories": int(train[c].nunique()),
                            "kept": int((train[c].value_counts() >= P.MIN_FREQ).sum()),
                            "coverage_kept_pct": r(100 * train[c].value_counts()[
                                train[c].value_counts() >= P.MIN_FREQ].sum() / train[c].notna().sum(), 2),
                            "top5_pct": r(100 * train[c].value_counts().head(5).sum()
                                          / train[c].notna().sum(), 2)}
                        for c in ["country", "agent"]}

# =============================================================================
# 3.A — numerical x numerical (Spearman)
# =============================================================================
corr_df = train[NUM_ALL + ["is_repeated_guest", "has_company"]].copy()
corr_df["arrival_month_num"] = train["arrival_date_month"].map({m: i for i, m in enumerate(MONTHS, 1)})
spear = corr_df.corr(method="spearman")
pear = corr_df.corr(method="pearson")
pairs = (spear.where(np.triu(np.ones(spear.shape, bool), 1)).stack()
         .rename("spearman").to_frame())
pairs["pearson"] = [pear.loc[a, b] for a, b in pairs.index]
pairs = pairs.reindex(pairs["spearman"].abs().sort_values(ascending=False).index)
results["corr_top"] = [{"a": a, "b": b, "spearman": r(s, 3), "pearson": r(p, 3)}
                       for (a, b), (s, p) in pairs.head(10).iterrows()]
table("Table 6 — the 10 most correlated numerical pairs (training set)", pairs.head(10), ".3f")

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(spear, dtype=bool), 1)
sns.heatmap(spear, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 7}, cbar_kws={"label": "Spearman ρ"}, ax=ax)
ax.set_title("Figure 6 — Spearman correlation between numerical features (training set)")
ax.set_xlabel("feature")
ax.set_ylabel("feature")
save(fig, "fig06_spearman.png")

samp = train.sample(6000, random_state=SEED)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
jit = np.random.default_rng(SEED)
for t in (0, 1):
    s = samp[samp[P.TARGET] == t]
    mnum = s["arrival_date_month"].map({m: i for i, m in enumerate(MONTHS, 1)})
    axes[0].scatter(mnum + jit.uniform(-.3, .3, len(s)), s["arrival_date_week_number"], s=4, alpha=.4,
                    color=TGT_COLORS[t], label=TGT_NAMES[t])
    axes[1].scatter(s["is_repeated_guest"] + jit.uniform(-.25, .25, len(s)),
                    s["previous_bookings_not_canceled"] + jit.uniform(-.2, .2, len(s)), s=4, alpha=.4,
                    color=TGT_COLORS[t], label=TGT_NAMES[t])
    axes[2].scatter(s["lead_time"], s["stays_in_week_nights"] + jit.uniform(-.3, .3, len(s)), s=4,
                    alpha=.4, color=TGT_COLORS[t], label=TGT_NAMES[t])
axes[0].set(xlabel="arrival month (1 = January)", ylabel="arrival_date_week_number",
            title=f"month × week: ρ = {spear.loc['arrival_month_num', 'arrival_date_week_number']:.3f}")
axes[1].set(xlabel="is_repeated_guest (jittered)", ylabel="previous_bookings_not_canceled (jittered)",
            title=f"repeated guest × previous bookings: ρ = "
                  f"{spear.loc['previous_bookings_not_canceled', 'is_repeated_guest']:.3f}", ylim=(-1, 25))
axes[2].set(xlabel="lead_time (days)", ylabel="stays_in_week_nights (jittered)",
            title=f"lead time × week nights: ρ = {spear.loc['lead_time', 'stays_in_week_nights']:.3f}",
            ylim=(-1, 16))
for ax in axes:
    ax.legend(markerscale=3, fontsize=8)
fig.suptitle("Figure 7 — the most correlated pairs (6 000 training bookings, coloured by target)")
fig.tight_layout()
save(fig, "fig07_scatter_pairs.png")

# =============================================================================
# 3.B — categorical x target
# =============================================================================
base = y_train.mean()
B_COLS = ["deposit_type", "market_segment", "customer_type", "hotel", "distribution_channel", "meal"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
rates = {}
for ax, c in zip(axes.ravel(), B_COLS):
    g = train.groupby(c)[P.TARGET].agg(["mean", "size"]).sort_values("mean")
    rates[c] = {str(k): {"rate": r(v["mean"]), "n": int(v["size"])} for k, v in g.iterrows()}
    ax.barh(g.index.astype(str), 100 * g["mean"],
            color=["tab:red" if m > base else "tab:blue" for m in g["mean"]])
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(100 * m + 1, i, f"{100 * m:.0f}%  (n={n:,})", va="center", fontsize=8)
    ax.axvline(100 * base, color="k", ls="--", lw=1, label=f"overall {100 * base:.1f}%")
    ax.set_xlim(0, 115)
    ax.set_xlabel("% canceled")
    ax.set_ylabel(c)
    ax.set_title(c)
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Figure 8 — cancellation rate per category (training set; red = above the overall rate)")
fig.tight_layout()
save(fig, "fig08_cat_target.png")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
top_c = train["country"].value_counts().head(15).index
g = train[train["country"].isin(top_c)].groupby("country")[P.TARGET].agg(["mean", "size"]).sort_values("mean")
rates["country_top15"] = {str(k): {"rate": r(v["mean"]), "n": int(v["size"])} for k, v in g.iterrows()}
axes[0].barh(g.index, 100 * g["mean"], color=["tab:red" if m > base else "tab:blue" for m in g["mean"]])
for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
    axes[0].text(100 * m + 0.5, i, f"{100 * m:.0f}% (n={n:,})", va="center", fontsize=8)
axes[0].axvline(100 * base, color="k", ls="--", lw=1, label=f"overall {100 * base:.1f}%")
axes[0].set(xlabel="% canceled", ylabel="country (15 most frequent)", xlim=(0, 60),
            title="country")
axes[0].legend(fontsize=8, loc="lower right")
gm = train.groupby(["arrival_date_month", "hotel"])[P.TARGET].mean().unstack().reindex(MONTHS)
rates["month_hotel"] = {h: {m: r(v) for m, v in gm[h].items()} for h in gm.columns}
for h, col in zip(gm.columns, ["tab:purple", "tab:olive"]):
    axes[1].plot(range(1, 13), 100 * gm[h], "o-", color=col, label=h)
axes[1].axhline(100 * base, color="k", ls="--", lw=1, label=f"overall {100 * base:.1f}%")
axes[1].set_xticks(range(1, 13), [m[:3] for m in MONTHS])
axes[1].set(xlabel="arrival month", ylabel="% canceled", title="arrival month × hotel")
axes[1].legend(fontsize=8)
fig.suptitle("Figure 9 — cancellation rate by country and by arrival month (training set)")
fig.tight_layout()
save(fig, "fig09_country_month.png")
results["cat_target"] = rates
results["base_rate_train"] = r(base)

# =============================================================================
# 3.C — numerical x categorical (grouped boxplots)
# =============================================================================
def group_stats(num, cat):
    g = train.groupby(cat)[num]
    return pd.DataFrame({"n": g.size(), "median": g.median(), "IQR": g.quantile(.75) - g.quantile(.25)})


fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
order_ms = train.groupby("market_segment")["lead_time"].median().sort_values().index
sns.boxplot(data=train, x="market_segment", y="lead_time", hue=P.TARGET, order=order_ms,
            palette=TGT_COLORS, showfliers=False, ax=axes[0])
axes[0].set(xlabel="market_segment", ylabel="lead_time (days)",
            title="lead_time by market segment and target (outliers hidden)")
axes[0].tick_params(axis="x", rotation=25)
order_rt = sorted(train["reserved_room_type"].unique())
sns.boxplot(data=train, x="reserved_room_type", y="adr", hue="hotel", order=order_rt,
            showfliers=False, ax=axes[1])
axes[1].set(xlabel="reserved_room_type", ylabel="adr (average daily rate, €)",
            title="adr by reserved room type and hotel (outliers hidden)")
for ax, title in zip(axes, ["is_canceled", "hotel"]):
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, [TGT_NAMES.get(int(x), x) if x in ("0", "1") else x for x in l], title=title, fontsize=8)
fig.suptitle("Figure 10 — grouped boxplots: location and spread of numerical features per category")
fig.tight_layout()
save(fig, "fig10_boxplots.png")

fig, ax = plt.subplots(figsize=(13, 4.8))
sns.boxplot(data=train, x="arrival_date_month", y="adr", hue="hotel", order=MONTHS,
            showfliers=False, ax=ax)
ax.set(xlabel="arrival month", ylabel="adr (average daily rate, €)",
       title="Figure 11 — adr by arrival month and hotel: the Resort's price is seasonal, the City's is not")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="hotel")
save(fig, "fig11_adr_month.png")

box = {}
s = group_stats("lead_time", ["market_segment", P.TARGET]).round(1)
box["lead_time_by_segment_target"] = {f"{a}|{b}": v for (a, b), v in s.to_dict(orient="index").items()}
s = group_stats("adr", ["reserved_room_type", "hotel"]).round(1)
box["adr_by_room_hotel"] = {f"{a}|{b}": v for (a, b), v in s.to_dict(orient="index").items()}
s = group_stats("adr", ["arrival_date_month", "hotel"]).round(1)
box["adr_by_month_hotel"] = {f"{a}|{b}": v for (a, b), v in s.to_dict(orient="index").items()}
s = group_stats("lead_time", [P.TARGET]).round(1)
box["lead_time_by_target"] = {str(k): v for k, v in s.to_dict(orient="index").items()}
results["boxplots"] = box

# =============================================================================
# 4.A — evidence for the strategies
# =============================================================================
X_train, X_test = X_train_all[P.FEATURES], X_test_all[P.FEATURES]
unseen = {c: int((~X_test[c].dropna().isin(set(X_train[c].dropna()))).sum()) for c in P.CATEGORICAL}
unseen_values = {c: sorted(map(str, set(X_test[c].dropna()) - set(X_train[c].dropna())))[:10]
                 for c in P.CATEGORICAL if unseen[c]}
results["strategies"] = {
    "train_missing": {c: int(n) for c, n in X_train.isna().sum().items() if n},
    "test_rows_with_unseen_category": unseen, "unseen_examples": unseen_values,
    "num_ranges": {c: [r(X_train[c].min(), 2), r(X_train[c].max(), 2)] for c in P.NUM_LOG + P.NUM_LIN},
}

# =============================================================================
# 4.C — the pipeline, fitted on train only
# =============================================================================
# --8<-- [start:fit]
pre = P.build_preprocessor()
Xt_train = pre.fit_transform(X_train)        # every statistic (median, q99.9, mean/std, categories)
Xt_test = pre.transform(X_test)              # ... is learned on X_train and only applied to X_test
names = pre.get_feature_names_out()
print("NaN train/test:", np.isnan(Xt_train).sum(), np.isnan(Xt_test).sum())
print("shape train/test:", Xt_train.shape, Xt_test.shape)
# --8<-- [end:fit]
ohe = pre.named_transformers_["cat"].named_steps["onehot"]
blocks = pd.Series([str(n).split("__", 1)[0] for n in names])
feat_tab = pd.DataFrame([{"block": blk, "columns": int((blocks == blk).sum()),
                          "names": ", ".join(f"`{str(n).split('__', 1)[1]}`"
                                             for n in names if str(n).startswith(blk + "__"))}
                         for blk in ["num_log", "num_lin", "bin", "cat"]]).set_index("block")
table("Table 8 — the 192 output columns of the pipeline, by block", feat_tab)
results["pipeline"] = {
    "shape_train": list(Xt_train.shape), "shape_test": list(Xt_test.shape),
    "nan_train": int(np.isnan(Xt_train).sum()), "nan_test": int(np.isnan(Xt_test).sum()),
    "n_features_in": len(P.FEATURES),
    "blocks": {"num_log": len(P.NUM_LOG), "num_lin": len(P.NUM_LIN), "bin": len(P.BINARY),
               "cat_onehot": int(sum(n.startswith("cat__") for n in names))},
    "onehot_per_feature": {c: len(cats) for c, cats in zip(P.CATEGORICAL, ohe.categories_)},
    "infrequent_per_feature": {c: (len(i) if i is not None else 0)
                               for c, i in zip(P.CATEGORICAL, ohe.infrequent_categories_)},
    "train_num_mean_after": r(np.abs(Xt_train[:, :len(P.NUM_LOG) + len(P.NUM_LIN)].mean(0)).max(), 6),
    "test_num_mean_after": r(np.abs(Xt_test[:, :len(P.NUM_LOG) + len(P.NUM_LIN)].mean(0)).max(), 4),
    "feature_names": list(map(str, names)),
}

# =============================================================================
# 4.B — dimensionality reduction on the scaled training features
# =============================================================================
pca = PCA(random_state=SEED).fit(Xt_train)
cum = np.cumsum(pca.explained_variance_ratio_)
load = pd.DataFrame(pca.components_[:2].T, index=names, columns=["PC1", "PC2"])
top_load = {pc: load[pc].reindex(load[pc].abs().sort_values(ascending=False).index).head(8).round(3)
            .to_dict() for pc in ["PC1", "PC2"]}
results["pca"] = {"pc1": r(pca.explained_variance_ratio_[0]), "pc2": r(pca.explained_variance_ratio_[1]),
                  "pc1_pc2": r(cum[1]), "n_for_80": int(np.searchsorted(cum, .80) + 1),
                  "n_for_95": int(np.searchsorted(cum, .95) + 1), "top_loadings": top_load}

# stratified sample of the training set for the 2-D projections
idx_s, _ = train_test_split(np.arange(len(Xt_train)), train_size=5000, stratify=y_train,
                            random_state=SEED)
Xs, ys = Xt_train[idx_s], y_train.to_numpy()[idx_s]
Z_pca = pca.transform(Xs)[:, :2]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
axes[0].plot(np.arange(1, len(cum) + 1), 100 * cum, "-", color="tab:blue", label="cumulative")
axes[0].bar(np.arange(1, 21), 100 * pca.explained_variance_ratio_[:20], color="tab:gray",
            label="per component (first 20)")
for lvl in (80, 95):
    k = int(np.searchsorted(cum, lvl / 100) + 1)
    axes[0].axhline(lvl, color="k", ls=":", lw=1)
    axes[0].text(len(cum) * 0.55, lvl + 1, f"{lvl}% with {k} PCs", fontsize=8)
axes[0].set(xlabel="number of principal components", ylabel="% of variance explained",
            title="explained variance", xscale="log")
axes[0].legend(fontsize=8, loc="center right")
for t in (0, 1):
    m = ys == t
    axes[1].scatter(Z_pca[m, 0], Z_pca[m, 1], s=4, alpha=.45, color=TGT_COLORS[t], label=TGT_NAMES[t])
axes[1].set(xlabel=f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)",
            ylabel=f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)", title="projection on PC1–PC2")
axes[1].legend(markerscale=3, fontsize=8)
top = sorted(set(top_load["PC1"]) | set(top_load["PC2"]), key=lambda n: -abs(load.loc[n]).max())[:12]
yy = np.arange(len(top))
axes[2].barh(yy - 0.2, load.loc[top, "PC1"], height=0.4, color="tab:blue", label="PC1")
axes[2].barh(yy + 0.2, load.loc[top, "PC2"], height=0.4, color="tab:orange", label="PC2")
axes[2].set_yticks(yy, [n.split("__", 1)[1] for n in top], fontsize=8)
axes[2].invert_yaxis()
axes[2].set(xlabel="loading", ylabel="feature after the pipeline", title="largest loadings of PC1 and PC2")
axes[2].legend(fontsize=8)
fig.suptitle(f"Figure 12 — PCA of the {Xt_train.shape[1]} preprocessed training features")
fig.tight_layout()
save(fig, "fig12_pca.png")


def scatter_panel(ax, Z, title, xl, yl):
    for t in (0, 1):
        m = ys == t
        ax.scatter(Z[m, 0], Z[m, 1], s=3, alpha=.5, color=TGT_COLORS[t], label=TGT_NAMES[t])
    ax.set(title=title, xlabel=xl, ylabel=yl)
    ax.legend(markerscale=4, fontsize=8)


# --8<-- [start:manifold]
import umap  # noqa: E402

emb = {}
for perp in (30, 100):
    emb[f"tsne_{perp}"] = TSNE(n_components=2, perplexity=perp, init="pca", random_state=SEED
                               ).fit_transform(Xs)
for nn in (15, 100):
    emb[f"umap_{nn}"] = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=SEED
                                  ).fit_transform(Xs)
# --8<-- [end:manifold]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, perp in zip(axes, (30, 100)):
    scatter_panel(ax, emb[f"tsne_{perp}"], f"t-SNE, perplexity = {perp}", "t-SNE 1", "t-SNE 2")
fig.suptitle("Figure 13 — t-SNE of 5 000 training bookings (stratified sample), coloured by target")
fig.tight_layout()
save(fig, "fig13_tsne.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, nn in zip(axes, (15, 100)):
    scatter_panel(ax, emb[f"umap_{nn}"], f"UMAP, n_neighbors = {nn}", "UMAP 1", "UMAP 2")
fig.suptitle("Figure 14 — UMAP of the same 5 000 training bookings, coloured by target")
fig.tight_layout()
save(fig, "fig14_umap.png")


# How well does each 2-D map separate the classes? 15-NN accuracy inside the sample (5-fold CV).
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

knn = {k: r(cross_val_score(KNeighborsClassifier(15), Z, ys, cv=5).mean())
       for k, Z in [("pca_2d", Z_pca)] + list(emb.items())}
knn[f"full_{Xs.shape[1]}d"] = r(cross_val_score(KNeighborsClassifier(15), Xs, ys, cv=5).mean())
knn["majority"] = r(1 - ys.mean())
results["projection_knn15_acc"] = knn
# What are the UMAP islands? DBSCAN on the 2-D UMAP (n_neighbors = 15) map, then describe each
# group by its size, cancellation rate and the dominant value of a few categorical features.
from sklearn.cluster import DBSCAN  # noqa: E402

lab = DBSCAN(eps=0.6, min_samples=20).fit_predict(emb["umap_15"])
Xs_raw = X_train_all.iloc[idx_s].reset_index(drop=True)
groups = []
for g in sorted(set(lab) - {-1}):
    m = lab == g
    row = {"group": int(g), "n": int(m.sum()), "cancel_rate": r(ys[m].mean(), 3),
           "centre": [r(v, 1) for v in emb["umap_15"][m].mean(0)]}
    for c in ["hotel", "market_segment", "deposit_type", "customer_type", "distribution_channel"]:
        vc = Xs_raw.loc[m, c].value_counts(normalize=True)
        row[c] = f"{vc.index[0]} ({100 * vc.iloc[0]:.0f}%)"
    for c in ["is_repeated_guest", "has_company"]:
        row[c] = r(Xs_raw.loc[m, c].mean(), 3)
    for c in ["required_car_parking_spaces", "previous_cancellations", "babies", "children",
              "days_in_waiting_list", "booking_changes"]:
        row[f"{c} > 0"] = r((Xs_raw.loc[m, c] > 0).mean(), 3)
    groups.append(row)
groups.sort(key=lambda d: -d["n"])
results["umap15_groups"] = groups
results["umap15_noise_points"] = int((lab == -1).sum())
results["rate_if_positive"] = {c: {"rate": r(train.loc[train[c] > 0, P.TARGET].mean()),
                                   "n": int((train[c] > 0).sum())}
                               for c in ["required_car_parking_spaces", "previous_cancellations",
                                         "babies", "children", "days_in_waiting_list",
                                         "booking_changes", "total_of_special_requests"]}
table("Table 7 — groups found by DBSCAN on the UMAP (n_neighbors = 15) map",
      pd.DataFrame(groups).set_index("group"), ".3f")

with open(FIG_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
(FIG_DIR / "tables.txt").write_text("\n".join(tables))
print(json.dumps({k: v for k, v in results.items()
                  if k not in ("num_desc", "cat", "cat_freq_top12_pct", "boxplots", "skew",
                               "cat_target")}, indent=1, default=str)[:6000])
