"""
Project 1 (EDA) — Hotel Booking Demand: data loading, cleaning, split and preprocessing pipeline.

Importable module used by eda.py and, later, by the classification deliverable:

    import sys; sys.path.insert(0, "docs/projects/eda/code")
    from pipeline import load_clean_split, build_preprocessor

    X_train, X_test, y_train, y_test = load_clean_split()
    pre = build_preprocessor().fit(X_train)       # every statistic comes from the training set
    Xt_train, Xt_test = pre.transform(X_train), pre.transform(X_test)

Every choice below is motivated in docs/projects/eda/index.md (section 4.A).
"""
# --8<-- [start:constants]
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hotel_bookings.csv"
TARGET = "is_canceled"

# Columns removed before modeling, with the reason (section 1.B)
DROP = {
    "reservation_status": "leaks the target (Canceled / No-Show <=> is_canceled = 1)",
    "reservation_status_date": "leaks the target (date of the cancellation or of the check-out)",
    "assigned_room_type": "leaks the target (the room is only assigned at check-in)",
    "required_car_parking_spaces": "suspected leak: 0 cancellations among 5,865 bookings with > 0",
    "company": "94% missing; replaced by the flag has_company",
    "arrival_date_year": "only 3 values (2015-2017); a deployed model only sees future years",
    "arrival_date_week_number": "redundant with arrival_date_month (Spearman 0.995, section 3.A)",
}

# Right-skewed features: log1p before scaling. Rule (checked in eda.py, section 4.A): on the
# training set, skewness after clipping > 1 AND log1p reduces |skewness|.
NUM_LOG = ["lead_time", "stays_in_week_nights", "total_of_special_requests", "booking_changes",
           "children", "previous_cancellations", "previous_bookings_not_canceled",
           "days_in_waiting_list"]
# The other numeric features: clipped and scaled only
NUM_LIN = ["adr", "adults", "arrival_date_day_of_month", "stays_in_weekend_nights", "babies"]
# Binary 0/1 flags: passed through unchanged
BINARY = ["is_repeated_guest", "has_company"]
# Categorical features (agent is a travel-agency ID, so it is a category, not a number)
CATEGORICAL = ["hotel", "arrival_date_month", "meal", "country", "market_segment",
               "distribution_channel", "reserved_room_type", "deposit_type",
               "customer_type", "agent"]
FEATURES = NUM_LOG + NUM_LIN + BINARY + CATEGORICAL

CLIP_Q = 0.999        # upper clipping quantile for numeric outliers (fitted on train)
MIN_FREQ = 50         # categories seen fewer times in train are merged into "infrequent"
# --8<-- [end:constants]


# --8<-- [start:load]
def load_raw(path=DATA_PATH):
    """The CSV as published (agent/company 'NULL' strings are read as NaN)."""
    return pd.read_csv(path)


def clean(df):
    """Row-level rules that need no statistic from the data (safe to apply before the split).

    1. exact duplicate rows are dropped (no booking ID exists to tell them apart; kept, the
       same row would land in train and in test);
    2. bookings with 0 guests (adults + children + babies = 0) are invalid records;
    3. has_company = 1 if a company paid / booked (company is 94% missing).
    """
    df = df.drop_duplicates().copy()
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    df = df[guests > 0].copy()
    df["has_company"] = df["company"].notna().astype(int)
    df["agent"] = df["agent"].astype("Int64").astype(str).replace("<NA>", np.nan)
    return df.reset_index(drop=True)


def split(df, test_size=0.2):
    """Stratified 80/20 split with a fixed seed. Returns X_train, X_test, y_train, y_test."""
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def load_clean_split():
    """Raw CSV -> clean rows -> stratified split -> model features only."""
    X_train, X_test, y_train, y_test = split(clean(load_raw()))
    return X_train[FEATURES], X_test[FEATURES], y_train, y_test
# --8<-- [end:load]


# --8<-- [start:pipeline]
class UpperClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [lower, q-quantile of the TRAINING data]."""

    def __init__(self, q=CLIP_Q, lower=0.0):
        self.q, self.lower = q, lower

    def fit(self, X, y=None):
        self.upper_ = np.nanquantile(np.asarray(X, dtype=float), self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor():
    """ColumnTransformer that turns the raw features into a dense, NaN-free, scaled matrix."""
    num_log = Pipeline([
        ("impute", SimpleImputer(strategy="median")),          # children: 4 NaN
        ("clip", UpperClipper()),                               # extreme tails (q 0.999)
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    num_lin = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", UpperClipper()),                               # adr < 0 -> 0, adults up to 50
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),  # country, agent
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=MIN_FREQ, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num_log", num_log, NUM_LOG),
        ("num_lin", num_lin, NUM_LIN),
        ("bin", "passthrough", BINARY),
        ("cat", cat, CATEGORICAL),
    ], verbose_feature_names_out=True)
# --8<-- [end:pipeline]
