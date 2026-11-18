"""
Utility functions for logging metrics, visualizing results, exporting reports,
and pipeline utilities for DDoS/Fraud detection.
"""

import os
import sys
import io
import json
import csv
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ── Canonical feature list (must match ddos_detector.py FEATURES exactly) ─────
FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Total",
    "Bwd IAT Total",
    "Packet Length Mean",
    "Packet Length Std",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
]

NUM_FEATURES = len(FEATURES)   # 22

# ── Project-relative path helpers ──────────────────────────────────────────────
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

def _proj(relpath: str) -> str:
    """Return an absolute path relative to the project root."""
    return os.path.join(_PROJECT_DIR, relpath)

# Well-known paths used across the live pipeline
MODEL_PATH        = _proj(os.path.join("docs", "global_model.h5"))
SCALER_PATH       = _proj("scaler.pkl")
TRAINING_CSV_PATH = _proj(
    os.path.join(
        "data",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    )
)
PCAP_PATH                  = _proj("traffic.pcap")
LIVE_FEATURES_CSV          = _proj("live_features.csv")
LIVE_DATASET_CSV           = _proj("live_dataset.csv")
LIVE_DATASET_PREDICTED_CSV = _proj("live_dataset_predicted.csv")
PREDICTIONS_CSV            = _proj("predictions.csv")


# ── Logger ─────────────────────────────────────────────────────────────────────
def get_logger(name: str = "ddos_live") -> logging.Logger:
    """
    Return a module-level logger with a UTF-8-safe stream handler.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# ── Scaler helpers ─────────────────────────────────────────────────────────────
def generate_scaler(csv_path: str = TRAINING_CSV_PATH,
                    save_path: str = SCALER_PATH) -> StandardScaler:
    """
    Fit a StandardScaler on the CICIDS2017 training CSV using the same 22
    features and preprocessing steps as ddos_detector.preprocess().
    """
    log = get_logger()
    log.info("Generating scaler from %s …", csv_path)

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    available = [f for f in FEATURES if f in df.columns]
    if len(available) < NUM_FEATURES:
        missing = set(FEATURES) - set(available)
        log.warning("CSV missing %d feature(s): %s", len(missing), missing)

    X = (
        df[available]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .values.astype(np.float32)
    )

    scaler = StandardScaler()
    scaler.fit(X)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(scaler, save_path)
    log.info("Scaler saved -> %s (fitted on %d rows × %d features)",
             save_path, X.shape[0], X.shape[1])
    return scaler


def load_scaler_strict(scaler_path: str = SCALER_PATH) -> StandardScaler:
    """
    INFERENCE-ONLY scaler loader.
    """
    log = get_logger()
    if not os.path.exists(scaler_path):
        log.error(
            "\n"
            "  ================================================================\n"
            "  ERROR: scaler.pkl not found at:\n"
            "         %s\n"
            "\n"
            "  The live testing pipeline requires a pre-fitted scaler.\n"
            "  scaler.pkl must be generated ONCE during the training phase:\n"
            "\n"
            "      python src/generate_scaler.py\n"
            "\n"
            "  The live pipeline will NOT create or fit a new scaler.\n"
            "  ================================================================",
            scaler_path,
        )
        raise SystemExit(1)

    scaler = joblib.load(scaler_path)
    log.info("Scaler loaded (inference-only): %s", scaler_path)
    return scaler


def load_or_create_scaler(scaler_path: str = SCALER_PATH,
                          csv_path: str = TRAINING_CSV_PATH) -> StandardScaler:
    """
    Load scaler.pkl if it exists; otherwise generate it from the training CSV.
    """
    log = get_logger()
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        log.info("Scaler loaded from %s", scaler_path)
        return scaler

    log.info("scaler.pkl not found — generating from training dataset …")
    return generate_scaler(csv_path=csv_path, save_path=scaler_path)


# ── Feature validation ─────────────────────────────────────────────────────────
def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and reorder df columns to match the canonical FEATURES list.
    """
    log = get_logger()

    if df.empty:
        raise ValueError("validate_features(): received an empty DataFrame.")

    for feat in FEATURES:
        if feat not in df.columns:
            log.warning("Feature '%s' missing from extracted data — filled with 0", feat)
            df[feat] = 0.0

    df = df[FEATURES].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = df.astype(np.float32)

    return df


def ensure_dir(path: str) -> None:
    """Create the directory for path (file or directory) if it doesn't exist."""
    target = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if target:
        os.makedirs(target, exist_ok=True)


def safe_div(numerator: float, denominator: float,
             default: float = 0.0) -> float:
    """Division that returns default when denominator is zero."""
    return numerator / denominator if denominator != 0 else default


# ── Metrics and evaluation helpers ─────────────────────────────────────────────
def log_metrics(metrics_dict, prefix=""):
    """
    Logs training or evaluation metrics in formatted output.
    """
    print(f"{prefix} Metrics:")
    for key, val in metrics_dict.items():
        print(f"  {key}: {val:.4f}")


def save_metrics(metrics_dict, filepath, fmt='json'):
    """
    Saves metrics to a file in JSON or CSV format.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if fmt == 'json':
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
    elif fmt == 'csv':
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in metrics_dict.items():
                writer.writerow([k, v])
    else:
        raise ValueError("Unsupported format. Use 'json' or 'csv'.")

    print(f"Metrics saved to {filepath}")


def display_classification_report(y_true, y_pred):
    """
    Prints classification report.
    """
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()
