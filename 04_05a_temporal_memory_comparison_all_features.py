from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# WEEK 4 - STEP 4.5A
# TEMPORAL-MEMORY COMPARISON ACROSS F2, F3, F4
# =============================================================================
#
# Purpose
# -------
# Preserve the complete classical benchmark surface over
#
#   Feature set F in {F2, F3, F4}
#   Window      W in {1, 2, 5, 7, 14, 21, 28}
#
# for
#
#   ESN
#   Vanilla RNN
#   GRU
#
# This is important because a later QRC model may prefer a smaller feature
# set (e.g. F2), even if F4 is best for classical models.
#
# The script produces:
#
# 1) all available architecture-level CV results,
# 2) best architecture per (model, feature_set, W),
# 3) fixed-capacity memory summaries,
# 4) best feature set / window per model,
# 5) one best-per-W plot for each feature set,
# 6) one global classical envelope over feature sets for reference.
#
# Selection metric:
#   2022-2024 chronological CV RMSE only.
#
# 2025 validation:
#   NOT used to select feature set, W, or capacity here.
#
# 2026 test:
#   NOT loaded / NOT used.
# =============================================================================


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_SETS = ["F2", "F3", "F4"]
WINDOWS = [1, 2, 5, 7, 14, 21, 28]

ESN_FILE = RESULTS_DIR / "04_02_esn_validation_metrics.csv"
RNN_FILE = RESULTS_DIR / "04_03_vanilla_rnn_cv_results.csv"
GRU_FILE = RESULTS_DIR / "04_04a_gru_screening_cv_results.csv"

OUT_ALL = RESULTS_DIR / "04_05a_temporal_memory_all_features_all_cv.csv"
OUT_BEST_FW = RESULTS_DIR / "04_05a_temporal_memory_best_per_feature_window.csv"
OUT_FIXED = RESULTS_DIR / "04_05a_temporal_memory_fixed_capacity_all_features.csv"
OUT_MODEL_BEST = RESULTS_DIR / "04_05a_temporal_memory_best_classical_configuration.csv"
OUT_GLOBAL_W = RESULTS_DIR / "04_05a_temporal_memory_global_best_per_W.csv"
OUT_SUMMARY = RESULTS_DIR / "04_05a_temporal_memory_all_features_summary.txt"

PLOT_TEMPLATE = "04_05a_temporal_memory_best_per_W_{feature}.png"
GLOBAL_PLOT = RESULTS_DIR / "04_05a_temporal_memory_global_classical_envelope.png"


# =============================================================================
# HELPERS
# =============================================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{path}"
        )


def find_column(df, candidates, label, required=True):
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(
            f"Could not identify {label} column.\n"
            f"Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_feature_value(value):
    """
    Normalize possible feature-set labels to F2 / F3 / F4.
    """
    text = str(value).strip().upper()

    if text in {"F2", "2"}:
        return "F2"
    if text in {"F3", "3"}:
        return "F3"
    if text in {"F4", "4"}:
        return "F4"

    return text


def standardize_model_table(
    df,
    model,
    feature_col,
    window_col,
    cv_col,
    hidden_col=None,
    parameter_col=None,
):
    result = pd.DataFrame({
        "model": model,
        "feature_set": df[feature_col].map(normalize_feature_value),
        "window": pd.to_numeric(
            df[window_col],
            errors="coerce"
        ),
        "cv_rmse": pd.to_numeric(
            df[cv_col],
            errors="coerce"
        ),
    })

    if hidden_col is not None:
        result["hidden_size"] = pd.to_numeric(
            df[hidden_col],
            errors="coerce"
        )
    else:
        result["hidden_size"] = np.nan

    if parameter_col is not None:
        result["parameter_count"] = pd.to_numeric(
            df[parameter_col],
            errors="coerce"
        )
    else:
        result["parameter_count"] = np.nan

    result = result.dropna(
        subset=["window", "cv_rmse"]
    )

    result["window"] = result["window"].astype(int)

    result = result[
        result["feature_set"].isin(FEATURE_SETS)
        & result["window"].isin(WINDOWS)
    ].copy()

    if result.empty:
        raise ValueError(
            f"No usable F2/F3/F4 rows found for {model}."
        )

    if result["hidden_size"].notna().any():
        result["capacity_label"] = (
            "H="
            + result["hidden_size"]
            .astype("Int64")
            .astype(str)
        )
    else:
        result["capacity_label"] = "selected"

    return result


# =============================================================================
# LOAD ESN
# =============================================================================

def load_esn():
    require_file(ESN_FILE)
    df = pd.read_csv(ESN_FILE)

    feature_col = find_column(
        df,
        ["feature_set_code", "feature_set"],
        "ESN feature set"
    )

    window_col = find_column(
        df,
        ["window_length", "window"],
        "ESN window"
    )

    cv_col = find_column(
        df,
        ["training_cv_rmse", "cv_rmse_mean", "cv_rmse"],
        "ESN CV RMSE"
    )

    hidden_col = find_column(
        df,
        [
            "reservoir_size",
            "n_reservoir",
            "reservoir_neurons",
            "n_units",
            "hidden_size",
        ],
        "ESN reservoir size",
        required=False
    )

    parameter_col = find_column(
        df,
        ["parameter_count", "n_parameters"],
        "ESN parameter count",
        required=False
    )

    return standardize_model_table(
        df=df,
        model="ESN",
        feature_col=feature_col,
        window_col=window_col,
        cv_col=cv_col,
        hidden_col=hidden_col,
        parameter_col=parameter_col,
    )


# =============================================================================
# LOAD VANILLA RNN
# =============================================================================

def load_rnn():
    require_file(RNN_FILE)
    df = pd.read_csv(RNN_FILE)

    feature_col = find_column(
        df,
        ["feature_set", "feature_set_code"],
        "RNN feature set"
    )

    window_col = find_column(
        df,
        ["window", "window_length"],
        "RNN window"
    )

    cv_col = find_column(
        df,
        ["cv_rmse_mean", "cv_rmse", "mean_cv_rmse"],
        "RNN CV RMSE"
    )

    hidden_col = find_column(
        df,
        ["hidden_size", "hidden_dim"],
        "RNN hidden size"
    )

    parameter_col = find_column(
        df,
        ["parameter_count", "n_parameters"],
        "RNN parameter count",
        required=False
    )

    return standardize_model_table(
        df=df,
        model="Vanilla RNN",
        feature_col=feature_col,
        window_col=window_col,
        cv_col=cv_col,
        hidden_col=hidden_col,
        parameter_col=parameter_col,
    )


# =============================================================================
# LOAD GRU
# =============================================================================

def load_gru():
    require_file(GRU_FILE)
    df = pd.read_csv(GRU_FILE)

    feature_col = find_column(
        df,
        ["feature_set", "feature_set_code"],
        "GRU feature set"
    )

    window_col = find_column(
        df,
        ["window", "window_length"],
        "GRU window"
    )

    cv_col = find_column(
        df,
        ["cv_rmse_mean", "cv_rmse", "mean_cv_rmse"],
        "GRU CV RMSE"
    )

    hidden_col = find_column(
        df,
        ["hidden_size", "hidden_dim"],
        "GRU hidden size"
    )

    parameter_col = find_column(
        df,
        ["parameter_count", "n_parameters"],
        "GRU parameter count",
        required=False
    )

    return standardize_model_table(
        df=df,
        model="GRU",
        feature_col=feature_col,
        window_col=window_col,
        cv_col=cv_col,
        hidden_col=hidden_col,
        parameter_col=parameter_col,
    )


# =============================================================================
# BEST ARCHITECTURE PER (MODEL, FEATURE SET, W)
# =============================================================================

def build_best_per_feature_window(all_results):
    best = (
        all_results
        .sort_values(
            [
                "model",
                "feature_set",
                "window",
                "cv_rmse",
            ]
        )
        .groupby(
            [
                "model",
                "feature_set",
                "window",
            ],
            as_index=False
        )
        .first()
    )

    rows = []

    for (model, feature_set), group in best.groupby(
        ["model", "feature_set"]
    ):
        group = group.sort_values("window").copy()

        w1 = group[
            group["window"] == 1
        ]

        if len(w1) == 1:
            rmse_w1 = float(
                w1.iloc[0]["cv_rmse"]
            )

            group["delta_rmse_vs_W1"] = (
                group["cv_rmse"] - rmse_w1
            )

            group["memory_gain_vs_W1_pct"] = (
                100.0
                * (
                    rmse_w1 - group["cv_rmse"]
                )
                / rmse_w1
            )
        else:
            group["delta_rmse_vs_W1"] = np.nan
            group["memory_gain_vs_W1_pct"] = np.nan

        rows.append(group)

    return pd.concat(
        rows,
        ignore_index=True
    )


# =============================================================================
# FIXED-CAPACITY MEMORY SUMMARY
# =============================================================================

def build_fixed_capacity_summary(all_results):
    rows = []

    finite = all_results[
        all_results["hidden_size"].notna()
    ].copy()

    for (
        model,
        feature_set,
        hidden_size
    ), group in finite.groupby(
        [
            "model",
            "feature_set",
            "hidden_size",
        ]
    ):
        if group["window"].nunique() < 2:
            continue

        curve = (
            group
            .sort_values("cv_rmse")
            .groupby(
                "window",
                as_index=False
            )
            .first()
            .sort_values("window")
        )

        w1 = curve[
            curve["window"] == 1
        ]

        if len(w1) != 1:
            continue

        rmse_w1 = float(
            w1.iloc[0]["cv_rmse"]
        )

        winner = (
            curve
            .sort_values("cv_rmse")
            .iloc[0]
        )

        best_rmse = float(
            winner["cv_rmse"]
        )

        best_window = int(
            winner["window"]
        )

        gain = (
            100.0
            * (
                rmse_w1 - best_rmse
            )
            / rmse_w1
        )

        rows.append({
            "model": model,
            "feature_set": feature_set,
            "hidden_size": int(hidden_size),
            "W1_cv_rmse": rmse_w1,
            "best_window": best_window,
            "best_cv_rmse": best_rmse,
            "best_memory_gain_vs_W1_pct": gain,
            "n_windows_available": int(
                curve["window"].nunique()
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 112)
    print("WEEK 4 - STEP 4.5A: TEMPORAL MEMORY ACROSS F2 / F3 / F4")
    print("=" * 112)

    print()
    print(f"Feature sets: {FEATURE_SETS}")
    print(f"Windows:      {WINDOWS}")
    print("Metric:       2022-2024 chronological CV RMSE")
    print("2025 validation is NOT used for F/W/capacity selection.")
    print("2026 TEST SET is NOT loaded / NOT used.")

    # -------------------------------------------------------------------------
    # Load full result surfaces
    # -------------------------------------------------------------------------

    esn = load_esn()
    rnn = load_rnn()
    gru = load_gru()

    all_results = pd.concat(
        [esn, rnn, gru],
        ignore_index=True,
        sort=False
    )

    all_results = all_results.sort_values(
        [
            "model",
            "feature_set",
            "hidden_size",
            "window",
            "cv_rmse",
        ],
        na_position="last"
    ).reset_index(drop=True)

    all_results.to_csv(
        OUT_ALL,
        index=False
    )

    print()
    print("=" * 112)
    print("AVAILABLE ARCHITECTURE-LEVEL DATA")
    print("=" * 112)

    counts = (
        all_results
        .groupby(
            ["model", "feature_set"]
        )
        .size()
        .rename("rows")
        .reset_index()
    )

    print(
        counts.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Best architecture per model / feature / W
    # -------------------------------------------------------------------------

    best_fw = build_best_per_feature_window(
        all_results
    )

    best_fw.to_csv(
        OUT_BEST_FW,
        index=False
    )

    # -------------------------------------------------------------------------
    # Print one best-per-W table for every feature set
    # -------------------------------------------------------------------------

    print()
    print("=" * 112)
    print("BEST-PER-W TABLES")
    print("=" * 112)

    for feature_set in FEATURE_SETS:
        sub = best_fw[
            best_fw["feature_set"] == feature_set
        ]

        pivot = sub.pivot(
            index="window",
            columns="model",
            values="cv_rmse"
        )

        gain = sub.pivot(
            index="window",
            columns="model",
            values="memory_gain_vs_W1_pct"
        )

        print()
        print("-" * 112)
        print(f"{feature_set}: BEST CV RMSE PER WINDOW")
        print("-" * 112)

        print(
            pivot.to_string(
                float_format=lambda x: f"{x:.6f}"
            )
        )

        print()
        print(f"{feature_set}: MEMORY GAIN VS W=1 (%)")

        print(
            gain.to_string(
                float_format=lambda x: f"{x:+.3f}"
            )
        )

    # -------------------------------------------------------------------------
    # Fixed-capacity summaries
    # -------------------------------------------------------------------------

    fixed = build_fixed_capacity_summary(
        all_results
    )

    fixed.to_csv(
        OUT_FIXED,
        index=False
    )

    print()
    print("=" * 112)
    print("FIXED-CAPACITY MEMORY SUMMARY")
    print("=" * 112)

    if fixed.empty:
        print("No repeated fixed-capacity curves available.")
    else:
        print(
            fixed.sort_values(
                [
                    "model",
                    "feature_set",
                    "hidden_size",
                ]
            ).to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    # -------------------------------------------------------------------------
    # Best classical configuration per model across ALL F and W
    # -------------------------------------------------------------------------

    model_best_rows = []

    for model, group in best_fw.groupby("model"):
        winner = (
            group
            .sort_values("cv_rmse")
            .iloc[0]
        )

        model_best_rows.append({
            "model": model,
            "feature_set": winner["feature_set"],
            "window": int(winner["window"]),
            "capacity": winner["capacity_label"],
            "cv_rmse": float(winner["cv_rmse"]),
            "parameter_count": winner["parameter_count"],
        })

    model_best = pd.DataFrame(
        model_best_rows
    ).sort_values("cv_rmse")

    model_best.to_csv(
        OUT_MODEL_BEST,
        index=False
    )

    print()
    print("=" * 112)
    print("BEST CLASSICAL CONFIGURATION PER MODEL ACROSS F2/F3/F4 AND ALL W")
    print("=" * 112)

    print(
        model_best.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # -------------------------------------------------------------------------
    # Global best-per-W classical envelope across ALL feature sets
    # -------------------------------------------------------------------------

    global_w = (
        best_fw
        .sort_values(
            [
                "model",
                "window",
                "cv_rmse",
            ]
        )
        .groupby(
            [
                "model",
                "window",
            ],
            as_index=False
        )
        .first()
    )

    global_w.to_csv(
        OUT_GLOBAL_W,
        index=False
    )

    print()
    print("=" * 112)
    print("GLOBAL CLASSICAL BEST-PER-W ENVELOPE")
    print("=" * 112)

    global_pivot = global_w.pivot(
        index="window",
        columns="model",
        values="cv_rmse"
    )

    print(
        global_pivot.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()
    print("Feature set selected at each point:")

    feature_pivot = global_w.pivot(
        index="window",
        columns="model",
        values="feature_set"
    )

    print(
        feature_pivot.to_string()
    )

    # -------------------------------------------------------------------------
    # PLOTS: one separate figure per feature set
    # -------------------------------------------------------------------------

    plot_paths = []

    for feature_set in FEATURE_SETS:
        sub = best_fw[
            best_fw["feature_set"] == feature_set
        ].copy()

        fig, ax = plt.subplots(
            figsize=(8.5, 5.5)
        )

        for model, group in sub.groupby("model"):
            group = group.sort_values("window")

            ax.plot(
                group["window"],
                group["cv_rmse"],
                marker="o",
                linewidth=2,
                label=model,
            )

        ax.set_xlabel(
            "Temporal window W (days)"
        )
        ax.set_ylabel(
            "Best chronological CV RMSE"
        )
        ax.set_title(
            f"{feature_set}: best model-family performance at each temporal window"
        )
        ax.set_xticks(WINDOWS)
        ax.grid(True, alpha=0.25)
        ax.legend()

        fig.tight_layout()

        path = RESULTS_DIR / PLOT_TEMPLATE.format(
            feature=feature_set
        )

        fig.savefig(
            path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        plot_paths.append(path)

    # -------------------------------------------------------------------------
    # GLOBAL CLASSICAL ENVELOPE PLOT
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    for model, group in global_w.groupby("model"):
        group = group.sort_values("window")

        ax.plot(
            group["window"],
            group["cv_rmse"],
            marker="o",
            linewidth=2,
            label=model,
        )

    ax.set_xlabel(
        "Temporal window W (days)"
    )
    ax.set_ylabel(
        "Best chronological CV RMSE"
    )
    ax.set_title(
        "Global classical envelope across F2, F3 and F4"
    )
    ax.set_xticks(WINDOWS)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        GLOBAL_PLOT,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # Text summary
    # -------------------------------------------------------------------------

    with open(
        OUT_SUMMARY,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "WEEK 4 - STEP 4.5A: TEMPORAL MEMORY ACROSS F2/F3/F4\n"
        )
        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "Metric: 2022-2024 chronological CV RMSE\n"
        )
        f.write(
            "2025 validation not used for F/W/capacity selection.\n"
        )
        f.write(
            "2026 test set not used.\n\n"
        )

        for feature_set in FEATURE_SETS:
            sub = best_fw[
                best_fw["feature_set"] == feature_set
            ]

            pivot = sub.pivot(
                index="window",
                columns="model",
                values="cv_rmse"
            )

            gain = sub.pivot(
                index="window",
                columns="model",
                values="memory_gain_vs_W1_pct"
            )

            f.write(
                f"\n{feature_set}: BEST CV RMSE PER W\n"
            )
            f.write(
                pivot.to_string(
                    float_format=lambda x: f"{x:.6f}"
                )
            )

            f.write(
                f"\n\n{feature_set}: MEMORY GAIN VS W=1 (%)\n"
            )
            f.write(
                gain.to_string(
                    float_format=lambda x: f"{x:+.3f}"
                )
            )
            f.write("\n")

        f.write(
            "\n\nBEST CLASSICAL CONFIGURATION PER MODEL\n"
        )
        f.write(
            model_best.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

        f.write(
            "\n\nGLOBAL BEST-PER-W ENVELOPE\n"
        )
        f.write(
            global_pivot.to_string(
                float_format=lambda x: f"{x:.6f}"
            )
        )

        f.write(
            "\n\nFEATURE SET USED IN GLOBAL ENVELOPE\n"
        )
        f.write(
            feature_pivot.to_string()
        )

        f.write(
            "\n\n2026 TEST SET WAS NOT USED.\n"
        )

    # -------------------------------------------------------------------------
    # Saved
    # -------------------------------------------------------------------------

    print()
    print("=" * 112)
    print("SAVED")
    print("=" * 112)

    print(OUT_ALL)
    print(OUT_BEST_FW)
    print(OUT_FIXED)
    print(OUT_MODEL_BEST)
    print(OUT_GLOBAL_W)
    print(OUT_SUMMARY)

    for path in plot_paths:
        print(path)

    print(GLOBAL_PLOT)

    print()
    print("2026 TEST SET WAS NOT USED.")


if __name__ == "__main__":
    main()

