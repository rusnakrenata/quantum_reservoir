from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# WEEK 4 - STEP 4.5B
# FINAL CLASSICAL BENCHMARK
# =============================================================================
#
# PURPOSE
# -------
# Freeze the final classical benchmark before moving to QRC.
#
# Model families:
#   - Seasonal naive
#   - Ridge
#   - Poisson
#   - Negative Binomial
#   - ESN
#   - Vanilla RNN
#   - GRU
#
# IMPORTANT MODEL-SELECTION RULE
# ------------------------------
# Temporal architectures are selected using 2022-2024 chronological CV only.
# 2025 is used only for official validation evaluation.
#
# In particular:
#   ESN must NOT be selected by minimum 2025 validation RMSE.
#   The script chooses the ESN row with minimum training_cv_rmse.
#
# COMMON NRMSE
# ------------
# Earlier scripts used different denominators in some places.
# Here every model uses exactly:
#
#       NRMSE = RMSE / std(y_2025)
#
# where std(y_2025) is calculated once from the same 365 validation targets.
#
# 2026 TEST SET IS NOT LOADED OR USED.
# =============================================================================


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

OUT_CSV = RESULTS_DIR / "04_05b_final_classical_benchmark.csv"
OUT_TXT = RESULTS_DIR / "04_05b_final_classical_benchmark_summary.txt"
OUT_PLOT = RESULTS_DIR / "04_05b_final_classical_rmse_ranking.png"


# =============================================================================
# CURRENT FINAL FALLBACKS
# =============================================================================
#
# These are used ONLY if the corresponding final prediction file cannot be
# found. If prediction files exist, metrics are recomputed directly.
#
# They reflect the final corrected Week 4 runs.
# =============================================================================

RNN_FALLBACK = {
    "feature_set": "F4",
    "window": 2,
    "capacity": "H=64",
    "parameter_count": 4609,
    "mae": 3.948176,
    "rmse": 5.027691,
    "bias": -1.206917,
}

GRU_FALLBACK = {
    "feature_set": "F4",
    "window": 1,
    "capacity": "H=1",
    "parameter_count": 26,
    "mae": 3.928557,
    "rmse": 4.920860,
    "bias": -0.887161,
}


# =============================================================================
# HELPERS
# =============================================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def find_column(df, candidates, label, required=True):
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(
            f"Could not identify {label}.\n"
            f"Tried columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_feature(value):
    text = str(value).strip().upper()

    if text in {"F2", "2"}:
        return "F2"
    if text in {"F3", "3"}:
        return "F3"
    if text in {"F4", "4"}:
        return "F4"

    if text in {"-", "NONE", "NAN"}:
        return "-"

    return text


def load_common_validation_target():
    """
    Load one canonical 2025 validation target array.

    W=1 is used only because it contains the same 365 validation targets
    without losing observations at the validation boundary.
    """

    path = RESULTS_DIR / "04_01_windows_F4_W01.npz"
    require_file(path)

    with np.load(path, allow_pickle=True) as data:

        y_key = None

        for candidate in [
            "y_validation",
            "y_val",
            "Y_validation",
            "Y_val",
        ]:
            if candidate in data.files:
                y_key = candidate
                break

        if y_key is None:
            raise KeyError(
                f"Could not find validation target in {path}.\n"
                f"Available keys: {data.files}"
            )

        y_val = np.asarray(
            data[y_key],
            dtype=float
        ).reshape(-1)

    if len(y_val) != 365:
        raise ValueError(
            f"Expected 365 validation targets, got {len(y_val)}."
        )

    return y_val


def metrics_from_prediction_file(path):
    """
    Read a validation prediction file and recompute MAE, RMSE and bias.

    Supports the column names used throughout Weeks 3 and 4.
    """

    df = pd.read_csv(path)

    actual_col = find_column(
        df,
        [
            "actual",
            "y_true",
            "target",
            "observed",
        ],
        "actual target column"
    )

    pred_col = find_column(
        df,
        [
            "prediction",
            "y_pred",
            "predicted",
            "forecast",
        ],
        "prediction column"
    )

    y = pd.to_numeric(
        df[actual_col],
        errors="coerce"
    ).to_numpy(dtype=float)

    p = pd.to_numeric(
        df[pred_col],
        errors="coerce"
    ).to_numpy(dtype=float)

    mask = np.isfinite(y) & np.isfinite(p)

    y = y[mask]
    p = p[mask]

    if len(y) != 365:
        raise ValueError(
            f"{path}: expected 365 usable validation rows, got {len(y)}."
        )

    residual = p - y

    return {
        "mae": float(
            np.mean(
                np.abs(residual)
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        ),
        "bias": float(
            np.mean(residual)
        ),
        "actual_mean": float(
            np.mean(y)
        ),
        "prediction_mean": float(
            np.mean(p)
        ),
    }


def extract_week3_model(benchmark_df, model_name):
    """
    Extract one frozen Week-3 row.

    Ridge/Poisson/NB are frozen at the best Week-3 feature-set result.
    Seasonal naive is the explicit seasonal_naive_7d row.
    """

    model_col = find_column(
        benchmark_df,
        ["model"],
        "Week-3 model column"
    )

    if model_name == "Seasonal naive":
        mask = (
            benchmark_df[model_col]
            .astype(str)
            .str.lower()
            .eq("seasonal_naive_7d")
        )
    else:
        mask = (
            benchmark_df[model_col]
            .astype(str)
            .str.lower()
            .eq(model_name.lower())
        )

    sub = benchmark_df[mask].copy()

    if sub.empty:
        raise ValueError(
            f"Could not find {model_name} in Week-3 benchmark."
        )

    rmse_col = find_column(
        sub,
        ["validation_rmse"],
        f"{model_name} RMSE"
    )

    # For model families with F2/F3/F4 rows, use the already frozen best
    # Week-3 feature-set result = smallest validation RMSE from that family.
    # This reproduces Week-3's final benchmark. For all three learned GLM /
    # Ridge families this is F4.
    row = (
        sub
        .sort_values(rmse_col)
        .iloc[0]
    )

    feature_col = find_column(
        sub,
        ["feature_set"],
        f"{model_name} feature set",
        required=False
    )

    mae_col = find_column(
        sub,
        ["validation_mae"],
        f"{model_name} MAE"
    )

    bias_col = find_column(
        sub,
        ["validation_bias"],
        f"{model_name} bias"
    )

    return {
        "model": model_name,
        "feature_set":
            normalize_feature(row[feature_col])
            if feature_col is not None
            else "-",
        "window": "-",
        "capacity": "-",
        "parameter_count": np.nan,
        "mae": float(row[mae_col]),
        "rmse": float(row[rmse_col]),
        "bias": float(row[bias_col]),
        "selection_source": "Week 3 frozen benchmark",
    }


# =============================================================================
# ESN
# =============================================================================

def extract_esn():
    """
    Select the ESN globally using 2022-2024 chronological CV only.

    DO NOT sort by validation_rmse.
    """

    path = RESULTS_DIR / "04_02_esn_validation_metrics.csv"
    require_file(path)

    df = pd.read_csv(path)

    cv_col = find_column(
        df,
        [
            "training_cv_rmse",
            "cv_rmse_mean",
            "cv_rmse",
        ],
        "ESN training CV RMSE"
    )

    feature_col = find_column(
        df,
        [
            "feature_set_code",
            "feature_set",
        ],
        "ESN feature set"
    )

    window_col = find_column(
        df,
        [
            "window_length",
            "window",
        ],
        "ESN window"
    )

    mae_col = find_column(
        df,
        ["validation_mae"],
        "ESN validation MAE"
    )

    rmse_col = find_column(
        df,
        ["validation_rmse"],
        "ESN validation RMSE"
    )

    # Bias may not be present in older exported validation-metrics files.
    bias_col = find_column(
        df,
        ["validation_bias", "bias"],
        "ESN validation bias",
        required=False
    )

    # Global CV winner over all F and W.
    row = (
        df
        .sort_values(cv_col)
        .iloc[0]
    )

    feature_set = normalize_feature(
        row[feature_col]
    )

    window = int(
        row[window_col]
    )

    # If validation bias is missing, recover the exact selected config from
    # the validation-predictions file when possible.
    mae = float(row[mae_col])
    rmse = float(row[rmse_col])
    bias = (
        float(row[bias_col])
        if bias_col is not None
        else np.nan
    )

    pred_path = RESULTS_DIR / "04_02_esn_validation_predictions.csv"

    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)

        fcol = find_column(
            pred_df,
            ["feature_set_code", "feature_set"],
            "ESN prediction feature set",
            required=False
        )

        wcol = find_column(
            pred_df,
            ["window_length", "window"],
            "ESN prediction window",
            required=False
        )

        if fcol is not None and wcol is not None:
            mask = (
                pred_df[fcol].map(normalize_feature).eq(feature_set)
                &
                pd.to_numeric(
                    pred_df[wcol],
                    errors="coerce"
                ).eq(window)
            )

            selected = pred_df[mask].copy()

            if not selected.empty:
                actual_col = find_column(
                    selected,
                    ["actual", "y_true", "target"],
                    "ESN actual"
                )
                pred_col = find_column(
                    selected,
                    ["prediction", "y_pred", "predicted"],
                    "ESN prediction"
                )

                y = pd.to_numeric(
                    selected[actual_col],
                    errors="coerce"
                ).to_numpy(dtype=float)

                p = pd.to_numeric(
                    selected[pred_col],
                    errors="coerce"
                ).to_numpy(dtype=float)

                good = np.isfinite(y) & np.isfinite(p)
                y = y[good]
                p = p[good]

                if len(y) == 365:
                    residual = p - y
                    mae = float(np.mean(np.abs(residual)))
                    rmse = float(np.sqrt(np.mean(residual ** 2)))
                    bias = float(np.mean(residual))

    reservoir_size_col = find_column(
        df,
        [
            "reservoir_size",
            "n_reservoir",
            "reservoir_neurons",
        ],
        "ESN reservoir size",
        required=False
    )

    reservoir_size = (
        int(row[reservoir_size_col])
        if reservoir_size_col is not None
        else 100
    )

    # Trainable parameters = ridge readout only:
    # one coefficient per reservoir feature + intercept.
    trainable_parameters = reservoir_size + 1

    return {
        "model": "ESN",
        "feature_set": feature_set,
        "window": window,
        "capacity": f"N={reservoir_size}",
        "parameter_count": trainable_parameters,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "selection_source":
            f"minimum 2022-2024 CV RMSE = {float(row[cv_col]):.6f}",
    }


# =============================================================================
# FINAL VANILLA RNN
# =============================================================================

def extract_rnn():
    """
    Use the final tuned RNN prediction file if available.
    Architecture metadata are taken from the final tuning CV table.
    """

    prediction_path = first_existing([
        RESULTS_DIR / "04_03c_vanilla_rnn_tuned_validation_predictions.csv",
        RESULTS_DIR / "04_03c_vanilla_rnn_final_validation_predictions.csv",
        RESULTS_DIR / "04_03_vanilla_rnn_final_validation_predictions.csv",
    ])

    result = dict(RNN_FALLBACK)

    if prediction_path is not None:
        m = metrics_from_prediction_file(
            prediction_path
        )

        result["mae"] = m["mae"]
        result["rmse"] = m["rmse"]
        result["bias"] = m["bias"]

    tuning_path = RESULTS_DIR / "04_03c_vanilla_rnn_tuning_cv.csv"

    selection_source = "final corrected RNN fallback"

    if tuning_path.exists():
        df = pd.read_csv(tuning_path)

        cv_col = find_column(
            df,
            ["cv_rmse_mean", "cv_rmse"],
            "RNN tuning CV RMSE"
        )

        row = (
            df
            .sort_values(cv_col)
            .iloc[0]
        )

        h_col = find_column(
            df,
            ["hidden_size", "hidden_dim"],
            "RNN hidden size"
        )

        param_col = find_column(
            df,
            ["parameter_count", "n_parameters"],
            "RNN parameter count",
            required=False
        )

        result["capacity"] = f"H={int(row[h_col])}"

        if param_col is not None:
            result["parameter_count"] = int(
                row[param_col]
            )

        selection_source = (
            f"minimum tuned 2022-2024 CV RMSE = "
            f"{float(row[cv_col]):.6f}"
        )

    return {
        "model": "Vanilla RNN",
        "feature_set": result["feature_set"],
        "window": result["window"],
        "capacity": result["capacity"],
        "parameter_count": result["parameter_count"],
        "mae": result["mae"],
        "rmse": result["rmse"],
        "bias": result["bias"],
        "selection_source": selection_source,
    }


# =============================================================================
# FINAL GRU
# =============================================================================

def extract_gru():
    """
    Use the final GRU prediction file if available.
    Architecture is selected from 2022-2024 CV only.
    """

    result = dict(GRU_FALLBACK)

    cv_path = RESULTS_DIR / "04_04a_gru_screening_cv_results.csv"

    selection_source = "final corrected GRU fallback"

    if cv_path.exists():
        df = pd.read_csv(cv_path)

        cv_col = find_column(
            df,
            ["cv_rmse_mean", "cv_rmse"],
            "GRU CV RMSE"
        )

        row = (
            df
            .sort_values(cv_col)
            .iloc[0]
        )

        f_col = find_column(
            df,
            ["feature_set", "feature_set_code"],
            "GRU feature set"
        )

        w_col = find_column(
            df,
            ["window", "window_length"],
            "GRU window"
        )

        h_col = find_column(
            df,
            ["hidden_size", "hidden_dim"],
            "GRU hidden size"
        )

        p_col = find_column(
            df,
            ["parameter_count", "n_parameters"],
            "GRU parameter count",
            required=False
        )

        result["feature_set"] = normalize_feature(
            row[f_col]
        )
        result["window"] = int(row[w_col])
        result["capacity"] = f"H={int(row[h_col])}"

        if p_col is not None:
            result["parameter_count"] = int(row[p_col])

        selection_source = (
            f"minimum 2022-2024 CV RMSE = "
            f"{float(row[cv_col]):.6f}"
        )

    prediction_path = first_existing([
        RESULTS_DIR / "04_04a_gru_validation_predictions.csv",
        RESULTS_DIR / "04_04a_gru_best_validation_predictions.csv",
    ])

    if prediction_path is not None:
        m = metrics_from_prediction_file(
            prediction_path
        )
        result["mae"] = m["mae"]
        result["rmse"] = m["rmse"]
        result["bias"] = m["bias"]

    return {
        "model": "GRU",
        "feature_set": result["feature_set"],
        "window": result["window"],
        "capacity": result["capacity"],
        "parameter_count": result["parameter_count"],
        "mae": result["mae"],
        "rmse": result["rmse"],
        "bias": result["bias"],
        "selection_source": selection_source,
    }


# =============================================================================
# PARAMETER COUNTS FOR WEEK-3 MODELS
# =============================================================================

def assign_simple_parameter_counts(row):
    """
    Interpretable trainable-parameter counts for the final F4 classical models.

    F4 has 5 numerical input columns:
      C_t_z, D_sin, D_cos, P_t_z, holiday

    + intercept -> 6 regression coefficients.

    Negative Binomial additionally estimates alpha -> 7.
    """

    if row["model"] == "Seasonal naive":
        return 0

    if row["model"] in {"Ridge", "Poisson"}:
        if row["feature_set"] == "F4":
            return 6

    if row["model"] == "Negative Binomial":
        if row["feature_set"] == "F4":
            return 7

    return row["parameter_count"]


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 112)
    print("WEEK 4 - STEP 4.5B: FINAL CLASSICAL BENCHMARK")
    print("=" * 112)

    print()
    print(
        "Temporal architectures are selected using "
        "2022-2024 chronological CV only."
    )
    print(
        "2025 is used only for official validation metrics."
    )
    print(
        "2026 TEST SET is NOT loaded / NOT used."
    )

    # =========================================================================
    # COMMON 2025 TARGET SCALE
    # =========================================================================

    y_2025 = load_common_validation_target()

    validation_mean = float(
        np.mean(y_2025)
    )

    validation_std = float(
        np.std(
            y_2025,
            ddof=0
        )
    )

    print()
    print("=" * 112)
    print("COMMON 2025 TARGET SCALE")
    print("=" * 112)

    print(
        f"N                 : {len(y_2025)}"
    )
    print(
        f"Mean              : {validation_mean:.6f}"
    )
    print(
        f"Std (ddof=0)      : {validation_std:.6f}"
    )
    print(
        "Common NRMSE rule : RMSE / std(y_2025)"
    )

    # =========================================================================
    # WEEK-3 FROZEN MODELS
    # =========================================================================

    week3_path = RESULTS_DIR / "03_07_final_validation_benchmark.csv"
    require_file(week3_path)

    week3 = pd.read_csv(
        week3_path
    )

    rows = []

    for model in [
        "Seasonal naive",
        "Ridge",
        "Poisson",
        "Negative Binomial",
    ]:
        rows.append(
            extract_week3_model(
                week3,
                model
            )
        )

    # =========================================================================
    # WEEK-4 TEMPORAL MODELS
    # =========================================================================

    rows.append(
        extract_esn()
    )

    rows.append(
        extract_rnn()
    )

    rows.append(
        extract_gru()
    )

    benchmark = pd.DataFrame(
        rows
    )

    # Parameter counts for simple models.
    benchmark["parameter_count"] = benchmark.apply(
        assign_simple_parameter_counts,
        axis=1
    )

    # =========================================================================
    # CONSISTENT NRMSE
    # =========================================================================

    benchmark["nrmse_common_2025_std"] = (
        benchmark["rmse"]
        / validation_std
    )

    # =========================================================================
    # IMPROVEMENTS
    # =========================================================================

    seasonal_rmse = float(
        benchmark.loc[
            benchmark["model"] == "Seasonal naive",
            "rmse"
        ].iloc[0]
    )

    seasonal_mae = float(
        benchmark.loc[
            benchmark["model"] == "Seasonal naive",
            "mae"
        ].iloc[0]
    )

    benchmark["rmse_improvement_vs_seasonal_pct"] = (
        100.0
        * (
            seasonal_rmse
            - benchmark["rmse"]
        )
        / seasonal_rmse
    )

    benchmark["mae_improvement_vs_seasonal_pct"] = (
        100.0
        * (
            seasonal_mae
            - benchmark["mae"]
        )
        / seasonal_mae
    )

    # =========================================================================
    # RANKING
    # =========================================================================

    benchmark = (
        benchmark
        .sort_values(
            [
                "rmse",
                "mae",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    benchmark.insert(
        0,
        "rmse_rank",
        np.arange(
            1,
            len(benchmark) + 1
        )
    )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    benchmark.to_csv(
        OUT_CSV,
        index=False
    )

    # =========================================================================
    # PRINT FINAL TABLE
    # =========================================================================

    print()
    print("=" * 112)
    print("FINAL OFFICIAL 2025 CLASSICAL BENCHMARK")
    print("=" * 112)

    display_cols = [
        "rmse_rank",
        "model",
        "feature_set",
        "window",
        "capacity",
        "parameter_count",
        "mae",
        "rmse",
        "nrmse_common_2025_std",
        "bias",
        "rmse_improvement_vs_seasonal_pct",
    ]

    print(
        benchmark[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # =========================================================================
    # MODEL-SELECTION AUDIT
    # =========================================================================

    print()
    print("=" * 112)
    print("MODEL-SELECTION AUDIT")
    print("=" * 112)

    for _, row in benchmark.iterrows():
        print(
            f"{row['model']:<20} | "
            f"{row['feature_set']:<2} | "
            f"W={str(row['window']):<2} | "
            f"{row['capacity']:<8} | "
            f"{row['selection_source']}"
        )

    # =========================================================================
    # BEST MODEL
    # =========================================================================

    best = benchmark.iloc[0]

    print()
    print("=" * 112)
    print("FROZEN CLASSICAL REFERENCE FOR LATER QRC")
    print("=" * 112)

    print(
        f"Best model          : {best['model']}"
    )
    print(
        f"Feature set         : {best['feature_set']}"
    )
    print(
        f"Temporal window     : {best['window']}"
    )
    print(
        f"Capacity            : {best['capacity']}"
    )
    print(
        f"2025 MAE            : {best['mae']:.6f}"
    )
    print(
        f"2025 RMSE           : {best['rmse']:.6f}"
    )
    print(
        f"2025 common NRMSE   : "
        f"{best['nrmse_common_2025_std']:.6f}"
    )
    print(
        f"2025 bias           : {best['bias']:+.6f}"
    )

    # =========================================================================
# RMSE RANKING PLOT
# =========================================================================

    plot_df = benchmark.sort_values(
        "rmse",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = ax.barh(
        plot_df["model"],
        plot_df["rmse"]
    )

    ax.set_xlabel(
        "2025 validation RMSE"
    )

    ax.set_ylabel(
        "Model"
    )

    ax.set_title(
        "Final classical benchmark before QRC"
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    # -------------------------------------------------------------------------
    # ADD NUMERICAL RMSE VALUE TO EACH BAR
    # -------------------------------------------------------------------------

    for bar, rmse in zip(
        bars,
        plot_df["rmse"]
    ):
        ax.text(
            bar.get_width() + 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{rmse:.4f}",
            va="center",
            ha="left"
        )

    # Add a little space on the right for the labels
    ax.set_xlim(
        0,
        plot_df["rmse"].max() * 1.10
    )

    fig.tight_layout()

    fig.savefig(
        OUT_PLOT,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # =========================================================================
    # TEXT SUMMARY
    # =========================================================================

    with open(
        OUT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "WEEK 4 - STEP 4.5B: FINAL CLASSICAL BENCHMARK\n"
        )
        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "Temporal architectures selected using "
            "2022-2024 chronological CV only.\n"
        )

        f.write(
            "2025 used only for official validation.\n"
        )

        f.write(
            "2026 test set not used.\n\n"
        )

        f.write(
            f"2025 target mean: {validation_mean:.6f}\n"
        )

        f.write(
            f"2025 target std:  {validation_std:.6f}\n"
        )

        f.write(
            "Common NRMSE = RMSE / std(y_2025)\n\n"
        )

        f.write(
            "FINAL RANKING\n"
        )

        f.write(
            benchmark[
                display_cols
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

        f.write(
            "\n\nMODEL-SELECTION AUDIT\n"
        )

        for _, row in benchmark.iterrows():
            f.write(
                f"{row['model']} | "
                f"{row['feature_set']} | "
                f"W={row['window']} | "
                f"{row['capacity']} | "
                f"{row['selection_source']}\n"
            )

        f.write(
            "\nFROZEN CLASSICAL REFERENCE\n"
        )

        f.write(
            f"Model: {best['model']}\n"
        )
        f.write(
            f"Feature set: {best['feature_set']}\n"
        )
        f.write(
            f"Window: {best['window']}\n"
        )
        f.write(
            f"Capacity: {best['capacity']}\n"
        )
        f.write(
            f"MAE: {best['mae']:.6f}\n"
        )
        f.write(
            f"RMSE: {best['rmse']:.6f}\n"
        )
        f.write(
            f"Common NRMSE: "
            f"{best['nrmse_common_2025_std']:.6f}\n"
        )
        f.write(
            f"Bias: {best['bias']:+.6f}\n"
        )

        f.write(
            "\n2026 TEST SET WAS NOT USED.\n"
        )

    # =========================================================================
    # SAVED
    # =========================================================================

    print()
    print("=" * 112)
    print("SAVED")
    print("=" * 112)

    print(OUT_CSV)
    print(OUT_TXT)
    print(OUT_PLOT)

    print()
    print(
        "2026 TEST SET WAS NOT USED."
    )


if __name__ == "__main__":
    main()
