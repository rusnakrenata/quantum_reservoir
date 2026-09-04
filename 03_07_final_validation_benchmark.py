from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Canonical column aliases
#
# Earlier Week 3 scripts may use:
#   mae, rmse, nrmse, bias
#
# Later scripts may use:
#   validation_mae, validation_rmse, ...
#
# This function accepts both.
# ---------------------------------------------------------------------

COLUMN_ALIASES = {
    "model": [
        "model",
        "baseline",
        "model_name",
    ],

    "feature_set": [
        "feature_set",
    ],

    "validation_mae": [
        "validation_mae",
        "mae",
    ],

    "validation_rmse": [
        "validation_rmse",
        "rmse",
    ],

    "validation_nrmse": [
        "validation_nrmse",
        "nrmse",
        "nrmse_train_std",
    ],

    "validation_bias": [
        "validation_bias",
        "bias",
        "bias_pred_minus_actual",
    ],
}


# ---------------------------------------------------------------------
# Normalize column names
# ---------------------------------------------------------------------

def canonicalize_columns(df):

    df = df.copy()

    # Map lowercase names to actual dataframe names
    lower_to_original = {
        col.lower().strip(): col
        for col in df.columns
    }

    rename_map = {}

    for canonical_name, aliases in COLUMN_ALIASES.items():

        # Already present
        if canonical_name in df.columns:
            continue

        for alias in aliases:

            alias_lower = alias.lower()

            if alias_lower in lower_to_original:

                original_name = lower_to_original[
                    alias_lower
                ]

                rename_map[
                    original_name
                ] = canonical_name

                break

    return df.rename(
        columns=rename_map
    )


# ---------------------------------------------------------------------
# Find suitable CSV
# ---------------------------------------------------------------------

def find_and_load_metrics(
    prefix,
    require_model=False,
    require_feature_set=False,
):

    candidates = sorted(
        RESULTS_DIR.glob(
            f"{prefix}*.csv"
        )
    )

    if not candidates:

        raise FileNotFoundError(
            f"No CSV files found for prefix {prefix}"
        )

    required = {
        "validation_mae",
        "validation_rmse",
        "validation_nrmse",
        "validation_bias",
    }

    if require_model:
        required.add("model")

    if require_feature_set:
        required.add("feature_set")


    for file_path in candidates:

        try:
            raw_df = pd.read_csv(
                file_path
            )

            df = canonicalize_columns(
                raw_df
            )

        except Exception:
            continue


        if required.issubset(
            set(df.columns)
        ):

            return file_path, df


    # Useful diagnostics if nothing matched
    print(
        f"\nCould not automatically identify "
        f"metrics file for {prefix}."
    )

    print(
        "Candidates and their columns:"
    )

    for file_path in candidates:

        try:

            temp = pd.read_csv(
                file_path,
                nrows=2,
            )

            print(
                f"\n{file_path}"
            )

            print(
                list(temp.columns)
            )

        except Exception as exc:

            print(
                f"{file_path}: "
                f"could not read ({exc})"
            )


    raise FileNotFoundError(
        f"No suitable validation metrics CSV "
        f"found for prefix {prefix}"
    )


# ---------------------------------------------------------------------
# Locate and load files
# ---------------------------------------------------------------------

baseline_file, baseline = find_and_load_metrics(
    "03_02",
    require_model=True,
)


ridge_file, ridge = find_and_load_metrics(
    "03_03",
    require_feature_set=True,
)


poisson_file, poisson = find_and_load_metrics(
    "03_04",
    require_feature_set=True,
)


nb_file, nb = find_and_load_metrics(
    "03_05",
    require_feature_set=True,
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

print("=" * 78)
print("WEEK 3 - STEP 7: FINAL VALIDATION BENCHMARK")
print("=" * 78)

print("\nUsing:")
print(
    f" Naive baselines:    {baseline_file}"
)
print(
    f" Ridge:              {ridge_file}"
)
print(
    f" Poisson:            {poisson_file}"
)
print(
    f" Negative Binomial:  {nb_file}"
)


# ---------------------------------------------------------------------
# Standardize baseline table
# ---------------------------------------------------------------------

baseline_table = baseline[
    [
        "model",
        "validation_mae",
        "validation_rmse",
        "validation_nrmse",
        "validation_bias",
    ]
].copy()


baseline_table[
    "feature_set"
] = "-"


# ---------------------------------------------------------------------
# Standardize Ridge
# ---------------------------------------------------------------------

ridge_table = ridge[
    [
        "feature_set",
        "validation_mae",
        "validation_rmse",
        "validation_nrmse",
        "validation_bias",
    ]
].copy()


ridge_table[
    "model"
] = "Ridge"


# ---------------------------------------------------------------------
# Standardize Poisson
# ---------------------------------------------------------------------

poisson_table = poisson[
    [
        "feature_set",
        "validation_mae",
        "validation_rmse",
        "validation_nrmse",
        "validation_bias",
    ]
].copy()


poisson_table[
    "model"
] = "Poisson"


# ---------------------------------------------------------------------
# Standardize Negative Binomial
# ---------------------------------------------------------------------

nb_table = nb[
    [
        "feature_set",
        "validation_mae",
        "validation_rmse",
        "validation_nrmse",
        "validation_bias",
    ]
].copy()


nb_table[
    "model"
] = "Negative Binomial"


# ---------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------

benchmark = pd.concat(
    [
        baseline_table,
        ridge_table,
        poisson_table,
        nb_table,
    ],
    ignore_index=True,
)


benchmark = benchmark[
    [
        "model",
        "feature_set",
        "validation_mae",
        "validation_rmse",
        "validation_nrmse",
        "validation_bias",
    ]
].copy()


# ---------------------------------------------------------------------
# Convert metrics explicitly to numeric
# ---------------------------------------------------------------------

metric_columns = [
    "validation_mae",
    "validation_rmse",
    "validation_nrmse",
    "validation_bias",
]


for column in metric_columns:

    benchmark[column] = pd.to_numeric(
        benchmark[column],
        errors="raise",
    )


# ---------------------------------------------------------------------
# Identify seasonal-naive reference
# ---------------------------------------------------------------------

seasonal_mask = (
    benchmark["model"]
    .astype(str)
    .str.lower()
    .str.contains(
        "seasonal",
        na=False,
    )
)


seasonal_rows = benchmark[
    seasonal_mask
]


if len(seasonal_rows) != 1:

    print(
        "\nBaseline model names found:"
    )

    print(
        baseline_table[
            "model"
        ].to_list()
    )

    raise RuntimeError(
        "Could not uniquely identify "
        "the seasonal-naive baseline."
    )


seasonal_rmse = float(
    seasonal_rows.iloc[0][
        "validation_rmse"
    ]
)


seasonal_mae = float(
    seasonal_rows.iloc[0][
        "validation_mae"
    ]
)


# ---------------------------------------------------------------------
# Improvements versus seasonal naive
# ---------------------------------------------------------------------

benchmark[
    "delta_rmse_vs_seasonal"
] = (
    seasonal_rmse
    - benchmark[
        "validation_rmse"
    ]
)


benchmark[
    "rmse_improvement_vs_seasonal_pct"
] = (
    100.0
    * benchmark[
        "delta_rmse_vs_seasonal"
    ]
    / seasonal_rmse
)


benchmark[
    "delta_mae_vs_seasonal"
] = (
    seasonal_mae
    - benchmark[
        "validation_mae"
    ]
)


benchmark[
    "mae_improvement_vs_seasonal_pct"
] = (
    100.0
    * benchmark[
        "delta_mae_vs_seasonal"
    ]
    / seasonal_mae
)


benchmark[
    "beats_seasonal_rmse"
] = (
    benchmark[
        "validation_rmse"
    ]
    < seasonal_rmse
)


# ---------------------------------------------------------------------
# Rank by validation RMSE
# ---------------------------------------------------------------------

benchmark = (
    benchmark
    .sort_values(
        "validation_rmse"
    )
    .reset_index(
        drop=True
    )
)


benchmark.insert(
    0,
    "rmse_rank",
    range(
        1,
        len(benchmark) + 1,
    ),
)


# ---------------------------------------------------------------------
# Full ranking
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("FINAL WEEK 3 VALIDATION RANKING")
print("=" * 78)


print(
    benchmark
    .round(6)
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Learned models only
# ---------------------------------------------------------------------

learned_models = benchmark[
    benchmark[
        "feature_set"
    ] != "-"
].copy()


print("\n" + "=" * 78)
print("LEARNED MODELS ONLY")
print("=" * 78)


print(
    learned_models[
        [
            "rmse_rank",
            "model",
            "feature_set",
            "validation_mae",
            "validation_rmse",
            "validation_nrmse",
            "validation_bias",
            "rmse_improvement_vs_seasonal_pct",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Best overall validation model
# ---------------------------------------------------------------------

best = benchmark.iloc[0]


print("\n" + "=" * 78)
print("BEST VALIDATION MODEL")
print("=" * 78)


print(
    f"Model:                  "
    f"{best['model']}"
)

print(
    f"Feature set:            "
    f"{best['feature_set']}"
)

print(
    f"Validation MAE:         "
    f"{best['validation_mae']:.6f}"
)

print(
    f"Validation RMSE:        "
    f"{best['validation_rmse']:.6f}"
)

print(
    f"Validation NRMSE:       "
    f"{best['validation_nrmse']:.6f}"
)

print(
    f"Validation bias:        "
    f"{best['validation_bias']:.6f}"
)

print(
    f"RMSE improvement vs SN: "
    f"{best['rmse_improvement_vs_seasonal_pct']:.6f}%"
)

print(
    f"MAE improvement vs SN:  "
    f"{best['mae_improvement_vs_seasonal_pct']:.6f}%"
)


# ---------------------------------------------------------------------
# Best feature set within each learned model family
# ---------------------------------------------------------------------

best_by_family = (
    learned_models
    .sort_values(
        "validation_rmse"
    )
    .groupby(
        "model",
        as_index=False,
    )
    .first()
)


best_by_family = (
    best_by_family
    .sort_values(
        "validation_rmse"
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 78)
print("BEST FEATURE SET WITHIN EACH MODEL FAMILY")
print("=" * 78)


print(
    best_by_family[
        [
            "model",
            "feature_set",
            "validation_mae",
            "validation_rmse",
            "validation_nrmse",
            "validation_bias",
            "rmse_improvement_vs_seasonal_pct",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Decision Gate 1
# ---------------------------------------------------------------------

best_learned = (
    learned_models
    .sort_values(
        "validation_rmse"
    )
    .iloc[0]
)


gate_pass = bool(
    best_learned[
        "validation_rmse"
    ]
    < seasonal_rmse
)


absolute_reduction = (
    seasonal_rmse
    - best_learned[
        "validation_rmse"
    ]
)


relative_improvement = (
    100.0
    * absolute_reduction
    / seasonal_rmse
)


print("\n" + "=" * 78)
print("DECISION GATE 1")
print("=" * 78)


print(
    f"Seasonal-naive RMSE:       "
    f"{seasonal_rmse:.6f}"
)

print(
    f"Best learned model:        "
    f"{best_learned['model']} "
    f"{best_learned['feature_set']}"
)

print(
    f"Best learned-model RMSE:   "
    f"{best_learned['validation_rmse']:.6f}"
)

print(
    f"Absolute RMSE reduction:   "
    f"{absolute_reduction:.6f}"
)

print(
    f"Relative RMSE improvement: "
    f"{relative_improvement:.6f}%"
)

print(
    f"Beats seasonal naive:      "
    f"{gate_pass}"
)


if gate_pass:

    print(
        "\nDECISION GATE 1 STATUS: PASS"
    )

else:

    print(
        "\nDECISION GATE 1 STATUS: FAIL"
    )


# ---------------------------------------------------------------------
# Save final benchmark
# ---------------------------------------------------------------------

benchmark.to_csv(
    RESULTS_DIR /
    "03_07_final_validation_benchmark.csv",
    index=False,
)


best_by_family.to_csv(
    RESULTS_DIR /
    "03_07_best_by_model_family.csv",
    index=False,
)


decision_gate = pd.DataFrame(
    [
        {
            "seasonal_naive_rmse":
                seasonal_rmse,

            "best_model":
                best_learned[
                    "model"
                ],

            "best_feature_set":
                best_learned[
                    "feature_set"
                ],

            "best_validation_rmse":
                best_learned[
                    "validation_rmse"
                ],

            "absolute_rmse_reduction":
                absolute_reduction,

            "relative_rmse_improvement_pct":
                relative_improvement,

            "decision_gate_pass":
                gate_pass,
        }
    ]
)


decision_gate.to_csv(
    RESULTS_DIR /
    "03_07_decision_gate_1.csv",
    index=False,
)


print("\nSaved:")

print(
    " results/"
    "03_07_final_validation_benchmark.csv"
)

print(
    " results/"
    "03_07_best_by_model_family.csv"
)

print(
    " results/"
    "03_07_decision_gate_1.csv"
)


print(
    "\nStep 7 final validation benchmark finished."
)