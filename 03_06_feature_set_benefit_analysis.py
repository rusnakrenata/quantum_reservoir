from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------
# Helper: find the saved metrics file for a given Week 3 step
# ---------------------------------------------------------------------

def find_metrics_file(step_prefix):

    candidates = sorted(
        RESULTS_DIR.glob(
            f"{step_prefix}*.csv"
        )
    )

    required_columns = {
        "feature_set",
        "validation_mae",
        "validation_rmse",
        "validation_nrmse",
        "validation_bias",
    }

    for file_path in candidates:

        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue

        if required_columns.issubset(
            set(df.columns)
        ):
            return file_path

    raise FileNotFoundError(
        f"No suitable validation metrics CSV found "
        f"for prefix {step_prefix}"
    )


# ---------------------------------------------------------------------
# Locate result files
# ---------------------------------------------------------------------

ridge_file = find_metrics_file("03_03")
poisson_file = find_metrics_file("03_04")
nb_file = find_metrics_file("03_05")


print("=" * 78)
print("WEEK 3 - STEP 6: FEATURE-SET BENEFIT ANALYSIS")
print("=" * 78)

print("\nUsing:")
print(f" Ridge:             {ridge_file}")
print(f" Poisson:           {poisson_file}")
print(f" Negative Binomial: {nb_file}")


# ---------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------

ridge = pd.read_csv(ridge_file)
poisson = pd.read_csv(poisson_file)
nb = pd.read_csv(nb_file)


ridge["model_family"] = "Ridge"
poisson["model_family"] = "Poisson"
nb["model_family"] = "Negative Binomial"


metrics = pd.concat(
    [
        ridge,
        poisson,
        nb,
    ],
    ignore_index=True,
)


# ---------------------------------------------------------------------
# Keep required fields only
# ---------------------------------------------------------------------

metrics = metrics[
    [
        "model_family",
        "feature_set",
        "validation_mae",
        "validation_rmse",
        "validation_nrmse",
        "validation_bias",
    ]
].copy()


# ---------------------------------------------------------------------
# Ensure correct feature-set order
# ---------------------------------------------------------------------

feature_order = {
    "F2": 2,
    "F3": 3,
    "F4": 4,
}

metrics["feature_order"] = (
    metrics["feature_set"]
    .map(feature_order)
)

metrics = (
    metrics
    .sort_values(
        [
            "model_family",
            "feature_order",
        ]
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------
# Print complete benchmark table
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("MODEL / FEATURE-SET VALIDATION RESULTS")
print("=" * 78)

print(
    metrics[
        [
            "model_family",
            "feature_set",
            "validation_mae",
            "validation_rmse",
            "validation_nrmse",
            "validation_bias",
        ]
    ]
    .round(6)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Incremental feature benefit
# ---------------------------------------------------------------------

transitions = [
    (
        "F2",
        "F3",
        "P_t",
    ),
    (
        "F3",
        "F4",
        "H_t_plus_1",
    ),
]


benefit_rows = []


for model_family in [
    "Ridge",
    "Poisson",
    "Negative Binomial",
]:

    model_df = (
        metrics[
            metrics["model_family"]
            == model_family
        ]
        .set_index("feature_set")
    )

    for (
        before_set,
        after_set,
        added_feature,
    ) in transitions:

        before = model_df.loc[
            before_set
        ]

        after = model_df.loc[
            after_set
        ]


        rmse_gain = (
            before["validation_rmse"]
            - after["validation_rmse"]
        )

        rmse_gain_pct = (
            100.0
            * rmse_gain
            / before["validation_rmse"]
        )


        mae_gain = (
            before["validation_mae"]
            - after["validation_mae"]
        )

        mae_gain_pct = (
            100.0
            * mae_gain
            / before["validation_mae"]
        )


        abs_bias_before = abs(
            before["validation_bias"]
        )

        abs_bias_after = abs(
            after["validation_bias"]
        )

        abs_bias_reduction = (
            abs_bias_before
            - abs_bias_after
        )


        benefit_rows.append(
            {
                "model_family":
                    model_family,

                "transition":
                    f"{before_set}->{after_set}",

                "added_feature":
                    added_feature,

                "rmse_before":
                    before[
                        "validation_rmse"
                    ],

                "rmse_after":
                    after[
                        "validation_rmse"
                    ],

                "delta_rmse":
                    rmse_gain,

                "rmse_improvement_pct":
                    rmse_gain_pct,

                "mae_before":
                    before[
                        "validation_mae"
                    ],

                "mae_after":
                    after[
                        "validation_mae"
                    ],

                "delta_mae":
                    mae_gain,

                "mae_improvement_pct":
                    mae_gain_pct,

                "abs_bias_before":
                    abs_bias_before,

                "abs_bias_after":
                    abs_bias_after,

                "abs_bias_reduction":
                    abs_bias_reduction,
            }
        )


benefits = pd.DataFrame(
    benefit_rows
)


# ---------------------------------------------------------------------
# Print incremental benefits
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("INCREMENTAL FEATURE BENEFIT")
print("=" * 78)

print(
    benefits.round(6)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------

summary = (
    benefits
    .groupby(
        [
            "transition",
            "added_feature",
        ]
    )
    .agg(
        mean_delta_rmse=(
            "delta_rmse",
            "mean",
        ),

        min_delta_rmse=(
            "delta_rmse",
            "min",
        ),

        max_delta_rmse=(
            "delta_rmse",
            "max",
        ),

        mean_rmse_improvement_pct=(
            "rmse_improvement_pct",
            "mean",
        ),

        min_rmse_improvement_pct=(
            "rmse_improvement_pct",
            "min",
        ),

        max_rmse_improvement_pct=(
            "rmse_improvement_pct",
            "max",
        ),
    )
    .reset_index()
)


# Did the added feature help in every model?
consistency = (
    benefits
    .assign(
        improved=lambda x:
            x["delta_rmse"] > 0
    )
    .groupby(
        [
            "transition",
            "added_feature",
        ]
    )["improved"]
    .all()
    .reset_index(
        name="improved_in_all_models"
    )
)


summary = summary.merge(
    consistency,
    on=[
        "transition",
        "added_feature",
    ],
    how="left",
)


print("\n" + "=" * 78)
print("CROSS-MODEL FEATURE BENEFIT SUMMARY")
print("=" * 78)

print(
    summary.round(6)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Total F2 -> F4 improvement
# ---------------------------------------------------------------------

total_rows = []


for model_family in [
    "Ridge",
    "Poisson",
    "Negative Binomial",
]:

    model_df = (
        metrics[
            metrics["model_family"]
            == model_family
        ]
        .set_index("feature_set")
    )

    rmse_f2 = model_df.loc[
        "F2",
        "validation_rmse",
    ]

    rmse_f4 = model_df.loc[
        "F4",
        "validation_rmse",
    ]

    delta = (
        rmse_f2
        - rmse_f4
    )

    improvement_pct = (
        100.0
        * delta
        / rmse_f2
    )


    total_rows.append(
        {
            "model_family":
                model_family,

            "rmse_F2":
                rmse_f2,

            "rmse_F4":
                rmse_f4,

            "delta_rmse_F2_to_F4":
                delta,

            "improvement_pct_F2_to_F4":
                improvement_pct,
        }
    )


total_improvement = pd.DataFrame(
    total_rows
)


print("\n" + "=" * 78)
print("TOTAL BENEFIT: F2 -> F4")
print("=" * 78)

print(
    total_improvement.round(6)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Semantic feature/resource table
# ---------------------------------------------------------------------

resource_table = pd.DataFrame(
    {
        "feature_set": [
            "F2",
            "F3",
            "F4",
        ],

        "semantic_input_count": [
            2,
            3,
            4,
        ],

        "classical_numeric_columns": [
            3,
            4,
            5,
        ],

        "semantic_inputs": [
            "C_t, D_t_plus_1",
            "C_t, D_t_plus_1, P_t",
            "C_t, D_t_plus_1, P_t, H_t_plus_1",
        ],
    }
)


print("\n" + "=" * 78)
print("SEMANTIC INPUT COMPLEXITY")
print("=" * 78)

print(
    resource_table
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

metrics.drop(
    columns=["feature_order"]
).to_csv(
    RESULTS_DIR /
    "03_06_model_feature_set_metrics.csv",
    index=False,
)


benefits.to_csv(
    RESULTS_DIR /
    "03_06_incremental_feature_benefit.csv",
    index=False,
)


summary.to_csv(
    RESULTS_DIR /
    "03_06_cross_model_feature_benefit_summary.csv",
    index=False,
)


total_improvement.to_csv(
    RESULTS_DIR /
    "03_06_total_F2_to_F4_improvement.csv",
    index=False,
)


resource_table.to_csv(
    RESULTS_DIR /
    "03_06_semantic_input_complexity.csv",
    index=False,
)


print("\nSaved:")
print(
    " results/"
    "03_06_model_feature_set_metrics.csv"
)

print(
    " results/"
    "03_06_incremental_feature_benefit.csv"
)

print(
    " results/"
    "03_06_cross_model_feature_benefit_summary.csv"
)

print(
    " results/"
    "03_06_total_F2_to_F4_improvement.csv"
)

print(
    " results/"
    "03_06_semantic_input_complexity.csv"
)


print(
    "\nStep 6 feature-set benefit analysis finished."
)