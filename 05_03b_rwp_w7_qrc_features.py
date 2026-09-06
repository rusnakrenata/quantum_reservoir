from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RESULTS_DIR / "03_01_preprocessed_samples.csv"

# IMPORTANT:
# Reuse exactly the same couplings as CONT / 5.3A.
COUPLING_FILE = RESULTS_DIR / "05_03a_reservoir_couplings.csv"

FEATURE_OUTPUT = RESULTS_DIR / "05_03b_rwp_w7_qrc_features.csv"

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "05_03b_rwp_w7_qrc_features_summary.txt"
)


# ============================================================
# Rewinding configuration
# ============================================================

WINDOW = 7

N_INPUT = 4
N_MEMORY = 2
N_QUBITS = N_INPUT + N_MEMORY

Q_MEMORY = [4, 5]

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]


# ============================================================
# Same reference parameters as CONT / 5.3A
# ============================================================

ALPHA = 1.0

H_X = 0.5

DELTA_T = 0.8

TROTTER_STEPS = 2


# ============================================================
# Required columns
# ============================================================

REQUIRED_COLUMNS = [
    "input_date",
    "target_date",
    "split",

    "C_t_z",
    "D_sin",
    "D_cos",
    "P_t_z",

    "is_public_holiday_t_plus_1",
]


# ============================================================
# Logging
# ============================================================

output_lines = []


def log(text=""):
    text = str(text)
    print(text)
    output_lines.append(text)


# ============================================================
# Pauli matrices
# ============================================================

I2 = np.array(
    [
        [1, 0],
        [0, 1],
    ],
    dtype=complex,
)

X = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=complex,
)

Z = np.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=complex,
)


# ============================================================
# Ry-encoded one-qubit density matrix
# ============================================================

def ry_density(theta):

    c = np.cos(
        theta / 2.0
    )

    s = np.sin(
        theta / 2.0
    )

    return np.array(
        [
            [c * c, c * s],
            [c * s, s * s],
        ],
        dtype=complex,
    )


# ============================================================
# Encode one insurance input row
# ============================================================

def encode_input_row(row):

    # --------------------------------------------------------
    # Claim count
    # --------------------------------------------------------

    c_scaled = (
        np.clip(
            float(row["C_t_z"]),
            -3.0,
            3.0,
        )
        / 3.0
    )

    theta_c = (
        ALPHA
        * c_scaled
    )

    # --------------------------------------------------------
    # Weekday
    # --------------------------------------------------------

    theta_d = np.arctan2(
        float(row["D_sin"]),
        float(row["D_cos"]),
    )

    theta_d = np.mod(
        theta_d,
        2.0 * np.pi,
    )

    # --------------------------------------------------------
    # Policy count
    # --------------------------------------------------------

    p_scaled = (
        np.clip(
            float(row["P_t_z"]),
            -3.0,
            3.0,
        )
        / 3.0
    )

    theta_p = (
        ALPHA
        * p_scaled
    )

    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    holiday = float(
        row[
            "is_public_holiday_t_plus_1"
        ]
    )

    theta_h = (
        np.pi
        * holiday
    )

    # --------------------------------------------------------
    # q0 = claim
    # q1 = weekday
    # q2 = policy
    # q3 = holiday
    #
    # Numerical order:
    #
    # q3 kron q2 kron q1 kron q0
    # --------------------------------------------------------

    rho_q0 = ry_density(
        theta_c
    )

    rho_q1 = ry_density(
        theta_d
    )

    rho_q2 = ry_density(
        theta_p
    )

    rho_q3 = ry_density(
        theta_h
    )

    rho_input = np.kron(
        rho_q3,
        np.kron(
            rho_q2,
            np.kron(
                rho_q1,
                rho_q0,
            ),
        ),
    )

    angles = {
        "theta_C": theta_c,
        "theta_D": theta_d,
        "theta_P": theta_p,
        "theta_H": theta_h,
    }

    return rho_input, angles


# ============================================================
# Initial memory = |00><00|
# ============================================================

def initial_memory_density():

    rho = np.zeros(
        (
            2 ** N_MEMORY,
            2 ** N_MEMORY,
        ),
        dtype=complex,
    )

    rho[0, 0] = 1.0

    return rho


# ============================================================
# Load the SAME couplings used by CONT
# ============================================================

def load_couplings():

    if not COUPLING_FILE.exists():
        raise FileNotFoundError(
            f"Missing coupling file: "
            f"{COUPLING_FILE}\n"
            "Run 05_03a first."
        )

    coupling_df = pd.read_csv(
        COUPLING_FILE
    )

    couplings = {}

    for _, row in coupling_df.iterrows():

        edge = (
            int(row["q_i"]),
            int(row["q_j"]),
        )

        couplings[edge] = float(
            row["J_ij"]
        )

    expected_edges = set(
        EDGES
    )

    loaded_edges = set(
        couplings.keys()
    )

    if expected_edges != loaded_edges:
        raise ValueError(
            "Loaded coupling topology does "
            "not match expected topology.\n"
            f"Expected: {expected_edges}\n"
            f"Loaded:   {loaded_edges}"
        )

    return couplings


# ============================================================
# Build same Trotter reservoir as CONT
# ============================================================

def build_trotter_unitary(
    couplings,
):

    qc = QuantumCircuit(
        N_QUBITS
    )

    dt_slice = (
        DELTA_T
        / TROTTER_STEPS
    )

    for _ in range(
        TROTTER_STEPS
    ):

        # ----------------------------------------------------
        # ZZ block
        # ----------------------------------------------------

        for (
            q_i,
            q_j,
        ), J_ij in couplings.items():

            qc.rzz(
                2.0
                * J_ij
                * dt_slice,
                q_i,
                q_j,
            )

        # ----------------------------------------------------
        # X-field block
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            qc.rx(
                2.0
                * H_X
                * dt_slice,
                q,
            )

    U = np.asarray(
        Operator(qc).data,
        dtype=complex,
    )

    return qc, U


# ============================================================
# Partial trace over the four input qubits
# ============================================================

def trace_out_input(
    rho_global,
):

    d_m = 2 ** N_MEMORY
    d_i = 2 ** N_INPUT

    reshaped = (
        rho_global.reshape(
            d_m,
            d_i,
            d_m,
            d_i,
        )
    )

    # rho_M[a,b] =
    #     sum_i rho[a,i,b,i]
    rho_memory = np.einsum(
        "aibi->ab",
        reshaped,
    )

    return rho_memory


# ============================================================
# Density diagnostics
# ============================================================

def density_purity(rho):

    return float(
        np.real(
            np.trace(
                rho @ rho
            )
        )
    )


def hermiticity_error(rho):

    return float(
        np.linalg.norm(
            rho
            - rho.conj().T,
            ord="fro",
        )
    )


# ============================================================
# Observable construction
# ============================================================

def single_qubit_operator(
    target_q,
    pauli_matrix,
):

    matrices = []

    for q in reversed(
        range(N_QUBITS)
    ):

        if q == target_q:
            matrices.append(
                pauli_matrix
            )
        else:
            matrices.append(
                I2
            )

    operator = matrices[0]

    for matrix in matrices[1:]:

        operator = np.kron(
            operator,
            matrix,
        )

    return operator


def build_observables():

    observables = []

    for q in range(
        N_QUBITS
    ):

        observables.append(
            (
                f"X{q}",
                single_qubit_operator(
                    q,
                    X,
                ),
            )
        )

        observables.append(
            (
                f"Z{q}",
                single_qubit_operator(
                    q,
                    Z,
                ),
            )
        )

    return observables


# ============================================================
# Expectation value
# ============================================================

def expectation_value(
    rho,
    operator,
):

    return np.sum(
        rho
        * operator.T
    )


# ============================================================
# Resolve target column
# ============================================================

def resolve_target_column(df):

    candidates = [
        "property_damage_claim_count_t_plus_1",
        "property_damage_claim_count_target",
        "target_property_damage_claim_count",
        "target_claim_count",
        "C_t_plus_1",
    ]

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not resolve target column.\n"
        + "\n".join(
            df.columns
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 5 - STEP 5.3B")
    log("REWINDING IDEAL-QRC FEATURE GENERATION")
    log(f"REFERENCE WINDOW W = {WINDOW}")
    log("=" * 80)
    log()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                missing
            )
        )

    target_column = (
        resolve_target_column(
            df
        )
    )

    df["input_date"] = (
        pd.to_datetime(
            df["input_date"]
        )
    )

    df["target_date"] = (
        pd.to_datetime(
            df["target_date"]
        )
    )

    df = (
        df
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    if len(df) != 1692:
        raise ValueError(
            f"Expected 1692 rows, "
            f"found {len(df)}."
        )

    if not (
        df["target_date"]
        .is_monotonic_increasing
    ):
        raise ValueError(
            "Target dates are "
            "not chronological."
        )

    # --------------------------------------------------------
    # Daily-continuity audit
    # --------------------------------------------------------

    input_day_diff = (
        df["input_date"]
        .diff()
        .dropna()
        .dt.days
    )

    daily_continuity_ok = bool(
        (input_day_diff == 1)
        .all()
    )

    if not daily_continuity_ok:
        raise ValueError(
            "Input dates are not "
            "strictly daily/contiguous."
        )

    log(
        f"Loaded samples             = "
        f"{len(df)}"
    )

    log(
        f"Reference window           = "
        f"{WINDOW}"
    )

    log(
        f"Full-window output samples = "
        f"{len(df) - WINDOW + 1}"
    )

    log(
        f"Daily continuity audit     = "
        f"{'PASS' if daily_continuity_ok else 'FAIL'}"
    )

    log()

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    log("Architecture:")

    log(
        f"  F4 injection qubits = "
        f"{N_INPUT}"
    )

    log(
        f"  memory qubits       = "
        f"{N_MEMORY}"
    )

    log(
        f"  total qubits        = "
        f"{N_QUBITS}"
    )

    log(
        f"  window W            = "
        f"{WINDOW}"
    )

    log(
        f"  alpha               = "
        f"{ALPHA}"
    )

    log(
        f"  h_x                 = "
        f"{H_X}"
    )

    log(
        f"  Delta_t             = "
        f"{DELTA_T}"
    )

    log(
        f"  Trotter steps       = "
        f"{TROTTER_STEPS}"
    )

    log(
        "  Y_memory            = OFF"
    )

    log()

    # --------------------------------------------------------
    # Reuse exact same J couplings
    # --------------------------------------------------------

    couplings = (
        load_couplings()
    )

    log(
        "Loaded fixed CONT couplings:"
    )

    for edge in EDGES:

        log(
            f"  J{edge} = "
            f"{couplings[edge]:+.12f}"
        )

    log()

    # --------------------------------------------------------
    # Fixed unitary
    # --------------------------------------------------------

    _, U = (
        build_trotter_unitary(
            couplings
        )
    )

    identity = np.eye(
        2 ** N_QUBITS,
        dtype=complex,
    )

    unitarity_error = float(
        np.linalg.norm(
            U.conj().T
            @ U
            - identity,
            ord="fro",
        )
    )

    log(
        "Reservoir-unitary check:"
    )

    log(
        f"  ||U^dagger U - I||_F "
        f"= {unitarity_error:.12e}"
    )

    log()

    # --------------------------------------------------------
    # Precompute all encoded daily inputs once
    # --------------------------------------------------------

    encoded_inputs = []

    encoded_angles = []

    for _, row in df.iterrows():

        rho_input, angles = (
            encode_input_row(
                row
            )
        )

        encoded_inputs.append(
            rho_input
        )

        encoded_angles.append(
            angles
        )

    # --------------------------------------------------------
    # Observables
    # --------------------------------------------------------

    observables = (
        build_observables()
    )

    feature_names = [
        name
        for name, _
        in observables
    ]

    log(
        "Reference observables:"
    )

    log(
        "  "
        + ", ".join(
            feature_names
        )
    )

    log()

    # --------------------------------------------------------
    # Rewinding loop
    # --------------------------------------------------------

    output_rows = []

    max_trace_error = 0.0
    max_hermiticity = 0.0
    max_feature_imaginary = 0.0
    max_bound_violation = 0.0

    final_memory_purities = []

    # First complete W=7 window ends at index 6.
    for end_idx in range(
        WINDOW - 1,
        len(df),
    ):

        start_idx = (
            end_idx
            - WINDOW
            + 1
        )

        # ====================================================
        # REWIND:
        # reset memory at start of EVERY forecasting window
        # ====================================================

        rho_memory = (
            initial_memory_density()
        )

        rho_global = None

        # ====================================================
        # Replay exactly W chronological inputs
        # ====================================================

        for replay_idx in range(
            start_idx,
            end_idx + 1,
        ):

            rho_input = (
                encoded_inputs[
                    replay_idx
                ]
            )

            # Numerical ordering:
            #
            # memory kron input
            rho_pre = np.kron(
                rho_memory,
                rho_input,
            )

            rho_global = (
                U
                @ rho_pre
                @ U.conj().T
            )

            rho_memory = (
                trace_out_input(
                    rho_global
                )
            )

        # ====================================================
        # Extract features ONLY after final input in window
        # ====================================================

        features = {}

        for (
            name,
            operator,
        ) in observables:

            value = (
                expectation_value(
                    rho_global,
                    operator,
                )
            )

            real_value = float(
                np.real(
                    value
                )
            )

            imag_value = abs(
                float(
                    np.imag(
                        value
                    )
                )
            )

            bound_violation = max(
                0.0,
                abs(real_value) - 1.0,
            )

            max_feature_imaginary = max(
                max_feature_imaginary,
                imag_value,
            )

            max_bound_violation = max(
                max_bound_violation,
                bound_violation,
            )

            features[name] = (
                real_value
            )

        # ====================================================
        # Final-memory diagnostics
        # ====================================================

        trace_error = abs(
            np.trace(
                rho_memory
            )
            - 1.0
        )

        herm_error = (
            hermiticity_error(
                rho_memory
            )
        )

        max_trace_error = max(
            max_trace_error,
            float(trace_error),
        )

        max_hermiticity = max(
            max_hermiticity,
            herm_error,
        )

        memory_purity = (
            density_purity(
                rho_memory
            )
        )

        final_memory_purities.append(
            memory_purity
        )

        # ====================================================
        # Output metadata refers to prediction at end_idx
        # ====================================================

        current_row = (
            df.iloc[
                end_idx
            ]
        )

        output_row = {
            "window_W": WINDOW,

            "window_start_input_date": (
                df.iloc[
                    start_idx
                ]["input_date"]
            ),

            "window_end_input_date": (
                current_row[
                    "input_date"
                ]
            ),

            "input_date": (
                current_row[
                    "input_date"
                ]
            ),

            "target_date": (
                current_row[
                    "target_date"
                ]
            ),

            "split": (
                current_row[
                    "split"
                ]
            ),

            target_column: (
                current_row[
                    target_column
                ]
            ),

            "theta_C": (
                encoded_angles[
                    end_idx
                ]["theta_C"]
            ),

            "theta_D": (
                encoded_angles[
                    end_idx
                ]["theta_D"]
            ),

            "theta_P": (
                encoded_angles[
                    end_idx
                ]["theta_P"]
            ),

            "theta_H": (
                encoded_angles[
                    end_idx
                ]["theta_H"]
            ),

            "memory_purity": (
                memory_purity
            ),
        }

        output_row.update(
            features
        )

        output_rows.append(
            output_row
        )

        completed = len(
            output_rows
        )

        if (
            completed % 250 == 0
            or
            end_idx == len(df) - 1
        ):

            log(
                f"Generated "
                f"{completed:4d}/"
                f"{len(df) - WINDOW + 1} "
                f"rewound windows..."
            )

    # --------------------------------------------------------
    # Output dataframe
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        output_rows
    )

    feature_df.to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    X_qrc = (
        feature_df[
            feature_names
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Audits
    # --------------------------------------------------------

    expected_shape = (
        len(df) - WINDOW + 1,
        2 * N_QUBITS,
    )

    shape_ok = (
        X_qrc.shape
        == expected_shape
    )

    finite_ok = bool(
        np.isfinite(
            X_qrc
        ).all()
    )

    # Every W=7 window should span exactly 6 calendar days
    # from first input to final input.
    window_span_days = (
        pd.to_datetime(
            feature_df[
                "window_end_input_date"
            ]
        )
        -
        pd.to_datetime(
            feature_df[
                "window_start_input_date"
            ]
        )
    ).dt.days

    window_length_ok = bool(
        (
            window_span_days
            == WINDOW - 1
        )
        .all()
    )

    split_counts = (
        feature_df[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    log()
    log("=" * 80)
    log("RWP W=7 SUMMARY")
    log("=" * 80)

    log(
        f"Feature matrix shape       = "
        f"{X_qrc.shape}"
    )

    log(
        f"Expected shape             = "
        f"{expected_shape}"
    )

    log(
        f"Shape check                = "
        f"{'PASS' if shape_ok else 'FAIL'}"
    )

    log(
        f"All features finite        = "
        f"{'PASS' if finite_ok else 'FAIL'}"
    )

    log(
        f"Window-length audit        = "
        f"{'PASS' if window_length_ok else 'FAIL'}"
    )

    log(
        f"Output split counts        = "
        f"{split_counts}"
    )

    log()

    log(
        f"First output target date   = "
        f"{feature_df['target_date'].min()}"
    )

    log(
        f"Last output target date    = "
        f"{feature_df['target_date'].max()}"
    )

    log()

    log(
        f"Maximum memory trace error = "
        f"{max_trace_error:.12e}"
    )

    log(
        f"Maximum memory Hermiticity = "
        f"{max_hermiticity:.12e}"
    )

    log(
        f"Maximum feature imaginary  = "
        f"{max_feature_imaginary:.12e}"
    )

    log(
        f"Maximum bound violation    = "
        f"{max_bound_violation:.12e}"
    )

    log()

    log(
        f"Minimum final memory purity= "
        f"{np.min(final_memory_purities):.12f}"
    )

    log(
        f"Mean final memory purity   = "
        f"{np.mean(final_memory_purities):.12f}"
    )

    log(
        f"Maximum final memory purity= "
        f"{np.max(final_memory_purities):.12f}"
    )

    log()

    log(
        f"QRC feature range          = "
        f"[{X_qrc.min():+.6f}, "
        f"{X_qrc.max():+.6f}]"
    )

    log()

    log(
        "Saved rewinding features to:"
    )

    log(
        f"  {FEATURE_OUTPUT}"
    )

    # --------------------------------------------------------
    # Hard correctness failures
    # --------------------------------------------------------

    if not shape_ok:
        raise RuntimeError(
            "Wrong RWP feature matrix shape."
        )

    if not finite_ok:
        raise RuntimeError(
            "RWP feature matrix contains "
            "NaN/inf."
        )

    if not window_length_ok:
        raise RuntimeError(
            "RWP window-length audit failed."
        )

    if (
        max_bound_violation
        > 1e-10
    ):
        raise RuntimeError(
            "Pauli expectation-value "
            "bound violation."
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    SUMMARY_OUTPUT.write_text(
        "\n".join(
            output_lines
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()