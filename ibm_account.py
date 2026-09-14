"""
ibm_account.py
--------------
IBM Quantum account setup, backend selection and Aer simulator helpers
PRIMARY authentication now uses a hardcoded IBM Cloud API key when configured.
for Quantum Reservoir Computing (QRC) experiments.

First-time setup
----------------
1. Create an IBM Cloud API key in the IBM Quantum Platform.

2. Install required packages:

       pip install qiskit qiskit-ibm-runtime qiskit-aer

3. Save your IBM Quantum credentials:

       python ibm_account.py --save YOUR_API_KEY

   Credentials are stored locally in:

       ~/.qiskit/qiskit-ibm.json

4. Verify the setup:

       python ibm_account.py --list

5. Find the least-busy available QPU:

       python ibm_account.py --least-busy

"""

import os
import argparse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHANNEL = "ibm_quantum_platform"

# Environment variable that can optionally contain the IBM Cloud API key.
ENV_API_KEY = "IBM_QUANTUM_API_KEY"

# ---------------------------------------------------------------------------
# PRIMARY IBM CLOUD API KEY
# ---------------------------------------------------------------------------
# Put the NEW IBM Cloud API key here.
#
# Authentication priority is now:
#   1. PRIMARY_API_KEY below
#   2. environment variable IBM_QUANTUM_API_KEY
#   3. previously saved Qiskit Runtime credentials
#
# IMPORTANT:
# - Do not commit a real API key to Git/source control.
# - Never print the key itself.
# - Leave this as an empty string if you want to disable the hardcoded key.
PRIMARY_API_KEY = "MQ00QFkABL3UPFe1S0nVQeX0VKccSlbAy4J0IGBsZU6x"


# ---------------------------------------------------------------------------
# Account management
# ---------------------------------------------------------------------------

def setup_account(api_key: str) -> None:
    """
    Save IBM Quantum Platform credentials locally.

    Parameters
    ----------
    api_key : str
        IBM Cloud API key associated with the IBM Quantum Platform account.

    Notes
    -----
    Credentials are stored by Qiskit Runtime, normally in:

        ~/.qiskit/qiskit-ibm.json
    """

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

    except ImportError:
        raise ImportError(
            "qiskit-ibm-runtime is not installed.\n"
            "Install it using:\n"
            "    pip install qiskit-ibm-runtime"
        )

    QiskitRuntimeService.save_account(
        token=api_key,
        channel=DEFAULT_CHANNEL,
        plans_preference=["open"],
        overwrite=True,
        set_as_default=True,
    )

    print(
        "[ibm_account] IBM Quantum Platform account saved successfully."
    )


def get_service():
    """
    Return an authenticated QiskitRuntimeService.

    Authentication priority
    -----------------------
    1. PRIMARY_API_KEY hardcoded in this file
    2. Environment variable IBM_QUANTUM_API_KEY
    3. Previously saved Qiskit Runtime credentials

    Returns
    -------
    QiskitRuntimeService
    """

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

    except ImportError:
        raise ImportError(
            "qiskit-ibm-runtime is not installed.\n"
            "Install it using:\n"
            "    pip install qiskit-ibm-runtime"
        )

    # ---------------------------------------------------------------
    # Option 1:
    # Hardcoded PRIMARY API key
    # ---------------------------------------------------------------

    primary_key = str(PRIMARY_API_KEY).strip()

    # Treat the placeholder as "not configured".
    primary_is_configured = (
        bool(primary_key)
        and primary_key != "PASTE_NEW_IBM_CLOUD_API_KEY_HERE"
    )

    if primary_is_configured:
        print(
            "[ibm_account] Using PRIMARY_API_KEY hardcoded in ibm_account.py."
        )

        try:
            return QiskitRuntimeService(
                token=primary_key,
                channel=DEFAULT_CHANNEL,
                instance="auto",
                plans_preference=["open"],
            )
        except Exception as exc:
            raise RuntimeError(
                "The hardcoded PRIMARY_API_KEY was found, but IBM Quantum "
                "authentication failed. Because this key is configured as "
                "PRIMARY, the script will not silently fall back to another "
                "account.\n\n"
                f"Original error:\n{exc}"
            ) from exc

    # ---------------------------------------------------------------
    # Option 2:
    # API key provided through environment variable
    # ---------------------------------------------------------------

    api_key = os.getenv(ENV_API_KEY)

    if api_key:
        print(
            f"[ibm_account] PRIMARY_API_KEY is not configured. "
            f"Using API key from environment variable {ENV_API_KEY}."
        )

        return QiskitRuntimeService(
            token=api_key,
            channel=DEFAULT_CHANNEL,
            instance="auto",
            plans_preference=["open"],
        )

    # ---------------------------------------------------------------
    # Option 3:
    # Load saved account
    # ---------------------------------------------------------------

    print(
        "[ibm_account] PRIMARY_API_KEY and environment API key are not "
        "configured. Loading saved Qiskit Runtime credentials."
    )

    try:
        service = QiskitRuntimeService(
            instance="auto",
            channel=DEFAULT_CHANNEL,
            plans_preference=["open"],
        )

    except Exception as exc:
        raise RuntimeError(
            "IBM Quantum credentials could not be loaded.\n\n"
            "Configure PRIMARY_API_KEY in ibm_account.py, set "
            "IBM_QUANTUM_API_KEY, or run:\n"
            "    python ibm_account.py --save YOUR_API_KEY\n\n"
            f"Original error:\n{exc}"
        ) from exc

    return service


# ---------------------------------------------------------------------------
# IBM hardware backends
# ---------------------------------------------------------------------------

def get_ibm_backend(name: str, service=None):
    """
    Return a specific IBM Quantum hardware backend.

    Parameters
    ----------
    name : str
        IBM backend name, for example:

            ibm_brisbane
            ibm_kingston

        Availability depends on the IBM Quantum account and plan.

    service : QiskitRuntimeService, optional
        Existing service instance.

    Returns
    -------
    IBMBackend
    """

    if service is None:
        service = get_service()

    try:
        backend = service.backend(name)

    except Exception as exc:
        raise RuntimeError(
            f"Could not access IBM backend '{name}'.\n"
            "Use:\n"
            "    python ibm_account.py --list\n"
            "to see currently available backends."
        ) from exc

    print(
        f"[ibm_account] Backend: {backend.name} "
        f"({backend.num_qubits} qubits)"
    )

    return backend


def list_available_backends(
    service=None,
    operational_only: bool = True,
):
    """
    Print IBM Quantum hardware backends available to the account.

    Parameters
    ----------
    service : QiskitRuntimeService, optional

    operational_only : bool
        If True, show only operational devices.

    Returns
    -------
    list
        List of available backend objects.
    """

    if service is None:
        service = get_service()

    try:

        if operational_only:

            backends = service.backends(
                operational=True,
                simulator=False,
            )

        else:

            backends = service.backends(
                simulator=False,
            )

    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve IBM Quantum backends."
        ) from exc

    print()

    print(
        f"{'Backend':<30}"
        f"{'Qubits':>10}"
        f"{'Pending jobs':>15}"
        f"{'Status':>20}"
    )

    print("-" * 75)

    # ---------------------------------------------------------------
    # Sort by number of qubits and then name
    # ---------------------------------------------------------------

    backends = sorted(
        backends,
        key=lambda backend: (
            backend.num_qubits,
            backend.name,
        ),
    )

    for backend in backends:

        try:

            status = backend.status()

            pending_jobs = getattr(
                status,
                "pending_jobs",
                "?",
            )

            status_msg = getattr(
                status,
                "status_msg",
                "?",
            )

        except Exception:

            pending_jobs = "?"
            status_msg = "?"

        print(
            f"{backend.name:<30}"
            f"{backend.num_qubits:>10}"
            f"{str(pending_jobs):>15}"
            f"{str(status_msg):>20}"
        )

    print()

    print(
        f"[ibm_account] Found {len(backends)} hardware backend(s)."
    )

    print()

    return backends


def get_least_busy_backend(
    service=None,
    min_qubits: int = 5,
):
    """
    Return the least-busy operational IBM QPU.

    Parameters
    ----------
    service : QiskitRuntimeService, optional

    min_qubits : int
        Minimum number of qubits required.

    Returns
    -------
    IBMBackend
    """

    if service is None:
        service = get_service()

    try:

        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=min_qubits,
        )

    except Exception as exc:
        raise RuntimeError(
            "Could not find a suitable IBM Quantum backend "
            f"with at least {min_qubits} qubits."
        ) from exc

    try:

        status = backend.status()

        pending_jobs = getattr(
            status,
            "pending_jobs",
            "?",
        )

    except Exception:

        pending_jobs = "?"

    print(
        f"[ibm_account] Least-busy backend: "
        f"{backend.name}"
    )

    print(
        f"[ibm_account] Qubits: "
        f"{backend.num_qubits}"
    )

    print(
        f"[ibm_account] Pending jobs: "
        f"{pending_jobs}"
    )

    return backend


# ---------------------------------------------------------------------------
# Aer simulators
# ---------------------------------------------------------------------------

def get_aer_backend(
    method: str = "statevector",
    device_backend=None,
    noise_model=None,
):
    """
    Return a Qiskit Aer simulator.

    Parameters
    ----------
    method : str
        Simulation method.

        Common choices:

            "statevector"
                Ideal pure-state simulation.

            "density_matrix"
                Useful for exact mixed-state/noisy simulation of
                relatively small circuits.

    device_backend : IBMBackend, optional
        If supplied, AerSimulator.from_backend() is used.

        This creates a simulator approximating the selected IBM device,
        including hardware information such as:

            - noise model
            - basis gates
            - coupling map

    noise_model : NoiseModel, optional
        Explicit custom Aer noise model.

        If both device_backend and noise_model are supplied,
        the explicit noise_model takes precedence.

    Returns
    -------
    AerSimulator
    """

    try:
        from qiskit_aer import AerSimulator

    except ImportError:
        raise ImportError(
            "qiskit-aer is not installed.\n"
            "Install it using:\n"
            "    pip install qiskit-aer"
        )

    # ---------------------------------------------------------------
    # Explicit custom noise model
    # ---------------------------------------------------------------

    if noise_model is not None:

        backend = AerSimulator(
            method=method,
            noise_model=noise_model,
        )

        print(
            f"[ibm_account] Backend: "
            f"AerSimulator({method}) + custom noise model"
        )

        return backend

    # ---------------------------------------------------------------
    # Simulator configured from real IBM hardware
    # ---------------------------------------------------------------

    if device_backend is not None:

        backend = AerSimulator.from_backend(
            device_backend,
            method=method,
        )

        print(
            f"[ibm_account] Backend: "
            f"AerSimulator({method}) configured from "
            f"{device_backend.name}"
        )

        return backend

    # ---------------------------------------------------------------
    # Ideal simulator
    # ---------------------------------------------------------------

    backend = AerSimulator(
        method=method,
    )

    print(
        f"[ibm_account] Backend: "
        f"AerSimulator({method})"
    )

    return backend


# ---------------------------------------------------------------------------
# Basic backend information
# ---------------------------------------------------------------------------

def print_backend_info(backend):
    """
    Print basic information about an IBM Quantum backend.

    More detailed hardware characterization will later be implemented
    in backend_profiler.py.
    """

    print()
    print("=" * 60)
    print("IBM QUANTUM BACKEND")
    print("=" * 60)

    print(
        f"Name:              {backend.name}"
    )

    print(
        f"Number of qubits:  {backend.num_qubits}"
    )

    try:

        status = backend.status()

        print(
            f"Operational:       "
            f"{getattr(status, 'operational', '?')}"
        )

        print(
            f"Pending jobs:      "
            f"{getattr(status, 'pending_jobs', '?')}"
        )

        print(
            f"Status:            "
            f"{getattr(status, 'status_msg', '?')}"
        )

    except Exception:

        pass

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "IBM Quantum account and backend helper "
            "for QRC experiments"
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    # ---------------------------------------------------------------
    # Save credentials
    # ---------------------------------------------------------------

    group.add_argument(
        "--save",
        metavar="API_KEY",
        help=(
            "Save IBM Quantum Platform / IBM Cloud API key "
            "to disk"
        ),
    )

    # ---------------------------------------------------------------
    # List hardware
    # ---------------------------------------------------------------

    group.add_argument(
        "--list",
        action="store_true",
        help="List available IBM Quantum hardware backends",
    )

    # ---------------------------------------------------------------
    # Least busy hardware
    # ---------------------------------------------------------------

    group.add_argument(
        "--least-busy",
        action="store_true",
        help="Find the least-busy available IBM QPU",
    )

    # ---------------------------------------------------------------
    # Specific backend
    # ---------------------------------------------------------------

    group.add_argument(
        "--backend",
        metavar="NAME",
        help=(
            "Show information about a specific backend, "
            "for example --backend ibm_brisbane"
        ),
    )

    parser.add_argument(
        "--min-qubits",
        type=int,
        default=5,
        help=(
            "Minimum number of qubits for --least-busy "
            "(default: 5)"
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Save IBM account
    # ---------------------------------------------------------------

    if args.save:

        setup_account(
            args.save
        )

        print()
        print(
            "[ibm_account] Done."
        )

        print(
            "[ibm_account] Test your account using:"
        )

        print()
        print(
            "    python ibm_account.py --list"
        )

    # ---------------------------------------------------------------
    # List available hardware
    # ---------------------------------------------------------------

    elif args.list:

        service = get_service()

        list_available_backends(
            service=service,
        )

    # ---------------------------------------------------------------
    # Least-busy hardware
    # ---------------------------------------------------------------

    elif args.least_busy:

        service = get_service()

        backend = get_least_busy_backend(
            service=service,
            min_qubits=args.min_qubits,
        )

        print_backend_info(
            backend
        )

    # ---------------------------------------------------------------
    # Specific hardware
    # ---------------------------------------------------------------

    elif args.backend:

        service = get_service()

        backend = get_ibm_backend(
            name=args.backend,
            service=service,
        )

        print_backend_info(
            backend
        )


# ---------------------------------------------------------------------------
# Program entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()