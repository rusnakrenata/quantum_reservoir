"""
ibm_account.py
--------------
IBM Quantum account setup and backend selection for QRC experiments.

Setup (first time only)
-----------------------
1. Get your IBM Quantum token from: https://quantum.ibm.com/account
2. Run this file directly, passing your token:

       python ibm_account.py --save <YOUR_TOKEN>

   This saves credentials to ~/.qiskit/qiskit-ibm.json permanently.
   You only need to do this once per machine.

3. Verify the setup:

       python ibm_account.py --list

Execution modes in 06_qrc_model.py
------------------------------------
    # Local simulation (no IBM account needed)
    backend = get_aer_backend()

    # IBM 127-qubit hardware
    backend = get_ibm_backend("ibm_brisbane")

    # IBM hardware with noise-model simulation (Aer mimics real device)
    backend = get_aer_backend(device_backend=get_ibm_backend("ibm_brisbane"))
"""

import os
import argparse

# ---------------------------------------------------------------------------
# Known IBM Quantum backends
# ---------------------------------------------------------------------------

IBM_BACKENDS_127Q = [
    "ibm_brisbane",    # Eagle r3, 127 qubits (open plan available)
    "ibm_kyoto",       # Eagle r3, 127 qubits
    "ibm_osaka",       # Eagle r3, 127 qubits
    "ibm_sherbrooke",  # Eagle r3, 127 qubits
]

IBM_BACKENDS_133Q = [
    "ibm_torino",      # Heron r2, 133 qubits (fast, low noise)
    "ibm_marrakesh",   # Heron r2, 156 qubits
]

DEFAULT_BACKEND = "ibm_brisbane"
DEFAULT_CHANNEL = "ibm_quantum"   # use "ibm_cloud" for IBM Cloud accounts


# ---------------------------------------------------------------------------
# Account management
# ---------------------------------------------------------------------------

def setup_account(
    token: str,
    channel: str = DEFAULT_CHANNEL,
    instance: str = "ibm-q/open/main",
    save: bool = True,
) -> None:
    """
    Save IBM Quantum credentials locally (writes to ~/.qiskit/qiskit-ibm.json).

    Parameters
    ----------
    token    : IBM Quantum API token from https://quantum.ibm.com/account
    channel  : "ibm_quantum" (default) or "ibm_cloud"
    instance : hub/group/project string; "ibm-q/open/main" for open plan
    save     : if True, persist credentials on disk (recommended)
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise ImportError(
            "qiskit-ibm-runtime is not installed. "
            "Run: pip install qiskit-ibm-runtime"
        )

    QiskitRuntimeService.save_account(
        channel=channel,
        token=token,
        instance=instance,
        overwrite=True,
        set_as_default=True,
    )
    print(f"[ibm_account] Account saved (channel={channel}, instance={instance})")


def get_service(channel: str = DEFAULT_CHANNEL):
    """
    Load saved IBM Quantum credentials and return a QiskitRuntimeService.

    Raises RuntimeError if credentials are not found -- run
    `python ibm_account.py --save <TOKEN>` first.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise ImportError(
            "qiskit-ibm-runtime is not installed. "
            "Run: pip install qiskit-ibm-runtime"
        )

    # Also accept token from environment variable IBM_QUANTUM_TOKEN
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if token:
        return QiskitRuntimeService(channel=channel, token=token)

    return QiskitRuntimeService(channel=channel)


# ---------------------------------------------------------------------------
# Backend getters
# ---------------------------------------------------------------------------

def get_ibm_backend(name: str = DEFAULT_BACKEND, service=None):
    """
    Return an IBM Quantum hardware backend.

    Parameters
    ----------
    name    : backend name, e.g. "ibm_brisbane"
    service : existing QiskitRuntimeService instance (optional; will load
              saved credentials if None)

    Returns
    -------
    IBMBackend -- pass directly to QiskitQRC(..., backend=backend)
    """
    if service is None:
        service = get_service()
    backend = service.backend(name)
    n_qubits = backend.num_qubits
    print(f"[ibm_account] Backend: {name}  ({n_qubits} qubits)")
    return backend


def get_aer_backend(
    method: str = "statevector",
    device_backend=None,
    noise_model=None,
):
    """
    Return an Aer simulation backend.

    Parameters
    ----------
    method         : "statevector" (fast, pure states) or
                     "density_matrix" (exact mixed states, slower)
    device_backend : if provided, import noise model from this real IBM backend
                     (simulates hardware noise locally)
    noise_model    : explicit NoiseModel object; takes precedence over
                     device_backend

    Returns
    -------
    AerSimulator -- pass directly to QiskitQRC(..., backend=backend)
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        raise ImportError(
            "qiskit-aer is not installed. "
            "Run: pip install qiskit-aer"
        )

    if device_backend is not None and noise_model is None:
        from qiskit_aer.noise import NoiseModel
        noise_model = NoiseModel.from_backend(device_backend)
        print(f"[ibm_account] Aer noise model loaded from {device_backend.name}")

    if noise_model is not None:
        backend = AerSimulator(method=method, noise_model=noise_model)
    else:
        backend = AerSimulator(method=method)

    label = f"AerSimulator({method})"
    if noise_model:
        label += "+noise"
    print(f"[ibm_account] Backend: {label}")
    return backend


# ---------------------------------------------------------------------------
# Backend info helpers
# ---------------------------------------------------------------------------

def list_available_backends(service=None, operational_only: bool = True):
    """Print all backends available to the account."""
    if service is None:
        service = get_service()

    backends = service.backends(operational=operational_only)
    print(f"\n{'Backend':<30} {'Qubits':>8} {'Status':<15}")
    print("-" * 55)
    for b in sorted(backends, key=lambda x: x.num_qubits):
        try:
            status = b.status()
            status_str = status.status_msg
        except Exception:
            status_str = "?"
        print(f"{b.name:<30} {b.num_qubits:>8}   {status_str}")
    print()


def get_best_backend(service=None, min_qubits: int = 5):
    """Return the least-busy operational IBM backend with at least min_qubits."""
    if service is None:
        service = get_service()

    backends = service.backends(
        operational=True,
        min_num_qubits=min_qubits,
    )
    best = min(backends, key=lambda b: b.status().pending_jobs)
    print(f"[ibm_account] Least-busy backend: {best.name} "
          f"({best.num_qubits} qubits, "
          f"{best.status().pending_jobs} pending jobs)")
    return best


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IBM Quantum account setup for QRC experiments"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save",   metavar="TOKEN",
                       help="Save IBM Quantum token to disk")
    group.add_argument("--list",   action="store_true",
                       help="List available backends")
    group.add_argument("--best",   action="store_true",
                       help="Print the least-busy available backend")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL,
                        help=f"Channel (default: {DEFAULT_CHANNEL})")
    parser.add_argument("--instance", default="ibm-q/open/main",
                        help="hub/group/project (default: ibm-q/open/main)")
    args = parser.parse_args()

    if args.save:
        setup_account(args.save, channel=args.channel, instance=args.instance)
        print("[ibm_account] Done. You can now run 06_qrc_model.py --hardware")
    elif args.list:
        list_available_backends()
    elif args.best:
        get_best_backend()


if __name__ == "__main__":
    main()




# Save IBM token (once)
#python ibm_account.py --save YOUR_TOKEN_HERE