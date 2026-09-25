"""
iqm_account.py
--------------
IQM Resonance account setup, quantum-computer selection and Qiskit backend
helpers for Quantum Reservoir Computing (QRC) experiments.

Authentication priority
-----------------------
1. PRIMARY_API_TOKEN hardcoded in this file
2. environment variable IQM_TOKEN
3. locally saved token created by --save

The IQM server URL defaults to:
    https://resonance.iqm.tech/

First-time setup
----------------
1. Install:

       pip install "iqm-client[qiskit]" requests

2. Obtain a personal API token from IQM Resonance.

3. Either paste it into PRIMARY_API_TOKEN below, OR save it locally:

       python iqm_account.py --save YOUR_API_TOKEN

4. Verify authentication:

       python iqm_account.py --list

5. Inspect a specific quantum computer:

       python iqm_account.py --backend garnet

Notes
-----
- IQM itself supports authentication through token=... or IQM_TOKEN.
- The --save mechanism in this helper is OUR local convenience layer; it is
  not an IQM-managed account store.
- Never print or commit the real token.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = "https://resonance.iqm.tech/"

ENV_API_TOKEN = "IQM_TOKEN"
ENV_SERVER_URL = "IQM_SERVER_URL"
ENV_QUANTUM_COMPUTER = "IQM_QUANTUM_COMPUTER"

DEFAULT_QUANTUM_COMPUTER = "emerald" #garnet

# ---------------------------------------------------------------------------
# PRIMARY IQM API TOKEN
# ---------------------------------------------------------------------------
# Put the IQM Resonance API token here if you want this file to behave like
# ibm_account.py with a primary hardcoded credential.
#
# Authentication priority:
#   1. PRIMARY_API_TOKEN below
#   2. environment variable IQM_TOKEN
#   3. token saved by `python iqm_account.py --save ...`
#
# IMPORTANT:
# - Do not commit a real token to Git/source control.
# - Never print the token.
# - Leave this placeholder unchanged to disable the hardcoded token.

PRIMARY_API_TOKEN = "d5dFcNNf80OrJnZCZtfY90+gpCWMGsOPEYI0f3FVtmABoJu17Ol9MYoT4vPDwfy8"


# Local account file used only by this helper.
ACCOUNT_FILE = Path.home() / ".iqm" / "qrc_iqm_account.json"


# ---------------------------------------------------------------------------
# Local account management
# ---------------------------------------------------------------------------

def setup_account(
    api_token: str,
    server_url: str = DEFAULT_SERVER_URL,
    quantum_computer: str | None = None,
) -> None:
    """
    Save IQM Resonance credentials locally for this QRC project helper.

    This is NOT an IQM-native credential store.  The helper later reads the
    token and passes it explicitly to IQMClient(token=...).

    The account is stored in:
        ~/.iqm/qrc_iqm_account.json
    """

    api_token = str(api_token).strip()

    if not api_token:
        raise ValueError("The IQM API token is empty.")

    server_url = str(server_url).strip().rstrip("/") + "/"

    ACCOUNT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "server_url": server_url,
        "token": api_token,
        "quantum_computer": quantum_computer,
    }

    ACCOUNT_FILE.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Best-effort restriction on platforms where chmod is meaningful.
    try:
        os.chmod(
            ACCOUNT_FILE,
            0o600,
        )
    except Exception:
        pass

    print(
        "[iqm_account] IQM Resonance account saved successfully."
    )
    print(
        f"[iqm_account] Account file: {ACCOUNT_FILE}"
    )


def _load_saved_account() -> dict[str, Any] | None:
    """
    Load the helper's locally saved account, if present.
    """

    if not ACCOUNT_FILE.exists():
        return None

    try:
        payload = json.loads(
            ACCOUNT_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not read saved IQM account file:\n"
            f"    {ACCOUNT_FILE}\n\n"
            f"Original error:\n{exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Invalid IQM account file format:\n"
            f"    {ACCOUNT_FILE}"
        )

    return payload


def _primary_token_is_configured() -> bool:
    token = str(
        PRIMARY_API_TOKEN
    ).strip()

    return (
        bool(token)
        and token != "PASTE_IQM_API_TOKEN_HERE"
    )


def get_authentication() -> tuple[str, str, str | None]:
    """
    Resolve IQM authentication and connection settings.

    Authentication priority
    -----------------------
    1. PRIMARY_API_TOKEN hardcoded in this file
    2. environment variable IQM_TOKEN
    3. locally saved helper account

    Returns
    -------
    token : str
    server_url : str
    default_quantum_computer : str | None
    """

    env_server_url = os.getenv(
        ENV_SERVER_URL
    )

    env_qc = os.getenv(
        ENV_QUANTUM_COMPUTER
    )

    saved = _load_saved_account()

    saved_server_url = None
    saved_qc = None
    saved_token = None

    if saved is not None:
        saved_server_url = saved.get(
            "server_url"
        )
        saved_qc = saved.get(
            "quantum_computer"
        )
        saved_token = saved.get(
            "token"
        )

    server_url = (
        env_server_url
        or saved_server_url
        or DEFAULT_SERVER_URL
    )

    server_url = str(
        server_url
    ).strip().rstrip("/") + "/"

    default_qc = (
        env_qc
        or saved_qc
        or DEFAULT_QUANTUM_COMPUTER
    )

    # ---------------------------------------------------------------
    # Option 1:
    # Hardcoded PRIMARY API token
    # ---------------------------------------------------------------

    if _primary_token_is_configured():

        print(
            "[iqm_account] Using PRIMARY_API_TOKEN "
            "hardcoded in iqm_account.py."
        )

        return (
            str(PRIMARY_API_TOKEN).strip(),
            server_url,
            default_qc,
        )

    # ---------------------------------------------------------------
    # Option 2:
    # API token provided through environment variable
    # ---------------------------------------------------------------

    env_token = os.getenv(
        ENV_API_TOKEN
    )

    if env_token:

        print(
            "[iqm_account] PRIMARY_API_TOKEN is not configured. "
            f"Using token from environment variable {ENV_API_TOKEN}."
        )

        return (
            env_token.strip(),
            server_url,
            default_qc,
        )

    # ---------------------------------------------------------------
    # Option 3:
    # Locally saved helper account
    # ---------------------------------------------------------------

    if saved_token:

        print(
            "[iqm_account] PRIMARY_API_TOKEN and environment token "
            "are not configured. Loading locally saved IQM credentials."
        )

        return (
            str(saved_token).strip(),
            server_url,
            default_qc,
        )

    raise RuntimeError(
        "IQM credentials are not configured.\n\n"
        "Configure PRIMARY_API_TOKEN in iqm_account.py, set IQM_TOKEN, "
        "or run:\n"
        "    python iqm_account.py --save YOUR_API_TOKEN"
    )


# ---------------------------------------------------------------------------
# IQM client
# ---------------------------------------------------------------------------

def get_client(
    quantum_computer: str | None = None,
):
    """
    Return an authenticated IQMClient.

    Parameters
    ----------
    quantum_computer : str, optional
        IQM quantum-computer alias or ID, for example 'garnet'.
        If omitted, IQM_QUANTUM_COMPUTER / saved default is used.
        If still None, IQM Server's default quantum computer is used.
    """

    try:
        from iqm.iqm_client import IQMClient

    except ImportError as exc:
        raise ImportError(
            "iqm-client is not installed.\n"
            "Install it using:\n"
            '    pip install "iqm-client[qiskit]"'
        ) from exc

    token, server_url, default_qc = get_authentication()

    qc = (
        quantum_computer
        or default_qc
    )

    try:
        client = IQMClient(
            server_url,
            quantum_computer=qc,
            token=token,
        )

        # Force a real authenticated request now rather than returning
        # an object that has not yet contacted the server.
        health = client.get_health()

    except Exception as exc:
        target = (
            qc
            if qc is not None
            else "<server default>"
        )

        raise RuntimeError(
            "IQM Resonance authentication/connection failed.\n"
            f"Server: {server_url}\n"
            f"Quantum computer: {target}\n\n"
            f"Original error:\n{exc}"
        ) from exc

    print(
        "[iqm_account] IQM Resonance authentication successful."
    )
    print(
        f"[iqm_account] Server: {server_url}"
    )
    print(
        f"[iqm_account] Quantum computer: "
        f"{client.quantum_computer_name}"
    )
    print(
        f"[iqm_account] Health: {health}"
    )

    return client


# ---------------------------------------------------------------------------
# Discover accessible IQM quantum computers
# ---------------------------------------------------------------------------

def list_available_quantum_computers():
    """
    List quantum computers visible to the IQM Resonance account.

    Uses the documented IQM Server endpoint:
        GET /v1/quantum-computers
    """

    try:
        import requests

    except ImportError as exc:
        raise ImportError(
            "requests is not installed.\n"
            "Install it using:\n"
            "    pip install requests"
        ) from exc

    token, server_url, _ = get_authentication()

    endpoint = (
        server_url.rstrip("/")
        + "/v1/quantum-computers"
    )

    try:
        response = requests.get(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=60,
        )

        response.raise_for_status()
        payload = response.json()

    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve IQM quantum computers from Resonance.\n"
            f"Endpoint: {endpoint}\n\n"
            f"Original error:\n{exc}"
        ) from exc

    # Be tolerant to API envelope differences.
    if isinstance(payload, list):
        systems = payload

    elif isinstance(payload, dict):
        systems = (
            payload.get("quantum_computers")
            or payload.get("items")
            or payload.get("data")
            or []
        )

        # If the response itself looks like one quantum-computer object.
        if not systems and (
            "id" in payload
            or "alias" in payload
            or "display_name" in payload
        ):
            systems = [payload]

    else:
        systems = []

    if not isinstance(systems, list):
        systems = []

    print()
    print(
        f"{'Alias':<24}"
        f"{'Display name':<32}"
        f"{'ID'}"
    )
    print("-" * 95)

    for item in systems:

        if not isinstance(item, dict):
            print(str(item))
            continue

        alias = item.get(
            "alias",
            "",
        )

        display_name = item.get(
            "display_name",
            "",
        )

        qc_id = item.get(
            "id",
            "",
        )

        print(
            f"{str(alias):<24}"
            f"{str(display_name):<32}"
            f"{str(qc_id)}"
        )

    print()
    print(
        f"[iqm_account] Found {len(systems)} "
        "accessible quantum computer(s)."
    )
    print()

    return systems


# ---------------------------------------------------------------------------
# Qiskit IQM backend
# ---------------------------------------------------------------------------

def get_iqm_backend(
    quantum_computer: str | None = None,
    use_metrics: bool = False,
):
    """
    Return an IQM Qiskit backend.

    Parameters
    ----------
    quantum_computer : str, optional
        IQM quantum-computer alias or ID.

    use_metrics : bool
        Whether to expose available calibration/quality metrics to the
        Qiskit transpilation target. Kept False by default because IQM's
        current documentation notes that Resonance metric availability
        may vary.
    """

    try:
        from iqm.qiskit_iqm.iqm_provider import IQMProvider

    except ImportError as exc:
        raise ImportError(
            "IQM Qiskit adapter is not installed.\n"
            "Install it using:\n"
            '    pip install "iqm-client[qiskit]"'
        ) from exc

    token, server_url, default_qc = get_authentication()

    qc = (
        quantum_computer
        or default_qc
    )

    try:
        provider = IQMProvider(
            server_url,
            quantum_computer=qc,
            token=token,
        )

        backend = provider.get_backend(
            use_metrics=use_metrics,
        )

    except Exception as exc:
        target = (
            qc
            if qc is not None
            else "<server default>"
        )

        raise RuntimeError(
            "Could not create IQM Qiskit backend.\n"
            f"Server: {server_url}\n"
            f"Quantum computer: {target}\n\n"
            f"Original error:\n{exc}"
        ) from exc

    print(
        f"[iqm_account] Qiskit backend ready: "
        f"{backend.name}"
    )

    return backend


# ---------------------------------------------------------------------------
# Basic backend information
# ---------------------------------------------------------------------------

def print_backend_info(
    quantum_computer: str,
):
    """
    Print basic IQM hardware information.

    Detailed calibration/topology profiling belongs to Week 12.1 and the
    dedicated profiler, not this account helper.
    """

    client = get_client(
        quantum_computer=quantum_computer,
    )

    try:
        architecture = (
            client.get_static_quantum_architecture()
        )

    except Exception as exc:
        raise RuntimeError(
            "Authenticated successfully, but static architecture "
            "could not be retrieved."
        ) from exc

    qubits = getattr(
        architecture,
        "qubits",
        [],
    )

    connectivity = getattr(
        architecture,
        "connectivity",
        [],
    )

    print()
    print("=" * 60)
    print("IQM QUANTUM COMPUTER")
    print("=" * 60)

    print(
        f"Name:              "
        f"{client.quantum_computer_name}"
    )

    print(
        f"Number of qubits:  "
        f"{len(qubits)}"
    )

    print(
        f"Connectivity edges:"
        f"  {len(connectivity)}"
    )

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "IQM Resonance account and backend helper "
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
        metavar="API_TOKEN",
        help=(
            "Save IQM Resonance API token locally "
            "for this QRC helper"
        ),
    )

    # ---------------------------------------------------------------
    # List hardware
    # ---------------------------------------------------------------

    group.add_argument(
        "--list",
        action="store_true",
        help=(
            "List IQM quantum computers accessible "
            "to this Resonance account"
        ),
    )

    # ---------------------------------------------------------------
    # Test default connection
    # ---------------------------------------------------------------

    group.add_argument(
        "--test",
        action="store_true",
        help=(
            "Test IQM authentication against the configured/default "
            "quantum computer"
        ),
    )

    # ---------------------------------------------------------------
    # Specific quantum computer
    # ---------------------------------------------------------------

    group.add_argument(
        "--backend",
        metavar="NAME",
        help=(
            "Show basic information about a specific IQM quantum "
            "computer, for example --backend garnet"
        ),
    )

    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=(
            "IQM server URL used by --save "
            f"(default: {DEFAULT_SERVER_URL})"
        ),
    )

    parser.add_argument(
        "--default-backend",
        metavar="NAME",
        default=None,
        help=(
            "Optional default quantum-computer alias stored by --save"
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Save IQM account
    # ---------------------------------------------------------------

    if args.save:

        setup_account(
            api_token=args.save,
            server_url=args.server_url,
            quantum_computer=args.default_backend,
        )

        print()
        print(
            "[iqm_account] Done."
        )

        print(
            "[iqm_account] Test your account using:"
        )

        print()
        print(
            "    python iqm_account.py --list"
        )

    # ---------------------------------------------------------------
    # List accessible hardware
    # ---------------------------------------------------------------

    elif args.list:

        list_available_quantum_computers()

    # ---------------------------------------------------------------
    # Test authentication
    # ---------------------------------------------------------------

    elif args.test:

        get_client()

    # ---------------------------------------------------------------
    # Specific hardware
    # ---------------------------------------------------------------

    elif args.backend:

        print_backend_info(
            quantum_computer=args.backend,
        )


# ---------------------------------------------------------------------------
# Program entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
