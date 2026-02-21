"""
NCLM v2 REST API Server

Exposes the DNA-Lang SDK over HTTP for use within Amazon VPCs,
AWS Lambda, or any sovereign air-gapped environment.

Endpoints:
    POST /api/dnalang-quantum/submit  — Submit GHZ/Bell/VQE quantum jobs
    GET  /api/ccce                    — Real-time Φ, Λ, Γ, Ξ metrics
    POST /api/noncausal-lm/chat       — Sovereign intent parsing (NCLM v2)
    POST /api/deploy_vercel           — Trigger Vercel/AWS Lambda deployment

Usage::

    python backend/nclm_api.py          # default port 8080
    PORT=5000 python backend/nclm_api.py
"""

import asyncio
import os
import re
import sys
from typing import Any, Dict

from flask import Flask, jsonify, request

# Resolve lib path relative to this file so the server can be started from
# any working directory.
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
if _LIB_PATH not in sys.path:
    sys.path.insert(0, _LIB_PATH)

from dnalang_sdk import (  # noqa: E402
    CCCEMetrics,
    LambdaPhiValidator,
    QuantumBackend,
    QuantumCircuit,
)

app = Flask(__name__)

_backend = QuantumBackend(ibm_token=os.getenv("IBM_QUANTUM_TOKEN"))
_validator = LambdaPhiValidator()

# In-memory CCCE state — updated after each successful /submit call.
_ccce_state: CCCEMetrics = CCCEMetrics(
    lambda_coherence=1.0,
    phi_consciousness=0.75,
    gamma_decoherence=0.0,
)


def _run_async(coro):
    """Run an async coroutine from a synchronous Flask route handler."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if not loop.is_running():
            return loop.run_until_complete(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# POST /api/dnalang-quantum/submit
# ---------------------------------------------------------------------------


@app.route("/api/dnalang-quantum/submit", methods=["POST"])
def submit_quantum_job():
    """
    Submit a GHZ, Bell, or VQE circuit job to a quantum backend.

    Request body (JSON):

    .. code-block:: json

        {
            "circuit_type": "bell",
            "num_qubits": 2,
            "backend": "local",
            "shots": 1024
        }

    ``circuit_type`` — one of ``"ghz"``, ``"bell"``, ``"vqe"``.
    ``backend``      — ``"local"`` (sovereign) or ``"ibm_torino"`` etc.
    """
    global _ccce_state

    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}

    circuit_type = str(data.get("circuit_type", "bell")).lower()
    num_qubits = max(1, int(data.get("num_qubits", 2)))
    backend_name = str(data.get("backend", "local"))
    shots = max(1, int(data.get("shots", 1024)))

    # Build circuit
    if circuit_type == "ghz":
        circuit = QuantumCircuit.ghz(num_qubits)
    elif circuit_type == "vqe":
        reps = max(1, int(data.get("reps", 1)))
        circuit = QuantumCircuit.vqe_ansatz(num_qubits, reps=reps)
    else:
        circuit = QuantumCircuit.bell()

    # ΛΦ Invariance gate check
    v_result = _run_async(_validator.validate(circuit))
    if not v_result.is_conserved:
        return (
            jsonify(
                {
                    "error": "Φ-Gate blocked: ΛΦ invariance not satisfied",
                    "details": v_result.details,
                    "phi": v_result.phi_val,
                    "gamma": v_result.gamma_val,
                }
            ),
            422,
        )

    # Execute circuit
    try:
        exec_result = _run_async(
            _backend.execute(circuit, backend=backend_name, shots=shots)
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Persist CCCE state for /api/ccce
    _ccce_state = CCCEMetrics(
        lambda_coherence=v_result.lambda_val,
        phi_consciousness=v_result.phi_val,
        gamma_decoherence=v_result.gamma_val,
    )

    return jsonify(
        {
            "job_id": exec_result.circuit_id,
            "circuit_type": circuit_type,
            "backend": exec_result.backend,
            "shots": exec_result.shots,
            "counts": exec_result.counts,
            "latency_ms": exec_result.latency_ms,
            "validation": {
                "is_conserved": v_result.is_conserved,
                "f_max": v_result.f_max,
            },
        }
    )


# ---------------------------------------------------------------------------
# GET /api/ccce
# ---------------------------------------------------------------------------


@app.route("/api/ccce", methods=["GET"])
def get_ccce_metrics():
    """
    Retrieve real-time CCCE metrics.

    Returns Λ (coherence), Φ (consciousness), Γ (decoherence),
    and Ξ (negentropic efficiency Ξ = ΓΛ·Φ).
    """
    return jsonify(
        {
            "metrics": _ccce_state.to_dict(),
            "description": {
                "lambda": "Coherence — quantum state fidelity",
                "phi": "Consciousness — integrated information (Φ)",
                "gamma": "Decoherence — noise/entropy",
                "xi": "Negentropic efficiency Ξ = ΓΛ·Φ",
            },
            "thresholds": {
                "phi_min": LambdaPhiValidator.PHI_THRESHOLD,
                "gamma_max": LambdaPhiValidator.GAMMA_THRESHOLD,
            },
        }
    )


# ---------------------------------------------------------------------------
# POST /api/noncausal-lm/chat
# ---------------------------------------------------------------------------


@app.route("/api/noncausal-lm/chat", methods=["POST"])
def nclm_chat():
    """
    Sovereign intent parsing via the local NCLM v2 engine.

    Parses quantum job parameters from natural language without
    any external API call.

    Request body (JSON):

    .. code-block:: json

        { "intent": "run navigator-32 on osaka with 1000 shots" }
    """
    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    intent: str = str(data.get("intent", "")).strip()

    if not intent:
        return jsonify({"error": "Field 'intent' is required"}), 400

    parsed = _parse_intent(intent)
    return jsonify(
        {
            "intent": intent,
            "parsed": parsed,
            "engine": "NCLM_v2_LOCAL",
            "planner": os.getenv("OSIRIS_PLANNER", "NCLM_v2_LOCAL"),
        }
    )


def _parse_intent(intent: str) -> Dict[str, Any]:
    """
    NCLM v2 local intent parser.

    Extracts quantum job parameters from a natural language string
    without requiring an external LLM API.
    """
    low = intent.lower()

    # Backend
    backend = "local"
    for name in ("torino", "osaka", "brisbane"):
        if name in low:
            backend = f"ibm_{name}"
            break

    # Shot count
    shots = 1024
    m = re.search(r"(\d+)\s*shots?", low)
    if m:
        shots = int(m.group(1))

    # Circuit type — "navigator-N" maps to GHZ with N qubits
    circuit_type = "bell"
    num_qubits = 2
    nav = re.search(r"navigator[-_]?(\d+)", low)
    if nav:
        circuit_type = "ghz"
        num_qubits = int(nav.group(1))
    else:
        for ct in ("ghz", "vqe", "bell"):
            if ct in low:
                circuit_type = ct
                break

    return {
        "circuit_type": circuit_type,
        "num_qubits": num_qubits,
        "backend": backend,
        "shots": shots,
    }


# ---------------------------------------------------------------------------
# POST /api/deploy_vercel
# ---------------------------------------------------------------------------


@app.route("/api/deploy_vercel", methods=["POST"])
def deploy_vercel():
    """
    Trigger a production deployment to Vercel/AWS Lambda.

    Request body (JSON):

    .. code-block:: json

        {
            "project": "dna-lang-nclm",
            "environment": "production",
            "vercel_token": "<optional — falls back to VERCEL_TOKEN env var>"
        }
    """
    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}

    project = str(data.get("project", "dna-lang-nclm"))
    environment = str(data.get("environment", "production"))
    token = data.get("vercel_token") or os.getenv("VERCEL_TOKEN")

    if not token:
        return (
            jsonify(
                {
                    "error": (
                        "Vercel token required. Provide 'vercel_token' in the "
                        "request body or set the VERCEL_TOKEN environment variable."
                    )
                }
            ),
            400,
        )

    # Call the Vercel Deployments API.
    # https://vercel.com/docs/rest-api/endpoints/deployments#create-a-new-deployment
    vercel_api_url = "https://api.vercel.com/v13/deployments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"name": project, "target": environment}

    try:
        import urllib.request
        import json as _json

        req = urllib.request.Request(
            vercel_api_url,
            data=_json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            deploy_data = _json.loads(resp.read().decode())
            deploy_id = deploy_data.get("id", "unknown")
            deploy_url = deploy_data.get("url", vercel_api_url)
    except Exception as exc:
        # Surface the upstream error so callers can diagnose issues.
        return jsonify({"error": f"Vercel API call failed: {exc}"}), 502

    return jsonify(
        {
            "status": "triggered",
            "project": project,
            "environment": environment,
            "deploy_id": deploy_id,
            "deploy_url": f"https://{deploy_url}" if not deploy_url.startswith("http") else deploy_url,
            "message": (
                f"Deployment of '{project}' to {environment} has been triggered. "
                "Monitor progress at https://vercel.com/dashboard."
            ),
        }
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe endpoint."""
    return jsonify({"status": "ok", "engine": "NCLM_v2_LOCAL"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
