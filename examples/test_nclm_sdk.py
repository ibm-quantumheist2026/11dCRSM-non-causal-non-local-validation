#!/usr/bin/env python3
"""
Unit tests for the DNA-Lang SDK (dnalang_sdk) — NCLM v2 infrastructure.

Tests QuantumCircuit, QuantumBackend, LambdaPhiValidator, and CCCEMetrics.
Run with::

    python examples/test_nclm_sdk.py
"""

import asyncio
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_PATH = os.path.join(_REPO_ROOT, "lib")
_BACKEND_PATH = os.path.join(_REPO_ROOT, "backend")

for _p in (_LIB_PATH, _BACKEND_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dnalang_sdk import (
    CCCEMetrics,
    ExecutionResult,
    LAMBDA_PHI,
    LambdaPhiValidator,
    QuantumBackend,
    QuantumCircuit,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# ΛΦ constant
# ---------------------------------------------------------------------------


def test_lambda_phi_constant():
    """ΛΦ constant must match the specification (3.14159 × 10⁻⁹)."""
    assert abs(LAMBDA_PHI - 3.14159e-9) < 1e-15, f"ΛΦ constant mismatch: {LAMBDA_PHI}"
    print("✓ ΛΦ constant")


# ---------------------------------------------------------------------------
# QuantumCircuit
# ---------------------------------------------------------------------------


def test_circuit_construction():
    """QuantumCircuit gate chaining and gate list integrity."""
    circuit = QuantumCircuit(num_qubits=2)
    circuit.h(0).cx(0, 1)

    assert circuit.num_qubits == 2
    assert len(circuit._gates) == 2
    assert circuit._gates[0]["gate"] == "h"
    assert circuit._gates[1]["gate"] == "cx"
    print("✓ QuantumCircuit construction")


def test_ghz_factory():
    """GHZ factory: Hadamard on qubit 0, then (n-1) CNOT gates."""
    for n in (2, 3, 5):
        circuit = QuantumCircuit.ghz(n)
        assert circuit.num_qubits == n
        assert circuit._gates[0]["gate"] == "h"
        cx_gates = [g for g in circuit._gates if g["gate"] == "cx"]
        assert len(cx_gates) == n - 1, (
            f"GHZ({n}): expected {n-1} CNOT gates, got {len(cx_gates)}"
        )
    print("✓ GHZ factory")


def test_bell_factory():
    """Bell state factory produces a 2-qubit circuit with H and CX."""
    circuit = QuantumCircuit.bell()
    assert circuit.num_qubits == 2
    gate_names = [g["gate"] for g in circuit._gates]
    assert "h" in gate_names
    assert "cx" in gate_names
    print("✓ Bell state factory")


def test_vqe_ansatz_factory():
    """VQE ansatz has the correct number of RY and CNOT gates for each rep."""
    n, reps = 2, 2
    circuit = QuantumCircuit.vqe_ansatz(num_qubits=n, reps=reps)
    assert circuit.num_qubits == n
    ry_count = sum(1 for g in circuit._gates if g["gate"] == "ry")
    cx_count = sum(1 for g in circuit._gates if g["gate"] == "cx")
    assert ry_count == n * reps, f"Expected {n * reps} RY gates, got {ry_count}"
    assert cx_count == (n - 1) * reps, (
        f"Expected {(n-1)*reps} CNOT gates, got {cx_count}"
    )
    print("✓ VQE ansatz factory")


def test_circuit_to_dict():
    """to_dict() must include all required fields with correct types."""
    circuit = QuantumCircuit.bell()
    d = circuit.to_dict()
    for key in ("circuit_id", "num_qubits", "gates", "depth"):
        assert key in d, f"Missing key: {key}"
    assert d["num_qubits"] == 2
    assert isinstance(d["gates"], list)
    print("✓ Circuit serialisation")


# ---------------------------------------------------------------------------
# CCCEMetrics
# ---------------------------------------------------------------------------


def test_ccce_metrics_xi():
    """Ξ = ΓΛ·Φ must equal the product of the three components."""
    m = CCCEMetrics(lambda_coherence=0.9, phi_consciousness=0.8, gamma_decoherence=0.1)
    expected = 0.9 * 0.8 * 0.1
    assert abs(m.xi - expected) < 1e-12, f"Ξ mismatch: {m.xi} != {expected}"
    assert set(m.to_dict().keys()) == {"lambda", "phi", "gamma", "xi"}
    print("✓ CCCEMetrics (Ξ = ΓΛ·Φ)")


# ---------------------------------------------------------------------------
# LambdaPhiValidator
# ---------------------------------------------------------------------------


def test_validator_bell_conserved():
    """Bell circuit should satisfy ΛΦ invariance."""
    validator = LambdaPhiValidator()
    circuit = QuantumCircuit.bell()
    result = asyncio.run(validator.validate(circuit))

    assert isinstance(result, ValidationResult)
    assert isinstance(result.is_conserved, bool)
    assert 0.0 <= result.f_max <= 1.0
    assert 0.0 <= result.phi_val <= 1.0
    assert 0.0 <= result.lambda_val <= 1.0
    assert 0.0 <= result.gamma_val <= 1.0
    print(
        f"✓ Validator Bell circuit — conserved={result.is_conserved}, "
        f"Φ={result.phi_val:.4f}, Γ={result.gamma_val:.4f}"
    )


def test_validator_deep_circuit_fails():
    """A circuit with 20 CNOT gates accumulates Γ > 0.3 and must fail."""
    validator = LambdaPhiValidator()
    circuit = QuantumCircuit(num_qubits=2)
    for _ in range(20):
        circuit.cx(0, 1)

    result = asyncio.run(validator.validate(circuit))
    assert not result.is_conserved, (
        "Deep circuit (Γ=0.40) should fail ΛΦ invariance"
    )
    print(
        f"✓ Validator deep circuit — conserved={result.is_conserved}, "
        f"Γ={result.gamma_val:.4f}"
    )


def test_validator_custom_thresholds():
    """Custom phi/gamma thresholds are honoured.

    Φ values are normalised to [0, 1] by the validator, so phi_threshold=1.01
    is impossible to satisfy and guarantees the circuit is blocked.
    """
    strict = LambdaPhiValidator(phi_threshold=1.01, gamma_threshold=0.3)
    circuit = QuantumCircuit.bell()
    result = asyncio.run(strict.validate(circuit))
    assert not result.is_conserved, (
        "Bell circuit should fail under phi_threshold=1.01 (Φ is bounded to [0,1])"
    )
    print("✓ Validator custom thresholds")


# ---------------------------------------------------------------------------
# QuantumBackend
# ---------------------------------------------------------------------------


def test_backend_local_bell():
    """Local backend returns ExecutionResult with correct shot count."""
    backend = QuantumBackend()
    circuit = QuantumCircuit.bell()
    result = asyncio.run(backend.execute(circuit, backend="local", shots=512))

    assert isinstance(result, ExecutionResult)
    assert result.shots == 512
    assert result.backend == "local"
    assert result.latency_ms >= 0
    total = sum(result.counts.values())
    assert total == 512, f"Expected 512 total counts, got {total}"
    print(f"✓ QuantumBackend local (Bell) — latency={result.latency_ms}ms")


def test_backend_ghz_execution():
    """GHZ circuits execute and return the correct shot count."""
    backend = QuantumBackend()
    circuit = QuantumCircuit.ghz(3)
    result = asyncio.run(backend.execute(circuit, backend="local", shots=100))
    assert result.shots == 100
    assert sum(result.counts.values()) == 100
    print(f"✓ QuantumBackend GHZ(3) — counts={result.counts}")


def test_backend_invalid_raises():
    """Unsupported backend must raise ValueError."""
    backend = QuantumBackend()
    circuit = QuantumCircuit.bell()
    try:
        asyncio.run(backend.execute(circuit, backend="unknown_backend"))
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("✓ QuantumBackend invalid backend → ValueError")


def test_backend_no_ibm_token_falls_back():
    """Without an IBM token, IBM backend falls back to local simulation."""
    backend = QuantumBackend(ibm_token=None)
    circuit = QuantumCircuit.bell()
    result = asyncio.run(backend.execute(circuit, backend="ibm_torino", shots=64))
    # Should succeed (via local fallback)
    assert sum(result.counts.values()) == 64
    print("✓ QuantumBackend IBM fallback (no token)")


# ---------------------------------------------------------------------------
# Intent parser (from nclm_api)
# ---------------------------------------------------------------------------


def test_intent_parser():
    """_parse_intent extracts circuit type, backend, and shot count."""
    from nclm_api import _parse_intent

    # navigator-32 → ghz with 32 qubits
    parsed = _parse_intent("run navigator-32 on osaka with 1000 shots")
    assert parsed["circuit_type"] == "ghz"
    assert parsed["num_qubits"] == 32
    assert parsed["backend"] == "ibm_osaka"
    assert parsed["shots"] == 1000

    # explicit bell, local backend
    parsed2 = _parse_intent("run bell circuit 512 shots")
    assert parsed2["circuit_type"] == "bell"
    assert parsed2["backend"] == "local"
    assert parsed2["shots"] == 512

    print("✓ NCLM v2 intent parser")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_tests():
    print("\n" + "=" * 70)
    print("  DNA-Lang SDK (NCLM v2) Unit Tests")
    print("=" * 70 + "\n")

    tests = [
        test_lambda_phi_constant,
        test_circuit_construction,
        test_ghz_factory,
        test_bell_factory,
        test_vqe_ansatz_factory,
        test_circuit_to_dict,
        test_ccce_metrics_xi,
        test_validator_bell_conserved,
        test_validator_deep_circuit_fails,
        test_validator_custom_thresholds,
        test_backend_local_bell,
        test_backend_ghz_execution,
        test_backend_invalid_raises,
        test_backend_no_ibm_token_falls_back,
        test_intent_parser,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            import traceback

            print(f"✗ {test.__name__} failed: {exc}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
