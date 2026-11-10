"""AQC prefix construction, profiling, and circuit-cache helpers."""

from __future__ import annotations

import os
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping


AQC_SCHEME_VERSION = "2026-09-ground-state-input-aware-v1"


@dataclass(frozen=True)
class AQCSettings:
    method: str = "L-BFGS-B"
    maxiter: int = 300
    stopping_fidelity: float = 0.999
    mps_max_bond_dimension: int | None = 100
    seed: int = 1234
    simulator: str = "quimb_jax"
    aqc_target_refinement: int = 2
    aqc_good_steps: int | None = None
    use_midpoint: bool = True
    decompose_reps: int = 4
    verbose: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.maxiter < 1:
            raise ValueError("maxiter must be positive.")
        if not 0 < self.stopping_fidelity <= 1:
            raise ValueError("stopping_fidelity must be in (0, 1].")
        if self.mps_max_bond_dimension is not None and self.mps_max_bond_dimension < 1:
            raise ValueError("mps_max_bond_dimension must be positive or None.")
        if self.aqc_target_refinement < 1:
            raise ValueError("aqc_target_refinement must be positive.")
        if self.aqc_good_steps is not None and self.aqc_good_steps < 1:
            raise ValueError("aqc_good_steps must be positive or None.")


@dataclass
class AQCProfileRecord:
    problem: str
    instance_id: str
    n_qubits: int
    num_steps: int
    alpha: int
    target_build_sec: float | None = None
    compression_sec: float | None = None
    transpile_sec: float | None = None
    peak_rss_mb: float | None = None
    optimizer_method: str | None = None
    optimizer_status_code: int | None = None
    optimizer_message: str | None = None
    optimizer_iterations: int | None = None
    objective_evaluations: int | None = None
    final_fidelity: float | None = None
    final_infidelity: float | None = None
    target_prefix_twoq_depth: int | None = None
    good_prefix_twoq_depth: int | None = None
    compressed_prefix_twoq_depth: int | None = None
    tail_twoq_depth: int | None = None
    full_twoq_depth: int | None = None
    simulator_backend: str | None = None
    ansatz_num_parameters: int | None = None
    backend_name: str | None = None
    cache_path: str | None = None
    status: str = "ok"
    scheme_version: str = AQC_SCHEME_VERSION
    fidelity_input: str | None = None
    annealing_time: float | None = None
    driver_convention: str | None = None
    mps_max_bond_dimension: int | None = None
    mps_bond_limit_scope: str = "target_circuit_simulation"
    stopping_fidelity_reached: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def two_qubit_depth(circuit: Any, *, decompose_reps: int = 0) -> int | None:
    """Return two-qubit depth for a Qiskit circuit-like object."""
    if circuit is None:
        return None
    if not hasattr(circuit, "depth"):
        return None
    if decompose_reps > 0 and hasattr(circuit, "decompose"):
        try:
            circuit = circuit.decompose(reps=decompose_reps)
        except Exception:
            pass
    try:
        return int(circuit.depth(lambda inst: inst.operation.num_qubits == 2))
    except Exception:
        try:
            return int(circuit.depth())
        except Exception:
            return None


def default_schedule_s(step_index: int, num_steps: int) -> float:
    """Linear schedule s in [0, 1] at the supplied step index."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    return float(step_index) / float(num_steps)


def build_driver_x_hamiltonian(num_qubits: int, coeff: float = -1.0) -> Any:
    """Build ``coeff * sum_i X_i``; the default has ``|+>`` as its ground state."""
    try:
        from qiskit.quantum_info import SparsePauliOp
    except ImportError as exc:
        raise ImportError("qiskit is required to build driver Hamiltonians.") from exc

    terms = []
    for idx in range(num_qubits):
        pauli = ["I"] * num_qubits
        pauli[idx] = "X"
        terms.append("".join(pauli[::-1]))
    return SparsePauliOp(terms, coeffs=[coeff] * len(terms))


def build_annealing_segment(
    num_qubits: int,
    H_initial: Any,
    H_problem: Any,
    delta_t: float,
    step_start: int,
    step_end: int,
    *,
    schedule_s: Callable[[float, int], float] = default_schedule_s,
    num_steps_total: int,
    s_scale: float = 1.0,
    use_midpoint: bool = False,
    insert_barriers: bool = False,
) -> Any:
    """Build trotterized annealing steps [step_start, step_end)."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
    except ImportError as exc:
        raise ImportError("qiskit is required for annealing circuit construction.") from exc

    if not (0 <= step_start <= step_end <= num_steps_total):
        raise ValueError("Require 0 <= step_start <= step_end <= num_steps_total.")
    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")

    circuit = QuantumCircuit(num_qubits)
    for step in range(step_start, step_end):
        step_eff = step + 0.5 if use_midpoint else step
        s = float(schedule_s(step_eff, num_steps_total)) * float(s_scale)
        H_t = (1.0 - s) * H_initial + s * H_problem
        circuit.append(PauliEvolutionGate(H_t, time=delta_t), range(num_qubits))
        if insert_barriers:
            circuit.barrier()
    return circuit


def build_annealing_circuit_with_aqc_prefix(
    num_qubits: int,
    H_problem: Any,
    *,
    num_steps: int,
    alpha: int,
    delta_t: float | None = None,
    annealing_time: float | None = None,
    H_initial: Any | None = None,
    schedule_s: Callable[[float, int], float] = default_schedule_s,
    s_scale: float = 1.0,
    use_midpoint: bool | None = None,
    aqc_target_refinement: int | None = None,
    aqc_good_steps: int | None = None,
    aqc_settings: AQCSettings | None = None,
    measure: bool = True,
) -> tuple[Any, dict[str, Any], dict[str, Any], AQCProfileRecord | None]:
    """Build an anneal initialized in ``|+>`` with an input-aware AQC prefix.

    ``target_prefix``, ``good_prefix`` and ``compressed_prefix`` are evolution
    circuits: callers apply them *after* ``remaining['initial_hadamards']``.
    AQC nevertheless optimizes their action on that prepared input, not on zero.
    The total duration defaults to one; specify ``annealing_time`` or ``delta_t``
    to vary it independently of the number of Trotter steps. Custom drivers are
    allowed, but their ground state must be compatible with this initialization.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError("qiskit is required for AQC annealing construction.") from exc

    aqc_settings = aqc_settings or AQCSettings()
    use_midpoint = aqc_settings.use_midpoint if use_midpoint is None else use_midpoint
    aqc_target_refinement = (
        aqc_settings.aqc_target_refinement
        if aqc_target_refinement is None else aqc_target_refinement
    )
    aqc_good_steps = aqc_settings.aqc_good_steps if aqc_good_steps is None else aqc_good_steps
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not (0 <= alpha <= num_steps):
        raise ValueError("alpha must satisfy 0 <= alpha <= num_steps.")
    if annealing_time is not None and (not math.isfinite(annealing_time) or annealing_time <= 0):
        raise ValueError("annealing_time must be finite and positive.")
    if delta_t is not None and (not math.isfinite(delta_t) or delta_t <= 0):
        raise ValueError("delta_t must be finite and positive.")
    if delta_t is None:
        delta_t = (1.0 if annealing_time is None else annealing_time) / num_steps
    elif annealing_time is not None and not math.isclose(delta_t * num_steps, annealing_time):
        raise ValueError("delta_t * num_steps must equal annealing_time when both are supplied.")
    annealing_time = float(delta_t * num_steps)
    driver_convention = "negative_x_ground_state_plus" if H_initial is None else "custom"
    if H_initial is None:
        H_initial = build_driver_x_hamiltonian(num_qubits)

    scheme_metadata = {
        "aqc_scheme_version": AQC_SCHEME_VERSION,
        "annealing_time": annealing_time,
        "driver_convention": driver_convention,
        "initial_state": "plus",
        "num_steps": int(num_steps),
        "compression_steps": int(alpha),
        "target_refinement": int(aqc_target_refinement),
    }

    circuit = QuantumCircuit(num_qubits)
    circuit.metadata = dict(scheme_metadata)
    init_circuit = QuantumCircuit(num_qubits)
    init_circuit.h(range(num_qubits))
    circuit.compose(init_circuit, inplace=True)
    circuit.barrier()

    prefixes: dict[str, Any] = {
        "target_prefix": None,
        "good_prefix": None,
        "compressed_prefix": None,
    }
    remaining: dict[str, Any] = {
        "initial_hadamards": init_circuit,
        "tail": None,
        "full_prefix_plus_tail_no_measure": None,
    }
    compression_record = None

    if alpha > 0:
        if aqc_target_refinement < 1:
            raise ValueError("aqc_target_refinement must be >= 1.")
        refined_steps = alpha * aqc_target_refinement
        refined_dt = delta_t / aqc_target_refinement
        target_prefix = QuantumCircuit(num_qubits)
        from qiskit.circuit.library import PauliEvolutionGate

        for idx in range(refined_steps):
            frac = (idx + (0.5 if use_midpoint else 0.0)) / refined_steps
            step_eff = frac * alpha
            s = float(schedule_s(step_eff, num_steps)) * float(s_scale)
            H_t = (1.0 - s) * H_initial + s * H_problem
            target_prefix.append(PauliEvolutionGate(H_t, time=refined_dt), range(num_qubits))

        if aqc_good_steps is None:
            aqc_good_steps = max(1, int(__import__("math").ceil(alpha / 3)))
        if aqc_good_steps < 1:
            raise ValueError("aqc_good_steps must be positive.")
        total_prefix_time = alpha * delta_t
        good_dt = total_prefix_time / aqc_good_steps
        good_prefix = QuantumCircuit(num_qubits)
        for idx in range(aqc_good_steps):
            frac = (idx + (0.5 if use_midpoint else 0.0)) / aqc_good_steps
            step_eff = frac * alpha
            s = float(schedule_s(step_eff, num_steps)) * float(s_scale)
            H_t = (1.0 - s) * H_initial + s * H_problem

            good_prefix.append(PauliEvolutionGate(H_t, time=good_dt), range(num_qubits))

        compressed_prefix, compression_record = aqc_tensor_compress_prefix(
            target_prefix,
            good_prefix,
            initial_state_preparation=init_circuit,
            aqc_settings=aqc_settings,
            decompose_reps=aqc_settings.decompose_reps,
        )
        compression_record.annealing_time = annealing_time
        compression_record.driver_convention = driver_convention
        compression_record.fidelity_input = "plus"
        prefixes.update(
            {
                "target_prefix": target_prefix,
                "good_prefix": good_prefix,
                "compressed_prefix": compressed_prefix,
            }
        )
        circuit.compose(compressed_prefix, inplace=True)
        circuit.barrier()

    if alpha < num_steps:
        tail = build_annealing_segment(
            num_qubits,
            H_initial,
            H_problem,
            delta_t,
            alpha,
            num_steps,
            schedule_s=schedule_s,
            num_steps_total=num_steps,
            s_scale=s_scale,
            use_midpoint=use_midpoint,
        )
        remaining["tail"] = tail
        circuit.compose(tail, inplace=True)

    circuit.barrier()
    if measure:
        circuit.measure_all()

    full_no_measure = QuantumCircuit(num_qubits)
    full_no_measure.metadata = dict(scheme_metadata)
    full_no_measure.compose(init_circuit, inplace=True)
    if prefixes["compressed_prefix"] is not None:
        full_no_measure.compose(prefixes["compressed_prefix"], inplace=True)
    if remaining["tail"] is not None:
        full_no_measure.compose(remaining["tail"], inplace=True)
    remaining["full_prefix_plus_tail_no_measure"] = full_no_measure

    for name, prefix in prefixes.items():
        if prefix is not None:
            prefix.metadata = {
                **(prefix.metadata or {}), **scheme_metadata,
                "circuit_role": name,
                "input_contract": "apply_after_initial_state_preparation",
            }
    init_circuit.metadata = {**scheme_metadata, "circuit_role": "initial_state_preparation"}
    if remaining["tail"] is not None:
        remaining["tail"].metadata = {**scheme_metadata, "circuit_role": "tail"}

    return circuit, prefixes, remaining, compression_record


def make_circuits_for_alphas(
    alphas: Iterable[int],
    num_qubits: int,
    H_problem: Any,
    *,
    num_steps: int,
    annealing_time: float | None = None,
    H_initial: Any | None = None,
    aqc_settings: AQCSettings | None = None,
    measure: bool = True,
) -> tuple[dict[int, Any], dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[int, AQCProfileRecord | None]]:
    """Build AQC-prefix circuits for a collection of compression levels."""
    settings = aqc_settings or AQCSettings()
    circuits = {}
    prefixes = {}
    remaining = {}
    compression_records = {}
    for alpha in alphas:
        circuit, pref, rem, record = build_annealing_circuit_with_aqc_prefix(
            num_qubits,
            H_problem,
            num_steps=num_steps,
            annealing_time=annealing_time,
            alpha=int(alpha),
            H_initial=H_initial,
            use_midpoint=settings.use_midpoint,
            aqc_target_refinement=settings.aqc_target_refinement,
            aqc_good_steps=settings.aqc_good_steps,
            aqc_settings=settings,
            measure=measure,
        )
        circuits[int(alpha)] = circuit
        prefixes[int(alpha)] = pref
        remaining[int(alpha)] = rem
        compression_records[int(alpha)] = record
    return circuits, prefixes, remaining, compression_records


def save_qpy(path: str | Path, circuits: Any) -> Path:
    """Save one or more Qiskit circuits as QPY."""
    try:
        from qiskit import qpy
    except ImportError as exc:
        raise ImportError("qiskit is required to save QPY circuit caches.") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        qpy.dump(circuits, handle)
    return path


def load_qpy(path: str | Path) -> list[Any]:
    """Load Qiskit circuits from QPY."""
    try:
        from qiskit import qpy
    except ImportError as exc:
        raise ImportError("qiskit is required to load QPY circuit caches.") from exc
    with Path(path).open("rb") as handle:
        return list(qpy.load(handle))


@contextmanager
def profile_block() -> Iterator[dict[str, float | None]]:
    """Measure elapsed time and best-effort peak RSS for a code block."""
    process = None
    try:
        import psutil

        process = psutil.Process()
        start_rss = process.memory_info().rss
    except Exception:
        start_rss = None
    start = time.perf_counter()
    data: dict[str, float | None] = {"elapsed_sec": None, "peak_rss_mb": None}
    try:
        yield data
    finally:
        data["elapsed_sec"] = time.perf_counter() - start
        if process is not None and start_rss is not None:
            end_rss = process.memory_info().rss
            data["peak_rss_mb"] = max(start_rss, end_rss) / (1024 * 1024)


def transpile_with_profile(circuit: Any, pass_manager: Any | None = None) -> tuple[Any, float]:
    """Run a Qiskit pass manager if supplied and return elapsed seconds."""
    start = time.perf_counter()
    out = pass_manager.run(circuit) if pass_manager is not None else circuit
    return out, time.perf_counter() - start


def prepare_quimb_jax_environment() -> None:
    """Set environment switches needed before importing quimb."""
    os.environ.setdefault("QUIMB_NUMBA_CACHE", "FALSE")
    os.environ.setdefault("JAX_ENABLE_X64", "True")


def build_quimb_jax_simulator(mps_max_bond_dimension: int | None = None) -> Any:
    """Construct Quimb/JAX and apply the requested MPS bond limit."""
    prepare_quimb_jax_environment()
    import quimb.tensor
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator

    return QuimbSimulator(
        partial(quimb.tensor.CircuitMPS, max_bond=mps_max_bond_dimension),
        autodiff_backend="jax",
    )


def aqc_tensor_compress_prefix(
    target_prefix_circuit: Any,
    good_prefix_circuit: Any,
    *,
    initial_state_preparation: Any | None = None,
    aqc_settings: AQCSettings | None = None,
    decompose_reps: int = 4,
) -> tuple[Any, AQCProfileRecord]:
    """Compress the action of a prefix on a specified prepared input.

    If ``S`` is ``initial_state_preparation``, optimize
    ``|<0|S† V(theta)† U_target S|0>|²`` and return the evolution circuit ``V``.
    Callers must still apply ``S`` exactly once before ``V``. Omitting ``S``
    explicitly means a zero input. This is state compilation, not a guarantee
    that ``V`` approximates the target unitary on other inputs.
    """
    aqc_settings = aqc_settings or AQCSettings()
    if aqc_settings.simulator != "quimb_jax":
        raise ValueError("This compression implementation supports simulator='quimb_jax' only.")
    try:
        import numpy as np
        from scipy.optimize import minimize
        from qiskit_addon_aqc_tensor.ansatz_generation import generate_ansatz_from_circuit
        from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
        from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError(
            "AQC compression requires qiskit-addon-aqc-tensor[quimb-jax]."
        ) from exc

    if target_prefix_circuit.num_qubits != good_prefix_circuit.num_qubits:
        raise ValueError("Target and good prefix must have equal qubit counts.")
    preparation = initial_state_preparation
    if preparation is None:
        preparation = QuantumCircuit(target_prefix_circuit.num_qubits)
    if preparation.num_qubits != target_prefix_circuit.num_qubits:
        raise ValueError("Initial preparation and prefix must have equal qubit counts.")
    if preparation.num_parameters or preparation.num_clbits:
        raise ValueError("Initial preparation must be bound and have no classical bits.")

    record = AQCProfileRecord(
        problem="",
        instance_id="",
        n_qubits=target_prefix_circuit.num_qubits,
        num_steps=0,
        alpha=0,
        optimizer_method=aqc_settings.method,
        simulator_backend=aqc_settings.simulator,
        fidelity_input="zero" if initial_state_preparation is None else "provided_preparation",
        mps_max_bond_dimension=aqc_settings.mps_max_bond_dimension,
    )

    simulator_settings = build_quimb_jax_simulator(aqc_settings.mps_max_bond_dimension)
    prepared_target = preparation.compose(target_prefix_circuit)
    with profile_block() as target_profile:
        target_state = tensornetwork_from_circuit(
            prepared_target.decompose(reps=decompose_reps),
            simulator_settings,
        )
    record.target_build_sec = target_profile["elapsed_sec"]
    record.peak_rss_mb = target_profile["peak_rss_mb"]
    record.target_prefix_twoq_depth = two_qubit_depth(
        target_prefix_circuit, decompose_reps=decompose_reps
    )
    record.good_prefix_twoq_depth = two_qubit_depth(
        good_prefix_circuit, decompose_reps=decompose_reps
    )

    ansatz, x0 = generate_ansatz_from_circuit(
        good_prefix_circuit.decompose(reps=decompose_reps)
    )
    record.ansatz_num_parameters = len(x0)
    prepared_ansatz = preparation.compose(ansatz)
    objective = MaximizeStateFidelity(target_state, prepared_ansatz, simulator_settings)
    stopping_point = 1.0 - float(aqc_settings.stopping_fidelity)

    history: list[float] = []

    def fun(theta: Any) -> tuple[float, Any]:
        value, gradient = objective.loss_function(np.asarray(theta, dtype=float))
        history.append(float(value))
        return value, gradient

    def callback(intermediate_result: Any = None, **kwargs: Any) -> None:
        if not history:
            return
        if aqc_settings.verbose:
            print(f"AQC fidelity={1.0 - history[-1]:.8f}")
        if history[-1] < stopping_point:
            raise StopIteration

    with profile_block() as compression_profile:
        result = minimize(
            fun,
            x0,
            method=aqc_settings.method,
            jac=True,
            options={"maxiter": int(aqc_settings.maxiter)},
            callback=callback,
        )

    record.compression_sec = compression_profile["elapsed_sec"]
    if compression_profile["peak_rss_mb"] is not None:
        record.peak_rss_mb = max(record.peak_rss_mb or 0.0, compression_profile["peak_rss_mb"])
    record.optimizer_iterations = getattr(result, "nit", None)
    record.objective_evaluations = getattr(result, "nfev", None)
    record.optimizer_status_code = int(getattr(result, "status", 0))
    record.optimizer_message = str(getattr(result, "message", ""))
    # Re-evaluate the returned parameters: callback termination can otherwise
    # leave a stale result.fun that does not describe the returned circuit.
    final_loss, _ = objective.loss_function(np.asarray(result.x, dtype=float))
    record.final_infidelity = float(final_loss)
    record.final_fidelity = 1.0 - float(final_loss)
    record.stopping_fidelity_reached = record.final_fidelity >= aqc_settings.stopping_fidelity
    if record.objective_evaluations is not None:
        record.objective_evaluations += 1
    if result.status not in (0, 1, 99):
        record.status = f"optimizer_status_{result.status}:{result.message}"

    compressed_prefix = ansatz.assign_parameters(result.x, inplace=False)
    compressed_prefix.metadata = {
        "aqc_scheme_version": AQC_SCHEME_VERSION,
        "input_contract": "apply_after_initial_state_preparation",
        "fidelity_input": record.fidelity_input,
        "mps_max_bond_dimension": aqc_settings.mps_max_bond_dimension,
        "mps_bond_limit_scope": "target_circuit_simulation",
        "final_fidelity": record.final_fidelity,
    }
    record.compressed_prefix_twoq_depth = two_qubit_depth(
        compressed_prefix, decompose_reps=decompose_reps
    )
    return compressed_prefix, record


def profile_aqc_prefix(
    *,
    problem: str,
    instance_id: str,
    n_qubits: int,
    num_steps: int,
    alpha: int,
    build_target_prefix: Callable[[], Any],
    compress_prefix: Callable[[Any], Any],
    build_full_circuit: Callable[[Any], Any],
    pass_manager: Any | None = None,
    settings: AQCSettings | None = None,
    backend_name: str | None = None,
    cache_path: str | Path | None = None,
) -> tuple[AQCProfileRecord, Any]:
    """Profile target build, AQC compression, and optional transpilation."""
    settings = settings or AQCSettings()
    record = AQCProfileRecord(
        problem=problem,
        instance_id=instance_id,
        n_qubits=n_qubits,
        num_steps=num_steps,
        alpha=alpha,
        optimizer_method=settings.method,
        backend_name=backend_name,
    )

    try:
        with profile_block() as target_prof:
            target = build_target_prefix()
        record.target_build_sec = target_prof["elapsed_sec"]
        record.peak_rss_mb = target_prof["peak_rss_mb"]
        record.target_prefix_twoq_depth = two_qubit_depth(target)

        with profile_block() as comp_prof:
            compressed = compress_prefix(target)
        record.compression_sec = comp_prof["elapsed_sec"]
        if comp_prof["peak_rss_mb"] is not None:
            record.peak_rss_mb = max(record.peak_rss_mb or 0.0, comp_prof["peak_rss_mb"])
        record.compressed_prefix_twoq_depth = two_qubit_depth(compressed)

        full = build_full_circuit(compressed)
        full_t, transpile_sec = transpile_with_profile(full, pass_manager)
        record.transpile_sec = transpile_sec
        record.full_twoq_depth = two_qubit_depth(full_t)

        if cache_path is not None:
            save_qpy(cache_path, [target, compressed, full_t])
            record.cache_path = str(cache_path)
        return record, full_t
    except Exception as exc:
        record.status = f"failed:{type(exc).__name__}:{exc}"
        return record, None
