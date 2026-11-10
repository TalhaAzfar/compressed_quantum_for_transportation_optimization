"""Mapping, decoding and fixed-instance inference for the contribution pilot."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/contribution_hardware_verification_2026'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temporary.replace(path)


def settings():
    return read(ROOT / 'configs/contribution_hardware_verification.json')


def connection_config():
    base = read(ROOT / 'configs/aqc_tensor_qaoa_v2.json')
    return {k: base[k] for k in ['account', 'instance_name', 'backend', 'processor_family_required']}


def measured_wires(qc):
    result = {}
    for inst in qc.data:
        if inst.operation.name == 'measure':
            bit = qc.find_bit(inst.clbits[0]).index
            if bit in result:
                raise ValueError('Repeated measurement bit')
            result[bit] = qc.find_bit(inst.qubits[0]).index
    if set(result) != set(range(qc.num_clbits)):
        raise ValueError('Incomplete measurement register')
    return [result[i] for i in range(qc.num_clbits)]


def embed_local(qc, physical, final_layout, width):
    from qiskit import QuantumCircuit, ClassicalRegister
    n = qc.num_qubits
    if len(physical) != n or len(set(physical)) != n or sorted(final_layout) != list(range(n)):
        raise ValueError('Invalid physical patch or final permutation')
    if qc.num_clbits or max(physical) >= width:
        raise ValueError('Expected an unmeasured local circuit fitting the backend')
    isa = QuantumCircuit(width)
    isa.compose(qc, qubits=physical, inplace=True)
    isa.add_register(ClassicalRegister(n, 'meas'))
    for logical, wire in enumerate(final_layout):
        isa.measure(physical[wire], logical)
    assert measured_wires(isa) == [physical[q] for q in final_layout]
    return isa


def check_isa(qc, target, physical, final_layout):
    if measured_wires(qc) != [physical[q] for q in final_layout]:
        raise ValueError('Measurement does not decode to logical variable order')
    for inst in qc.data:
        wires = tuple(qc.find_bit(q).index for q in inst.qubits)
        if not all(q in physical for q in wires):
            raise ValueError('Operation outside the frozen physical patch')
        if inst.operation.name == 'barrier':
            continue
        if not target.instruction_supported(operation_name=inst.operation.name, qargs=wires):
            raise ValueError(f'Unsupported instruction {inst.operation.name} {wires}')
    return dict(cz_count=int(qc.count_ops().get('cz', 0)),
                cz_depth=int(qc.depth(lambda inst: inst.operation.name == 'cz')))


def scalar_metrics(model, counts):
    from .validate_tensor_prefix import independent_decode
    from .tensor_prefix_models import count_metrics
    expected = {event: 0 for event in ['feasible', 'good', 'optimal']}
    for bits, count in counts.items():
        if len(bits) != model['n_qubits'] or set(bits) - {'0', '1'} or count <= 0:
            raise ValueError('Invalid measured bitstring or multiplicity')
        value = independent_decode(model, bits)
        for event in expected:
            expected[event] += count * value[event]
    vector = count_metrics(model, counts)
    assert all(vector[event + '_count'] == count for event, count in expected.items())
    return vector


def policy_method(plan, group, cid, policy):
    return plan['selector_choices'][cid][policy] if group == 'selector' else policy


def stratified_comparison(plan, data, spec, cases=None, event=None):
    """Shared selector choices have exactly zero effect and are not re-counted."""
    from .audit_tensor_prefix_statistics import conditional_two_sided
    group = spec['group']; event = event or spec['event']
    cases = cases or plan[group + '_cases']
    tables = []; differences = []; case_rows = []
    for cid in cases:
        left = policy_method(plan, group, cid, spec['treatment'])
        right = policy_method(plan, group, cid, spec['control'])
        ka = kb = na = nb = 0
        for block in range(plan['blocks']):
            a = data[group, cid, left, block]; b = data[group, cid, right, block]
            ai, bi = a[event + '_count'], b[event + '_count']
            assert a['shots'] == b['shots']
            ka += ai; kb += bi; na += a['shots']; nb += b['shots']
            if left != right:
                tables.append((ai, a['shots'], bi, b['shots']))
        delta = ka / na - kb / nb
        differences.append(delta)
        case_rows.append(dict(case_id=cid, treatment_method=left, control_method=right,
                              treatment_count=ka, control_count=kb, shots_each=na,
                              difference=delta, shared_circuit=left == right))
    return dict(**{k: v for k, v in spec.items() if k != 'event'}, event=event,
                difference=sum(differences) / len(differences),
                p_value=conditional_two_sided(tables) if tables else 1.,
                informative_case_count=sum(not r['shared_circuit'] for r in case_rows),
                cases=case_rows)
