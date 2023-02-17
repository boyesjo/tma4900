# %%
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from complete_unitary import complete_unitary
from qiskit import (
    Aer,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    assemble,
    transpile,
)
from qiskit.algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    FasterAmplitudeEstimation,
)
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance

sampler = Sampler()
# set shots to 1
# sampler.set_options(shots=1)


class Oracle(QuantumCircuit):
    def __init__(
        self,
        p: float,
        w_len: int,
        name: str = "$O_x$",
    ):
        self.p_ = p
        self.w_len = w_len
        self.w_reg = QuantumRegister(w_len, name="w")
        self.y_reg = QuantumRegister(1, name="y")
        super().__init__(self.y_reg, self.w_reg, name=name)
        self._build()

    def _matrix(self) -> np.ndarray:
        p_w = np.zeros(2**self.w_len)

        def y_x(w: int) -> int:
            return 1 if w == 0 else 0

        p_w[0] = self.p_
        p_w[1:] = (1 - self.p_) / (2**self.w_len - 1)

        row = np.zeros(2 ** (self.w_len + 1))

        for w in range(2**self.w_len):

            for y in range(2):
                col = 2 * w + y
                row[col] = np.sqrt(p_w[w]) if y == y_x(w) else 0

        return complete_unitary({0: row}).T

    def _build(self) -> None:
        mat = self._matrix()

        self.unitary(
            mat,
            self.qubits,
            label=self.name,
        )

    def inverse(self) -> QuantumCircuit:
        inv = super().inverse()
        inv.name = self.name + "$^\\dagger$"
        return inv


qc = Oracle(0.9, 3)

backend = Aer.get_backend("statevector_simulator")
job = backend.run(assemble(qc))
result = job.result()
statevector = Statevector(result.get_statevector())
probs = statevector.probabilities_dict()
for k, v in probs.items():
    print(f"{k}: {v:.3f}")
print()


# %%
qc = Oracle(0.8, 2)

grov = QuantumCircuit(len(qc.qubits))
# reflect about |0>
grov.z(0)
# apply inverse oracle
grov.compose(qc.inverse(), inplace=True)
# reflect about |0>
grov.z(0)
# apply oracle
grov.compose(qc, inplace=True)
grov.draw("mpl")


def is_good_state(bitstring: str) -> bool:
    raise NotImplementedError
    print(bitstring)
    return bitstring == "1"


problem = EstimationProblem(
    state_preparation=qc,
    # grover_operator=grov,
    objective_qubits=[0],
    # is_good_state=is_good_state,
)

n_qubits = 4
ae = AmplitudeEstimation(
    num_eval_qubits=n_qubits,
    sampler=sampler,
    # quantum_instance=QuantumInstance(
    #     Aer.get_backend("qasm_simulator"), shots=1
    # ),
)
shots = 100
ae_circ = ae.construct_circuit(problem)
eval_reg = ae_circ.qregs[0]
c_reg = ClassicalRegister(len(eval_reg), "c")
ae_circ.add_register(c_reg)
ae_circ.measure(eval_reg, c_reg)
ae_circ = transpile(ae_circ, Aer.get_backend("aer_simulator"))
backend = Aer.get_backend("aer_simulator")
job = backend.run(assemble(ae_circ), shots=shots)
result = job.result()
counts = result.get_counts()

est = {}
for i in range(2**n_qubits):
    est[np.sin(np.pi * i / 2**n_qubits) ** 2] = (
        counts.get(f"{i:0{n_qubits}b}", 0) / shots
    )

print(est)
plt.bar(est.keys(), est.values())

# ae_result = ae.estimate(problem)
# print(ae_result.estimation)
# ae_result.__dict__

# %%
x = np.random.binomial(1, 0.8, size=(2**n_qubits, shots))
x = np.mean(x, axis=1)
plt.hist(x, bins=10)
plt.xlim(0, 1)
# %%
problem.grover_operator.decompose().draw("mpl")

# %%
def qmc(oracle: Oracle, eps: float, delta: float) -> float:
    """QMC as per Wan2022

    Args:
        oracle (Oracle): Oracle st. O|0> = sqrt(p(w))7 |w> |y(w)> summed over w
        eps (float): Bound for estimate of E(y)
        delta (float): Error probability for estimate of E(y)

    Returns:
        float: Estimate of E(y)
    """
    t = int(1 / eps)  # t = O(1/eps) [montanaro2015]
    assert t >= 1

    problem = EstimationProblem(
        state_preparation=oracle,
        objective_qubits=[0],
    )

    eval_qubits = int(np.ceil(np.log2(t)))

    est_list = []

    n_iter = int(np.log(1 / delta))
    assert n_iter >= 1

    print(f"{t=}, {eval_qubits=}, {n_iter=}")

    # ae = AmplitudeEstimation(
    #     num_eval_qubits=eval_qubits,
    #     # sampler=sampler,
    #     quantum_instance=QuantumInstance(
    #         Aer.get_backend("qasm_simulator"), shots=t
    #     ),
    # )
    # ae_result = ae.estimate(problem)
    # print(ae_result.mle)
    # est_list.append(ae_result.mle)
    # oracle_calls = ae_result.num_oracle_queries + 1

    n_iter = int(
        4.1e3 * np.log(4 * np.log2(2 * np.pi / (3 * eps)) / delta) / eps
    )

    print(f"{n_iter=}")

    ae = FasterAmplitudeEstimation(
        maxiter=2,
        delta=delta,
        rescale=False,
        # quantum_instance=QuantumInstance(
        #     Aer.get_backend("qasm_simulator"), shots=2
        # ),
        sampler=sampler,
    )
    ae_result = ae.estimate(problem)
    print(ae_result.__dict__)
    oracle_calls = ae_result.num_oracle_queries  # + 1

    est_list.append(ae_result.estimation)

    print(f"{oracle_calls=}")
    print(est_list)
    return np.median(est_list)


qmc(Oracle(0.17, 3), 0.999, 0.1)

# %%
