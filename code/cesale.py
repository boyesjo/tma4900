# %%
import numpy as np
import matplotlib.pyplot as plt
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.algorithms import AmplitudeAmplifier, Grover, AmplificationProblem
from qiskit import Aer


# %%
n_arms = 8
n_qubits = int(np.log2(n_arms))


def nu_x(x: int):
    P_LIST = np.linspace(0, 0.8, n_arms)
    return P_LIST[x]


def f(x: int, y: int):
    return y == 1


def init_state(qc, n):
    for i in range(n):
        qc.h(i)
    return qc


x = QuantumRegister(n_qubits, name="x")
y = QuantumRegister(1, name="y")
c = ClassicalRegister(n_qubits, name="c")

qc = QuantumCircuit(x, y, c)


def a_gate():
    qc = QuantumCircuit(x)
    qc.h(x)
    return qc


def oracle_e():

    # map |x>|0> to |x>|nu(x)>
    qc = QuantumCircuit(x, y)
    for i in range(n_qubits):
        qc.u3(2 * nu_x(i), 0, 0, x[i])
    return qc


def oracle_f():
    qc = QuantumCircuit(x, y)
    # flip amplitude of |xy> state if f(x, y) = 1
    for i in range(n_qubits):
        qc.cx(x[i], y[i]).c_if(f())
    return qc.to_gate()


qc = qc.compose(a_gate(), qubits=x)
qc.measure(x, c)
qc.draw("mpl")
# %%
aer_sim = Aer.get_backend("qasm_simulator")
qobj = assemble(qc)
result = aer_sim.run(qobj).result()
counts = result.get_counts()
counts

# %%


def o():
    def _is_good_state(bitstring: str):
        x = int(bitstring[:n_qubits], 2)
        y = int(bitstring[n_qubits:], 2)
        return f(x, y)

    qc = QuantumCircuit(x, y)

    prob = AmplificationProblem(
        is_good_state=_is_good_state,
    )

    grover = Grover(quantum_instance=aer_sim)
    qc = grover.construct_circuit(prob, measurement=False)
    return qc


o()

# %%
