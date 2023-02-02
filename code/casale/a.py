from qiskit import QuantumCircuit, QuantumRegister


def circ(x_reg: QuantumRegister) -> QuantumCircuit:
    circ = QuantumCircuit(x_reg, name="$A$")
    circ.h(x_reg)
    return circ
