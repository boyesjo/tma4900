# %%
import numpy as np
import matplotlib.pyplot as plt
from qiskit.algorithms import Grover
from qiskit.algorithms import AmplificationProblem
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector

sampler = Sampler()
N_QUBITS = 5
good_state = "1" * N_QUBITS
oracle = Statevector.from_label(good_state)


problem = AmplificationProblem(
    oracle=oracle,
)
max_iterations = 20
prob_correct = np.zeros(max_iterations)
for i in range(0, 20):
    grover = Grover(sampler=sampler, iterations=i)
    results = grover.amplify(problem)
    # print probability of finding the good state
    prob_correct[i] = results.circuit_results[0][good_state]

    print(f"n_iter: {i}, prob_correct: {prob_correct[i]}")


plt.plot(prob_correct)
plt.xlabel("Number of iterations")
plt.ylabel("Probability of finding the good state")
plt.xticks(range(0, max_iterations, 2))
# optimal number of iterations is pi/4 * sqrt(N)
plt.axvline(np.pi / 4 * np.sqrt(2**N_QUBITS), color="red", linestyle="--")
# add explanation for the red line
plt.text(
    np.pi / 4 * np.sqrt(2**N_QUBITS) + 0.5,
    1,
    "Optimal number of iterations",
    color="red",
    fontsize=12,
)
plt.show()
