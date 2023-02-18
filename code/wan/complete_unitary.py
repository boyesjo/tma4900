import numpy as np
import scipy.linalg as la


def ortonormalise(
    new: np.ndarray, dict_of_rows: dict[int, np.ndarray]
) -> np.ndarray:
    for row in dict_of_rows.values():
        new -= (row @ new) * row

    assert la.norm(new) > 1e-10
    new /= la.norm(new)
    return new


def complete_unitary(dict_of_rows: dict[int, np.ndarray]) -> np.ndarray:
    """Given a dictionary of orthonormal rows,
    return a complete unitary matrix."""

    # assert ortormality
    for row in dict_of_rows.values():
        for other_row in dict_of_rows.values():
            if np.isclose(row @ other_row, 1 if row is other_row else 0):
                continue
            raise ValueError("Not orthonormal")

    n_cols = len(list(dict_of_rows.values())[0])
    mat = np.zeros((n_cols, n_cols), dtype=complex)

    d = dict_of_rows.copy()
    rs = np.random.RandomState(0)

    for i in range(n_cols):

        if i in d:
            continue

        try:
            # assert False
            # attempt using basis vectors, preserving some sparsity
            new_row = np.zeros(n_cols, dtype=complex)
            new_row[i] = 1
            new_row = ortonormalise(new_row, d)
        except AssertionError:
            new_row = rs.rand(n_cols)
            new_row = new_row.astype(complex)
            new_row = ortonormalise(new_row, d)

        d[i] = new_row

    for i, row in d.items():
        mat[i] = row

    return mat


def main() -> None:
    d = {
        0: np.array([0, 1, 0, 0]),
        # 1: np.array([0, np.sqrt(0.5), np.sqrt(0.5), 0]),
    }
    mat = complete_unitary(d)
    print(mat)
    print("\n" * 3)
    print(mat @ mat.T.conj())
    assert np.allclose(mat @ mat.T.conj(), np.eye(4))


if __name__ == "__main__":
    main()