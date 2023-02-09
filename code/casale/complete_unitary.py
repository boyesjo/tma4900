import numpy as np
import scipy.linalg as la


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
    np.random.seed(0)  # ensure consistent results

    for i in range(n_cols):

        if i in d:
            continue

        new_row = np.random.rand(n_cols)
        new_row = new_row.astype(complex)

        for row in d.values():
            new_row -= (row @ new_row) * row

        assert la.norm(new_row) > 1e-10
        new_row /= la.norm(new_row)
        d[i] = new_row

    for i, row in d.items():
        mat[i] = row

    return mat


def main() -> None:
    d = {
        0: np.array([1, 0, 0, 0]),
        1: np.array([0, np.sqrt(0.5), np.sqrt(0.5), 0]),
    }
    mat = complete_unitary(d)
    print(mat)
    print("\n" * 3)
    print(mat @ mat.T.conj())
    assert np.allclose(mat @ mat.T.conj(), np.eye(4))


if __name__ == "__main__":
    main()
