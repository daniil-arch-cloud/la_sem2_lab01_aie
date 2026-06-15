# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.
    """
    if tt.order == 1:
        return tt.copy()

    cores = [core.copy() for core in tt.cores]
    d = len(cores)

    for k in range(d - 1):
        core = cores[k]
        r_left, n, r_right = core.shape

        matrix = core.reshape((r_left * n, r_right))

        U, S, Vt = backend.svd(matrix, full_matrices=False)
        rank = _numerical_rank(S)
        rank = min(rank, S.shape[0])

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        cores[k] = U_trunc.reshape((r_left, n, rank))

        transfer = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)

        next_core = cores[k + 1]
        _, n_next, r_next = next_core.shape
        new_next = backend.zeros((rank, n_next, r_next))

        for i_next in range(n_next):
            old_slice = backend.zeros((r_right, r_next))
            for a in range(r_right):
                for b in range(r_next):
                    old_slice[a, b] = next_core[a, i_next, b]

            updated_slice = backend.matmul(transfer, old_slice)

            for a in range(rank):
                for b in range(r_next):
                    new_next[a, i_next, b] = updated_slice[a, b]

        cores[k + 1] = new_next

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.
    """
    if tt.order == 1:
        return tt.copy()

    cores = [core.copy() for core in tt.cores]
    d = len(cores)

    for k in range(d - 1, 0, -1):
        core = cores[k]
        r_left, n, r_right = core.shape

        matrix = core.reshape((r_left, n * r_right))

        U, S, Vt = backend.svd(matrix, full_matrices=False)
        rank = _numerical_rank(S)
        rank = min(rank, S.shape[0])

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        cores[k] = Vt_trunc.reshape((rank, n, r_right))

        transfer = _multiply_columns_by_diag(U_trunc, S_trunc, backend)

        prev_core = cores[k - 1]
        r_prev, n_prev, _ = prev_core.shape
        new_prev = backend.zeros((r_prev, n_prev, rank))

        for i_prev in range(n_prev):
            old_slice = backend.zeros((r_prev, r_left))
            for a in range(r_prev):
                for b in range(r_left):
                    old_slice[a, b] = prev_core[a, i_prev, b]

            updated_slice = backend.matmul(old_slice, transfer)

            for a in range(r_prev):
                for b in range(rank):
                    new_prev[a, i_prev, b] = updated_slice[a, b]

        cores[k - 1] = new_prev

    return TTTensor(cores)


def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    if S.ndim != 1:
        raise ValueError("S должен быть одномерным тензором")

    if S.shape[0] == 0:
        return 0

    max_s = max(abs(value) for value in S.data)
    threshold = max(abs_tol, rel_tol * max_s)

    rank = 0
    for value in S.data:
        if abs(value) > threshold:
            rank += 1

    return max(1, rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")
    if rank < 0 or rank > matrix.shape[1]:
        raise ValueError("некорректный rank")

    rows, _ = matrix.shape
    result = backend.zeros((rows, rank))

    for i in range(rows):
        for j in range(rank):
            result[i, j] = matrix[i, j]

    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")
    if rank < 0 or rank > matrix.shape[0]:
        raise ValueError("некорректный rank")

    _, cols = matrix.shape
    result = backend.zeros((rank, cols))

    for i in range(rank):
        for j in range(cols):
            result[i, j] = matrix[i, j]

    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    if vector.ndim != 1:
        raise ValueError("vector должен быть 1D")
    if rank < 0 or rank > vector.shape[0]:
        raise ValueError("некорректный rank")

    result = backend.zeros((rank,))

    for i in range(rank):
        result[i] = vector[i]

    return result


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    if diag_vec.ndim != 1 or matrix.ndim != 2:
        raise ValueError("ожидались diag_vec 1D и matrix 2D")
    if diag_vec.shape[0] < rank or matrix.shape[0] < rank:
        raise ValueError("rank несовместим с размерами")

    _, cols = matrix.shape
    result = backend.zeros((rank, cols))

    for i in range(rank):
        scale = diag_vec[i]
        for j in range(cols):
            result[i, j] = scale * matrix[i, j]

    return result


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    if matrix.ndim != 2 or diag_vec.ndim != 1:
        raise ValueError("ожидались matrix 2D и diag_vec 1D")

    rows, cols = matrix.shape
    if diag_vec.shape[0] != cols:
        raise ValueError("длина diag_vec должна совпадать с числом столбцов matrix")

    result = backend.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            result[i, j] = matrix[i, j] * diag_vec[j]

    return result