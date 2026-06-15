# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize
from algorithms.tensor_operations import tt_norm


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    if eps < 0:
        raise ValueError("eps должен быть неотрицательным")
    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank должен быть положительным или None")

    if tt.order == 1:
        return tt.copy()

    rounded = right_canonicalize(tt, backend)
    cores = [core.copy() for core in rounded.cores]
    d = len(cores)

    norm_value = tt_norm(rounded, backend)
    delta = eps * norm_value / math.sqrt(d - 1) if norm_value > 1e-30 else 0.0

    for k in range(d - 1):
        core = cores[k]
        r_left, n, r_right = core.shape

        matrix = core.reshape((r_left * n, r_right))

        U, S, Vt = backend.svd(matrix, full_matrices=False)
        rank = _compute_rank(S, delta, max_rank)

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


def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    if S.ndim != 1:
        raise ValueError("S должен быть одномерным тензором")

    count = S.shape[0]
    if count == 0:
        return 1

    sigma_1 = abs(S[0])
    threshold = max(1e-12, 1e-8 * sigma_1)

    numerical_rank = 0
    for value in S.data:
        if abs(value) > threshold:
            numerical_rank += 1

    numerical_rank = max(1, numerical_rank)
    rank = numerical_rank

    if delta > 0.0:
        for candidate in range(1, numerical_rank + 1):
            tail = 0.0
            for idx in range(candidate, numerical_rank):
                tail += S[idx] * S[idx]

            if tail <= delta * delta:
                rank = candidate
                break

    if max_rank is not None:
        rank = min(rank, max_rank)

    rank = max(1, rank)
    rank = min(rank, count)

    return rank


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