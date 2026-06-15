# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    if not isinstance(tensor, DenseTensor):
        raise TypeError("tensor должен быть DenseTensor")
    if eps < 0:
        raise ValueError("eps должен быть неотрицательным")
    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank должен быть положительным или None")

    shape = tensor.shape
    d = tensor.ndim

    if d == 1:
        return TTTensor([tensor.reshape((1, shape[0], 1))])

    norm_a = backend.norm(tensor)
    delta = eps * norm_a / math.sqrt(d - 1) if norm_a > 1e-30 else 0.0

    cores: list[DenseTensor] = []
    C = tensor.copy()
    r_prev = 1

    for k in range(d - 1):
        n_k = shape[k]
        rows = r_prev * n_k
        cols = 1
        for value in shape[k + 1:]:
            cols *= value

        matrix = C.reshape((rows, cols))

        U, S, Vt = backend.svd(matrix, full_matrices=False)
        rank = _compute_truncated_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        core = U_trunc.reshape((r_prev, n_k, rank))
        cores.append(core)

        C_matrix = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)
        C = C_matrix.reshape((rank,) + shape[k + 1:])
        r_prev = rank

    last_core = C.reshape((r_prev, shape[-1], 1))
    cores.append(last_core)

    return TTTensor(cores)


def _compute_truncated_rank(
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