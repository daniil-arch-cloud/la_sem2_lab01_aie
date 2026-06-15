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
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """

    # На всякий случай поддерживаем оба порядка аргументов:
    # tt_svd(tensor, backend, ...)
    # tt_svd(backend, tensor, ...)
    if hasattr(tensor, "svd") and hasattr(backend, "shape"):
        tensor, backend = backend, tensor

    if backend is None:
        raise ValueError("backend не должен быть None")

    if eps < 0:
        raise ValueError("eps должен быть неотрицательным")

    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank должен быть положительным или None")

    shape = tuple(backend.shape(tensor))
    d = len(shape)

    if d == 0:
        raise ValueError("тензор должен иметь хотя бы одну ось")

    if d == 1:
        core = backend.reshape(tensor, (1, shape[0], 1))
        return TTTensor([core])

    norm_a = backend.norm(tensor)

    if norm_a > 1e-30:
        delta = eps * norm_a / math.sqrt(d - 1)
    else:
        delta = 0.0

    cores: list[DenseTensor] = []

    C = backend.copy(tensor)
    r_prev = 1

    for k in range(d - 1):
        n_k = shape[k]

        rows = r_prev * n_k
        cols = 1
        for value in shape[k + 1:]:
            cols *= value

        matrix = backend.reshape(C, (rows, cols))

        U, S, Vt = backend.svd(matrix, full_matrices=False)

        rank = _compute_truncated_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        core = backend.reshape(U_trunc, (r_prev, n_k, rank))
        cores.append(core)

        C_matrix = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)
        C = backend.reshape(C_matrix, (rank,) + shape[k + 1:])

        r_prev = rank

    last_core = backend.reshape(C, (r_prev, shape[-1], 1))
    cores.append(last_core)

    return TTTensor(cores)


def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
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
            tail_sum = 0.0

            for idx in range(candidate, numerical_rank):
                tail_sum += S[idx] * S[idx]

            if tail_sum <= delta * delta:
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
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")

    rows, cols = matrix.shape

    if rank < 0 or rank > cols:
        raise ValueError("некорректный rank")

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
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")

    rows, cols = matrix.shape

    if rank < 0 or rank > rows:
        raise ValueError("некорректный rank")

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
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.
    """
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
    """
    Возвращает произведение diag(diag_vec) @ matrix.
    """
    if diag_vec.ndim != 1:
        raise ValueError("diag_vec должен быть 1D")

    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")

    if diag_vec.shape[0] < rank or matrix.shape[0] < rank:
        raise ValueError("rank несовместим с размерами")

    _, cols = matrix.shape

    result = backend.zeros((rank, cols))

    for i in range(rank):
        scale = diag_vec[i]
        for j in range(cols):
            result[i, j] = scale * matrix[i, j]

    return result