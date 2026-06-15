# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.tensor_operations import tt_norm


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами.

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if eps < 0:
        raise ValueError("eps должен быть неотрицательным")

    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank должен быть положительным или None")

    if tt.order == 1:
        return tt.copy()

    # Важно: здесь нельзя использовать QR-based right_canonicalize,
    # потому что после tt_add / tt_hadamard TT-ранги могут быть больше,
    # чем n_k * r_k справа. Тогда thin QR падает на матрицах m < n.
    # Поэтому для TT-rounding делаем правую ортогонализацию через SVD:
    # она одновременно ортогонализует и безопасно уменьшает невозможные ранги.
    rounded = _right_orthogonalize_for_rounding(tt, backend)

    cores = [core.copy() for core in rounded.cores]
    d = len(cores)

    norm_value = tt_norm(rounded, backend)

    if norm_value > 1e-30:
        delta = eps * norm_value / math.sqrt(d - 1)
    else:
        delta = 0.0

    # Левый проход: SVD-усечение слева направо.
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

            for alpha in range(r_right):
                for beta in range(r_next):
                    old_slice[alpha, beta] = next_core[alpha, i_next, beta]

            updated_slice = backend.matmul(transfer, old_slice)

            for alpha in range(rank):
                for beta in range(r_next):
                    new_next[alpha, i_next, beta] = updated_slice[alpha, beta]

        cores[k + 1] = new_next

    return TTTensor(cores)


def _right_orthogonalize_for_rounding(
    tt: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Правая ортогонализация для TT-rounding через SVD.

    Отличие от QR-based right_canonicalize:
    эта версия безопасна, когда левый ранг ядра больше, чем n_k * r_k.
    Такое часто возникает после tt_add и tt_hadamard.
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

            for alpha in range(r_prev):
                for beta in range(r_left):
                    old_slice[alpha, beta] = prev_core[alpha, i_prev, beta]

            updated_slice = backend.matmul(old_slice, transfer)

            for alpha in range(r_prev):
                for beta in range(rank):
                    new_prev[alpha, i_prev, beta] = updated_slice[alpha, beta]

        cores[k - 1] = new_prev

    return TTTensor(cores)


def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг по сингулярным значениям.
    """
    if S.ndim != 1:
        raise ValueError("S должен быть одномерным тензором")

    if S.shape[0] == 0:
        return 1

    max_s = max(abs(value) for value in S.data)
    threshold = max(abs_tol, rel_tol * max_s)

    rank = 0

    for value in S.data:
        if abs(value) > threshold:
            rank += 1

    return max(1, rank)


def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
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
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix
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


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат:
        matrix @ diag(diag_vec)
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")

    if diag_vec.ndim != 1:
        raise ValueError("diag_vec должен быть 1D")

    rows, cols = matrix.shape

    if diag_vec.shape[0] != cols:
        raise ValueError("длина diag_vec должна совпадать с числом столбцов matrix")

    result = backend.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            result[i, j] = matrix[i, j] * diag_vec[j]

    return result