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

    Левая канонизация делается QR-разложением слева направо.
    """
    if tt.order == 1:
        return tt.copy()

    cores = [core.copy() for core in tt.cores]
    d = len(cores)

    for k in range(d - 1):
        core = cores[k]
        r_left, n, r_right = core.shape

        matrix = core.reshape((r_left * n, r_right))

        Q, R = backend.qr(matrix)

        cores[k] = Q.reshape((r_left, n, r_right))

        next_core = cores[k + 1]
        _, n_next, r_next = next_core.shape

        new_next = backend.zeros((r_right, n_next, r_next))

        for i_next in range(n_next):
            old_slice = backend.zeros((r_right, r_next))

            for alpha in range(r_right):
                for beta in range(r_next):
                    old_slice[alpha, beta] = next_core[alpha, i_next, beta]

            updated_slice = backend.matmul(R, old_slice)

            for alpha in range(r_right):
                for beta in range(r_next):
                    new_next[alpha, i_next, beta] = updated_slice[alpha, beta]

        cores[k + 1] = new_next

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Правая канонизация делается RQ-разложением справа налево.
    RQ получаем как QR от транспонированной матрицы:
        A.T = Q_t R_t
        A   = R_t.T Q_t.T
    """
    if tt.order == 1:
        return tt.copy()

    cores = [core.copy() for core in tt.cores]
    d = len(cores)

    for k in range(d - 1, 0, -1):
        core = cores[k]
        r_left, n, r_right = core.shape

        matrix = core.reshape((r_left, n * r_right))

        matrix_t = backend.transpose(matrix)
        Q_t, R_t = backend.qr(matrix_t)

        Q = backend.transpose(Q_t)
        R = backend.transpose(R_t)

        cores[k] = Q.reshape((r_left, n, r_right))

        prev_core = cores[k - 1]
        r_prev, n_prev, _ = prev_core.shape

        new_prev = backend.zeros((r_prev, n_prev, r_left))

        for i_prev in range(n_prev):
            old_slice = backend.zeros((r_prev, r_left))

            for alpha in range(r_prev):
                for beta in range(r_left):
                    old_slice[alpha, beta] = prev_core[alpha, i_prev, beta]

            updated_slice = backend.matmul(old_slice, R)

            for alpha in range(r_prev):
                for beta in range(r_left):
                    new_prev[alpha, i_prev, beta] = updated_slice[alpha, beta]

        cores[k - 1] = new_prev

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# Оставлены, чтобы не ломать ожидаемую структуру файла.
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.
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
    Возвращает diag(diag_vec) @ matrix.
    """
    if diag_vec.ndim != 1:
        raise ValueError("diag_vec должен быть 1D")

    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")

    _, cols = matrix.shape

    result = backend.zeros((rank, cols))

    for i in range(rank):
        for j in range(cols):
            result[i, j] = diag_vec[i] * matrix[i, j]

    return result


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает matrix @ diag(diag_vec).
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть 2D")

    if diag_vec.ndim != 1:
        raise ValueError("diag_vec должен быть 1D")

    rows, cols = matrix.shape

    if diag_vec.shape[0] != cols:
        raise ValueError("длина diag_vec должна совпадать с числом столбцов")

    result = backend.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            result[i, j] = matrix[i, j] * diag_vec[j]

    return result