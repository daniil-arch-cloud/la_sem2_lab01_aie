# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковую shape")
    if tt1.order != tt2.order:
        raise ValueError("TT-тензоры должны иметь одинаковый порядок")

    d = tt1.order

    if d == 1:
        core1 = tt1.cores[0]
        core2 = tt2.cores[0]
        return TTTensor([backend.add(core1, core2)])

    result_cores: list[DenseTensor] = []

    for k in range(d):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n, r1_right = core1.shape
        r2_left, n2, r2_right = core2.shape

        if n != n2:
            raise ValueError("размеры физических мод TT-ядер не совпадают")

        if k == 0:
            result = backend.zeros((1, n, r1_right + r2_right))

            for i in range(n):
                for beta in range(r1_right):
                    result[0, i, beta] = core1[0, i, beta]
                for beta in range(r2_right):
                    result[0, i, r1_right + beta] = core2[0, i, beta]

        elif k == d - 1:
            result = backend.zeros((r1_left + r2_left, n, 1))

            for i in range(n):
                for alpha in range(r1_left):
                    result[alpha, i, 0] = core1[alpha, i, 0]
                for alpha in range(r2_left):
                    result[r1_left + alpha, i, 0] = core2[alpha, i, 0]

        else:
            result = backend.zeros(
                (r1_left + r2_left, n, r1_right + r2_right)
            )

            for alpha in range(r1_left):
                for i in range(n):
                    for beta in range(r1_right):
                        result[alpha, i, beta] = core1[alpha, i, beta]

            for alpha in range(r2_left):
                for i in range(n):
                    for beta in range(r2_right):
                        result[
                            r1_left + alpha,
                            i,
                            r1_right + beta
                        ] = core2[alpha, i, beta]

        result_cores.append(result)

    return TTTensor(result_cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    result = tt.copy()
    result.cores[0] = backend.scale(result.cores[0], alpha)
    return TTTensor(result.cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковую shape")
    if tt1.order != tt2.order:
        raise ValueError("TT-тензоры должны иметь одинаковый порядок")

    result_cores: list[DenseTensor] = []

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n, r1_right = core1.shape
        r2_left, n2, r2_right = core2.shape

        if n != n2:
            raise ValueError("размеры физических мод TT-ядер не совпадают")

        result = backend.zeros(
            (r1_left * r2_left, n, r1_right * r2_right)
        )

        for alpha1 in range(r1_left):
            for alpha2 in range(r2_left):
                left_index = alpha1 * r2_left + alpha2

                for i in range(n):
                    for beta1 in range(r1_right):
                        for beta2 in range(r2_right):
                            right_index = beta1 * r2_right + beta2
                            result[left_index, i, right_index] = (
                                core1[alpha1, i, beta1]
                                * core2[alpha2, i, beta2]
                            )

        result_cores.append(result)

    return TTTensor(result_cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковую shape")
    if tt1.order != tt2.order:
        raise ValueError("TT-тензоры должны иметь одинаковый порядок")

    Z = [[1.0]]

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n, r1_right = core1.shape
        r2_left, n2, r2_right = core2.shape

        if n != n2:
            raise ValueError("размеры физических мод TT-ядер не совпадают")

        new_Z = [[0.0 for _ in range(r2_right)] for _ in range(r1_right)]

        for beta1 in range(r1_right):
            for beta2 in range(r2_right):
                acc = 0.0

                for alpha1 in range(r1_left):
                    for alpha2 in range(r2_left):
                        z_value = Z[alpha1][alpha2]

                        if z_value == 0.0:
                            continue

                        for i in range(n):
                            acc += (
                                z_value
                                * core1[alpha1, i, beta1]
                                * core2[alpha2, i, beta2]
                            )

                new_Z[beta1][beta2] = acc

        Z = new_Z

    return Z[0][0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    value = tt_dot(tt, tt, backend)
    return math.sqrt(max(float(value), 0.0))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    value = (
        tt_dot(tt1, tt1, backend)
        + tt_dot(tt2, tt2, backend)
        - 2.0 * tt_dot(tt1, tt2, backend)
    )
    return math.sqrt(max(float(value), 0.0))