# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).
"""

from __future__ import annotations

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size, flat_to_multi_index


class TTTensor:
    """
    Тензор в TT-формате.
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    def __init__(self, cores: list[DenseTensor]) -> None:
        if not isinstance(cores, list):
            raise TypeError("cores должен быть списком DenseTensor")
        if len(cores) == 0:
            raise ValueError("TT-тензор должен иметь хотя бы одно ядро")

        copied_cores: list[DenseTensor] = []
        for core in cores:
            if not isinstance(core, DenseTensor):
                raise TypeError("каждое TT-ядро должно быть DenseTensor")
            if core.ndim != 3:
                raise ValueError("каждое TT-ядро должно иметь форму (r_k, n_k, r_{k+1})")
            copied_cores.append(core.copy())

        if copied_cores[0].shape[0] != 1:
            raise ValueError("первый TT-ранг r_0 должен быть равен 1")
        if copied_cores[-1].shape[2] != 1:
            raise ValueError("последний TT-ранг r_d должен быть равен 1")

        for k in range(len(copied_cores) - 1):
            if copied_cores[k].shape[2] != copied_cores[k + 1].shape[0]:
                raise ValueError(
                    f"соседние TT-ранги не согласованы между ядрами {k} и {k + 1}"
                )

        self.cores = copied_cores
        self.order = len(copied_cores)
        self.shape = tuple(core.shape[1] for core in copied_cores)
        self.ranks = (copied_cores[0].shape[0],) + tuple(
            core.shape[2] for core in copied_cores
        )

    @staticmethod
    def random(shape, ranks, seed=None):
        import random

        checked_shape = validate_shape(shape)
        d = len(checked_shape)

        if len(ranks) == d - 1:
            full_ranks = (1,) + tuple(ranks) + (1,)
        elif len(ranks) == d + 1:
            full_ranks = tuple(ranks)
        else:
            raise ValueError("ranks должен содержать d-1 внутренних или d+1 полных TT-рангов")

        if full_ranks[0] != 1 or full_ranks[-1] != 1:
            raise ValueError("граничные TT-ранги должны быть равны 1")

        rng = random.Random(seed)
        cores: list[DenseTensor] = []

        for k in range(d):
            core_shape = (full_ranks[k], checked_shape[k], full_ranks[k + 1])
            size = compute_size(core_shape)
            data = [rng.uniform(-1.0, 1.0) for _ in range(size)]
            cores.append(DenseTensor(core_shape, data=data))

        return TTTensor(cores)

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        if isinstance(indices, list):
            indices = tuple(indices)
        if len(indices) != self.order:
            raise ValueError(f"длина indices должна быть равна {self.order}")

        for axis, index in enumerate(indices):
            if not isinstance(index, int):
                raise TypeError("индексы должны быть целыми числами")
            if index < 0 or index >= self.shape[axis]:
                raise IndexError(
                    f"индекс {index} выходит за границы оси {axis} размера {self.shape[axis]}"
                )

        current = [1.0]

        for k, index in enumerate(indices):
            core = self.cores[k]
            r_left, _, r_right = core.shape
            next_values = [0.0] * r_right

            for beta in range(r_right):
                acc = 0.0
                for alpha in range(r_left):
                    acc += current[alpha] * core[alpha, index, beta]
                next_values[beta] = acc

            current = next_values

        return current[0]

    def full(self) -> DenseTensor:
        result = DenseTensor.zeros(self.shape)

        for flat_index in range(result.size):
            multi_index = flat_to_multi_index(flat_index, self.shape)
            result.data[flat_index] = self.get_element(multi_index)

        return result

    def core_sizes(self) -> list[tuple[int, ...]]:
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        storage = self.total_storage()
        if storage == 0:
            return float("inf")
        return compute_size(self.shape) / storage

    def copy(self) -> TTTensor:
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        return (
            "TTTensor("
            f"order={self.order}, "
            f"shape={self.shape}, "
            f"ranks={self.ranks}, "
            f"core_sizes={self.core_sizes()}, "
            f"storage={self.total_storage()}"
            ")"
        )

    def __str__(self) -> str:
        return self.__repr__()