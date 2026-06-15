# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""

from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
    check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)

        if data is None:
            self.data = [float(fill)] * self.size
        else:
            if len(data) != self.size:
                raise ValueError(
                    f"Размер data={len(data)} не совпадает с shape={self.shape}, "
                    f"ожидалось {self.size}"
                )
            self.data = [float(value) for value in data]

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        rng = random.Random(seed)
        checked_shape = validate_shape(shape)
        size = compute_size(checked_shape)

        if integer:
            data = [float(rng.randint(low, high)) for _ in range(size)]
        else:
            data = [float(rng.uniform(low, high)) for _ in range(size)]

        return DenseTensor(checked_shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список или кортеж
        """
        def is_sequence(obj) -> bool:
            return isinstance(obj, (list, tuple))

        def infer_shape(obj) -> tuple[int, ...]:
            if not is_sequence(obj):
                return ()

            if len(obj) == 0:
                raise ValueError("пустые списки/кортежи не задают корректный тензор")

            first_shape = infer_shape(obj[0])

            for item in obj[1:]:
                if infer_shape(item) != first_shape:
                    raise ValueError("вложенная структура должна быть прямоугольной")

            return (len(obj),) + first_shape

        def flatten(obj) -> list[float]:
            if not is_sequence(obj):
                return [float(obj)]

            result: list[float] = []
            for item in obj:
                result.extend(flatten(item))
            return result

        shape = infer_shape(nested)

        if len(shape) == 0:
            return DenseTensor((1,), data=[float(nested)])

        return DenseTensor(shape, data=flatten(nested))

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | list[int] | int
    ) -> tuple[int, ...]:
        if isinstance(multi_index, int):
            if self.ndim != 1:
                raise TypeError("целочисленный индекс допустим только для 1D-тензора")
            multi_index = (multi_index,)
        elif isinstance(multi_index, list):
            multi_index = tuple(multi_index)
        elif not isinstance(multi_index, tuple):
            raise TypeError("индекс должен быть int, tuple или list")

        if len(multi_index) != self.ndim:
            raise ValueError(
                f"Неверная длина индекса: {len(multi_index)}, ожидалось {self.ndim}"
            )

        for axis, index in enumerate(multi_index):
            if not isinstance(index, int):
                raise TypeError("все индексы должны быть целыми числами")
            if index < 0 or index >= self.shape[axis]:
                raise IndexError(
                    f"Индекс {index} выходит за границы оси {axis} "
                    f"размера {self.shape[axis]}"
                )

        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | list[int] | int) -> float:
        index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(index, self.strides)
        return self.data[flat_index]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | list[int] | int,
        value: float
    ) -> None:
        index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(index, self.strides)
        self.data[flat_index] = float(value)

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        checked_shape = validate_shape(new_shape)
        if compute_size(checked_shape) != self.size:
            raise ValueError(
                f"Нельзя reshape {self.shape} в {checked_shape}: "
                "число элементов не совпадает"
            )

        return DenseTensor(checked_shape, data=self.data[:])

    def unfolding(self, mode: int) -> DenseTensor:
        if not isinstance(mode, int):
            raise TypeError("mode должен быть целым числом")
        if mode < 0 or mode >= self.ndim:
            raise ValueError(f"mode должен быть от 0 до {self.ndim - 1}")

        row_count = self.shape[mode]
        col_shape = self.shape[:mode] + self.shape[mode + 1:]
        col_count = compute_size(col_shape) if col_shape else 1
        col_strides = compute_strides(col_shape) if col_shape else ()

        result = DenseTensor.zeros((row_count, col_count))

        for flat_index, value in enumerate(self.data):
            multi_index = flat_to_multi_index(flat_index, self.shape)
            row = multi_index[mode]
            col_multi = multi_index[:mode] + multi_index[mode + 1:]
            col = multi_index_to_flat(col_multi, col_strides) if col_shape else 0
            result[row, col] = value

        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        if not isinstance(k, int):
            raise TypeError("k должен быть целым числом")
        if k < 0 or k >= self.ndim - 1:
            raise ValueError(f"k должен быть от 0 до {self.ndim - 2}")

        rows = compute_size(self.shape[:k + 1])
        cols = compute_size(self.shape[k + 1:])
        return self.reshape((rows, cols))

    def copy(self) -> DenseTensor:
        return DenseTensor(self.shape, data=self.data[:])

    def norm(self) -> float:
        return math.sqrt(sum(value * value for value in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(
            self.shape,
            data=[a + b for a, b in zip(self.data, other.data)]
        )

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(
            self.shape,
            data=[a - b for a, b in zip(self.data, other.data)]
        )

    def __mul__(self, scalar: float | int) -> DenseTensor:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return DenseTensor(self.shape, data=[scalar * value for value in self.data])

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        return self * -1.0

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        if self.shape != other.shape:
            return False

        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False

        return True

    def to_nested_list(self) -> list:
        def build(offset: int, shape: tuple[int, ...]) -> list | float:
            if len(shape) == 1:
                return [self.data[offset + i] for i in range(shape[0])]

            step = compute_size(shape[1:])
            return [
                build(offset + i * step, shape[1:])
                for i in range(shape[0])
            ]

        return build(0, self.shape)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"DenseTensor(shape={self.shape}, data={self.data})"

    def __str__(self) -> str:
        return self.__repr__()