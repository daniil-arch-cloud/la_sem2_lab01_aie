# core/utils.py

"""Вспомогательные функции для работы с тензорами."""


def validate_shape(
    shape: tuple[int, ...] | list[int]
) -> tuple[int, ...]:
    """
    Проверяет корректность формы тензора и приводит её к стандартному виду.
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape должен быть tuple или list")

    if len(shape) == 0:
        raise ValueError("shape не должен быть пустым")

    result: list[int] = []
    for value in shape:
        if not isinstance(value, int):
            raise ValueError("все размеры shape должны быть целыми числами")
        if value <= 0:
            raise ValueError("все размеры shape должны быть положительными")
        result.append(value)

    return tuple(result)


def compute_size(shape: tuple[int, ...]) -> int:
    """
    Возвращает общее число элементов тензора заданной формы.
    """
    size = 1
    for dim in shape:
        size *= dim
    return size


def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Возвращает row-major / C-order strides.
    """
    strides: list[int] = [1] * len(shape)
    current = 1

    for idx in range(len(shape) - 1, -1, -1):
        strides[idx] = current
        current *= shape[idx]

    return tuple(strides)


def multi_index_to_flat(
    multi_index: tuple[int, ...],
    strides: tuple[int, ...]
) -> int:
    """
    Возвращает позицию элемента в плоском списке данных по мультииндексу.
    """
    if len(multi_index) != len(strides):
        raise ValueError("длина multi_index должна совпадать с длиной strides")

    flat_index = 0
    for index, stride in zip(multi_index, strides):
        if not isinstance(index, int):
            raise TypeError("индексы должны быть целыми числами")
        if index < 0:
            raise IndexError("индекс не может быть отрицательным")
        flat_index += index * stride

    return flat_index


def flat_to_multi_index(
    flat_index: int,
    shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Возвращает мультииндекс на основе плоского индекса.
    """
    if not isinstance(flat_index, int):
        raise TypeError("flat_index должен быть целым числом")

    size = compute_size(shape)
    if flat_index < 0 or flat_index >= size:
        raise IndexError("flat_index выходит за границы тензора")

    strides = compute_strides(shape)
    result: list[int] = []
    remainder = flat_index

    for dim, stride in zip(shape, strides):
        value = remainder // stride
        if value >= dim:
            raise IndexError("flat_index выходит за границы тензора")
        result.append(value)
        remainder %= stride

    return tuple(result)


def check_shapes_match(
    shape1: tuple[int, ...],
    shape2: tuple[int, ...]
) -> None:
    """
    Проверяет совпадение форм двух тензоров.
    """
    if tuple(shape1) != tuple(shape2):
        raise ValueError(f"Формы тензоров не совпадают: {shape1} и {shape2}")