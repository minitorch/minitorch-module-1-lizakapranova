"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


def mul(num1: float, num2: float) -> float:
    """Multiply two numbers"""
    return num1 * num2


def id(smth: float) -> float:
    """Return itself"""
    return smth


def add(num1: float, num2: float) -> float:
    """Add two numbers"""
    return num1 + num2


def neg(num: float) -> float:
    """Return negative number"""
    return float(-num)


def lt(num1: float, num2: float) -> bool:
    """Check whether first is less than second"""
    return num1 < num2


def eq(num1: float, num2: float) -> bool:
    """Check whether numbers are equal"""
    return num1 == num2


def max(num1: float, num2: float) -> float:
    """Return max number"""
    return num1 if num1 > num2 else num2


def is_close(num1: float, num2: float) -> bool:
    """Check whether numbers are close enough"""
    return abs(num1 - num2) < 1e-2


def sigmoid(num: float) -> float:
    """Calculate sigmoid function"""
    return 1.0 / (1.0 + exp(-num)) if num >= 0 else exp(num) / (1.0 + exp(num))


def relu(num: float) -> float:
    """Calculate ReLu function"""
    return max(0.0, num)


def log(num: float) -> float:
    """Calculate log function"""
    return math.log(num)


def exp(num: float) -> float:
    """Calculate exponent function"""
    return math.exp(num)


def inv(num: float) -> float:
    """Inverse number (1 / num)"""
    return 1.0 / num


def log_back(num1: float, num2: float) -> float:
    """Calculate derivative log function times second argument"""
    return (1.0 / num1) * num2


def inv_back(num1: float, num2: float) -> float:
    """Calculate derivative inverse times second argument"""
    return (-1.0 / (num1 * num1)) * num2


def relu_back(num1: float, num2: float) -> float:
    """Calculate derivative ReLu function times second argument"""
    return num2 if num1 > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable[[float], float], items: Iterable[float]) -> Iterable[float]:
    """Apply function to every item"""
    return [func(item) for item in items]


def zipWith(
    func: Callable[[float, float], float],
    items1: Iterable[float],
    items2: Iterable[float],
) -> Iterable[float]:
    """Zip two iterables using function"""
    return [func(item1, item2) for item1, item2 in zip(items1, items2)]


def reduce(func: Callable[[float, float], float], items: Iterable[float]) -> float:
    """Accumulate items using function"""
    result = None
    for item in items:
        if result is None:
            result = item
        else:
            result = func(result, item)
    return result or 0


def negList(items: Iterable[float]) -> Iterable[float]:
    """Make list negative"""
    return map(neg, items)


def addLists(items1: Iterable[float], items2: Iterable[float]) -> Iterable[float]:
    """Add elements of two iterables"""
    return zipWith(add, items1, items2)


def sum(items: Iterable[float]) -> float:
    """Sum all elements in iterable"""
    return reduce(add, items)


def prod(items: Iterable[float]) -> float:
    """Multiply all elements in iterable"""
    return reduce(mul, items)
