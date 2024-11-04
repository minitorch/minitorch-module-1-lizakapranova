from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Generator

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    moved_vals = list(vals)
    moved_vals[arg] += epsilon
    return (f(*moved_vals) - f(*vals)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    def check_all_parents_visited(var: Variable, visited_vars: set[int]) -> bool:
        for parent in var.parents:
            if parent.unique_id not in visited_vars:
                return False
        return True

    def _top_sort(root_var: Variable) -> Generator[Variable, None, None]:
        visited_vars: set[int] = set()
        stack: list[Variable] = [root_var]

        while len(stack) > 0:
            cur_var = stack[-1]
            if cur_var.unique_id in visited_vars or cur_var.is_constant():
                stack.pop()
                continue
            if cur_var.is_leaf() or check_all_parents_visited(cur_var, visited_vars):
                stack.pop()
                visited_vars.add(cur_var.unique_id)
                yield cur_var
            else:
                stack.extend([parent for parent in cur_var.parents if parent.unique_id not in visited_vars])
    return list(_top_sort(variable))[::-1]



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_vars = topological_sort(variable)
    outputs: dict[int, Any] = {variable.unique_id: deriv}

    for var in sorted_vars:
        cur_deriv = outputs[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(cur_deriv)
        else:
            for parent, d_output in var.chain_rule(cur_deriv):
                outputs[parent.unique_id] = outputs.get(parent.unique_id, 0) + d_output


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
