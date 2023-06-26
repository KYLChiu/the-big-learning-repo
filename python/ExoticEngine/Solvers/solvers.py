from inspect import isfunction
from collections.abc import Callable


# Bisection can be slow
# to do: Code up Newton-Raphson method
def bisection(f: Callable,
              target: float,
              lower_bound: float = 1e-8,
              upper_bound: float = 10,
              tolerance: float = 1e-8,
              max_iteration: int = 30):
    """
    Assumes f: R -> R is a monotonically increasing function
    f must only take 1 argument
    default max_iteration = 30 (~1e-9 min tolerance)
    """
    assert isfunction(f)
    assert upper_bound > lower_bound
    assert tolerance >= 0
    assert 2 < max_iteration < 100
    counter = 0
    mid_point = 0.5 * (lower_bound + upper_bound)
    y = f(mid_point)
    assert f(upper_bound) > y > f(lower_bound)
    def termination_condition(x):
        if counter >= max_iteration:
            print(f"WARNING: max number of iterations ({max_iteration}) reached!! \
                    target={target}, current={x}, tolerance={tolerance}")
            return True
        else:
            return abs(x - target) < tolerance

    terminate = termination_condition(y)
    while not terminate:
        if y < target:
            lower_bound = mid_point
        elif y > target:
            upper_bound = mid_point
        mid_point = 0.5 * (lower_bound + upper_bound)
        y = f(mid_point)
        counter += 1
        terminate = termination_condition(y)
    return mid_point
