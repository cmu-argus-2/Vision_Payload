from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

def unpack_and_call(func: Callable[P, R], args: P.args) -> R:
    """
    Unpacks the arguments, calls the provided function with those arguments, and returns the result.

    Args:
        func: The function to call.
        args: An iterable of positional arguments to unpack.

    Returns:
        The result of calling the function with the unpacked arguments.
    """
    return func(*args)
