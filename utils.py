"""
Utility functions for functional composition and higher-order operations
"""

import functools
import time
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Callable, List, TypeVar, Generator

T = TypeVar('T')
R = TypeVar('R')


def compose(*functions):
    """Compose functions right-to-left"""
    def composed(x):
        for f in reversed(functions):
            x = f(x)
        return x
    return composed


def pipe(*functions):
    """Compose functions left-to-right (Unix pipe style)"""
    def piped(x):
        for f in functions:
            x = f(x)
        return x
    return piped


def partial(func: Callable, *args, **kwargs) -> Callable:
    """Create partial function application"""
    return functools.partial(func, *args, **kwargs)


def map_parallel(func: Callable[[T], R], items: List[T], max_workers: int = 4) -> List[R]:
    """Map function over items in parallel"""
    if len(items) == 1:
        return [func(items[0])]
    
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        return [f.result() for f in as_completed(futures)]


def with_logging(logger: Callable[[str], None]):
    """Decorator to add logging to any function"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger(f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger(f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


def with_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator to add retry logic with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if i == max_retries - 1:
                        raise
                    time.sleep(delay * (2 ** i))  # Exponential backoff
            raise last_exception
        return wrapper
    return decorator


@contextmanager
def timeout_context(seconds: int):
    """Context manager for function timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler and a timeout alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def with_timeout(seconds: int):
    """Decorator to add timeout to function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timeout_context(seconds):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def memoize(func: Callable) -> Callable:
    """Simple memoization decorator"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from args and kwargs
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper


def curry(func: Callable, arity: int = None) -> Callable:
    """Convert a function to curried form"""
    if arity is None:
        arity = func.__code__.co_argcount
    
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= arity:
            return func(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(*(args + more_args), **{**kwargs, **more_kwargs})
    
    return curried


def flatten(nested_list: List[List[T]]) -> List[T]:
    """Flatten a nested list"""
    return [item for sublist in nested_list for item in sublist]


def safe_get(dictionary: dict, key: str, default: Any = None) -> Any:
    """Safe dictionary access"""
    try:
        return dictionary[key]
    except (KeyError, TypeError):
        return default


def substitute_variables(data: Any, variables: dict) -> Any:
    """Recursively substitute variables in data structures"""
    if isinstance(data, dict):
        return {k: substitute_variables(v, variables) for k, v in data.items()}
    elif isinstance(data, list):
        return [substitute_variables(item, variables) for item in data]
    elif isinstance(data, str) and data in variables:
        return variables[data]
    else:
        return data


def merge_dicts(*dicts: dict) -> dict:
    """Merge multiple dictionaries, with later ones taking precedence"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def chunks(lst: List[T], n: int) -> Generator[List[T], None, None]:
    """Yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def take(n: int, iterable) -> List[T]:
    """Take first n items from an iterable"""
    return list(iterable)[:n]


def identity(x: T) -> T:
    """Identity function - returns input unchanged"""
    return x


def const(value: T) -> Callable[..., T]:
    """Create a constant function that always returns the same value"""
    def constant_func(*args, **kwargs):
        return value
    return constant_func


def tap(func: Callable[[T], Any]) -> Callable[[T], T]:
    """Tap function - calls func with input but returns input unchanged (useful for side effects)"""
    def tapper(x: T) -> T:
        func(x)
        return x
    return tapper


def trace(label: str = ""):
    """Decorator to trace function calls (useful for debugging pipelines)"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[TRACE{' ' + label if label else ''}] Calling {func.__name__}")
            result = func(*args, **kwargs)
            print(f"[TRACE{' ' + label if label else ''}] {func.__name__} returned: {type(result)}")
            return result
        return wrapper
    return decorator
