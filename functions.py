# 1. Basic Function Definition
def greet(name):
    return f"Hello, {name}!"

# Example usage
print(greet("Alice"))  # Output: Hello, Alice!

# 2. Default Parameters
def power(base, exponent=2):
    return base ** exponent

# Example usage
print(power(4))      # Output: 16 (4^2)
print(power(2, 3))   # Output: 8 (2^3)

# 3. *args - Variable Number of Arguments
def sum_all(*numbers):
    return sum(numbers)

# Example usage
print(sum_all(1, 2, 3, 4))  # Output: 10

# 4. **kwargs - Keyword Arguments
def user_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Example usage
user_info(name="John", age=30, city="New York")

# 5. Lambda Functions (Anonymous Functions)
square = lambda x: x * x
print(square(5))  # Output: 25

# 6. Decorators - A More Advanced Concept
def timer(func):
    from time import time
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Function {func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

@timer
def slow_function(n):
    import time
    time.sleep(n)
    return "Done!"

# Example usage
print(slow_function(1))  # Will wait 1 second and show execution time

# 7. Closure - Function Factory
def multiplier(x):
    def multiply(y):
        return x * y
    return multiply

# Example usage
double = multiplier(2)
triple = multiplier(3)
print(double(5))  # Output: 10
print(triple(5))  # Output: 15

# 8. Type Hints (Python 3.5+)
from typing import List, Dict

def process_data(numbers: List[int], config: Dict[str, str]) -> int:
    return sum(numbers)

# Example usage
print(process_data([1, 2, 3], {"mode": "sum"}))  # Output: 6
