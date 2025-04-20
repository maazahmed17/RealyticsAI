# List Comprehension
# Example: Create a list of squares of numbers from 0 to 9
squares = [x**2 for x in range(10)]
print("List of squares:", squares)

# Set Comprehension
# Example: Create a set of unique vowels from a string
vowels = {char for char in "comprehensions" if char in "aeiou"}
print("Set of vowels:", vowels)

# Dictionary Comprehension
# Example: Create a dictionary mapping numbers to their squares
squares_dict = {x: x**2 for x in range(5)}
print("Dictionary of squares:", squares_dict)

# Generator Expression
# Example: Create a generator for even numbers from 0 to 9
even_numbers = (x for x in range(10) if x % 2 == 0)
print("Even numbers (from generator):", list(even_numbers))