print("Today we are gonna learn about lists, and it is represented by square brackets")

Masjid=["Masjid Al-Hidayah", "Masjid Al-Ikhlas", "Masjid Al-Furqan", "Masjid Al-Badr"]

# Creating a sample list to demonstrate list functions
fruits = ['apple', 'banana', 'orange']
numbers = [5, 2, 8, 1, 9, 3]

print("Original Lists:")
print("Fruits:", fruits)
print("Numbers:", numbers)
print("\n" + "="*50 + "\n")

# 1. Adding Elements
print("1. Adding Elements:")
fruits.append('mango')
print("After append('mango'):", fruits)

fruits.insert(1, 'grape')
print("After insert(1, 'grape'):", fruits)

more_fruits = ['kiwi', 'pear']
fruits.extend(more_fruits)
print("After extend(['kiwi', 'pear']):", fruits)
print("\n" + "="*50 + "\n")

# 2. Removing Elements
print("2. Removing Elements:")
fruits.remove('apple')
print("After remove('apple'):", fruits)

popped_fruit = fruits.pop()
print(f"Popped fruit: {popped_fruit}")
print("After pop():", fruits)

popped_index = fruits.pop(1)
print(f"Popped fruit at index 1: {popped_index}")
print("After pop(1):", fruits)
print("\n" + "="*50 + "\n")

# 3. List Information
print("3. List Information:")
print(f"Length of fruits list: {len(fruits)}")
fruits.append('banana')  # Adding another banana for count demo
print(f"Count of 'banana': {fruits.count('banana')}")
print(f"Index of 'banana': {fruits.index('banana')}")
print("\n" + "="*50 + "\n")

# 4. Ordering
print("4. Ordering:")
print("Original numbers:", numbers)
numbers.sort()
print("After sort():", numbers)
numbers.reverse()
print("After reverse():", numbers)
print("\n" + "="*50 + "\n")

# 5. Other Operations
print("5. Other Operations:")
# Slicing
print("Slicing numbers[1:4]:", numbers[1:4])

# List comprehension
squares = [x**2 for x in numbers]
print("Squares using list comprehension:", squares)

# Copy
fruits_copy = fruits.copy()
print("Copy of fruits list:", fruits_copy)
print("\n" + "="*50 + "\n")

# 6. Checking Elements
print("6. Checking Elements:")
print(f"Is 'banana' in fruits? {'banana' in fruits}")
print(f"Is 'watermelon' in fruits? {'watermelon' in fruits}")
print("\n" + "="*50 + "\n")

# Clear demonstration (at the end since it empties the list)
print("7. Clearing a List:")
fruits_copy.clear()
print("After clear():", fruits_copy)

print("\n=== DIFFERENCES BETWEEN LISTS AND TUPLES ===\n")

# 1. Lists are mutable (can be changed), Tuples are immutable (cannot be changed)
print("1. Mutability Difference:")
my_list = [1, 2, 3]
my_tuple = (1, 2, 3)

# List can be modified
my_list[0] = 10
print("List after modification:", my_list)

# Tuple cannot be modified
print("Tuple - attempting to modify will cause error")
print("Original tuple:", my_tuple)
# my_tuple[0] = 10  # This would raise an error

print("\n" + "="*50 + "\n")

# 2. Syntax Difference
print("2. Syntax Difference:")
empty_list = []      # Square brackets for lists
empty_tuple = ()     # Parentheses for tuples
single_item_tuple = (1,)  # Note the comma for single-item tuple
print("Empty list:", empty_list)
print("Empty tuple:", empty_tuple)
print("Single item tuple:", single_item_tuple)

print("\n" + "="*50 + "\n")

# 3. Memory and Performance
print("3. Memory Usage:")
import sys
list_example = [1, 2, 3, 4, 5]
tuple_example = (1, 2, 3, 4, 5)
print(f"List size: {sys.getsizeof(list_example)} bytes")
print(f"Tuple size: {sys.getsizeof(tuple_example)} bytes")

print("\n" + "="*50 + "\n")

# 4. Methods Available
print("4. Available Methods:")
print("List methods:", dir(list_example)[-5:])  # Showing last 5 methods
print("Tuple methods:", dir(tuple_example)[-5:])  # Showing last 5 methods

print("\n" + "="*50 + "\n")

# 5. Use Cases
print("5. Typical Use Cases:")
# Tuple for coordinates (unchangeable)
point = (3, 4)
print("Coordinate tuple:", point)

# List for collection that needs modification
scores = [85, 92, 78]
scores.append(95)
print("Scores list after append:", scores)





