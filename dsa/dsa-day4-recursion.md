# Day 4: Recursion & Backtracking - Complete Tutorial

## Today's Goals
- Understand recursion fundamentals
- Master recursive problem-solving
- Learn backtracking pattern
- Build recursive thinking

**Time:** 2-3 hours
**Difficulty:** Medium-Hard

---

## Concept: What is Recursion?

A function that calls itself to solve smaller versions of the same problem.

**Three Requirements:**
1. **Base Case:** When to stop
2. **Recursive Case:** How to break down problem
3. **Progress:** Each call gets closer to base case

### Simple Example

```python
def factorial(n):
    # Base case
    if n == 0 or n == 1:
        return 1

    # Recursive case
    return n * factorial(n - 1)

# factorial(5) = 5 * factorial(4)
#              = 5 * 4 * factorial(3)
#              = 5 * 4 * 3 * factorial(2)
#              = 5 * 4 * 3 * 2 * factorial(1)
#              = 5 * 4 * 3 * 2 * 1 = 120
```

---

## Problem 1: Fibonacci Number (Easy)

```python
def fibonacci(n):
    """
    Time: O(2^n) - very slow!
    Space: O(n) - recursion depth
    """
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Optimized with memoization
def fib_memo(n, memo={}):
    """
    Time: O(n), Space: O(n)
    """
    if n in memo:
        return memo[n]
    if n <= 1:
        return n

    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

print(fib_memo(10))  # 55
```

---

## Problem 2: Power Function (Medium)

```python
def power(x, n):
    """
    Calculate x^n using recursion

    Time: O(log n), Space: O(log n)
    """
    # Base cases
    if n == 0:
        return 1
    if n < 0:
        return 1 / power(x, -n)

    # If n is even: x^n = (x^(n/2))^2
    if n % 2 == 0:
        half = power(x, n // 2)
        return half * half
    # If n is odd: x^n = x * x^(n-1)
    else:
        return x * power(x, n - 1)

print(power(2, 10))  # 1024
```

---

## Problem 3: Generate Parentheses (Medium)

### Problem
Generate all valid combinations of n pairs of parentheses.

```
Input: n = 3
Output: ["((()))", "(()())", "(())()", "()(())", "()()()"]
```

### Solution

```python
def generate_parentheses(n):
    """
    Backtracking solution

    Time: O(4^n / sqrt(n)) - Catalan number
    Space: O(n) - recursion depth
    """
    result = []

    def backtrack(current, open_count, close_count):
        # Base case: used all parentheses
        if len(current) == 2 * n:
            result.append(current)
            return

        # Add opening parenthesis if we can
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)

        # Add closing parenthesis if valid
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)

    backtrack('', 0, 0)
    return result

print(generate_parentheses(3))
```

---

## Problem 4: Subsets (Medium)

### Problem
Given array of unique integers, return all possible subsets.

```
Input: [1, 2, 3]
Output: [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
```

### Solution

```python
def subsets(nums):
    """
    Backtracking to generate all subsets

    Time: O(2^n), Space: O(n)
    """
    result = []

    def backtrack(start, current):
        # Add current subset to result
        result.append(current[:])

        # Try adding each remaining element
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()  # Backtrack

    backtrack(0, [])
    return result

print(subsets([1, 2, 3]))
```

---

## Problem 5: Permutations (Medium)

```python
def permutations(nums):
    """
    Generate all permutations

    Time: O(n!), Space: O(n)
    """
    result = []

    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return

        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()

    backtrack([], nums)
    return result

print(permutations([1, 2, 3]))
```

---

## Problem 6: Letter Combinations of Phone Number (Medium)

```python
def letter_combinations(digits):
    """
    Phone keypad letter combinations

    Time: O(4^n), Space: O(n)
    """
    if not digits:
        return []

    phone = {
        '2': 'abc', '3': 'def', '4': 'ghi',
        '5': 'jkl', '6': 'mno', '7': 'pqrs',
        '8': 'tuv', '9': 'wxyz'
    }

    result = []

    def backtrack(index, current):
        if index == len(digits):
            result.append(current)
            return

        for letter in phone[digits[index]]:
            backtrack(index + 1, current + letter)

    backtrack(0, '')
    return result

print(letter_combinations("23"))
# ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
```

---

## Day 4 Summary

### Key Patterns
1. **Simple Recursion:** Factorial, Fibonacci
2. **Divide and Conquer:** Power function
3. **Backtracking:** Generate combinations
4. **Tree Recursion:** Multiple recursive calls

### Backtracking Template
```python
def backtrack(current, ...):
    # Base case
    if condition:
        result.append(current)
        return

    # Try all choices
    for choice in choices:
        # Make choice
        current.append(choice)
        # Recurse
        backtrack(current, ...)
        # Undo choice (backtrack!)
        current.pop()
```

### Time Complexity
- Factorial: O(n)
- Fibonacci (naive): O(2^n)
- Fibonacci (memo): O(n)
- Subsets: O(2^n)
- Permutations: O(n!)

Tomorrow: Hash Maps and Problem Patterns! 📊
