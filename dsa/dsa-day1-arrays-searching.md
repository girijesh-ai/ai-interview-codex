# Day 1: Arrays and Searching - Complete Tutorial

## Today's Goals
- Understand array fundamentals and indexing
- Master linear search technique
- Learn binary search pattern (very important!)
- Practice interview communication

**Time:** 2-3 hours
**Difficulty:** Easy to Medium

---

## Concept 1: Arrays - The Foundation

### What is an Array?
An array is a collection of elements stored in contiguous memory locations. Think of it like a row of boxes numbered 0, 1, 2, 3...

```
Index:  0    1    2    3    4
Array: [10] [20] [30] [40] [50]
```

### Key Properties
- **Fixed size** (in most languages)
- **O(1) access time** - Can access any element directly by index
- **O(n) search time** - Need to check each element (without sorting)
- **Contiguous memory** - Elements are stored next to each other

### Common Array Operations
```python
# Creation
arr = [1, 2, 3, 4, 5]

# Access - O(1)
first = arr[0]      # 1
last = arr[-1]      # 5 (Python allows negative indexing)

# Length - O(1)
length = len(arr)   # 5

# Iteration - O(n)
for element in arr:
    print(element)

# Slicing - O(k) where k is slice size
sub_array = arr[1:4]  # [2, 3, 4]
```

---

## Concept 2: Linear Search

### What is Linear Search?
Check each element one by one until you find what you're looking for.

### When to Use
- Unsorted arrays
- Small arrays
- When you need to find ALL occurrences

### Time Complexity
- **Best case:** O(1) - Found at first position
- **Worst case:** O(n) - Found at last position or not found
- **Average case:** O(n/2) → O(n)

### Space Complexity
- O(1) - No extra space needed

---

## Problem 1: Find Element in Array (Easy)

### Problem Statement
Given an array of integers and a target value, return the index of the target if it exists, otherwise return -1.

**Example:**
```
Input: arr = [4, 2, 7, 1, 9], target = 7
Output: 2

Input: arr = [4, 2, 7, 1, 9], target = 5
Output: -1
```

### How to Think About This

**Step 1: Understand the problem**
- We need to find if target exists in array
- Return its position (index)
- Return -1 if not found

**Step 2: Think of approach**
- We have an unsorted array
- No better option than checking each element
- Use linear search

**Step 3: Consider edge cases**
- Empty array → return -1
- Target at first position → return 0
- Target at last position → return len(arr)-1
- Target not in array → return -1

**Step 4: Plan the algorithm**
```
1. Loop through array with index
2. If current element == target, return index
3. If loop ends, return -1
```

### Your Turn to Try!
Spend 10 minutes trying to solve this yourself before looking at the solution.

```python
def linear_search(arr, target):
    # Your code here
    pass

# Test
arr = [4, 2, 7, 1, 9]
print(linear_search(arr, 7))  # Should print 2
print(linear_search(arr, 5))  # Should print -1
```

### Solution with Explanation

```python
def linear_search(arr, target):
    """
    Search for target in array using linear search

    Time Complexity: O(n) where n is length of array
    Space Complexity: O(1) - only using a counter variable
    """
    # Iterate through array with index
    for i in range(len(arr)):
        # Check if current element matches target
        if arr[i] == target:
            return i  # Found! Return index

    # If we reach here, target not found
    return -1

# Test cases
arr = [4, 2, 7, 1, 9]
print(linear_search(arr, 7))   # Output: 2
print(linear_search(arr, 5))   # Output: -1
print(linear_search([], 5))    # Output: -1 (edge case: empty array)
print(linear_search([5], 5))   # Output: 0 (edge case: single element)
```

### Interview Approach
When solving in interview, say:

> "I'll use linear search here. Since the array is unsorted, I need to check
> each element. I'll iterate through the array and return the index when I
> find the target. If the loop completes without finding it, I'll return -1.
> The time complexity is O(n) and space complexity is O(1)."

---

## Concept 3: Binary Search - The Power of Sorted Arrays

### What is Binary Search?
When array is **sorted**, we can eliminate half the search space in each step by checking the middle element.

### The Key Insight
```
If array is sorted and we check the middle:
- If target < middle: target must be in LEFT half
- If target > middle: target must be in RIGHT half
- If target == middle: Found it!
```

### Visualization
```
Array: [1, 3, 5, 7, 9, 11, 13, 15, 17]
Target: 11

Step 1: Check middle (9)
[1, 3, 5, 7, 9, 11, 13, 15, 17]
             ↑
Target (11) > 9, so search RIGHT half

Step 2: Check middle of right half (13)
[11, 13, 15, 17]
     ↑
Target (11) < 13, so search LEFT half

Step 3: Check middle of left half (11)
[11]
 ↑
Target == 11, FOUND!
```

### Time Complexity
- **O(log n)** - We halve the search space each time
- Much faster than O(n) for large arrays

Example:
- Array of 1 million elements
- Linear search: up to 1,000,000 comparisons
- Binary search: up to 20 comparisons! (log₂(1,000,000) ≈ 20)

### Space Complexity
- **O(1)** for iterative approach
- **O(log n)** for recursive approach (due to call stack)

### Requirements
- Array **MUST be sorted**
- If unsorted, need to sort first (O(n log n))

---

## Problem 2: Binary Search Implementation (Medium)

### Problem Statement
Given a **sorted** array of integers and a target value, return the index of the target if it exists, otherwise return -1.

**Example:**
```
Input: arr = [1, 3, 5, 7, 9, 11], target = 7
Output: 3

Input: arr = [1, 3, 5, 7, 9, 11], target = 6
Output: -1
```

### How to Think About This

**The Binary Search Template (MEMORIZE THIS!)**
```
1. Initialize left = 0, right = len(arr) - 1
2. While left <= right:
   a. Calculate mid = (left + right) // 2
   b. If arr[mid] == target: return mid
   c. If arr[mid] < target: search right, left = mid + 1
   d. If arr[mid] > target: search left, right = mid - 1
3. If not found, return -1
```

**Common Mistakes to Avoid:**
- ❌ `mid = (left + right) / 2` → Use `//` for integer division in Python
- ❌ `while left < right` → Should be `left <= right`
- ❌ `left = mid` or `right = mid` → Should be `mid + 1` or `mid - 1`
- ❌ Integer overflow in other languages → `mid = left + (right - left) // 2`

### Your Turn to Try!
Spend 15 minutes implementing binary search before checking the solution.

```python
def binary_search(arr, target):
    # Your code here
    pass

# Test
arr = [1, 3, 5, 7, 9, 11]
print(binary_search(arr, 7))  # Should print 3
print(binary_search(arr, 6))  # Should print -1
```

### Solution with Detailed Explanation

```python
def binary_search(arr, target):
    """
    Binary search on sorted array

    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    # Initialize pointers
    left = 0
    right = len(arr) - 1

    # Continue while there's a valid search space
    while left <= right:
        # Calculate middle index
        # Using (left + right) // 2 can cause overflow in some languages
        # Safer: mid = left + (right - left) // 2
        mid = (left + right) // 2

        # Check if we found the target
        if arr[mid] == target:
            return mid

        # If target is greater, ignore left half
        elif arr[mid] < target:
            left = mid + 1  # Search right half

        # If target is smaller, ignore right half
        else:
            right = mid - 1  # Search left half

    # Target not found
    return -1

# Test cases
arr = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search(arr, 7))    # Output: 3
print(binary_search(arr, 6))    # Output: -1
print(binary_search(arr, 1))    # Output: 0 (first element)
print(binary_search(arr, 15))   # Output: 7 (last element)
print(binary_search([], 5))     # Output: -1 (empty array)
```

### Step-by-Step Walkthrough

Let's trace through `binary_search([1, 3, 5, 7, 9, 11], 7)`:

```
Iteration 1:
left = 0, right = 5
mid = (0 + 5) // 2 = 2
arr[2] = 5
5 < 7, so left = mid + 1 = 3

Iteration 2:
left = 3, right = 5
mid = (3 + 5) // 2 = 4
arr[4] = 9
9 > 7, so right = mid - 1 = 3

Iteration 3:
left = 3, right = 3
mid = (3 + 3) // 2 = 3
arr[3] = 7
7 == 7, FOUND! Return 3
```

### Interview Communication
> "Since the array is sorted, I'll use binary search which is O(log n). I'll
> maintain two pointers, left and right, and repeatedly check the middle element.
> If the middle is less than target, I search the right half by moving left
> pointer. If greater, I search the left half by moving right pointer. This
> continues until I find the target or the pointers cross."

---

## Problem 3: Find First and Last Position (Medium)

### Problem Statement
Given a sorted array with possible duplicates, find the starting and ending position of a given target value. If not found, return [-1, -1].

**Example:**
```
Input: arr = [5, 7, 7, 8, 8, 8, 10], target = 8
Output: [3, 5]

Input: arr = [5, 7, 7, 8, 8, 8, 10], target = 6
Output: [-1, -1]
```

### How to Think About This

**Key Insight:**
This is a variation of binary search! We need to find:
1. **First occurrence** - leftmost position of target
2. **Last occurrence** - rightmost position of target

**Approach:**
- Use binary search TWICE:
  - Once to find first occurrence
  - Once to find last occurrence

**Modified Binary Search for First Occurrence:**
- When we find target, don't return immediately
- Continue searching in LEFT half to find earlier occurrence
- Keep track of the position we found

**Modified Binary Search for Last Occurrence:**
- When we find target, don't return immediately
- Continue searching in RIGHT half to find later occurrence
- Keep track of the position we found

### Your Turn to Try!

```python
def search_range(arr, target):
    # Your code here
    # Hint: You need two separate binary searches
    pass

# Test
arr = [5, 7, 7, 8, 8, 8, 10]
print(search_range(arr, 8))  # Should print [3, 5]
```

### Solution with Explanation

```python
def search_range(arr, target):
    """
    Find first and last position of target in sorted array

    Time Complexity: O(log n) - Two binary searches
    Space Complexity: O(1)
    """
    def find_first(arr, target):
        """Find first (leftmost) occurrence of target"""
        left, right = 0, len(arr) - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2

            if arr[mid] == target:
                result = mid  # Found it, but continue searching left
                right = mid - 1  # Search left half for earlier occurrence
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return result

    def find_last(arr, target):
        """Find last (rightmost) occurrence of target"""
        left, right = 0, len(arr) - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2

            if arr[mid] == target:
                result = mid  # Found it, but continue searching right
                left = mid + 1  # Search right half for later occurrence
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return result

    # Find first and last positions
    first = find_first(arr, target)

    # If first is -1, target doesn't exist
    if first == -1:
        return [-1, -1]

    last = find_last(arr, target)

    return [first, last]

# Test cases
arr = [5, 7, 7, 8, 8, 8, 10]
print(search_range(arr, 8))   # Output: [3, 5]
print(search_range(arr, 7))   # Output: [1, 2]
print(search_range(arr, 6))   # Output: [-1, -1]
print(search_range(arr, 10))  # Output: [6, 6]
```

### Key Pattern: Binary Search Variations

This teaches an important pattern:
> When you need to find BOUNDARIES in a sorted array, use modified binary
> search where you continue searching even after finding the target.

---

## Problem 4: Search in Rotated Sorted Array (Medium-Hard)

### Problem Statement
A sorted array has been rotated at some pivot point. Find a target value.

**Example:**
```
Original: [0, 1, 2, 4, 5, 6, 7]
Rotated:  [4, 5, 6, 7, 0, 1, 2]  (rotated at index 4)

Input: arr = [4, 5, 6, 7, 0, 1, 2], target = 0
Output: 4

Input: arr = [4, 5, 6, 7, 0, 1, 2], target = 3
Output: -1
```

### How to Think About This

**Key Insights:**
1. Array has TWO sorted portions: [4,5,6,7] and [0,1,2]
2. We can still use binary search!
3. At least ONE half is always sorted

**Strategy:**
```
When we pick middle:
- One side is definitely sorted (increasing)
- Check which side is sorted
- Determine if target is in sorted side
- If yes, search that side
- If no, search other side
```

**Visual Example:**
```
[4, 5, 6, 7, 0, 1, 2], target = 0
             ↑
            mid=7

Left side [4,5,6,7] is sorted (4 < 7)
Is target (0) in range [4,7]? No
So search RIGHT side

[0, 1, 2]
 ↑
mid=1

Left side [0] and right side [2] - left is "sorted"
Is target (0) in range [0,0]? Yes!
Found at index 4 (in original array)
```

### Solution with Detailed Explanation

```python
def search_rotated(arr, target):
    """
    Search in rotated sorted array

    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        # Found target
        if arr[mid] == target:
            return mid

        # Determine which side is sorted
        # Left side is sorted
        if arr[left] <= arr[mid]:
            # Check if target is in sorted left side
            if arr[left] <= target < arr[mid]:
                right = mid - 1  # Search left
            else:
                left = mid + 1   # Search right

        # Right side is sorted
        else:
            # Check if target is in sorted right side
            if arr[mid] < target <= arr[right]:
                left = mid + 1   # Search right
            else:
                right = mid - 1  # Search left

    return -1

# Test cases
arr = [4, 5, 6, 7, 0, 1, 2]
print(search_rotated(arr, 0))  # Output: 4
print(search_rotated(arr, 3))  # Output: -1
print(search_rotated(arr, 4))  # Output: 0
```

### Why This Works

The key is recognizing that even though the array is rotated, one half is always sorted:
- If `arr[left] <= arr[mid]`: left half is sorted
- Otherwise: right half is sorted

Once we know which half is sorted, we can check if target falls in that sorted range using normal comparison.

### Interview Tips for This Problem
1. Draw a diagram of the rotated array
2. Explain that one half is always sorted
3. Walk through the logic of determining which half to search
4. This is a HARD problem - it's okay to get hints!

---

## Problem 5: Find Peak Element (Medium)

### Problem Statement
A peak element is an element that is strictly greater than its neighbors. Find any peak element's index.

**Example:**
```
Input: arr = [1, 2, 3, 1]
Output: 2 (element 3 is a peak)

Input: arr = [1, 2, 1, 3, 5, 6, 4]
Output: 5 (element 6 is a peak, though 2 is also a peak)
```

### How to Think About This

**Key Insights:**
1. Array might have multiple peaks - we just need to find ONE
2. If arr[mid] < arr[mid + 1], there MUST be a peak on the right
3. If arr[mid] > arr[mid + 1], there MUST be a peak on the left

**Why?**
- If going uphill (mid < mid+1), peak is ahead
- If going downhill (mid > mid+1), peak is behind
- We're guaranteed to find a peak!

### Solution

```python
def find_peak(arr):
    """
    Find a peak element using binary search

    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:  # Note: < not <=
        mid = (left + right) // 2

        # If going uphill, peak is on right
        if arr[mid] < arr[mid + 1]:
            left = mid + 1
        # If going downhill, peak is on left (or at mid)
        else:
            right = mid

    # left == right, this is a peak
    return left

# Test cases
print(find_peak([1, 2, 3, 1]))           # Output: 2
print(find_peak([1, 2, 1, 3, 5, 6, 4]))  # Output: 1 or 5 (both valid)
print(find_peak([1, 2, 3, 4, 5]))        # Output: 4 (last element)
```

---

## Day 1 Summary

### Patterns You Learned
1. **Linear Search** - O(n) for unsorted arrays
2. **Binary Search** - O(log n) for sorted arrays
3. **Modified Binary Search** - Finding boundaries
4. **Binary Search on Rotated Array** - Identifying sorted portion
5. **Binary Search for Peak** - Using mid point comparison

### Key Takeaways
- Binary search requires sorted array
- Always consider edge cases (empty array, single element)
- Binary search can be modified for many variations
- Practice the template until it becomes second nature

### Time Complexity Quick Reference
- Linear Search: O(n)
- Binary Search: O(log n)
- All Day 1 problems: O(n) or O(log n)

### Interview Communication Tips
1. Always state your approach before coding
2. Mention time and space complexity
3. Discuss edge cases
4. Draw diagrams when helpful
5. Think aloud!

### Practice Checklist
- [ ] Solved all 5 problems
- [ ] Understood binary search template
- [ ] Can explain time complexity
- [ ] Tried problems before looking at solutions
- [ ] Can solve Problem 2 (basic binary search) in < 10 minutes

---

## Tomorrow: Day 2 - Sorting Algorithms

We'll learn:
- Selection Sort, Bubble Sort (O(n²))
- Merge Sort (O(n log n))
- Quick Sort (O(n log n) average)
- When to use which sorting algorithm
- Sorting-based problem solving patterns

Great job completing Day 1! 🎉
