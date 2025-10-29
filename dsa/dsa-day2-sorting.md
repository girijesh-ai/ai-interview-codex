# Day 2: Sorting Algorithms - Complete Tutorial

## Today's Goals
- Understand different sorting algorithms and when to use each
- Master merge sort (most important for interviews!)
- Learn quick sort concepts
- Solve sorting-based problems
- Understand time/space complexity trade-offs

**Time:** 2-3 hours
**Difficulty:** Medium

---

## Why Learn Sorting?

Sorting is fundamental because:
1. **Many problems require sorted data** (binary search, finding duplicates, etc.)
2. **Sorting algorithms teach recursion and divide-and-conquer**
3. **Frequently asked in interviews** - especially merge sort
4. **Understanding trade-offs** - time vs space complexity

---

## Concept 1: Sorting Basics

### What is Sorting?
Arranging elements in a specific order (ascending or descending).

```python
Unsorted: [5, 2, 8, 1, 9]
Sorted:   [1, 2, 5, 8, 9]
```

### Key Questions for Any Sorting Algorithm
1. **Time Complexity** - How fast is it?
2. **Space Complexity** - How much extra memory?
3. **Stability** - Does it preserve relative order of equal elements?
4. **In-place** - Does it need extra space?

### Sorting Algorithms Overview

| Algorithm | Best | Average | Worst | Space | Stable | Notes |
|-----------|------|---------|-------|-------|--------|-------|
| **Bubble Sort** | O(n) | O(n²) | O(n²) | O(1) | Yes | Simple, rarely used |
| **Selection Sort** | O(n²) | O(n²) | O(n²) | O(1) | No | Simple, rarely used |
| **Insertion Sort** | O(n) | O(n²) | O(n²) | O(1) | Yes | Good for small/nearly sorted |
| **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | **Most important!** |
| **Quick Sort** | O(n log n) | O(n log n) | O(n²) | O(log n) | No | Fast in practice |
| **Python's sort()** | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | Timsort (hybrid) |

---

## Problem 1: Bubble Sort Implementation (Easy)

### Concept: Bubble Sort

**How it works:**
- Compare adjacent elements
- Swap if they're in wrong order
- Repeat until no swaps needed

**Visualization:**
```
Pass 1: [5, 2, 8, 1, 9]
        [2, 5, 8, 1, 9]  (swap 5,2)
        [2, 5, 8, 1, 9]  (no swap)
        [2, 5, 1, 8, 9]  (swap 8,1)
        [2, 5, 1, 8, 9]  (no swap)

Pass 2: [2, 5, 1, 8, 9]
        [2, 5, 1, 8, 9]  (no swap)
        [2, 1, 5, 8, 9]  (swap 5,1)
        [2, 1, 5, 8, 9]  (no swap)

Pass 3: [1, 2, 5, 8, 9]  (sorted!)
```

**Why "Bubble"?**
Large elements "bubble up" to the end like bubbles in water.

### Solution

```python
def bubble_sort(arr):
    """
    Bubble sort implementation

    Time Complexity:
    - Best: O(n) when already sorted
    - Average/Worst: O(n²)
    Space Complexity: O(1) - in-place sorting
    """
    n = len(arr)

    # Outer loop - number of passes
    for i in range(n):
        # Flag to detect if any swap happened
        swapped = False

        # Inner loop - compare adjacent elements
        # -i because last i elements are already sorted
        for j in range(n - i - 1):
            # If current > next, swap them
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # If no swaps, array is sorted
        if not swapped:
            break

    return arr

# Test
arr = [5, 2, 8, 1, 9]
print(bubble_sort(arr))  # [1, 2, 5, 8, 9]
```

### When to Use Bubble Sort
- **Almost never in practice!**
- Good for: Teaching, very small arrays (< 10 elements)
- Not good for: Anything else

---

## Problem 2: Merge Sort Implementation (IMPORTANT!)

### Concept: Merge Sort (Divide and Conquer)

**Strategy:**
1. **Divide:** Split array into two halves
2. **Conquer:** Recursively sort each half
3. **Combine:** Merge the sorted halves

**Visualization:**
```
Original: [5, 2, 8, 1, 9, 3]

DIVIDE:
[5, 2, 8, 1, 9, 3]
    /          \
[5, 2, 8]    [1, 9, 3]
  /    \       /    \
[5]  [2,8]   [1]  [9,3]
       |           |
     [2] [8]     [9] [3]

CONQUER (Merge):
[2] [8] → [2, 8]
[9] [3] → [3, 9]

[5] [2, 8] → [2, 5, 8]
[1] [3, 9] → [1, 3, 9]

[2, 5, 8] [1, 3, 9] → [1, 2, 3, 5, 8, 9]
```

**Key Insight:**
- Merging two sorted arrays is easy (O(n))
- Recursively sort smaller pieces
- Always O(n log n) - guaranteed!

### How to Think About Merge Sort

**The Two Key Functions:**
1. `merge_sort(arr)` - Splits array and recursively sorts
2. `merge(left, right)` - Merges two sorted arrays

**Merge Function Logic:**
```
left = [2, 5, 8]
right = [1, 3, 9]

Compare left[0]=2 vs right[0]=1 → 1 is smaller → result=[1]
Compare left[0]=2 vs right[1]=3 → 2 is smaller → result=[1,2]
Compare left[1]=5 vs right[1]=3 → 3 is smaller → result=[1,2,3]
Compare left[1]=5 vs right[2]=9 → 5 is smaller → result=[1,2,3,5]
Compare left[2]=8 vs right[2]=9 → 8 is smaller → result=[1,2,3,5,8]
Only right[2]=9 left → result=[1,2,3,5,8,9]
```

### Solution

```python
def merge_sort(arr):
    """
    Merge sort using divide and conquer

    Time Complexity: O(n log n) - always!
    Space Complexity: O(n) - for temporary arrays
    """
    # Base case: array of 0 or 1 element is already sorted
    if len(arr) <= 1:
        return arr

    # Divide: Find middle and split
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    # Conquer: Recursively sort both halves
    left_sorted = merge_sort(left)
    right_sorted = merge_sort(right)

    # Combine: Merge the sorted halves
    return merge(left_sorted, right_sorted)


def merge(left, right):
    """
    Merge two sorted arrays into one sorted array

    Time Complexity: O(n) where n = len(left) + len(right)
    Space Complexity: O(n) for result array
    """
    result = []
    i, j = 0, 0

    # Compare elements from left and right, add smaller to result
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Add remaining elements from left (if any)
    while i < len(left):
        result.append(left[i])
        i += 1

    # Add remaining elements from right (if any)
    while j < len(right):
        result.append(right[j])
        j += 1

    return result


# Test
arr = [5, 2, 8, 1, 9, 3]
print(merge_sort(arr))  # [1, 2, 3, 5, 8, 9]

# Edge cases
print(merge_sort([]))        # []
print(merge_sort([1]))       # [1]
print(merge_sort([2, 1]))    # [1, 2]
```

### Step-by-Step Trace

Let's trace `merge_sort([5, 2, 8, 1])`:

```
Call 1: merge_sort([5, 2, 8, 1])
  mid = 2
  left = [5, 2]
  right = [8, 1]

  Call 2: merge_sort([5, 2])
    mid = 1
    left = [5]
    right = [2]

    Call 3: merge_sort([5]) → returns [5] (base case)
    Call 4: merge_sort([2]) → returns [2] (base case)

    merge([5], [2]) → [2, 5]
  Returns: [2, 5]

  Call 5: merge_sort([8, 1])
    mid = 1
    left = [8]
    right = [1]

    Call 6: merge_sort([8]) → returns [8] (base case)
    Call 7: merge_sort([1]) → returns [1] (base case)

    merge([8], [1]) → [1, 8]
  Returns: [1, 8]

  merge([2, 5], [1, 8]) → [1, 2, 5, 8]
Returns: [1, 2, 5, 8]
```

### Why Merge Sort is Important

1. **Guaranteed O(n log n)** - No worst case like quick sort
2. **Stable** - Preserves order of equal elements
3. **Predictable** - Always same performance
4. **Used in practice** - Part of Timsort (Python's default)
5. **Most asked sorting algorithm in interviews!**

### Interview Communication

> "I'll use merge sort which is O(n log n). It's a divide-and-conquer algorithm.
> I'll recursively split the array in half until I have single elements, then
> merge them back together in sorted order. The merge operation takes O(n) time,
> and we have log n levels of recursion, giving us O(n log n) total."

---

## Problem 3: Quick Sort Concept (Medium)

### Concept: Quick Sort

**Strategy:**
1. **Pick a pivot** element
2. **Partition:** Move smaller elements left, larger right
3. **Recursively** sort left and right partitions

**Visualization:**
```
Array: [5, 2, 8, 1, 9, 3]
Pick pivot: 3 (last element)

Partition:
[2, 1, 3, 5, 8, 9]
        ↑
     pivot at correct position

Recursively sort:
[2, 1] and [5, 8, 9]

Final: [1, 2, 3, 5, 8, 9]
```

### Solution

```python
def quick_sort(arr, low=0, high=None):
    """
    Quick sort implementation

    Time Complexity:
    - Best/Average: O(n log n)
    - Worst: O(n²) when array is already sorted
    Space Complexity: O(log n) for recursion stack
    """
    if high is None:
        high = len(arr) - 1

    if low < high:
        # Partition and get pivot index
        pivot_index = partition(arr, low, high)

        # Recursively sort left and right of pivot
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)

    return arr


def partition(arr, low, high):
    """
    Partition array around pivot (last element)
    Returns final pivot position
    """
    pivot = arr[high]  # Choose last element as pivot
    i = low - 1  # Index of smaller element

    # Move all elements smaller than pivot to left
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # Place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    return i + 1


# Test
arr = [5, 2, 8, 1, 9, 3]
print(quick_sort(arr.copy()))  # [1, 2, 3, 5, 8, 9]
```

### Merge Sort vs Quick Sort

| Feature | Merge Sort | Quick Sort |
|---------|------------|------------|
| **Time (Best)** | O(n log n) | O(n log n) |
| **Time (Worst)** | O(n log n) | O(n²) |
| **Space** | O(n) | O(log n) |
| **Stable** | Yes | No |
| **In-place** | No | Yes |
| **Use when** | Guaranteed performance | Fast average case |

**Interview Tip:** If not specified, use Merge Sort - it's safer!

---

## Problem 4: Merge Two Sorted Arrays (Easy-Medium)

### Problem Statement
Given two sorted arrays, merge them into one sorted array.

```
Input: arr1 = [1, 3, 5], arr2 = [2, 4, 6]
Output: [1, 2, 3, 4, 5, 6]
```

### How to Think About This

This is the **merge** function from merge sort! We already learned it.

**Two-pointer approach:**
- Pointer i for arr1
- Pointer j for arr2
- Compare and add smaller element

### Solution

```python
def merge_sorted_arrays(arr1, arr2):
    """
    Merge two sorted arrays

    Time Complexity: O(n + m) where n, m are array lengths
    Space Complexity: O(n + m) for result
    """
    result = []
    i, j = 0, 0

    # Merge while both arrays have elements
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # Add remaining from arr1
    result.extend(arr1[i:])

    # Add remaining from arr2
    result.extend(arr2[j:])

    return result

# Test
print(merge_sorted_arrays([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]
print(merge_sorted_arrays([1, 2, 3], [4, 5, 6]))  # [1, 2, 3, 4, 5, 6]
print(merge_sorted_arrays([], [1, 2]))            # [1, 2]
```

---

## Problem 5: Sort Colors (Dutch National Flag) - Medium

### Problem Statement
Given an array with only 0s, 1s, and 2s, sort it in-place.

```
Input: [2, 0, 2, 1, 1, 0]
Output: [0, 0, 1, 1, 2, 2]
```

**Constraint:** Do it in O(n) time and O(1) space (in-place).

### How to Think About This

**Naive approach:** Use any sorting → O(n log n)

**Better approach:** We only have 3 values (0, 1, 2)!
- Count how many 0s, 1s, 2s
- Rebuild array

**Best approach: Dutch National Flag (Three Pointers)**
- `low` pointer: boundary of 0s
- `mid` pointer: current element
- `high` pointer: boundary of 2s

**Strategy:**
- If arr[mid] == 0: swap with low, move both forward
- If arr[mid] == 1: just move mid forward
- If arr[mid] == 2: swap with high, move high backward

### Solution

```python
def sort_colors(arr):
    """
    Sort array of 0s, 1s, 2s using Dutch National Flag algorithm

    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - in-place
    """
    low = 0       # Boundary of 0s
    mid = 0       # Current element
    high = len(arr) - 1  # Boundary of 2s

    while mid <= high:
        if arr[mid] == 0:
            # Swap with low, move both forward
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            # 1 is in correct position, move forward
            mid += 1
        else:  # arr[mid] == 2
            # Swap with high, move high backward
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
            # Don't move mid, need to check swapped element

    return arr

# Test
print(sort_colors([2, 0, 2, 1, 1, 0]))  # [0, 0, 1, 1, 2, 2]
print(sort_colors([2, 0, 1]))           # [0, 1, 2]
```

### Visualization

```
[2, 0, 2, 1, 1, 0]
 ↑           ↑
low,mid    high

arr[mid]=2, swap with high:
[0, 0, 2, 1, 1, 2]
 ↑        ↑
low,mid  high

arr[mid]=0, swap with low:
[0, 0, 2, 1, 1, 2]
    ↑     ↑
   low   mid,high

Continue...
[0, 0, 1, 1, 2, 2]
```

---

## Day 2 Summary

### Algorithms Learned
1. **Bubble Sort** - O(n²), simple but inefficient
2. **Merge Sort** - O(n log n), most important!
3. **Quick Sort** - O(n log n) average, O(n²) worst
4. **Merge Sorted Arrays** - O(n), two-pointer technique
5. **Dutch National Flag** - O(n), three-pointer technique

### Key Takeaways
- Merge sort is most important for interviews
- Understanding how merge works is crucial
- Sometimes custom sorting logic beats standard algorithms
- Consider time/space trade-offs

### Patterns You Learned
- **Divide and Conquer** (Merge Sort, Quick Sort)
- **Two Pointers** (Merging arrays)
- **Three Pointers** (Dutch National Flag)

### Time Complexity Summary
- O(n²): Bubble, Selection, Insertion Sort
- O(n log n): Merge Sort, Quick Sort (average)
- O(n): Merge sorted arrays, Dutch National Flag

### Practice Checklist
- [ ] Implemented merge sort from scratch
- [ ] Understand merge function thoroughly
- [ ] Can explain why merge sort is O(n log n)
- [ ] Solved Dutch National Flag problem
- [ ] Know when to use which sorting algorithm

---

## Tomorrow: Day 3 - Two Pointers & Sliding Window

We'll learn powerful array traversal patterns:
- Two pointers from both ends
- Two pointers same direction
- Sliding window for subarrays
- Fast and slow pointers

Great work on Day 2! Sorting is fundamental! 🎯
