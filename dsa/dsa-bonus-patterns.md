# Bonus DSA Patterns - Additional Interview Prep

## Overview

This supplements Days 1-6 with additional important patterns commonly asked in interviews.

---

## Pattern 1: Sliding Window - Advanced Variations

### Problem 1: Minimum Size Subarray Sum (Medium)

**Problem:** Find minimum length subarray with sum ≥ target.

```
Input: arr = [2,3,1,2,4,3], target = 7
Output: 2  (subarray [4,3])
```

**Pattern:** Variable size sliding window with shrinking

```python
def min_subarray_len(target, nums):
    """
    Sliding window - find minimum length

    Time: O(n), Space: O(1)
    """
    left = 0
    current_sum = 0
    min_length = float('inf')

    for right in range(len(nums)):
        # Expand window
        current_sum += nums[right]

        # Shrink window while condition met
        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1

    return min_length if min_length != float('inf') else 0

# Test
print(min_subarray_len(7, [2,3,1,2,4,3]))  # 2
```

**Key Pattern:**
```python
# Variable Sliding Window Template
left = 0
for right in range(len(arr)):
    # Add right element
    window.add(arr[right])

    # Shrink while condition violated
    while condition_violated:
        window.remove(arr[left])
        left += 1

    # Update result
    result = update(result, window)
```

---

### Problem 2: Longest Repeating Character Replacement (Medium)

**Problem:** Can replace k characters. Find longest substring with same character.

```
Input: s = "AABABBA", k = 1
Output: 4  (replace one B → "AAAA")
```

```python
def character_replacement(s, k):
    """
    Sliding window with character frequency

    Time: O(n), Space: O(26) = O(1)
    """
    from collections import Counter

    left = 0
    max_freq = 0
    max_length = 0
    char_count = Counter()

    for right in range(len(s)):
        char_count[s[right]] += 1
        max_freq = max(max_freq, char_count[s[right]])

        # Current window size - most frequent char > k
        # means we need more than k replacements
        window_size = right - left + 1
        if window_size - max_freq > k:
            char_count[s[left]] -= 1
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length

# Test
print(character_replacement("AABABBA", 1))  # 4
```

---

### Problem 3: Permutation in String (Medium)

**Problem:** Check if s2 contains permutation of s1.

```
Input: s1 = "ab", s2 = "eidbaooo"
Output: True  (s2 contains "ba" which is permutation of "ab")
```

```python
def check_inclusion(s1, s2):
    """
    Fixed size sliding window + frequency matching

    Time: O(n), Space: O(1)
    """
    from collections import Counter

    if len(s1) > len(s2):
        return False

    s1_count = Counter(s1)
    window_count = Counter()

    # Size of sliding window
    k = len(s1)

    for i in range(len(s2)):
        # Add new character
        window_count[s2[i]] += 1

        # Remove old character if window > k
        if i >= k:
            if window_count[s2[i - k]] == 1:
                del window_count[s2[i - k]]
            else:
                window_count[s2[i - k]] -= 1

        # Check if window matches s1
        if window_count == s1_count:
            return True

    return False

# Test
print(check_inclusion("ab", "eidbaooo"))  # True
```

---

## Pattern 2: Fast & Slow Pointers (Floyd's Cycle Detection)

### Problem 4: Linked List Cycle Detection (Easy)

**Pattern:** Two pointers moving at different speeds

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    """
    Detect cycle using fast and slow pointers

    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next       # Move 1 step
        fast = fast.next.next  # Move 2 steps

        if slow == fast:
            return True  # Cycle detected

    return False
```

**Application to Arrays:** Find duplicate in array (1 to n)

```python
def find_duplicate(nums):
    """
    Find duplicate using cycle detection
    Array with n+1 elements, values 1 to n

    Time: O(n), Space: O(1)
    """
    # Phase 1: Find intersection point
    slow = nums[0]
    fast = nums[0]

    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # Phase 2: Find entrance to cycle (duplicate)
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow

# Test
print(find_duplicate([1,3,4,2,2]))  # 2
```

---

## Pattern 3: Prefix Sum (Cumulative Sum)

### Problem 5: Subarray Sum Equals K (Medium)

**Problem:** Count subarrays with sum = k.

```
Input: nums = [1,1,1], k = 2
Output: 2  (subarrays [1,1] at two positions)
```

```python
def subarray_sum(nums, k):
    """
    Prefix sum + hash map

    Time: O(n), Space: O(n)
    """
    from collections import defaultdict

    # prefix_sum -> count
    prefix_sums = defaultdict(int)
    prefix_sums[0] = 1  # Empty prefix

    current_sum = 0
    count = 0

    for num in nums:
        current_sum += num

        # If (current_sum - k) exists, we found subarray(s)
        # Because: current_sum - previous_sum = k
        if current_sum - k in prefix_sums:
            count += prefix_sums[current_sum - k]

        # Add current sum to map
        prefix_sums[current_sum] += 1

    return count

# Test
print(subarray_sum([1,1,1], 2))  # 2
print(subarray_sum([1,2,3], 3))  # 2
```

**Pattern Explanation:**
```
Array: [1, 2, 3, 4]
Prefix: [1, 3, 6, 10]

To find subarray sum from index i to j:
sum[i:j] = prefix[j] - prefix[i-1]

If we want sum = k:
prefix[j] - prefix[i-1] = k
prefix[i-1] = prefix[j] - k
```

---

### Problem 6: Range Sum Query (Easy)

```python
class NumArray:
    """
    Precompute prefix sums for O(1) range queries

    Initialization: O(n)
    Query: O(1)
    """
    def __init__(self, nums):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)

    def sumRange(self, left, right):
        """Sum from index left to right inclusive"""
        return self.prefix[right + 1] - self.prefix[left]

# Test
obj = NumArray([1, 2, 3, 4, 5])
print(obj.sumRange(0, 2))  # 6 (1+2+3)
print(obj.sumRange(2, 4))  # 12 (3+4+5)
```

---

## Pattern 4: Monotonic Stack

### Problem 7: Next Greater Element (Medium)

**Problem:** For each element, find next greater element to its right.

```
Input: [4, 5, 2, 10, 8]
Output: [5, 10, 10, -1, -1]
```

```python
def next_greater_elements(nums):
    """
    Monotonic stack - decreasing order

    Time: O(n), Space: O(n)
    """
    result = [-1] * len(nums)
    stack = []  # Store indices

    for i in range(len(nums)):
        # While current element is greater than stack top
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]

        stack.append(i)

    return result

# Test
print(next_greater_elements([4, 5, 2, 10, 8]))
# [5, 10, 10, -1, -1]
```

**Pattern:**
```python
# Monotonic Stack Template
stack = []
for i, num in enumerate(arr):
    # Maintain stack property
    while stack and condition(arr[stack[-1]], num):
        idx = stack.pop()
        # Process using idx and current num

    stack.append(i)
```

---

### Problem 8: Daily Temperatures (Medium)

**Problem:** How many days until warmer temperature?

```
Input: [73, 74, 75, 71, 69, 72, 76, 73]
Output: [1, 1, 4, 2, 1, 1, 0, 0]
```

```python
def daily_temperatures(temperatures):
    """
    Monotonic stack - find next warmer day

    Time: O(n), Space: O(n)
    """
    result = [0] * len(temperatures)
    stack = []

    for i, temp in enumerate(temperatures):
        # While current temp is warmer
        while stack and temp > temperatures[stack[-1]]:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx

        stack.append(i)

    return result

# Test
print(daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]))
# [1, 1, 4, 2, 1, 1, 0, 0]
```

---

## Pattern 5: String Patterns

### Problem 9: Longest Palindromic Substring (Medium)

```python
def longest_palindrome(s):
    """
    Expand around center

    Time: O(n²), Space: O(1)
    """
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    if not s:
        return ""

    start = 0
    max_len = 0

    for i in range(len(s)):
        # Odd length palindrome (center is single char)
        len1 = expand_around_center(i, i)
        # Even length palindrome (center is between chars)
        len2 = expand_around_center(i, i + 1)

        current_max = max(len1, len2)
        if current_max > max_len:
            max_len = current_max
            start = i - (current_max - 1) // 2

    return s[start:start + max_len]

# Test
print(longest_palindrome("babad"))  # "bab" or "aba"
```

---

### Problem 10: Valid Palindrome (Easy)

```python
def is_palindrome(s):
    """
    Two pointers from both ends

    Time: O(n), Space: O(1)
    """
    # Clean string: only alphanumeric, lowercase
    cleaned = ''.join(c.lower() for c in s if c.isalnum())

    left, right = 0, len(cleaned) - 1

    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1

    return True

# Test
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
```

---

## Pattern 6: Matrix Traversal

### Problem 11: Spiral Matrix (Medium)

```python
def spiral_order(matrix):
    """
    Traverse matrix in spiral order

    Time: O(m*n), Space: O(1)
    """
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # Traverse down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        # Traverse left (if still rows left)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        # Traverse up (if still columns left)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result

# Test
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(spiral_order(matrix))  # [1,2,3,6,9,8,7,4,5]
```

---

### Problem 12: Rotate Image 90 Degrees (Medium)

```python
def rotate(matrix):
    """
    Rotate matrix 90 degrees clockwise in-place

    Time: O(n²), Space: O(1)
    """
    n = len(matrix)

    # Step 1: Transpose (swap matrix[i][j] with matrix[j][i])
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse each row
    for i in range(n):
        matrix[i].reverse()

    return matrix

# Test
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
rotate(matrix)
print(matrix)
# [[7, 4, 1],
#  [8, 5, 2],
#  [9, 6, 3]]
```

---

## Pattern 7: Bit Manipulation

### Problem 13: Single Number (Easy)

**Problem:** Every element appears twice except one. Find it.

```python
def single_number(nums):
    """
    XOR property: a ^ a = 0, a ^ 0 = a

    Time: O(n), Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result

# Test
print(single_number([4,1,2,1,2]))  # 4
```

---

### Problem 14: Counting Bits (Easy)

```python
def count_bits(n):
    """
    Count 1s in binary representation for 0 to n

    Time: O(n), Space: O(1) excluding output
    """
    result = [0] * (n + 1)

    for i in range(1, n + 1):
        # i >> 1 is i // 2
        # i & 1 checks if last bit is 1
        result[i] = result[i >> 1] + (i & 1)

    return result

# Test
print(count_bits(5))  # [0,1,1,2,1,2]
# 0:0, 1:1, 2:10, 3:11, 4:100, 5:101
```

---

## All Patterns Summary

| Pattern | Use When | Time | Key Technique |
|---------|----------|------|---------------|
| **Sliding Window (Fixed)** | Fixed-size subarray | O(n) | Window sum/properties |
| **Sliding Window (Variable)** | Dynamic subarray/substring | O(n) | Expand/shrink window |
| **Two Pointers** | Sorted array, pairs | O(n) | Left/right pointers |
| **Fast & Slow Pointers** | Cycle detection | O(n) | Different speeds |
| **Prefix Sum** | Range queries | O(1) query | Cumulative sum |
| **Monotonic Stack** | Next greater/smaller | O(n) | Maintain order |
| **Hash Map** | Frequency, pairs | O(n) | O(1) lookup |
| **Recursion/Backtracking** | Combinations | Exponential | Explore + undo |
| **Binary Search** | Sorted search | O(log n) | Divide search space |
| **Sorting** | Order matters | O(n log n) | Merge/Quick sort |
| **Bit Manipulation** | Binary operations | O(1) | XOR, AND, OR |

---

## Pattern Recognition Guide

**When you see:**
- "Subarray/substring" → Sliding Window
- "Sorted array" → Two Pointers or Binary Search
- "Find pair/triplet" → Two Pointers or Hash Map
- "Count frequency" → Hash Map
- "Next greater/smaller" → Monotonic Stack
- "Range sum query" → Prefix Sum
- "Generate combinations" → Backtracking
- "Cycle detection" → Fast & Slow Pointers
- "Matrix traversal" → DFS/BFS or Boundary tracking

---

## Practice Strategy

1. **Identify pattern** before coding
2. **Recall template** for that pattern
3. **Adapt template** to specific problem
4. **Test with examples**
5. **Analyze complexity**

Master these patterns and you can solve 80%+ of interview problems!
