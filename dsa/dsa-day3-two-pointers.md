# Day 3: Two Pointers & Sliding Window - Complete Tutorial

## Today's Goals
- Master two-pointer technique (opposite ends)
- Learn two-pointer same direction
- Understand sliding window pattern
- Solve subarray problems efficiently

**Time:** 2-3 hours
**Difficulty:** Medium

---

## Concept 1: Two Pointers Pattern

### What is Two Pointers?
Using two pointers to traverse array instead of nested loops.

**Benefits:**
- Reduces O(n²) to O(n)
- Elegant and efficient
- Common in interviews!

### Pattern 1: Opposite Direction
Pointers start at both ends, move toward center.

```python
left = 0
right = len(arr) - 1

while left < right:
    # Process arr[left] and arr[right]
    # Move pointers based on condition
    left += 1  # or
    right -= 1
```

**Use when:** Need to check pairs, reverse, palindrome

---

## Problem 1: Two Sum II - Sorted Array (Easy-Medium)

### Problem
Given **sorted** array, find two numbers that sum to target.

```
Input: arr = [2, 7, 11, 15], target = 9
Output: [0, 1]  (arr[0] + arr[1] = 2 + 7 = 9)
```

### How to Think About This

**Brute Force:** Check all pairs → O(n²)

**Better:** Two pointers on sorted array!
- If sum < target: need bigger number, move left++
- If sum > target: need smaller number, move right--
- If sum == target: found!

### Solution

```python
def two_sum_sorted(arr, target):
    """
    Find two numbers that sum to target in sorted array

    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum

    return [-1, -1]  # Not found

# Test
print(two_sum_sorted([2, 7, 11, 15], 9))  # [0, 1]
```

---

## Problem 2: Remove Duplicates from Sorted Array (Easy)

### Problem
Remove duplicates in-place from sorted array. Return new length.

```
Input: [1, 1, 2, 2, 3]
Output: 3, array becomes [1, 2, 3, _, _]
```

### Pattern: Two Pointers Same Direction

```python
def remove_duplicates(arr):
    """
    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0

    # Slow pointer: position to place next unique element
    slow = 0

    # Fast pointer: scanning array
    for fast in range(1, len(arr)):
        # Found new unique element
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]

    return slow + 1  # Length of unique elements

# Test
arr = [1, 1, 2, 2, 3]
length = remove_duplicates(arr)
print(length, arr[:length])  # 3 [1, 2, 3]
```

---

## Concept 2: Sliding Window Pattern

### What is Sliding Window?
Maintain a "window" that slides through array.

**Use for:** Subarrays, substrings with specific properties

**Two types:**
1. **Fixed size window**
2. **Variable size window**

---

## Problem 3: Maximum Sum Subarray of Size K (Fixed Window)

### Problem
Find maximum sum of any subarray of size k.

```
Input: arr = [2, 1, 5, 1, 3, 2], k = 3
Output: 9  (subarray [5, 1, 3])
```

### Solution

```python
def max_sum_subarray(arr, k):
    """
    Sliding window - fixed size

    Time: O(n), Space: O(1)
    """
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide window
    for i in range(k, len(arr)):
        # Add new element, remove old element
        window_sum = window_sum + arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Test
print(max_sum_subarray([2, 1, 5, 1, 3, 2], 3))  # 9
```

---

## Problem 4: Longest Substring Without Repeating Characters (Variable Window)

### Problem
Find length of longest substring without repeating characters.

```
Input: "abcabcbb"
Output: 3  ("abc")

Input: "bbbbb"
Output: 1  ("b")
```

### Solution

```python
def longest_substring_no_repeat(s):
    """
    Sliding window with hash set

    Time: O(n), Space: O(min(n, charset))
    """
    char_set = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Shrink window while duplicate exists
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        # Add current character
        char_set.add(s[right])

        # Update max length
        max_length = max(max_length, right - left + 1)

    return max_length

# Test
print(longest_substring_no_repeat("abcabcbb"))  # 3
print(longest_substring_no_repeat("bbbbb"))     # 1
```

---

## Problem 5: Container With Most Water (Medium)

### Problem
Given array where each element is height, find two lines that form container with most water.

```
Input: [1,8,6,2,5,4,8,3,7]
Output: 49  (lines at index 1 and 8)
```

### Solution

```python
def max_area(heights):
    """
    Two pointers from both ends

    Time: O(n), Space: O(1)
    """
    left, right = 0, len(heights) - 1
    max_water = 0

    while left < right:
        # Calculate current area
        width = right - left
        height = min(heights[left], heights[right])
        area = width * height

        max_water = max(max_water, area)

        # Move pointer with smaller height
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1

    return max_water

# Test
print(max_area([1,8,6,2,5,4,8,3,7]))  # 49
```

---

## Problem 6: Minimum Window Substring (Hard)

### Problem
Find minimum window in S that contains all characters of T.

```
Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"
```

### Solution

```python
def min_window(s, t):
    """
    Sliding window with hash maps

    Time: O(n + m), Space: O(m)
    """
    from collections import Counter

    if not s or not t:
        return ""

    # Character frequencies in t
    target_count = Counter(t)
    required = len(target_count)

    # Sliding window
    left = 0
    formed = 0  # Unique characters matched
    window_count = {}

    # Result: (window length, left, right)
    result = float('inf'), None, None

    for right in range(len(s)):
        # Add character from right
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1

        # Check if frequency matches
        if char in target_count and window_count[char] == target_count[char]:
            formed += 1

        # Try to shrink window
        while left <= right and formed == required:
            char = s[left]

            # Update result if smaller window
            if right - left + 1 < result[0]:
                result = (right - left + 1, left, right)

            # Remove from left
            window_count[char] -= 1
            if char in target_count and window_count[char] < target_count[char]:
                formed -= 1

            left += 1

    return "" if result[0] == float('inf') else s[result[1]:result[2] + 1]

# Test
print(min_window("ADOBECODEBANC", "ABC"))  # "BANC"
```

---

## Day 3 Summary

### Patterns Learned
1. **Two Pointers - Opposite:** O(n²) → O(n)
2. **Two Pointers - Same Direction:** In-place modifications
3. **Sliding Window - Fixed:** Subarray problems
4. **Sliding Window - Variable:** Dynamic size problems

### Key Takeaways
- Two pointers eliminate nested loops
- Sliding window for contiguous subarray problems
- Hash maps complement sliding window
- These patterns appear in 30%+ of interviews!

### Time Complexities
- All problems: O(n) instead of O(n²)
- Space: Usually O(1) or O(charset)

Tomorrow: Recursion and Backtracking! 🚀
