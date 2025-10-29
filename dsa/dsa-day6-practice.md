# Day 6: Mixed Practice - Interview Ready!

## Today's Goals
- Apply all learned patterns
- Solve mixed difficulty problems
- Practice interview communication
- Build confidence

**Time:** 2-3 hours
**Difficulty:** Easy to Hard

---

## Problem 1: Move Zeroes (Easy)

**Pattern:** Two Pointers

```python
def move_zeroes(nums):
    """
    Move all zeros to end, maintain order

    Time: O(n), Space: O(1)
    """
    slow = 0  # Position for next non-zero

    # Move non-zeros to front
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1

    return nums

print(move_zeroes([0,1,0,3,12]))  # [1,3,12,0,0]
```

---

## Problem 2: Product of Array Except Self (Medium)

**Pattern:** Array Manipulation

```python
def product_except_self(nums):
    """
    Without division operator

    Time: O(n), Space: O(1) excluding output
    """
    n = len(nums)
    result = [1] * n

    # Left products
    left = 1
    for i in range(n):
        result[i] = left
        left *= nums[i]

    # Right products
    right = 1
    for i in range(n-1, -1, -1):
        result[i] *= right
        right *= nums[i]

    return result

print(product_except_self([1,2,3,4]))  # [24,12,8,6]
```

---

## Problem 3: Kth Largest Element (Medium)

**Pattern:** Sorting / QuickSelect

```python
def find_kth_largest(nums, k):
    """
    Using sorting (simple approach)

    Time: O(n log n), Space: O(1)
    """
    nums.sort()
    return nums[-k]

    # Heap solution (better)
    # import heapq
    # return heapq.nlargest(k, nums)[-1]

print(find_kth_largest([3,2,1,5,6,4], 2))  # 5
```

---

## Problem 4: Valid Parentheses (Easy)

**Pattern:** Stack

```python
def is_valid_parentheses(s):
    """
    Check if parentheses are balanced

    Time: O(n), Space: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            # Closing bracket
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            # Opening bracket
            stack.append(char)

    return len(stack) == 0

print(is_valid_parentheses("()[]{}"))  # True
print(is_valid_parentheses("([)]"))    # False
```

---

## Problem 5: Merge Intervals (Medium)

**Pattern:** Sorting + Intervals

```python
def merge_intervals(intervals):
    """
    Merge overlapping intervals

    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return []

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]

        # Overlapping intervals
        if current[0] <= last[1]:
            # Merge
            last[1] = max(last[1], current[1])
        else:
            # Non-overlapping
            merged.append(current)

    return merged

print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))
# [[1,6],[8,10],[15,18]]
```

---

## Problem 6: Three Sum (Medium-Hard)

**Pattern:** Two Pointers + Sorting

```python
def three_sum(nums):
    """
    Find all unique triplets that sum to zero

    Time: O(n²), Space: O(1)
    """
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates
        if i > 0 and nums[i] == nums[i-1]:
            continue

        # Two pointer for remaining
        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])

                # Skip duplicates
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1

                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result

print(three_sum([-1,0,1,2,-1,-4]))
# [[-1,-1,2],[-1,0,1]]
```

---

## Complete DSA Journey Summary

### Patterns Mastered
1. **Binary Search** - O(log n) searching
2. **Two Pointers** - O(n) pair problems
3. **Sliding Window** - Subarray problems
4. **Hash Maps** - O(1) lookup, frequencies
5. **Recursion** - Divide and conquer
6. **Backtracking** - Generate combinations
7. **Sorting** - O(n log n) ordering

### Time Complexity Hierarchy
```
O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n) < O(n!)
```

### Interview Checklist
- [ ] Can solve easy problems in 10-15 min
- [ ] Can solve medium problems in 20-30 min
- [ ] Understand time/space complexity
- [ ] Can communicate approach clearly
- [ ] Test code with examples
- [ ] Handle edge cases

---

## Final Interview Tips

### Problem-Solving Framework
1. **Understand** - Clarify requirements, examples
2. **Plan** - Discuss approach, mention trade-offs
3. **Implement** - Write clean code, explain as you go
4. **Test** - Walk through examples, edge cases
5. **Optimize** - Discuss improvements

### Common Edge Cases
- Empty input
- Single element
- Duplicates
- Negative numbers
- Very large numbers
- Sorted vs unsorted

### Communication Tips
- Think aloud
- Start with brute force
- Explain optimizations
- Ask clarifying questions
- It's okay to get hints!

---

## You're Ready!

You've completed 33 problems across 6 days covering all fundamental DSA patterns. For your Cisco interview on Oct 23, focus on:

1. **Binary Search** (Day 1) - Most important algorithm
2. **Merge Sort** (Day 2) - Understand thoroughly
3. **Two Pointers** (Day 3) - Very common pattern
4. **Hash Maps** (Day 5) - Frequency problems

**Practice Strategy:**
- Review 2-3 problems daily
- Focus on explaining your approach
- Time yourself (20-30 min per problem)
- Practice on whiteboard or paper

You've got this! 🚀
