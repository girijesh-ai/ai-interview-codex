# Day 5: Hash Maps & Problem Solving Patterns - Complete Tutorial

## Today's Goals
- Master hash map (dictionary) usage
- Learn frequency counting patterns
- Understand two-sum pattern variations
- Solve common interview problems

**Time:** 2-3 hours
**Difficulty:** Easy-Medium

---

## Concept: Hash Maps (Dictionaries)

### What is a Hash Map?
Key-value data structure with O(1) average lookup, insert, delete.

```python
# Creating hash maps in Python
hash_map = {}
hash_map['key'] = 'value'

# Or using Counter for frequencies
from collections import Counter
freq = Counter([1, 2, 2, 3, 3, 3])  # {1: 1, 2: 2, 3: 3}
```

### When to Use Hash Maps
- Need O(1) lookup
- Counting frequencies
- Finding pairs/complements
- Checking existence
- Grouping elements

---

## Problem 1: Two Sum (Easy)

### Problem
Find two numbers in array that sum to target.

```
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]  (nums[0] + nums[1] = 9)
```

### Solution

```python
def two_sum(nums, target):
    """
    Hash map for O(n) solution

    Time: O(n), Space: O(n)
    """
    seen = {}  # value -> index

    for i, num in enumerate(nums):
        complement = target - num

        # Check if complement exists
        if complement in seen:
            return [seen[complement], i]

        # Store current number
        seen[num] = i

    return [-1, -1]

print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
```

---

## Problem 2: Group Anagrams (Medium)

### Problem
Group strings that are anagrams of each other.

```
Input: ["eat", "tea", "tan", "ate", "nat", "bat"]
Output: [["eat","tea","ate"], ["tan","nat"], ["bat"]]
```

### Solution

```python
def group_anagrams(strs):
    """
    Use sorted string as key

    Time: O(n * k log k) where k = max string length
    Space: O(n * k)
    """
    from collections import defaultdict

    groups = defaultdict(list)

    for s in strs:
        # Sorted string as key
        key = ''.join(sorted(s))
        groups[key].append(s)

    return list(groups.values())

print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
```

---

## Problem 3: Top K Frequent Elements (Medium)

### Problem
Find k most frequent elements in array.

```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1, 2]
```

### Solution

```python
def top_k_frequent(nums, k):
    """
    Using Counter and sorting

    Time: O(n log n), Space: O(n)
    """
    from collections import Counter

    # Count frequencies
    freq = Counter(nums)

    # Sort by frequency
    return [num for num, count in freq.most_common(k)]

print(top_k_frequent([1,1,1,2,2,3], 2))  # [1, 2]
```

---

## Problem 4: Longest Consecutive Sequence (Medium)

### Problem
Find length of longest consecutive sequence.

```
Input: [100, 4, 200, 1, 3, 2]
Output: 4  (sequence: [1, 2, 3, 4])
```

### Solution

```python
def longest_consecutive(nums):
    """
    Hash set for O(n) solution

    Time: O(n), Space: O(n)
    """
    num_set = set(nums)
    longest = 0

    for num in num_set:
        # Only start counting if it's start of sequence
        if num - 1 not in num_set:
            current_num = num
            current_length = 1

            # Count consecutive numbers
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            longest = max(longest, current_length)

    return longest

print(longest_consecutive([100, 4, 200, 1, 3, 2]))  # 4
```

---

## Problem 5: Valid Anagram (Easy)

```python
def is_anagram(s, t):
    """
    Check if two strings are anagrams

    Time: O(n), Space: O(1) - max 26 letters
    """
    from collections import Counter

    return Counter(s) == Counter(t)

    # Alternative without Counter
    # return sorted(s) == sorted(t)

print(is_anagram("anagram", "nagaram"))  # True
```

---

## Day 5 Summary

### Key Patterns
1. **Two Sum Pattern:** complement = target - num
2. **Frequency Counting:** Counter/hash map
3. **Grouping:** Use key to group similar items
4. **Set for O(1) lookup:** Check existence

### Hash Map Operations
- Insert: O(1) average
- Lookup: O(1) average
- Delete: O(1) average
- Iteration: O(n)

### Common Use Cases
- Finding pairs/triplets
- Counting frequencies
- Detecting duplicates
- Grouping anagrams
- Caching/memoization

Tomorrow: Mixed Practice Problems! 🎯
