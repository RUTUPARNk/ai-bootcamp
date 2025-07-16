#  H-Index Problem — Difference-Based Approach

This repository implements a clean, efficient solution to the **H-Index** problem using a **difference-based observation**, leveraging a sorted citations array to determine the maximum `h` such that the researcher has published `h` papers each with at least `h` citations.

---

##  Problem Statement

Given an array `citations` of integers representing the number of citations each paper has received, compute the researcher's **H-Index**.

> The H-Index is defined as the maximum value `h` such that the researcher has at least `h` papers with at least `h` citations each.

---
## Constraints

    n == citations.length

    1 <= n <= 5000

    0 <= citations[i] <= 1000

 ## Approach

We observe that for a sorted list of citation counts:

    If there are n papers total, then citations[i] must be greater than or equal to n - i (i.e., number of papers from i to end) to qualify as a valid h.

This observation allows us to find the first index i such that:

citations[i] >= n - i

Then the H-Index = n - i.

This difference-based threshold tracking leads to a concise and performant implementation.
---
## Time and Space Complexity
| Metric           | Complexity                                             |
| ---------------- | ------------------------------------------------------ |
| Time Complexity  | `O(n log n)` — due to sorting                          |
| Space Complexity | `O(1)` — in-place sorting and constant auxiliary space |

---
## Code
```python
def hIndex_by_difference(citations):
    citations.sort()
    n = len(citations)
    for i in range(n):
        if citations[i] >= n - i:
            return n - i
    return 0
---


## 🧪 Example

### Example 1:
```python
Input: citations = [3, 0, 6, 1, 5]
Sorted: [0, 1, 3, 5, 6]
Output: 3
Explanation:
There are 3 papers with at least 3 citations.

Input: citations = [1, 2, 100]
Sorted: [1, 2, 100]
Output: 2
Explanation:
There are 2 papers with at least 2 citations.

