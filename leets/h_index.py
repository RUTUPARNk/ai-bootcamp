'''i got this solution for time complexity O(n log n) 
Space complexity O(1) in-place sort
'''
def hIndex_by_difference(citations):
    citations.sort()
    n = len(citations)
    for i in range(n):
        if citations[i] >= n - i:
            return n - i
    return 0
