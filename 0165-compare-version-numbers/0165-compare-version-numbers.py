class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        nums1 = [int(x) for x in version1.split('.')]
        nums2 = [int(x) for x in version2.split('.')]
        lmax = max(len(nums1), len(nums2))
        for i in range(lmax):
            n1 = nums1[i] if i < len(nums1) else 0
            n2 = nums2[i] if i < len(nums2) else 0
            if n1 > n2:
                return 1
            elif n1 < n2:
                return -1
        return 0