import numpy as np


class BinaryRatio:
    def __init__(self, r0, r1):
        self.r0 = r0
        self.r1 = r1

    def get_n_effective(self, acc):
        if max(self.r0, self.r1) == 0:
            return np.inf

        if self.r0 == self.r1:
            return 0

        if acc == 1 or np.abs(self.r0 - self.r1) == 1:
            return np.inf

        num = np.log(1 - ((2 * acc - 1) ** 2))
        ratio_0 = min(self.r0, self.r1) / max(self.r0, self.r1)
        ratio_1 = (1 - max(self.r0, self.r1)) / (1 - min(self.r0, self.r1))
        den = np.log(max(ratio_0, ratio_1))
        return num / den

    def bound(self, n):
        if max(self.r0, self.r1) == 0:
            return 0.5

        def bound_1():
            # Handle 0/0 form gracefully
            # if x == 0 and y == 0:
            #     return 0
            ratio = min(self.r0, self.r1) / max(self.r0, self.r1)
            return np.sqrt(1 - (ratio ** n))

        def bound_2():
            ratio = (1 - max(self.r0, self.r1)) / (1 - min(self.r0, self.r1))
            return np.sqrt(1 - (ratio ** n))

        l1 = bound_1()
        l2 = bound_2()
        pick = min(l1, l2) / 2
        return 0.5 + pick
