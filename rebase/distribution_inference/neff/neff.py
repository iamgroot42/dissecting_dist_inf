from distribution_inference.datasets.arxiv import DatasetInformation
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


class Regression:
    def __init__(self, r):
        self.r = r

    def get_n_effective(self, mse):
        return (self.r * (1 - self.r)) / mse
    
    def bound(self, n_leaked):
        return self.get_n_effective(n_leaked)


class GraphBinary(BinaryRatio):
    def __init__(self, deg0, deg1):
        self.info_obj = DatasetInformation()
        super().__init__(
            self.info_obj.param_mapping[deg0],
            self.info_obj.param_mapping[deg1])

    def get_n_effective(self, acc):
        if acc == 1:
            return np.inf

        if self.r0[0] > self.r1[0]:
            n0, s0 = self.r1
            n1, s1 = self.r0
        else:
            n0, s0 = self.r0
            n1, s1 = self.r1

        def gen_harmonic(n, s):
            return np.sum(np.arange(1, n + 1) ** (-s))

        numerator = np.log(1 - ((2 * acc - 1) ** 2))
        denominator = np.log(gen_harmonic(n0, s0) / gen_harmonic(n1, s1))

        if s1 > s0:
            denominator += (s0 - s1) * np.log(n0)

        if denominator == 0:
            return np.inf

        return numerator / denominator
    
    def bound(self, n):
        # Determine which of two Ns are larger:
        if self.r0[0] > self.r1[0]:
            n0, m0 = self.r1
            n1, m1 = self.r0
        else:
            n0, m0 = self.r0
            n1, m1 = self.r1

        c0 = np.sum(np.arange(1, n0 + 1) ** -m0)
        c1 = np.sum(np.arange(1, n1 + 1) ** -m1)

        inner = (c0 / c1)
        if m1 > m0:
            inner *= (n0 ** (m0 - m1))
    
        return (1 + np.sqrt(1 - (inner ** n))) / 2
