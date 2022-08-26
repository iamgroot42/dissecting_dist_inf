from distribution_inference.attacks.utils import ATTACK_MAPPING, get_attack_name
from distribution_inference.utils import warning_string, get_arxiv_node_params_mapping
import numpy as np
import warnings


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
        self.param_mapping = get_arxiv_node_params_mapping()
        super().__init__(
            self.param_mapping[deg0],
            self.param_mapping[deg1])

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


def process_logfile_for_neffs(data, logger, attack_res, ratios_wanted, is_regression: bool=False):
    attack_names = get_attack_name(attack_res)

    # Loss & Threshold attacks
    if (attack_res == "loss_and_threshold"):
        for ratio in logger['result'][attack_res]:
            if ratios_wanted is not None and ratio not in ratios_wanted:
                continue

            ratio_name = ratio
            if ratio_name != "regression":
                ratio_name = float(ratio)

            victim_results = logger['result'][attack_res][ratio]['victim_acc']
            for results in victim_results:
                loss = results[1]
                threshold = results[0]
                if type(loss) == list:
                    assert len(loss) == len(threshold)
                    for epoch, (l, t) in enumerate(zip(loss, threshold)):
                        data.append({
                            "prop_val": ratio_name,
                            "acc_or_loss": l,
                            "attack": attack_names[0],
                            "epoch": epoch + 1})
                        data.append({
                            "prop_val": ratio_name,
                            "acc_or_loss": t,
                            "attack": attack_names[1],
                            "epoch": epoch + 1})
                else:
                    assert type(threshold) != list
                    data.append({
                        "prop_val": ratio_name,
                        "acc_or_loss": loss,
                        "attack": attack_names[0]})
                    data.append({
                        "prop_val": ratio_name,
                        "acc_or_loss": threshold,
                        "attack": attack_names[1]})

    # Per-point threshold attack, or white-box attack
    elif attack_res in ATTACK_MAPPING.keys():
        for ratio in logger['result'][attack_res]:

            ratio_name = float(ratio)

            if ratios_wanted is not None and ratio not in ratios_wanted:
                continue
            victim_results = logger['result'][attack_res][ratio]['victim_acc']
            
            for results in victim_results:
                if type(results) == list:
                    for epoch, result in enumerate(results):
                        data.append({
                            "prop_val": ratio_name,
                            # Temporary (below) - ideally all results should be in [0, 100] across entire module
                            "acc_or_loss": result,  # * 100,
                            "attack": attack_names,
                            "epoch": epoch + 1})
                else:
                    data.append({
                        "prop_val": ratio_name,
                        # Temporary (below) - ideally all results should be in [0, 100] across entire module
                        "acc_or_loss": results*100 if (results <= 1 and not is_regression) else results,  # * 100,
                        "attack": attack_names})
    else:
        warnings.warn(warning_string(
            f"\nAttack type {attack_res} not supported\n"))

    return data
