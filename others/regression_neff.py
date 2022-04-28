"""
    Compute lower bound on L2 loss for the case of direct ratio regression
"""


def compute_mse(N, alpha):
    """
        Compute lower bound on MSE loss
    """
    return (alpha * (1 - alpha)) / N


def find_neff(loss, alpha):
    return compute_mse(loss, alpha)


if __name__ == "__main__":
    ratio = 0.2
    # Plot for a given ratio
    N = 200
    print(compute_mse(N, ratio))
