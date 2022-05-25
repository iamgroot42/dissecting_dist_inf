import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def visualize_dt_contour(clf):
    """
        Given decision tree that works on 2 input features in [0, 1]
        visualize its decision boundary. Will be helpful in visualizing
        just how much importance it gives to each of the two features.
    """
    granularity = 0.01
    xx, yy = np.meshgrid(np.arange(0, 1 + granularity, granularity), np.arange(0, 1 + granularity, granularity))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.xlabel("White-Box Prediction")
    plt.ylabel("Black-Box Prediction")
    plt.savefig('dt_contour.png')


if __name__ == "__main__":
    # Load decision-tree classifier
    clf = pkl.load(open('log/nc_aff.p', 'rb'))
    # Visualize decision tree
    visualize_dt_contour(clf)
