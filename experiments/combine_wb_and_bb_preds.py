import numpy as np
from sklearn.tree import DecisionTreeClassifier
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
    plt.savefig('dt_contour.png')


if __name__ == "__main__":
    # Load predictions from WB models
    wb_preds_adv = np.load('wb_preds_adv.npy')
    wb_preds_vic = np.load('wb_preds_vic.npy')
    bb_preds_adv = np.load('bb_preds_adv.npy')
    bb_preds_vic = np.load('bb_preds_vic.npy')
    # Combined WB, BB preds
    preds_adv = np.concatenate((wb_preds_adv, bb_preds_adv), axis=1)
    preds_vic = np.concatenate((wb_preds_vic, bb_preds_vic), axis=1)
    num_each_adv = wb_preds_adv.shape[0] // 2
    labels_adv = np.concatenate((np.zeros(num_each_adv), np.ones(num_each_adv)))
    num_each_vic = wb_preds_vic.shape[0] // 2
    labels_vic = np.concatenate((np.zeros(num_each_vic), np.ones(num_each_vic)))
    # Train decision tree classifier
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(preds_adv, labels_adv)
    # Print out test & train accs
    print('Train acc:', clf.score(preds_vic, labels_vic))
    print('Test acc:', clf.score(preds_adv, labels_adv))
    # Visualize decision tree
    visualize_dt_contour(clf)
