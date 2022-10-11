import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


def bound(alpha, n):
    return alpha * (1 - alpha) / n


def find_n_eff(mapping):
    neff = np.inf
    for k, v in mapping.items():
        if k == 0 or k == 1:
            continue
        neff = min(k * (1 - k) / v, neff)
    return neff


def get_boneage_data():
    mapping = {0.2: 0.00087064866, 0.3: 0.0004569574, 0.4: 0.0006438183, 0.5: 0.0007685996, 0.6: 0.0008711952, 0.7: 0.0011653631, 0.8: 0.0018011187}
    return mapping

def get_census_race():
    mapping = {0.0: 0.0654676, 0.1: 0.06229961, 0.2: 0.029624252, 0.3: 0.012912554, 0.4: 0.0143639855, 0.5: 0.031330504, 0.6: 0.06570812, 0.7: 0.12373179, 0.8: 0.1903304, 0.9: 0.27961597, 1.0: 0.25872186}
    # Skip 0, 1- no theoretical bound
    del mapping[0.0]
    del mapping[1.0]
    return mapping

def get_census_sex():
    mapping = {0.0: 0.09151385, 0.1: 0.080922775, 0.2: 0.048298728, 0.3: 0.024282401, 0.4: 0.01142508, 0.5: 0.010248317, 0.6: 0.019268725, 0.7: 0.03726791, 0.8: 0.07382312, 0.9: 0.11668019, 1.0: 0.15792367}
    # Skip 0, 1- no theoretical bound
    del mapping[0.0]
    del mapping[1.0]
    return mapping

def get_celeba_sex_data():
    mapping = {0.0: 0.013693116, 0.1: 0.007828232, 0.2: 0.008433977, 0.3: 0.013606352, 0.4: 0.02016941, 0.5: 0.027217364, 0.6: 0.033584125, 0.7: 0.043971337, 0.8: 0.062406037, 0.9: 0.08795085, 1.0: 0.13489045}
    # Skip 0, 1- no theoretical bound
    del mapping[0.0]
    del mapping[1.0]
    return mapping

def get_celeba_young_data():
    mapping = {0.0: 0.1324398, 0.1: 0.11734961, 0.2: 0.09645328, 0.3: 0.0802868, 0.4: 0.0598498, 0.5: 0.042930175, 0.6: 0.029001104, 0.7: 0.020008726, 0.8: 0.017675962, 0.9: 0.0262954, 1.0: 0.05070238}
    # Skip 0, 1- no theoretical bound
    del mapping[0.0]
    del mapping[1.0]
    return mapping



if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes', labelsize=14)

    plt.style.use('dark_background')

    # Plot curves corresponding to ratios
    colors = ['blue', 'orange', 'green', 'lightcoral', 'purple']
    # curve_colors = ['mediumpurple', 'olive', 'firebrick']
    curve_colors = ['darkred', 'lightsalmon']
    n_effs = []
    x_axis = np.linspace(0.1, 0.9, 1000)

    # BoneAge
    boneage = get_boneage_data()
    for k, v in boneage.items():
        plt.scatter(k, v, color=colors[0], edgecolors='face', marker='D')
    n_eff = find_n_eff(boneage)
    n_effs.append(n_eff)
    print("N-eff Boneage: %d" % n_eff)

    # CelebA: female
    celeba_sex = get_celeba_sex_data()
    for k, v in celeba_sex.items():
        # Ratios are actually 1 - K
        plt.scatter(round(1 - k, 1), v, color=colors[1], edgecolors='face', marker='*')
        print(round(1 - k, 1), v)
    n_eff = find_n_eff(celeba_sex)
    n_effs.append(n_eff)
    print("N-eff Celeba (female): %d" % n_eff)

    # Celeba: old
    celeba_young = get_celeba_young_data()
    for k, v in celeba_young.items():
        # Ratios are actually 1 - K
        plt.scatter(round(1 - k, 1), v, color=colors[2], edgecolors='face', marker='*')
    n_eff = find_n_eff(celeba_young)
    n_effs.append(n_eff)
    print("N-eff Celeba (old): %d" % n_eff)

    # Census: sex
    census_sex = get_census_sex()
    for k, v in celeba_young.items():
        plt.scatter(k, v, color=colors[3], edgecolors='face', marker='s')
    n_eff = find_n_eff(celeba_young)
    n_effs.append(n_eff)
    print("N-eff Census (sex): %d" % n_eff)

    # Census: race
    census_race = get_census_race()
    for k, v in census_race.items():
        plt.scatter(k, v, color=colors[4], edgecolors='face', marker='s')
    n_eff = find_n_eff(census_race)
    n_effs.append(n_eff)
    print("N-eff Census (race): %d" % n_eff)

    n_effs = [10, 1]
    for cc, n_eff in zip(curve_colors, n_effs):

        plt.plot(x_axis, [bound(x_, n_eff)
                          for x_ in x_axis], '--', color=cc, label=r"$n_{leaked}=%d$" % n_eff)

    # Trick to get desired legend
    # plt.plot([], [], color=colors[0], marker="D", ms=10, ls="", label="RSNA Bone Age")
    # plt.plot([],[], color=colors[1], marker="*", ms=10, ls="", label="CelebA (female)")
    # plt.plot([], [], color=colors[2], marker="*", ms=10, ls="", label="CelebA (young)")
    # plt.plot([], [], color=colors[3], marker="s", ms=10, ls="", label="Census (female)")
    # plt.plot([], [], color=colors[4], marker="s", ms=10, ls="", label="Census (white)")

    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'MSE')
    plt.ylim(-0.01, 0.3)
    plt.xticks(np.arange(0.1, 1.0, 0.1))
    # plt.grid()
    plt.style.use('seaborn')
    plt.gca().invert_yaxis()
    plt.legend()
    # plt.savefig("./bound_curves_regression.pdf")
    plt.savefig("./bound_curves_regression.png")

    # print(bound(0.1, 0.2, 2))
    # print(bound(0.5, 0.6, 2))
