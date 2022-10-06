"""
    Generic utility functions useful for writing Python code in general
"""
import os
from colorama import Fore, Style
import dataclasses


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log_string(x, ttype="log"):
    color_mapping = {"train": Fore.YELLOW, "val": Fore.GREEN}
    return color_mapping.get(ttype, Fore.MAGENTA) + x + Style.RESET_ALL


def log_statement(x, ttype="log"):
    print(log_string(x, ttype))


def warning_string(x):
    return f"{bcolors.WARNING}%s{bcolors.ENDC}" % x


def log(x):
    print(warning_string(x))


def check_if_inside_cluster():
    """
        Check if current code is being run inside a cluster.
    """
    if os.environ.get('ISRIVANNA') == "1":
        return True
    return False


def check_user():
    return os.environ.get('USER')


def flash_utils(args, root: bool = True, num_tabs: int = 0):
    prefix = "    " * num_tabs
    if root:
        log_statement("==> Arguments:")
    for arg in vars(args):
        arg_val = getattr(args, arg)
        if dataclasses.is_dataclass(arg_val):
            print(prefix + arg, " : ")
            flash_utils(arg_val, root=False, num_tabs=num_tabs + 1)
        else:
            print(prefix + arg, " : ", arg_val)


def ensure_dir_exists(dir):
    """
        Create necessary folders if given path does not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_save_path():
    """
        Path where results/trained meta-models are stored
    """
    # return "./log"
    return "./log_new"


def get_arxiv_node_params_mapping():
    """
        Get parameters for Zipf distribution estimation
        of arXiv graphs
    """
    return {
        9: (86, 1.843366),
        10: (133, 1.881514),
        11: (214, 1.929022),
        12: (257, 1.927125),
        13: (253, 1.918591),
        14: (267, 1.914586),
        15: (265, 1.903623),
        16: (263, 1.886148),
        17: (263, 1.876854)
    }
