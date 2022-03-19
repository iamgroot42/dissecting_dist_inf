"""
Generic utility functions useful for writing Python code in general
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from os import environ


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


def log(x):
    print(f"{bcolors.WARNING}%s{bcolors.ENDC}" % x)


def check_if_inside_cluster():
    """
        Check if current code is being run inside a cluster.
    """
    if environ.get('ISRIVANNA') == "1":
        return True
    return False


def flash_utils(args):
    log_statement("==> Arguments:")
    for arg in vars(args):
        print(arg, " : ", getattr(args, arg))
