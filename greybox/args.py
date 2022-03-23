import argparse
from random import paretovariate


def common_args(parser):
    parser.add_argument(
        "--bs", default=1, type=int, help="Batch Size (default: 1)",
    )


def swin_args(parser):
    parser.add_argument("--")
