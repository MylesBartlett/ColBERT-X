import os
import torch
import ujson

from math import ceil
from itertools import accumulate
from xlmr_colbert.utils.utils import print_message


def get_parts(directory):
    extension = '.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [
        os.path.join(directory, f'{filename}{extension}') for filename in parts
    ]

    samples_paths = [
        os.path.join(directory, f'{filename}.sample') for filename in parts
    ]


    return parts, parts_paths, samples_paths


def load_doclens(directory, flatten=True):
    parts, _, _ = get_parts(directory)

    doclens_filenames = [
        os.path.join(directory, f'doclens.{filename}.json')
        for filename in parts
    ]

    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    return all_doclens
