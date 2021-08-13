# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import jenkspy
import numpy as np


def quantize_values_by_jenks(numeric_values, k, vectorize=False, break_profile=None, return_break_profile=False):
    """
    @param numeric_values: list(numeric)
        Values to quantize.
    @param k: int
        Total level of quantization.
    @param vectorize: bool
        If true return a numpy array which encode the numeric values. Otherwise return a list of quantize levels.
    @param break_profile: list(numeric)
        If None, this funciton will create a new break profile using the numeric_values.
    """

    if break_profile is None:
        break_profile = jenkspy.jenks_breaks(list(numeric_values), nb_class=k)[1:-1]
    else:
        assert k == len(break_profile) + 1

    quantize_levels = [find_quantize_level_bs(val, break_profile) for val in numeric_values]

    if vectorize:
        vector = np.zeros(len(break_profile)+1)

        for quantize_level in quantize_levels:
            vector[quantize_level] += 1.0

        vector /= len(numeric_values)

        return vector if not return_break_profile else (vector, break_profile)
    else:
        return quantize_levels if not return_break_profile else (quantize_levels, break_profile)


def find_quantize_level_bs(num, break_profile):
    """
    Binary search 'num' in break_profile list. The output level range is [0, len(break_profile)].

    @param num: numeric
        Value to quantize.
    @param break_profile: list(float)
        Break points list.
    @return: int
        Quantize level.
    """

    lo, hi = 0, len(break_profile)

    if num > break_profile[-1]:
        return len(break_profile)

    while lo < hi:
        mid = lo + int((hi - lo) / 2)

        if num < break_profile[mid]:
            hi = mid
        elif num > break_profile[mid]:
            lo = mid + 1
        else:
            return mid

    return lo if hi > 0 else 0