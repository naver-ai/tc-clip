# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from typing import List, Tuple, Union


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


def schedule_r_constant(start_number, final_number, r):
    default_iter = int((start_number - final_number) / r)
    cnt = start_number
    cnt_list = []
    r_list = []
    for i in range(default_iter):
        r_list.append(min(r, cnt // 2))
        cnt = cnt - r_list[-1]
        cnt_list.append(cnt)

    while cnt > final_number:
        r_list.append(min(cnt - final_number, cnt // 2))
        cnt = cnt - r_list[-1]
        cnt_list.append(cnt)

    return r_list, cnt_list
