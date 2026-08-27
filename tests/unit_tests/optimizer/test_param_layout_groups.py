# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.optimizer.param_layout import layout_group_key, order_params_for_layout


def _param(numel: int, group=None):
    param = torch.nn.Parameter(torch.zeros(numel))
    if group is not None:
        param._contiguous_layout_group = group
    return param


def test_order_params_for_layout_restores_group_ascending_order():
    """Global order stays reversed while each contiguous group becomes ascending."""
    group_key = "experts0"
    dense_before = _param(7)
    experts = [_param(5, group=(group_key, i, 3)) for i in range(3)]
    dense_after = _param(9)

    params = [dense_before, *experts, dense_after]
    ordered = order_params_for_layout(params)

    # Reverse order across modules: the param registered last is packed first.
    assert ordered[0] is dense_after
    assert ordered[-1] is dense_before
    # Ascending member order within the group (global reversal alone would
    # yield weight2, weight1, weight0).
    assert ordered[1:4] == experts


def test_order_params_for_layout_handles_multiple_and_adjacent_groups():
    group_a = [_param(4, group=("a", i, 2)) for i in range(2)]
    group_b = [_param(4, group=("b", i, 2)) for i in range(2)]

    ordered = order_params_for_layout([*group_a, *group_b])

    # Groups swap (reverse order) but stay internally ascending.
    assert ordered == [*group_b, *group_a]


def test_order_params_for_layout_without_groups_matches_plain_reverse():
    params = [_param(3), _param(4), _param(5)]
    assert order_params_for_layout(params) == params[::-1]


def test_layout_group_key():
    grouped = _param(2, group=("g", 0, 1))
    assert layout_group_key(grouped) == "g"
    assert layout_group_key(_param(2)) is None
