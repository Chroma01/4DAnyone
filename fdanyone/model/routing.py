"""Target-context routing across view groups."""

from __future__ import annotations

from itertools import pairwise


def _validate_grouping(num_views: int, group_size: int) -> int:
    if num_views <= 0 or group_size <= 0 or num_views % group_size:
        raise ValueError(f"num_views={num_views} must be divisible by positive group_size={group_size}.")
    return num_views // group_size


def view_groups(
    num_views: int,
    group_size: int,
    offset: int = 0,
    *,
    circular: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Partition one camera layer, optionally without joining its endpoints."""

    num_groups = _validate_grouping(num_views, group_size)
    if circular:
        return tuple(
            tuple((group_index * group_size + offset + local_index) % num_views for local_index in range(group_size))
            for group_index in range(num_groups)
        )

    # Shifting an open sequence creates smaller boundary groups instead of a
    # false neighborhood between the two ends of a partial yaw span.
    offset %= group_size
    boundaries = [0]
    if offset:
        boundaries.append(offset)
    boundaries.extend(range(offset + group_size, num_views, group_size))
    boundaries.append(num_views)
    return tuple(tuple(range(start, end)) for start, end in pairwise(boundaries))


def routing_steps(
    *,
    views_per_layer: int,
    num_layers: int,
    group_size: int,
    num_steps: int,
    enable_tcr: bool,
    circular: bool,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return layer-local target groups for every denoising step."""

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")
    _validate_grouping(views_per_layer, group_size)
    return tuple(
        tuple(
            tuple(layer_index * views_per_layer + view_index for view_index in group)
            for layer_index in range(num_layers)
            for group in view_groups(
                views_per_layer,
                group_size,
                step_index if enable_tcr else 0,
                circular=circular,
            )
        )
        for step_index in range(num_steps)
    )
