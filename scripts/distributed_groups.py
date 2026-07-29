"""Create independent MLX payload and control communication groups."""

from __future__ import annotations

import os

import mlx.core as mx


def secondary_coordinator(primary: str) -> str:
    host, separator, port = primary.rpartition(":")
    if not separator:
        raise ValueError(f"invalid coordinator: {primary!r}")
    secondary_port = int(port) + 1
    if secondary_port > 65535:
        raise ValueError(f"coordinator port has no control peer: {primary!r}")
    return f"{host}:{secondary_port}"


def init_distributed_groups(backend: str):
    """Initialize a payload group and a separate JACCL control group.

    JACCL's point-to-point receive may become locally visible before the
    sender retires its work request.  A collective on the same queue pair in
    that window can deadlock.  A second backend instance gives barriers and
    all-gather an independent RDMA queue pair.
    """
    payload = mx.distributed.init(strict=True, backend=backend)
    if backend != "jaccl":
        return payload, payload

    primary = os.environ["MLX_JACCL_COORDINATOR"]
    control_address = os.environ.get(
        "KIMI_JACCL_CONTROL_COORDINATOR", secondary_coordinator(primary)
    )
    if control_address == primary:
        raise ValueError("payload and control coordinators must be different")
    os.environ["MLX_JACCL_COORDINATOR"] = control_address
    try:
        control = mx.distributed.init(strict=True, backend="jaccl")
    finally:
        os.environ["MLX_JACCL_COORDINATOR"] = primary
    if (control.rank(), control.size()) != (payload.rank(), payload.size()):
        raise RuntimeError("payload and control JACCL groups do not match")
    return payload, control
