# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Generic remote policy inference SDK."""

from importlib.metadata import PackageNotFoundError, version

from praxis_remote.client import PolicyClient
from praxis_remote.protocol import PolicyHandler
from praxis_remote.serialization import (
    Observation,
    ObservationValue,
    PolicyKwargValue,
    build_observation,
)
from praxis_remote.server import PolicyServer

try:
    __version__ = version("praxis-remote")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "Observation",
    "ObservationValue",
    "PolicyClient",
    "PolicyHandler",
    "PolicyServer",
    "PolicyKwargValue",
    "__version__",
    "build_observation",
]
