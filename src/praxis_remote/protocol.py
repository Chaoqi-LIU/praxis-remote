# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Public protocol types for remote policy inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from praxis_remote.serialization import ObservationValue, PolicyKwargValue

Observation = Mapping[str, ObservationValue]


class PolicyHandler(Protocol):
    """Server-side policy interface consumed by :class:`PolicyServer`."""

    def predict_action(
        self,
        observations: Sequence[Observation],
        *,
        policy_kwargs: Mapping[str, PolicyKwargValue] | None = None,
        episode_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return a batched action array for a batch of observations."""
        ...

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        """Reset policy rollout state between episodes."""
        ...

    def model_info(self) -> str:
        """Return a short human-readable readiness string."""
        ...
