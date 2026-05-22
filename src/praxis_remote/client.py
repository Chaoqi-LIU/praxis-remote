# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""gRPC client for generic remote policy inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType

import grpc
import numpy as np

from praxis_remote.proto import policy_service_pb2, policy_service_pb2_grpc
from praxis_remote.serialization import (
    ObservationValue,
    PolicyKwargValue,
    build_observation,
    decode_numpy_array,
    encode_observations,
    encode_policy_kwargs,
)

_GRPC_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class PolicyClient:
    """Remote client for the policy inference service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        *,
        timeout: float | None = None,
    ) -> None:
        self._timeout = timeout
        self.channel = grpc.insecure_channel(
            f"{host}:{port}",
            options=[
                ("grpc.max_send_message_length", _GRPC_MAX_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _GRPC_MAX_MESSAGE_BYTES),
            ],
        )
        self.stub = policy_service_pb2_grpc.PolicyServiceStub(self.channel)

    def __enter__(self) -> PolicyClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def predict_action_from_parts(
        self,
        state: np.ndarray | None = None,
        images: Mapping[str, np.ndarray] | None = None,
        task_description: str | None = None,
        *,
        policy_kwargs: Mapping[str, PolicyKwargValue] | None = None,
        episode_id: str | None = None,
        timeout: float | None = None,
    ) -> np.ndarray:
        """Build one observation from common fields and request its action.

        Args:
            state: Flat state vector, shape (state_dim,), float32.
            images: {key: array} where array is shaped as the policy expects.
            task_description: Optional language instruction for the task.
            policy_kwargs: Optional predict_action kwargs shared across the request.
            episode_id: Optional rollout slot id for this single observation.
            timeout: Optional per-call gRPC deadline in seconds.

        Returns:
            Action as numpy array. Shape matches local ``predict_action`` output.
        """
        observation = build_observation(
            state=state,
            images=images,
            task_description=task_description,
        )
        return self.predict_action(
            [observation],
            policy_kwargs=policy_kwargs,
            episode_ids=[episode_id] if episode_id is not None else None,
            timeout=timeout,
        )

    def predict_action(
        self,
        observations: Sequence[Mapping[str, ObservationValue]],
        *,
        policy_kwargs: Mapping[str, PolicyKwargValue] | None = None,
        episode_ids: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> np.ndarray:
        """Request actions for a non-empty batch of observations."""
        if len(observations) < 1:
            raise ValueError("observations must be non-empty.")
        request = policy_service_pb2.PredictRequest()
        request.observations.extend(encode_observations(observations))
        if episode_ids is not None:
            if len(episode_ids) != len(observations):
                raise ValueError(
                    "episode_ids must match observations length: "
                    f"{len(episode_ids)} != {len(observations)}."
                )
            request.episode_ids.extend(str(item) for item in episode_ids)
        for key, value in encode_policy_kwargs(policy_kwargs).items():
            request.policy_kwargs[key].CopyFrom(value)

        response = self.stub.Predict(request, timeout=self._resolve_timeout(timeout))
        return decode_numpy_array(response.action)

    def reset(
        self,
        *,
        episode_ids: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Reset policy state between episodes."""
        request = policy_service_pb2.ResetRequest()
        if episode_ids is not None:
            request.episode_ids.extend(str(item) for item in episode_ids)
        response = self.stub.Reset(
            request,
            timeout=self._resolve_timeout(timeout),
        )
        return response.success

    def health_check(self, *, timeout: float | None = None) -> tuple[bool, str]:
        """Check server readiness."""
        response = self.stub.HealthCheck(
            policy_service_pb2.Empty(),
            timeout=self._resolve_timeout(timeout),
        )
        return response.ready, response.model_info

    def close(self) -> None:
        """Close the channel."""
        self.channel.close()

    def _resolve_timeout(self, timeout: float | None) -> float | None:
        return self._timeout if timeout is None else timeout
