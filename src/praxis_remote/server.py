# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Generic gRPC policy server for remote policy inference."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator
from concurrent import futures

import grpc
import numpy as np

from praxis_remote.proto import policy_service_pb2, policy_service_pb2_grpc
from praxis_remote.protocol import PolicyHandler
from praxis_remote.serialization import (
    decode_observations,
    decode_policy_kwargs,
    encode_numpy_array,
)

_GRPC_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_LOCK_POLL_SECONDS = 0.05
_LOG = logging.getLogger(__name__)


class PolicyServicer(policy_service_pb2_grpc.PolicyServiceServicer):
    """gRPC servicer wrapping a generic policy handler."""

    def __init__(self, handler: PolicyHandler) -> None:
        self.handler = handler
        self._policy_lock = threading.Lock()

    @contextlib.contextmanager
    def _locked_policy(self, context) -> Iterator[None]:
        """Acquire the policy lock without running cancelled queued requests."""
        while context.is_active():
            remaining = context.time_remaining()
            timeout = _LOCK_POLL_SECONDS
            if remaining is not None:
                timeout = min(timeout, max(0.0, float(remaining)))
            if timeout <= 0:
                break
            if self._policy_lock.acquire(timeout=timeout):
                if not context.is_active():
                    self._policy_lock.release()
                    break
                try:
                    yield
                finally:
                    self._policy_lock.release()
                return

        context.abort(
            grpc.StatusCode.CANCELLED,
            "Request was cancelled before the policy handler became available.",
        )

    def Predict(self, request, context):
        try:
            observations = decode_observations(request.observations)
            policy_kwargs = decode_policy_kwargs(request.policy_kwargs)
            if len(observations) < 1:
                raise ValueError("observations must be non-empty.")
            if request.episode_ids and len(request.episode_ids) != len(observations):
                raise ValueError(
                    "episode_ids must be empty or match observations length: "
                    f"{len(request.episode_ids)} != {len(observations)}."
                )
        except (TypeError, ValueError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        with self._locked_policy(context):
            try:
                episode_ids = (
                    tuple(str(item) for item in request.episode_ids)
                    if request.episode_ids
                    else None
                )
                predict_kwargs = {
                    "policy_kwargs": policy_kwargs,
                }
                if episode_ids is not None:
                    predict_kwargs["episode_ids"] = episode_ids
                action = self.handler.predict_action(
                    observations,
                    **predict_kwargs,
                )
                response_action = encode_numpy_array(np.asarray(action))
            except (TypeError, ValueError) as exc:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except Exception as exc:
                context.abort(grpc.StatusCode.INTERNAL, str(exc))

        return policy_service_pb2.PredictResponse(action=response_action)

    def Reset(self, request, context):
        try:
            with self._locked_policy(context):
                episode_ids = (
                    tuple(str(item) for item in request.episode_ids)
                    if request.episode_ids
                    else None
                )
                if episode_ids is None:
                    self.handler.reset()
                else:
                    self.handler.reset(episode_ids=episode_ids)
        except (TypeError, ValueError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        return policy_service_pb2.ResetResponse(success=True)

    def HealthCheck(self, request, context):
        _ = (request, context)
        with self._locked_policy(context):
            model_info = self.handler.model_info()
        return policy_service_pb2.HealthResponse(ready=True, model_info=str(model_info))


class PolicyServer:
    """Manages the gRPC policy-server lifecycle."""

    def __init__(
        self,
        handler: PolicyHandler,
        *,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 4,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.servicer = PolicyServicer(handler)
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_send_message_length", _GRPC_MAX_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _GRPC_MAX_MESSAGE_BYTES),
            ],
        )
        policy_service_pb2_grpc.add_PolicyServiceServicer_to_server(
            self.servicer,
            self.server,
        )
        self.server.add_insecure_port(f"{self.host}:{self.port}")

    def serve(self, *, block: bool = True) -> None:
        """Start the server."""
        self.server.start()
        _LOG.info("PolicyServer listening on %s:%s", self.host, self.port)
        if block:
            self.server.wait_for_termination()

    def stop(self, *, grace: float = 0) -> None:
        """Stop the server."""
        self.server.stop(grace)
