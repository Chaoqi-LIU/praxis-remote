from __future__ import annotations

import socket
import threading
import time

import grpc
import numpy as np

from praxis_remote import PolicyClient, PolicyServer
from praxis_remote.proto import policy_service_pb2, policy_service_pb2_grpc


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class EchoShapeHandler:
    def __init__(self) -> None:
        self.reset_count = 0
        self.last_policy_kwargs = None
        self.last_episode_ids = None
        self.last_observation = None

    def predict_action(self, observations, *, policy_kwargs=None, episode_ids=None):
        self.last_policy_kwargs = dict(policy_kwargs or {})
        self.last_episode_ids = tuple(episode_ids or ())
        self.last_observation = dict(observations[0])
        return np.asarray(
            [[len(observations), len(observations[0])]],
            dtype=np.float32,
        )

    def reset(self, episode_ids=None) -> None:
        self.last_episode_ids = tuple(episode_ids or ())
        self.reset_count += 1

    def model_info(self) -> str:
        return "echo-shape"


def test_policy_server_client_roundtrip() -> None:
    handler = EchoShapeHandler()
    port = _free_port()
    server = PolicyServer(handler, host="127.0.0.1", port=port)
    server.serve(block=False)
    try:
        client = PolicyClient(host="127.0.0.1", port=port)
        try:
            ready, info = client.health_check(timeout=5)
            assert ready is True
            assert info == "echo-shape"

            action = client.predict_action(
                [
                    {
                        "task": "pick",
                        "observation.state": np.zeros(3, dtype=np.float32),
                        "metadata.episode_index": 7,
                    }
                ],
                policy_kwargs={"decode_keep_k": 2},
                episode_ids=["slot-0"],
                timeout=5,
            )
            np.testing.assert_array_equal(
                action,
                np.asarray([[1, 3]], dtype=np.float32),
            )
            assert handler.last_observation is not None
            assert handler.last_observation["metadata.episode_index"] == 7
            assert handler.last_policy_kwargs == {"decode_keep_k": 2}
            assert handler.last_episode_ids == ("slot-0",)

            convenience_action = client.predict_action_from_parts(
                state=np.zeros(3, dtype=np.float32),
                task_description="pick",
                policy_kwargs={"decode_keep_k": 4},
                episode_id="slot-1",
                timeout=5,
            )
            np.testing.assert_array_equal(
                convenience_action,
                np.asarray([[1, 2]], dtype=np.float32),
            )
            assert handler.last_policy_kwargs == {"decode_keep_k": 4}
            assert handler.last_episode_ids == ("slot-1",)

            assert client.reset(episode_ids=["slot-0"], timeout=5) is True
            assert handler.reset_count == 1
            assert handler.last_episode_ids == ("slot-0",)
        finally:
            client.close()
    finally:
        server.stop(grace=0)


def test_client_rejects_empty_observation_batch() -> None:
    client = PolicyClient(host="127.0.0.1", port=_free_port())
    try:
        try:
            client.predict_action([])
        except ValueError as exc:
            assert "non-empty" in str(exc)
        else:  # pragma: no cover - assertion branch
            raise AssertionError("empty observation batch was accepted")
    finally:
        client.close()


class SimpleHandler:
    def __init__(self) -> None:
        self.reset_count = 0

    def predict_action(self, observations, *, policy_kwargs=None):
        _ = policy_kwargs
        return np.zeros((len(observations), 1), dtype=np.float32)

    def reset(self) -> None:
        self.reset_count += 1

    def model_info(self) -> str:
        return "simple"


class SlowHandler:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.seen_episode_ids: list[tuple[str, ...]] = []

    def predict_action(self, observations, *, policy_kwargs=None, episode_ids=None):
        del observations, policy_kwargs
        self.seen_episode_ids.append(tuple(episode_ids or ()))
        self.started.set()
        if tuple(episode_ids or ()) == ("slow",):
            assert self.release.wait(timeout=5)
        return np.zeros((1, 1), dtype=np.float32)

    def reset(self, episode_ids=None) -> None:
        del episode_ids

    def model_info(self) -> str:
        return "slow"


def test_server_allows_simple_handler_when_no_episode_ids() -> None:
    handler = SimpleHandler()
    port = _free_port()
    server = PolicyServer(handler, host="127.0.0.1", port=port)
    server.serve(block=False)
    try:
        client = PolicyClient(host="127.0.0.1", port=port)
        try:
            action = client.predict_action(
                [{"task": "pick"}],
                policy_kwargs={},
                timeout=5,
            )
            np.testing.assert_array_equal(action, np.zeros((1, 1), dtype=np.float32))
            assert client.reset(timeout=5) is True
            assert handler.reset_count == 1
        finally:
            client.close()
    finally:
        server.stop(grace=0)


def test_server_rejects_raw_request_with_mismatched_episode_ids() -> None:
    handler = SimpleHandler()
    port = _free_port()
    server = PolicyServer(handler, host="127.0.0.1", port=port)
    server.serve(block=False)
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = policy_service_pb2_grpc.PolicyServiceStub(channel)
        request = policy_service_pb2.PredictRequest(
            observations=[policy_service_pb2.Observation()],
            episode_ids=["slot-0", "slot-1"],
        )

        try:
            stub.Predict(request, timeout=5)
        except grpc.RpcError as exc:
            assert exc.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "episode_ids" in str(exc.details())
        else:  # pragma: no cover - assertion branch
            raise AssertionError("mismatched episode_ids were accepted")
    finally:
        channel.close()
        server.stop(grace=0)


def test_server_does_not_run_timed_out_request_after_policy_lock_wait() -> None:
    handler = SlowHandler()
    port = _free_port()
    server = PolicyServer(handler, host="127.0.0.1", port=port)
    server.serve(block=False)
    first_done = threading.Event()
    first_error: list[BaseException] = []

    def run_slow_request() -> None:
        client = PolicyClient(host="127.0.0.1", port=port, timeout=5)
        try:
            client.predict_action([{"task": "slow"}], episode_ids=["slow"])
        except BaseException as exc:  # pragma: no cover - diagnostic path
            first_error.append(exc)
        finally:
            client.close()
            first_done.set()

    thread = threading.Thread(target=run_slow_request)
    thread.start()
    try:
        assert handler.started.wait(timeout=5)
        client = PolicyClient(host="127.0.0.1", port=port)
        try:
            try:
                client.predict_action(
                    [{"task": "timed out"}],
                    episode_ids=["timed-out"],
                    timeout=0.05,
                )
            except grpc.RpcError as exc:
                assert exc.code() in {
                    grpc.StatusCode.CANCELLED,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                }
            else:  # pragma: no cover - assertion branch
                raise AssertionError("queued request did not time out")
        finally:
            client.close()

        time.sleep(0.2)
        assert ("timed-out",) not in handler.seen_episode_ids

        handler.release.set()
        assert first_done.wait(timeout=5)
        assert first_error == []
        thread.join(timeout=5)

        time.sleep(0.2)
        assert handler.seen_episode_ids == [("slow",)]
    finally:
        handler.release.set()
        thread.join(timeout=5)
        server.stop(grace=0)
