from __future__ import annotations

import numpy as np

from praxis_remote.serialization import (
    decode_observations,
    decode_policy_kwargs,
    encode_observations,
    encode_policy_kwargs,
)


def test_observation_roundtrip_preserves_arrays_and_strings() -> None:
    observations = [
        {
            "task": "pick the cube",
            "observation.state": np.arange(3, dtype=np.float32),
            "observation.images.front": np.zeros((2, 3, 4), dtype=np.uint8),
        }
    ]

    decoded = decode_observations(encode_observations(observations))

    assert decoded[0]["task"] == "pick the cube"
    np.testing.assert_array_equal(
        decoded[0]["observation.state"],
        observations[0]["observation.state"],
    )
    np.testing.assert_array_equal(
        decoded[0]["observation.images.front"],
        observations[0]["observation.images.front"],
    )


def test_observation_roundtrip_preserves_scalar_metadata() -> None:
    observations = [
        {
            "metadata.done": False,
            "metadata.episode_index": 7,
            "metadata.score": 1.25,
            "metadata.numpy_bool": np.bool_(True),
            "metadata.numpy_int": np.int64(8),
            "metadata.numpy_float": np.float32(2.5),
        }
    ]

    decoded = decode_observations(encode_observations(observations))[0]

    assert decoded == {
        "metadata.done": False,
        "metadata.episode_index": 7,
        "metadata.score": 1.25,
        "metadata.numpy_bool": True,
        "metadata.numpy_int": 8,
        "metadata.numpy_float": 2.5,
    }


def test_policy_kwargs_roundtrip_nested_values() -> None:
    kwargs = {
        "decode_keep_k": 4,
        "strict": True,
        "nested": {"temperature": 0.7, "tags": ["a", "b"], "none": None},
    }

    decoded = decode_policy_kwargs(encode_policy_kwargs(kwargs))

    assert decoded == kwargs


def test_policy_kwargs_rejects_nested_duplicate_keys_after_string_conversion() -> None:
    try:
        encode_policy_kwargs({"nested": {1: "a", "1": "b"}})
    except ValueError as exc:
        assert "duplicate nested key '1'" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("duplicate nested policy kwarg keys were accepted")


def test_build_observation_prefixes_image_camera_names() -> None:
    from praxis_remote.serialization import build_observation

    observation = build_observation(
        images={
            "front": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation.images.wrist": np.ones((2, 2, 3), dtype=np.uint8),
        },
        task_description="pick",
    )

    assert set(observation) == {
        "observation.images.front",
        "observation.images.wrist",
        "task",
    }
