# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Serialization helpers for generic remote policy inference."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np

from praxis_remote.proto import policy_service_pb2

NumpyScalar: TypeAlias = np.bool_ | np.integer[Any] | np.floating[Any]
ObservationValue: TypeAlias = np.ndarray | str | bool | int | float | NumpyScalar
Observation: TypeAlias = dict[str, ObservationValue]
NumpyPolicyKwargScalar: TypeAlias = NumpyScalar
PolicyKwargValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | NumpyPolicyKwargScalar
    | list["PolicyKwargValue"]
    | dict[str, "PolicyKwargValue"]
)

_NUMPY_DTYPE_TO_PROTO = {
    np.dtype(np.float32): policy_service_pb2.DTYPE_FLOAT32,
    np.dtype(np.float64): policy_service_pb2.DTYPE_FLOAT64,
    np.dtype(np.int32): policy_service_pb2.DTYPE_INT32,
    np.dtype(np.int64): policy_service_pb2.DTYPE_INT64,
    np.dtype(np.uint8): policy_service_pb2.DTYPE_UINT8,
    np.dtype(np.bool_): policy_service_pb2.DTYPE_BOOL,
}
_PROTO_DTYPE_TO_NUMPY = {
    proto_dtype: numpy_dtype
    for numpy_dtype, proto_dtype in _NUMPY_DTYPE_TO_PROTO.items()
}


def build_observation(
    *,
    state: np.ndarray | None = None,
    images: Mapping[str, np.ndarray] | None = None,
    task_description: str | None = None,
) -> Observation:
    """Build a single observation dict from common image/state/text inputs."""
    observation: Observation = {}

    if state is not None:
        observation["observation.state"] = np.asarray(state)

    if images is not None:
        for key, value in images.items():
            image_key = str(key)
            if not image_key.startswith("observation."):
                image_key = f"observation.images.{image_key}"
            observation[image_key] = np.asarray(value)

    if task_description is not None:
        observation["task"] = str(task_description)

    return observation


def encode_observations(
    observations: Sequence[Mapping[str, ObservationValue]],
) -> list[policy_service_pb2.Observation]:
    """Serialize a batch of observations into proto messages."""
    encoded: list[policy_service_pb2.Observation] = []
    for observation in observations:
        obs_msg = policy_service_pb2.Observation()
        encoded_keys: set[str] = set()
        for raw_key, value in observation.items():
            key = str(raw_key)
            if key in encoded_keys:
                raise ValueError(
                    f"Observation contains duplicate key {key!r} after string conversion."
                )
            encoded_keys.add(key)

            if isinstance(value, str):
                obs_msg.strings.append(
                    policy_service_pb2.StringField(key=key, value=value)
                )
                continue

            if isinstance(value, (bool, int, float, np.generic)):
                obs_msg.scalars.append(
                    policy_service_pb2.ValueField(
                        key=key,
                        value=encode_observation_scalar(value),
                    )
                )
                continue

            arr = np.asarray(value)
            obs_msg.tensors.append(
                policy_service_pb2.NamedTensor(
                    key=key,
                    tensor=encode_numpy_array(arr),
                )
            )
        encoded.append(obs_msg)
    return encoded


def decode_observations(
    observations: Sequence[policy_service_pb2.Observation],
) -> list[Observation]:
    """Deserialize proto observations back into Python observation dicts."""
    decoded: list[Observation] = []
    for observation in observations:
        item: Observation = {}
        for tensor_field in observation.tensors:
            if tensor_field.key in item:
                raise ValueError(
                    f"Duplicate observation key {tensor_field.key!r} in PredictRequest."
                )
            item[tensor_field.key] = decode_numpy_array(tensor_field.tensor)
        for string_field in observation.strings:
            if string_field.key in item:
                raise ValueError(
                    f"Duplicate observation key {string_field.key!r} in PredictRequest."
                )
            item[string_field.key] = string_field.value
        for scalar_field in observation.scalars:
            if scalar_field.key in item:
                raise ValueError(
                    f"Duplicate observation key {scalar_field.key!r} in PredictRequest."
                )
            item[scalar_field.key] = decode_observation_scalar(scalar_field.value)
        decoded.append(item)
    return decoded


def encode_observation_scalar(value: ObservationValue) -> policy_service_pb2.Value:
    """Serialize one non-string scalar observation value."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return policy_service_pb2.Value(bool_value=value)
    if isinstance(value, int):
        return policy_service_pb2.Value(int_value=value)
    if isinstance(value, float):
        return policy_service_pb2.Value(float_value=value)
    raise TypeError(
        f"Unsupported observation scalar type {type(value)!r} for remote transport."
    )


def decode_observation_scalar(value: policy_service_pb2.Value) -> bool | int | float:
    """Deserialize one scalar observation value."""
    kind = value.WhichOneof("kind")
    if kind == "bool_value":
        return bool(value.bool_value)
    if kind == "int_value":
        return int(value.int_value)
    if kind == "float_value":
        return float(value.float_value)
    raise TypeError(f"Observation scalar Value cannot use kind {kind!r}.")


def encode_policy_kwargs(
    policy_kwargs: Mapping[str, PolicyKwargValue] | None,
) -> dict[str, policy_service_pb2.Value]:
    """Serialize predict_action kwargs into proto values."""
    if policy_kwargs is None:
        return {}
    encoded: dict[str, policy_service_pb2.Value] = {}
    for raw_key, value in policy_kwargs.items():
        key = str(raw_key)
        if key in encoded:
            raise ValueError(
                f"Policy kwargs contain duplicate key {key!r} after string conversion."
            )
        encoded[key] = encode_policy_kwarg_value(value)
    return encoded


def decode_policy_kwargs(
    policy_kwargs: Mapping[str, policy_service_pb2.Value],
) -> dict[str, PolicyKwargValue]:
    """Deserialize request kwargs into native Python values."""
    return {
        str(key): decode_policy_kwarg_value(value)
        for key, value in policy_kwargs.items()
    }


def encode_numpy_array(array: np.ndarray) -> policy_service_pb2.Tensor:
    """Serialize a numpy array with dtype metadata."""
    arr = np.asarray(array)
    dtype = arr.dtype
    proto_dtype = _NUMPY_DTYPE_TO_PROTO.get(dtype)
    if proto_dtype is None:
        raise TypeError(
            f"Unsupported tensor dtype {dtype!r} for remote serving transport."
        )
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return policy_service_pb2.Tensor(
        data=arr.tobytes(order="C"),
        shape=list(arr.shape),
        dtype=proto_dtype,
    )


def decode_numpy_array(tensor: policy_service_pb2.Tensor) -> np.ndarray:
    """Deserialize a proto tensor into a numpy array."""
    dtype = _PROTO_DTYPE_TO_NUMPY.get(tensor.dtype)
    if dtype is None:
        raise TypeError(
            f"Unsupported tensor dtype enum {tensor.dtype!r} in remote serving transport."
        )
    array = np.frombuffer(tensor.data, dtype=dtype).copy()
    shape = tuple(int(dim) for dim in tensor.shape)
    if any(dim < 0 for dim in shape):
        raise ValueError(f"Tensor shape cannot contain negative dimensions: {shape}.")
    expected_size = math.prod(shape) if shape else 1
    if array.size != expected_size:
        raise ValueError(
            f"Tensor data has {array.size} elements but shape {shape} requires "
            f"{expected_size}."
        )
    return array.reshape(shape)


def encode_policy_kwarg_value(value: PolicyKwargValue) -> policy_service_pb2.Value:
    """Serialize one policy kwarg value."""
    if isinstance(value, np.generic):
        value = value.item()

    if value is None:
        return policy_service_pb2.Value(null_value=policy_service_pb2.NULL_VALUE)
    if isinstance(value, bool):
        return policy_service_pb2.Value(bool_value=value)
    if isinstance(value, int):
        return policy_service_pb2.Value(int_value=value)
    if isinstance(value, float):
        return policy_service_pb2.Value(float_value=value)
    if isinstance(value, str):
        return policy_service_pb2.Value(string_value=value)
    if isinstance(value, Mapping):
        struct_value = policy_service_pb2.ValueStruct()
        encoded_keys: set[str] = set()
        for raw_key, item in value.items():
            key = str(raw_key)
            if key in encoded_keys:
                raise ValueError(
                    f"Policy kwargs contain duplicate nested key {key!r} "
                    "after string conversion."
                )
            encoded_keys.add(key)
            struct_value.fields[key].CopyFrom(encode_policy_kwarg_value(item))
        return policy_service_pb2.Value(struct_value=struct_value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        list_value = policy_service_pb2.ValueList()
        list_value.values.extend(encode_policy_kwarg_value(item) for item in value)
        return policy_service_pb2.Value(list_value=list_value)

    raise TypeError(
        f"Unsupported policy kwarg value type {type(value)!r} for remote serving transport."
    )


def decode_policy_kwarg_value(value: policy_service_pb2.Value) -> PolicyKwargValue:
    """Deserialize one policy kwarg value."""
    kind = value.WhichOneof("kind")
    if kind == "null_value":
        return None
    if kind == "bool_value":
        return bool(value.bool_value)
    if kind == "int_value":
        return int(value.int_value)
    if kind == "float_value":
        return float(value.float_value)
    if kind == "string_value":
        return str(value.string_value)
    if kind == "list_value":
        return [decode_policy_kwarg_value(item) for item in value.list_value.values]
    if kind == "struct_value":
        return {
            str(key): decode_policy_kwarg_value(item)
            for key, item in value.struct_value.fields.items()
        }
    raise TypeError("Policy kwarg Value message is missing a populated kind.")
