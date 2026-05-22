from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DTYPE_UNSPECIFIED: _ClassVar[DType]
    DTYPE_FLOAT32: _ClassVar[DType]
    DTYPE_FLOAT64: _ClassVar[DType]
    DTYPE_INT32: _ClassVar[DType]
    DTYPE_INT64: _ClassVar[DType]
    DTYPE_UINT8: _ClassVar[DType]
    DTYPE_BOOL: _ClassVar[DType]

class NullValue(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NULL_VALUE: _ClassVar[NullValue]
DTYPE_UNSPECIFIED: DType
DTYPE_FLOAT32: DType
DTYPE_FLOAT64: DType
DTYPE_INT32: DType
DTYPE_INT64: DType
DTYPE_UINT8: DType
DTYPE_BOOL: DType
NULL_VALUE: NullValue

class Tensor(_message.Message):
    __slots__ = ("data", "shape", "dtype")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: DType
    def __init__(self, data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[_Union[DType, str]] = ...) -> None: ...

class NamedTensor(_message.Message):
    __slots__ = ("key", "tensor")
    KEY_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    key: str
    tensor: Tensor
    def __init__(self, key: _Optional[str] = ..., tensor: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class StringField(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class Observation(_message.Message):
    __slots__ = ("tensors", "strings", "scalars")
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    STRINGS_FIELD_NUMBER: _ClassVar[int]
    SCALARS_FIELD_NUMBER: _ClassVar[int]
    tensors: _containers.RepeatedCompositeFieldContainer[NamedTensor]
    strings: _containers.RepeatedCompositeFieldContainer[StringField]
    scalars: _containers.RepeatedCompositeFieldContainer[ValueField]
    def __init__(self, tensors: _Optional[_Iterable[_Union[NamedTensor, _Mapping]]] = ..., strings: _Optional[_Iterable[_Union[StringField, _Mapping]]] = ..., scalars: _Optional[_Iterable[_Union[ValueField, _Mapping]]] = ...) -> None: ...

class Value(_message.Message):
    __slots__ = ("null_value", "bool_value", "int_value", "float_value", "string_value", "list_value", "struct_value")
    NULL_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    LIST_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRUCT_VALUE_FIELD_NUMBER: _ClassVar[int]
    null_value: NullValue
    bool_value: bool
    int_value: int
    float_value: float
    string_value: str
    list_value: ValueList
    struct_value: ValueStruct
    def __init__(self, null_value: _Optional[_Union[NullValue, str]] = ..., bool_value: bool = ..., int_value: _Optional[int] = ..., float_value: _Optional[float] = ..., string_value: _Optional[str] = ..., list_value: _Optional[_Union[ValueList, _Mapping]] = ..., struct_value: _Optional[_Union[ValueStruct, _Mapping]] = ...) -> None: ...

class ValueList(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[Value]
    def __init__(self, values: _Optional[_Iterable[_Union[Value, _Mapping]]] = ...) -> None: ...

class ValueStruct(_message.Message):
    __slots__ = ("fields",)
    class FieldsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.MessageMap[str, Value]
    def __init__(self, fields: _Optional[_Mapping[str, Value]] = ...) -> None: ...

class ValueField(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: Value
    def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ("observations", "policy_kwargs", "episode_ids")
    class PolicyKwargsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    POLICY_KWARGS_FIELD_NUMBER: _ClassVar[int]
    EPISODE_IDS_FIELD_NUMBER: _ClassVar[int]
    observations: _containers.RepeatedCompositeFieldContainer[Observation]
    policy_kwargs: _containers.MessageMap[str, Value]
    episode_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, observations: _Optional[_Iterable[_Union[Observation, _Mapping]]] = ..., policy_kwargs: _Optional[_Mapping[str, Value]] = ..., episode_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("action",)
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: Tensor
    def __init__(self, action: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("episode_ids",)
    EPISODE_IDS_FIELD_NUMBER: _ClassVar[int]
    episode_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, episode_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("ready", "model_info")
    READY_FIELD_NUMBER: _ClassVar[int]
    MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    model_info: str
    def __init__(self, ready: bool = ..., model_info: _Optional[str] = ...) -> None: ...
