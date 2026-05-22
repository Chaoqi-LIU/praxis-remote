# praxis-remote

`praxis-remote` is a standalone gRPC transport package for policy inference. It
does not depend on Praxis training code or benchmark environments. It defines a
small generic contract:

- A client sends a batch of observations.
- A server-side policy handler returns an action tensor.
- Optional `episode_ids` let stateful policies keep vectorized rollout lanes
  independent.
- Optional `policy_kwargs` carry request-level inference options.

The package is intentionally about transport and serialization only.
Application-specific observation semantics and policy adaptation live in the
caller.

## Install

Install from PyPI:

```bash
pip install praxis-remote==0.1.1
```

## Observation Contract

An observation is a mapping from string keys to supported values:

```python
{
    "task": "put the mug on the plate",
    "observation.images.front": front_rgb,
    "observation.state": proprio,
    "metadata.episode_index": 7,
}
```

Supported observation values:

- `numpy.ndarray`
- `str`
- `bool`
- `int`
- `float`
- numpy scalar values

Policy kwargs support `None`, booleans, integers, floats, strings, numpy scalar
values, lists, and dictionaries composed from those types.

Supported tensor dtypes:

- `float32`
- `float64`
- `int32`
- `int64`
- `uint8`
- `bool`

The transport preserves keys exactly after string conversion. The caller and
policy adapter are responsible for agreeing on the meaning of those keys.

## Serving a Policy

Implement the `PolicyHandler` protocol and pass it to `PolicyServer`:

```python
import numpy as np
from praxis_remote import PolicyServer


class MyHandler:
    def predict_action(self, observations, *, policy_kwargs=None, episode_ids=None):
        del policy_kwargs, episode_ids
        return np.zeros((len(observations), 7), dtype=np.float32)

    def reset(self, episode_ids=None):
        del episode_ids

    def model_info(self):
        return "my-policy"


PolicyServer(MyHandler(), host="0.0.0.0", port=50051).serve()
```

`PolicyServer` serializes requests, validates basic transport shape, and guards
policy calls with a lock so one handler instance is not called concurrently.
Use separate server processes if your policy stack needs independent workers.

`host="0.0.0.0"` binds the server on all network interfaces. This is useful when
another process, container, or node needs to connect to it. Use
`host="127.0.0.1"` when the server should accept only same-machine connections.

## Calling a Policy

Use `PolicyClient.predict_action` for an explicit observation batch:

```python
from praxis_remote import PolicyClient

with PolicyClient(host="127.0.0.1", port=50051, timeout=30.0) as client:
    action = client.predict_action(
        [
            {
                "task": "pick up the cube",
                "observation.state": state,
            }
        ],
        episode_ids=["env-0"],
        policy_kwargs={"temperature": 0.0},
    )
```

`127.0.0.1` is correct when the client runs on the same machine as the server,
or when a remote server is exposed through an SSH tunnel or port-forward. If the
server is running on another reachable host, pass that hostname or IP address
instead.

Use `predict_action_from_parts` for a single common image/state/text
observation:

```python
action = client.predict_action_from_parts(
    state=state,
    images={"front": front_rgb, "wrist": wrist_rgb},
    task_description="pick up the cube",
    episode_id="env-0",
)
```

Check server readiness:

```python
ready, model_info = client.health_check()
```

Reset all policy state or selected rollout lanes:

```python
client.reset()
client.reset(episode_ids=["env-0", "env-1"])
```

## Using with praxis-eval

[`praxis-eval`](https://github.com/Chaoqi-LIU/praxis-eval) exposes
`RemotePolicy`, which wraps `PolicyClient` behind its generic eval policy
protocol:

```python
from praxis_eval import EvalConfig, RemotePolicy, evaluate

result = evaluate(
    "libero",
    policy=RemotePolicy("127.0.0.1:50051", timeout=30.0),
    config=EvalConfig(
        task="libero_spatial",
        num_eval_per_task=10,
        output_dir="eval/libero_spatial",
    ),
)
```

The remote server does not need to import `praxis-eval` unless the policy
adapter chooses to reuse its types. It only needs to implement
`PolicyHandler.predict_action`.

## Protocol

The gRPC schema is in `src/praxis_remote/proto/policy_service.proto`. Generated
Python files are committed so users do not need `protoc` for normal installs.

Service methods:

- `Predict`: batch observations plus policy kwargs to one action tensor.
- `Reset`: reset all state or selected `episode_ids`.
- `HealthCheck`: readiness and short model information string.

The default gRPC send and receive message limit is 64 MiB. Keep image batches
and rollout parallelism sized accordingly, or run multiple servers for larger
throughput.

## Repository Structure

```text
src/praxis_remote/
  client.py               # PolicyClient
  server.py               # PolicyServer and gRPC servicer
  protocol.py             # PolicyHandler protocol
  serialization.py        # observation, kwarg, and numpy tensor codecs
  proto/
    policy_service.proto  # source gRPC schema
    *_pb2*.py             # generated protobuf/grpc modules
tests/                    # serialization and client/server tests
```

## Development Checks

Run the same checks as CI:

```bash
uv run --extra dev pre-commit run check-license-headers --all-files
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev pytest --strict-markers -m "not manual"
uv build --sdist --wheel
```

Regenerate protobuf files after changing `policy_service.proto`:

```bash
uv run python -m grpc_tools.protoc \
  -Isrc/praxis_remote/proto \
  --python_out=src/praxis_remote/proto \
  --pyi_out=src/praxis_remote/proto \
  --grpc_python_out=src/praxis_remote/proto \
  src/praxis_remote/proto/policy_service.proto
```

## License and Attribution

`praxis-remote` is licensed under Apache-2.0. Redistribution must preserve the
license, copyright notices, and `NOTICE` file. If this package supports your
research or product work, please cite the repository using `CITATION.cff`.
