# Contributing

## Development Setup

```bash
git clone https://github.com/Chaoqi-LIU/praxis-remote.git
cd praxis-remote
uv sync --extra dev
uv run --extra dev pre-commit install
uv run --extra dev pytest --strict-markers -m "not manual"
```

## Project Boundary

`praxis-remote` owns generic remote policy transport: client, server, protocol,
and serialization. Application-specific observation/action contracts and
model-specific policy adaptation belong in higher-level packages or callers.
For robot benchmark evaluation, [`praxis-eval`](https://github.com/Chaoqi-LIU/praxis-eval)
is one such integration package.

Keep the wire format generic. Do not add benchmark names, policy names, or
Praxis training concepts to the transport protocol unless they are truly
cross-cutting transport concerns.

## License Headers

Add explicit SPDX headers to new manually maintained Python files under `src/`:

```text
SPDX-FileCopyrightText: 2027 Your Name
SPDX-License-Identifier: Apache-2.0
```

Use the current year when creating a new file. Do not update every file just
because the calendar year changed. If an existing file receives copyrightable
changes in a later year, update only that file's year range or copyright holder
when appropriate, for example:

```text
SPDX-FileCopyrightText: 2026-2027 Chaoqi Liu and contributors
SPDX-License-Identifier: Apache-2.0
```

Generated protobuf files, tests, documentation, workflow files, project config,
lockfiles, and files that should not be edited directly are licensed through the
root `LICENSE` and `NOTICE` files, but they should not carry visible SPDX
headers.

## Checks

Run the same checks as CI before sending changes:

```bash
uv run --extra dev pre-commit run check-license-headers --all-files
uv run --extra dev pre-commit run --all-files
uv run --extra dev pytest --strict-markers -m "not manual"
uv build --sdist --wheel
```
