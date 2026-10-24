# Development and release

The public repository is [mit-han-lab/cti-kernel](https://github.com/mit-han-lab/cti-kernel).
Open issues and pull requests there. Wheels are published as
[GitHub releases](https://github.com/mit-han-lab/cti-kernel/releases); there is no
package-index publication.

## Contributing

Development tools use Python 3.12. The installed package supports Python 3.10
and newer, as declared in `pyproject.toml`. Install with the `dev` extra and run
the CPU tests and the pre-commit hooks before opening a pull request:

```bash
python -m pip install --no-build-isolation -e '.[dev,cute]'
python -m pytest -q tests/modules tests/benchmarks tests/quality tests/ci tests/norm -m "not gpu"
pre-commit run --all-files
```

A change to a GPU operation also needs the GPU suite on a supported device
(RTX 5090 or RTX PRO 6000 Server Edition, sm_120; GB10, sm_121). The full suite
is `python -m pytest -q tests/gpu tests/norm`; add `-m "not exhaustive"` for the
shorter selection. The [cookbook](../cookbook/README.md) lists the extra
providers the GPU suite compares against. A green CPU run or a successful import
does not qualify a device or provider; report the device, driver, CUDA, Torch
and provider versions with GPU results.

## Packaging

Build an independent source archive and wheel with `python -m build`. Native JIT
sources and headers are included in both artifacts. The internal C++ namespaces
and FFI type names retain their existing identifiers; they are not Python engine
dependencies. The kernel version is defined in `pyproject.toml`. Kernel releases
are versioned independently: each engine release selects its own kernel package
version, and kernel releases do not require engine updates or matching version
numbers. Each GitHub release lists the SHA256 of its wheel.
