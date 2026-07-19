# vLLM WSL2 ROCm patch

`vllm_patch.py` is a source patcher for starting vLLM with an AMD GPU under
WSL2 when `amdsmi` cannot initialize even though HIP can access the GPU. Its
target source layout is vLLM **v0.25.1**.

The default target is **gfx1151 (RDNA 3.5)**. Set `PYTORCH_ROCM_ARCH` before
starting vLLM to override that default for another GPU architecture.

## What it changes

The script updates the installed vLLM package in two locations:

- `vllm/platforms/__init__.py`: preserves the normal amdsmi ROCm platform
  check and falls back to `hipGetDeviceCount` from `libamdhip64.so` on WSL2.
- `vllm/platforms/rocm.py`: uses `PYTORCH_ROCM_ARCH`, or `gfx1151` when it is
  unset, before vLLM attempts to initialize amdsmi for the GCN architecture.
  It also patches v0.25.1's `_rocm_device_count_stateless()` to use HIP when
  its amdsmi-backed counter does not find a device, and falls back to PyTorch
  device properties for a device-name query that amdsmi cannot satisfy.

This is a local workaround, not an upstream vLLM change. It modifies installed
source files and therefore needs a writable vLLM installation.

## Usage in WSL2

Run the script with the same Python environment that contains vLLM. Provide
the vLLM checkout root or installed package directory explicitly when automatic
discovery cannot find it:

```bash
python vllm_patch.py --vllm-root /path/to/vllm
```

For an installed package, `--vllm-root` may point directly to the directory
containing `platforms/`, for example a virtual environment's
`site-packages/vllm` directory. Alternatively, set `VLLM_ROOT` to either form
of path.

To inspect whether the expected patch locations exist without changing files:

```bash
python vllm_patch.py --vllm-root /path/to/vllm --dry-run
```

The script creates one-time backups next to changed files with the suffix
`.wsl2-amdsmi.bak` and clears the affected Python bytecode caches.

## Architecture override

`gfx1151` is used only when amdsmi fails and no architecture is provided by
the environment. To select another architecture, set the variable before
starting vLLM:

```bash
export PYTORCH_ROCM_ARCH=gfx1151
vllm serve /path/to/model
```

Use `HIP_VISIBLE_DEVICES` to restrict visibility on current ROCm vLLM builds;
do not introduce `CUDA_VISIBLE_DEVICES` unless another component requires it.

## Scope and limitations

- This repository contains no Docker or Kubernetes deployment files.
- The patch expects the relevant ROCm library, `libamdhip64.so`, to be visible
  to the WSL2 process.
- The patcher stops with a non-zero exit code if it cannot recognize the v0.25.1
  target source layout; this is preferable to writing a partial patch for an
  incompatible vLLM release.
- Patches are marked in the target source and can be run again without adding
  duplicate device-count code. A legacy pre-v0.25 global torch device-count
  patch is removed automatically before the v0.25.1 patch is installed.

## License

MIT. See [LICENSE](LICENSE).
