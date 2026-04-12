# vllm-rdna4-container-patches

Runtime patches + one critical CLI flag that let [vLLM](https://github.com/vllm-project/vllm) run on a **Radeon RX 9070 XT (gfx1201 / RDNA 4)** inside an unprivileged container on stock k3s, including loading a 14B Q4\_K\_M GGUF model on 16 GB of VRAM.

> *AI disclosure: I used Claude to help write this repo up. Every patch has been verified on my own hardware (RX 9070 XT, ROCm 7.12 nightly, vLLM v0.16.0rc0). I can answer follow-ups.*

Built on top of [@bluefalcon13/vllm-rocm](https://github.com/bluefalcon13/vllm-rocm), which handles the hard part — compiling vLLM against AMD's TheRock nightly toolchain for gfx1201 with a Composable Kernel build of Flash Attention. This repo is the thin container-runtime layer that gets that image to actually start inside k3s/k8s and serve a model on a 16 GB card.

## What this fixes

If you try to run bluefalcon13's image unmodified inside a container on RDNA 4, you will hit all of the following in order:

1. **vLLM refuses to detect the GPU.** `amdsmi` fails to initialize inside the container (`AMDSMI_STATUS_NOT_INIT`), so vLLM's `rocm_platform_plugin()` never returns and the engine bails out with "no platform found." The GPU is perfectly functional via HIP — `amdsmi` just doesn't work in this environment.
2. **A circular import while resolving the GCN architecture.** Once you get past step 1, vLLM tries to call `torch.cuda.get_device_properties().gcnArchName` as a fallback, which trips `logger.warning_once()` → `vllm.distributed.parallel_state` → `vllm.utils.system_utils` → `vllm.platforms` during module init. You get `ImportError: cannot import name 'current_platform'`.
3. **`torch.cuda.device_count()` returns 0** despite the GPU working fine via HIP. vLLM's DP code then asserts `rank 0 is out of bounds` and crashes.
4. **GGUF weight-loading OOM on 16 GB.** Once vLLM finally starts and begins loading a 14B Q4\_K\_M GGUF (~9 GB on disk), `_create_padded_weight_param` allocates a temp FP16 buffer for merged/padded weights and you OOM during loading. PyTorch has ~6.8 GB allocated, needs ~96 MiB more, has ~6 MiB free. **This happens before KV cache allocation**, so the usual `--gpu-memory-utilization` and `--max-model-len` knobs do not help.

The three runtime patches in `vllm_rdna4_patches.py` fix (1)-(3). The `--cpu-offload-gb 4` flag in the reference `k8s/deployment.yaml` fixes (4).

## What's in the box

```
.
├── vllm_rdna4_patches.py   # the three runtime patches (applied at image build time)
├── Dockerfile.diff         # diff on top of bluefalcon13/vllm-rocm Dockerfile
├── k8s/deployment.yaml     # reference k8s manifest with all needed flags + env vars
└── LICENSE                 # MIT
```

## The three patches

All three are applied at image build time by running `python vllm_rdna4_patches.py` against the vLLM source tree inside the container. They modify two files:

- `/app/vllm/vllm/platforms/__init__.py`
- `/app/vllm/vllm/platforms/rocm.py`

### Patch 1 — HIP-based platform detection

Replaces the `amdsmi`-based `rocm_platform_plugin()` with a `ctypes` call to `libamdhip64.so`'s `hipGetDeviceCount`. If HIP reports at least one device, vLLM is told the ROCm platform is available.

### Patch 2 — `PYTORCH_ROCM_ARCH` env var fallback for `_get_gcn_arch()`

When `amdsmi` fails to report the GCN arch, vLLM falls back to `torch.cuda.get_device_properties()`, but `logger.warning_once()` on the way in triggers a circular import during module init. The patch does two things:

1. Checks `PYTORCH_ROCM_ARCH` **before** `warning_once` is called. If set (e.g. `gfx1201`), return it immediately.
2. Replaces `logger.warning_once()` with regular `logger.warning()` to avoid the circular import entirely if we do fall through.

### Patch 3 — `torch.cuda.device_count()` monkey-patch via HIP

Injects a `_patch_torch_device_count()` function into `rocm.py` that runs at module import time. If `torch.cuda.device_count()` returns 0, it loads `libamdhip64.so` directly, calls `hipGetDeviceCount()`, and overwrites `torch.cuda.device_count` (and `torch.accelerator.device_count` if present) to return the HIP-reported count.

**⚠️ Known side-effect:** this is a global mutation of `torch.cuda.device_count` that persists for the lifetime of the Python process. Anything else running in the same process that imports torch and expects the original function will see the patched version. For vLLM itself and for spawned subprocesses (EngineCore, workers) this is intentional and correct. For more general use, be aware.

## The GGUF OOM workaround

Even with the platform-detection patches applied and vLLM starting cleanly, loading a 14B Q4\_K\_M GGUF on a 16 GB card fails with:

```
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 96.00 MiB.
GPU 0 has a total capacity of 15.92 GiB of which 5.71 MiB is free.
Of the allocated memory 6.84 GiB is allocated by PyTorch, ...
```

The ~9 GB gap between "total capacity" and "PyTorch allocated" is ROCm runtime overhead + PyTorch caching allocator reserved blocks + HIP context. The OOM itself happens inside `_create_padded_weight_param`, which allocates a FP16 buffer for the merged/padded weight before the quantized data is loaded into it. This is **during weight loading**, not during KV cache allocation, so the usual memory knobs do not help:

- `--gpu-memory-utilization 0.5` — doesn't help (KV cache hasn't been allocated yet)
- `--max-model-len 1024` — doesn't help (same reason)
- `--enforce-eager` — doesn't help (same reason)

**What works: `--cpu-offload-gb 4`.** vLLM's [`OffloadConfig`](https://github.com/vllm-project/vllm/blob/main/vllm/config/offload.py) in v0.16 exposes a UVA (Unified Virtual Addressing) zero-copy offload path via this CLI flag. It tells vLLM to keep 4 GB of weights in pinned CPU RAM and stream them to GPU on each forward pass. That frees exactly enough headroom for the weight-loading spike.

With `--cpu-offload-gb 4`, on my hardware:

```
Model loading took 8.75 GiB memory and 24.077499 seconds
INFO ... Application startup complete.
```

And end-to-end inference through the OpenAI-compatible endpoint works correctly.

This is likely related to the `torch.tensor()` vs `torch.from_numpy()` GGUF loader regression tracked in [vllm-project/vllm#22814](https://github.com/vllm-project/vllm/issues/22814). That issue is about *system RAM* bloat during GGUF load; I hit a symptom with the same shape in *GPU VRAM*. I have not confirmed the root cause is the same.

## How to use this

### 1. Build the image

Clone bluefalcon13's repo, apply this repo's `Dockerfile.diff`, drop `vllm_rdna4_patches.py` next to the Dockerfile, and build:

```bash
git clone https://github.com/bluefalcon13/vllm-rocm.git
cd vllm-rocm
curl -O https://raw.githubusercontent.com/sleeepss/vllm-rdna4-container-patches/main/Dockerfile.diff
curl -O https://raw.githubusercontent.com/sleeepss/vllm-rdna4-container-patches/main/vllm_rdna4_patches.py
git apply Dockerfile.diff
podman build -t localhost/vllm-rocm-gfx1201:latest .
```

### 2. Verify the patches applied

```bash
podman run --rm --entrypoint /bin/bash \
  --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  localhost/vllm-rocm-gfx1201:latest -l -c \
  "/app/.venv/bin/vllm serve --help 2>&1 | tail -5"
```

You should see `rocm.py` emit these two log lines at startup:

```
INFO [rocm.py:176] Using PYTORCH_ROCM_ARCH=gfx1201 for GCN arch
INFO [rocm.py:206] Patched torch device_count to 1 via HIP
```

### 3. Deploy on k3s/k8s

See `k8s/deployment.yaml` for the full manifest. Key points:

- `PYTORCH_ROCM_ARCH=gfx1201` must be set. The GCN arch patch depends on it.
- **Do not name your Service `vllm`.** k8s auto-injects `VLLM_PORT` env vars from the Service, which collide with vLLM's own config and crash the engine. Name it `llm-api` or similar.
- `--cpu-offload-gb 4` is required for 14B GGUF models on 16 GB cards.
- `--enforce-eager` — CUDA graphs consume extra VRAM and we need every byte.
- `--dtype float16` — GGUF does not support bfloat16.
- Drop the liveness probe; use `/ping` as the readiness path (cheap HTTP 200 that doesn't touch the engine).

## What this does not claim to do

- **This does not give you FP8 WMMA performance.** The patches that unlock RDNA 4's FP8 hardware path are discussed at length in [vllm-project/vllm#28649](https://github.com/vllm-project/vllm/issues/28649) and the [vLLM forum](https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900). Those are separate changes. This repo is about getting the engine to *start* and *load a model*. You can (and should) layer the FP8 patches on top.
- **This is not a clean upstream PR.** The three runtime patches are monkey-patches over the vLLM source tree. They work, they're minimal, they're documented, but they are not the shape maintainers want code merged in. If someone with vLLM-internals chops wants to reshape them into a proper platform-detection refactor, please do.
- **This is not tested on any GPU other than RX 9070 XT.** I only own one RDNA 4 card. It should work on R9700 / gfx1200 variants with minimal changes, but I have not verified.

## Credits

- [@bluefalcon13](https://github.com/bluefalcon13) — for [vllm-rocm](https://github.com/bluefalcon13/vllm-rocm), which is the actual build and without which none of this exists.
- [@hyoon1](https://github.com/hyoon1) — for the `enable-ck-gfx12` branch of Flash Attention.
- **AMD ROCm team** — for the TheRock nightly toolchain that makes gfx1201 buildable at all.
- **vLLM maintainers** — for the `--cpu-offload-gb` UVA path, which turned out to be exactly what a 16 GB consumer card with a 14B GGUF needs.

## License

MIT (matching bluefalcon13's upstream). See `LICENSE`.
