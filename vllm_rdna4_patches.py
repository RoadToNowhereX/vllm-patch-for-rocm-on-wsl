#!/usr/bin/env python3
"""
vLLM RDNA 4 (gfx1201) compatibility patches.

On RDNA 4, amdsmi fails to initialize in containers (AMDSMI_STATUS_NOT_INIT)
and torch.cuda.device_count() returns 0 despite the GPU being fully functional
via HIP. These patches:

1. Replace amdsmi-based platform detection with HIP-based detection
2. Add PYTORCH_ROCM_ARCH env var fallback for GCN architecture lookup
3. Monkey-patch torch.cuda.device_count / torch.accelerator.device_count via HIP
"""
import os
import shutil


def patch_platform_detection():
    """Replace amdsmi platform detection with HIP ctypes-based detection."""
    path = '/app/vllm/vllm/platforms/__init__.py'
    with open(path) as f:
        lines = f.readlines()

    new_lines = []
    skip = False
    patched = False

    for i, line in enumerate(lines):
        if 'def rocm_platform_plugin' in line and not skip:
            skip = True
            patched = True
            new_lines.extend([
                'def rocm_platform_plugin() -> str | None:\n',
                '    logger.debug("Checking if ROCm platform is available.")\n',
                '    try:\n',
                '        import ctypes\n',
                '        hip = ctypes.CDLL("libamdhip64.so")\n',
                '        count = ctypes.c_int()\n',
                '        result = hip.hipGetDeviceCount(ctypes.byref(count))\n',
                '        if result == 0 and count.value > 0:\n',
                '            logger.debug("ROCm platform available via HIP.")\n',
                '            return "vllm.platforms.rocm.RocmPlatform"\n',
                '        logger.debug("ROCm not available via HIP")\n',
                '    except Exception as e:\n',
                '        logger.debug("ROCm not available: " + str(e))\n',
                '    return None\n',
            ])
            continue

        if skip:
            # End of function: next non-indented, non-empty line
            if line.strip() == '':
                # Check if next line is top-level
                if i + 1 < len(lines) and lines[i + 1] and not lines[i + 1][0].isspace():
                    skip = False
                    new_lines.append('\n')
            elif not line[0].isspace():
                skip = False
                new_lines.append(line)
            continue

        new_lines.append(line)

    with open(path, 'w') as f:
        f.writelines(new_lines)

    return patched


def patch_gcn_arch_fallback():
    """Add PYTORCH_ROCM_ARCH env var fallback for GCN architecture lookup.

    Must patch BEFORE logger.warning_once() which triggers a circular import
    back to vllm.platforms during module init.
    """
    path = '/app/vllm/vllm/platforms/rocm.py'
    with open(path) as f:
        content = f.read()

    # Add os import if not present
    if '\nimport os\n' not in content:
        content = content.replace('import logging\n', 'import logging\nimport os\n', 1)

    # Replace the entire except block: put env var check before warning_once
    # to avoid the circular import that warning_once triggers
    old_block = (
        '    except Exception as e:\n'
        '        logger.debug("Failed to get GCN arch via amdsmi: %s", e)\n'
        '        logger.warning_once(\n'
        '            "Failed to get GCN arch via amdsmi, falling back to torch.cuda. "\n'
        '            "This will initialize CUDA and may cause "\n'
        '            "issues if CUDA_VISIBLE_DEVICES is not set yet."\n'
        '        )\n'
        '    # Ultimate fallback: use torch.cuda (will initialize CUDA)\n'
        '    return torch.cuda.get_device_properties("cuda").gcnArchName'
    )
    new_block = (
        '    except Exception as e:\n'
        '        logger.debug("Failed to get GCN arch via amdsmi: %s", e)\n'
        '    # RDNA 4 workaround: env var fallback before torch.cuda\n'
        '    # (avoids circular import from logger.warning_once during module init)\n'
        '    arch_env = os.environ.get("PYTORCH_ROCM_ARCH", "")\n'
        '    if arch_env:\n'
        '        logger.info("Using PYTORCH_ROCM_ARCH=%s for GCN arch", arch_env)\n'
        '        return arch_env\n'
        '    logger.warning(\n'
        '        "Failed to get GCN arch via amdsmi, falling back to torch.cuda. "\n'
        '        "This will initialize CUDA and may cause "\n'
        '        "issues if CUDA_VISIBLE_DEVICES is not set yet."\n'
        '    )\n'
        '    return torch.cuda.get_device_properties("cuda").gcnArchName'
    )

    patched = old_block in content
    content = content.replace(old_block, new_block)

    with open(path, 'w') as f:
        f.write(content)

    return patched


def patch_torch_device_count():
    """Inject HIP-based monkey-patch for torch.cuda.device_count into rocm.py.

    torch.cuda.device_count() returns 0 on RDNA 4 in containers, but HIP
    hipGetDeviceCount works. This injects a monkey-patch at module load time
    in rocm.py so that all vLLM code (including subprocesses) gets the fix.
    """
    path = '/app/vllm/vllm/platforms/rocm.py'
    with open(path) as f:
        content = f.read()

    # Inject the monkey-patch right after _GCN_ARCH is resolved
    marker = '_GCN_ARCH = _get_gcn_arch()'
    patch_code = (
        '_GCN_ARCH = _get_gcn_arch()\n'
        '\n'
        '# RDNA 4 workaround: torch.cuda.device_count() returns 0 but HIP works.\n'
        '# Monkey-patch torch to use HIP device count globally.\n'
        'def _patch_torch_device_count():\n'
        '    import torch\n'
        '    if torch.cuda.device_count() == 0:\n'
        '        try:\n'
        '            import ctypes\n'
        '            _hip = ctypes.CDLL("libamdhip64.so")\n'
        '            _count = ctypes.c_int()\n'
        '            _result = _hip.hipGetDeviceCount(ctypes.byref(_count))\n'
        '            if _result == 0 and _count.value > 0:\n'
        '                _n = _count.value\n'
        '                torch.cuda.device_count = lambda: _n\n'
        '                if hasattr(torch, "accelerator"):\n'
        '                    torch.accelerator.device_count = lambda: _n\n'
        '                logger.info("Patched torch device_count to %d via HIP", _n)\n'
        '        except Exception as e:\n'
        '            logger.debug("HIP device_count patch failed: %s", e)\n'
        '\n'
        '_patch_torch_device_count()\n'
    )

    patched = marker in content
    content = content.replace(marker, patch_code, 1)

    with open(path, 'w') as f:
        f.write(content)

    return patched


def clear_pycache():
    """Remove compiled bytecache so patches take effect."""
    for d in [
        '/app/vllm/vllm/platforms/__pycache__',
        '/app/vllm/vllm/v1/worker/__pycache__',
    ]:
        if os.path.exists(d):
            shutil.rmtree(d)


if __name__ == '__main__':
    p1 = patch_platform_detection()
    p2 = patch_gcn_arch_fallback()
    p3 = patch_torch_device_count()
    clear_pycache()
    print(f'Platform detection (HIP): {"PATCHED" if p1 else "FAILED"}')
    print(f'GCN arch fallback (env): {"PATCHED" if p2 else "FAILED"}')
    print(f'torch.device_count (HIP): {"PATCHED" if p3 else "FAILED"}')
