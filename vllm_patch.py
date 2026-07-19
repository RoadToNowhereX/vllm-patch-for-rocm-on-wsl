#!/usr/bin/env python3
"""Apply WSL2 ROCm compatibility patches to an installed vLLM package.

WSL2 can expose a usable HIP device while amdsmi cannot initialize.  Some
vLLM releases use amdsmi during ROCm platform discovery and GCN architecture
resolution, which prevents vLLM from starting.  This script changes the
installed vLLM source to use HIP for platform detection and to use gfx1151 as
the architecture fallback when PYTORCH_ROCM_ARCH is not explicitly set.

The script changes two files in the target package:

* vllm/platforms/__init__.py
* vllm/platforms/rocm.py

Pass --vllm-root with either a vLLM checkout root or the installed ``vllm``
package directory.  If it is omitted, VLLM_ROOT, the current directory, and
the Python environment's installed vLLM package are considered in that order.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path


DEFAULT_GCN_ARCH = "gfx1151"
PLATFORM_MARKER = "WSL2_AMDSMI_HIP_PLATFORM_PATCH"
ARCH_MARKER = "WSL2_AMDSMI_ARCH_FALLBACK"
DEVICE_COUNT_MARKER = "WSL2_HIP_DEVICE_COUNT_PATCH"


class PatchError(RuntimeError):
    """The target vLLM source does not have the expected patch location."""


def _is_vllm_package(path: Path) -> bool:
    return (path / "platforms" / "__init__.py").is_file() and (
        path / "platforms" / "rocm.py"
    ).is_file()


def _package_candidates(root: Path) -> list[Path]:
    """Return both supported interpretations of a user-supplied root."""
    return [root, root / "vllm"]


def find_vllm_package(root_hint: Path | None) -> Path:
    """Locate a writable vLLM package without importing vLLM itself."""
    roots: list[Path] = []
    if root_hint is not None:
        roots.append(root_hint)
    elif env_root := os.environ.get("VLLM_ROOT"):
        roots.append(Path(env_root))
    else:
        roots.append(Path.cwd())
        spec = importlib.util.find_spec("vllm")
        if spec and spec.submodule_search_locations:
            roots.extend(Path(location) for location in spec.submodule_search_locations)

    checked: list[Path] = []
    for root in roots:
        for candidate in _package_candidates(root.expanduser().resolve()):
            if candidate not in checked:
                checked.append(candidate)
            if _is_vllm_package(candidate):
                return candidate

    locations = "\n  ".join(str(path) for path in checked)
    raise PatchError(
        "Could not locate a vLLM package. Pass --vllm-root /path/to/vllm "
        "(or its checkout root). Checked:\n  " + locations
    )


def _find_top_level_function(source: str, name: str) -> ast.FunctionDef:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PatchError(f"Could not parse target source: {exc}") from exc

    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise PatchError(
            f"Expected exactly one top-level function named {name!r}; found {len(matches)}."
        )
    return matches[0]


def _replace_lines(source: str, start: int, end: int, replacement: str) -> str:
    """Replace a 1-based, inclusive line range while preserving all other text."""
    lines = source.splitlines(keepends=True)
    replacement_lines = replacement.splitlines(keepends=True)
    if replacement_lines and not replacement_lines[-1].endswith(("\n", "\r")):
        replacement_lines[-1] += "\n"
    return "".join(lines[: start - 1] + replacement_lines + lines[end:])


def patch_platform_detection(source: str) -> tuple[str, str]:
    """Replace amdsmi platform discovery with HIP device enumeration."""
    function = _find_top_level_function(source, "rocm_platform_plugin")
    original = ast.get_source_segment(source, function) or ""
    if PLATFORM_MARKER in original:
        return source, "already applied"

    replacement = f'''def rocm_platform_plugin() -> str | None:
    # {PLATFORM_MARKER}
    """Report ROCm when HIP can enumerate at least one visible device."""
    logger.debug("Checking if ROCm platform is available via HIP.")
    try:
        import ctypes

        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        hip.hipGetDeviceCount.restype = ctypes.c_int
        count = ctypes.c_int()
        result = hip.hipGetDeviceCount(ctypes.byref(count))
        if result == 0 and count.value > 0:
            logger.debug("ROCm platform available via HIP with %d device(s).", count.value)
            return "vllm.platforms.rocm.RocmPlatform"
        logger.debug("ROCm unavailable via HIP (result=%d, count=%d).", result, count.value)
    except Exception as exc:
        logger.debug("ROCm unavailable via HIP: %s", exc)
    return None
'''
    return _replace_lines(source, function.lineno, function.end_lineno, replacement), "applied"


def _ensure_os_import(source: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names):
            return source

    logging_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.Import)
        and any(alias.name == "logging" for alias in node.names)
    ]
    if not logging_imports:
        raise PatchError("Could not add 'import os': target has no 'import logging'.")
    insertion_line = logging_imports[-1].end_lineno
    return _replace_lines(source, insertion_line + 1, insertion_line, "import os\n")


def patch_gcn_arch_fallback(source: str) -> tuple[str, str]:
    """Avoid the amdsmi failure path when resolving the active GCN architecture."""
    function = _find_top_level_function(source, "_get_gcn_arch")
    original = ast.get_source_segment(source, function) or ""
    if ARCH_MARKER in original:
        return source, "already applied"

    try_nodes = [node for node in function.body if isinstance(node, ast.Try)]
    if len(try_nodes) != 1 or len(try_nodes[0].handlers) != 1:
        raise PatchError("Expected one amdsmi try/except block in _get_gcn_arch().")
    try_node = try_nodes[0]
    if "amdsmi" not in (ast.get_source_segment(source, try_node) or ""):
        raise PatchError("The _get_gcn_arch() try/except block is not amdsmi-based.")

    handler = try_node.handlers[0]
    replacement = f'''    except Exception as exc:
        logger.debug("Failed to get GCN arch via amdsmi: %s", exc)

    # {ARCH_MARKER}
    # WSL2 fallback: amdsmi may fail even though HIP can use the GPU.
    # Explicit user configuration always takes precedence over gfx1151.
    arch = os.environ.get("PYTORCH_ROCM_ARCH") or "{DEFAULT_GCN_ARCH}"
    logger.info("Using %s for GCN arch after amdsmi fallback", arch)
    return arch
'''
    source = _replace_lines(source, handler.lineno, function.end_lineno, replacement)
    return _ensure_os_import(source), "applied"


def patch_torch_device_count(source: str) -> tuple[str, str]:
    """Make vLLM use HIP enumeration when torch reports zero visible devices."""
    if DEVICE_COUNT_MARKER in source:
        return source, "already applied"

    marker = "_GCN_ARCH = _get_gcn_arch()"
    occurrences = source.count(marker)
    if occurrences != 1:
        raise PatchError(
            f"Expected exactly one {marker!r} marker in rocm.py; found {occurrences}."
        )

    injection = f'''{marker}

# {DEVICE_COUNT_MARKER}
def _hip_device_count() -> int:
    """Return the number of HIP-visible devices, or zero when HIP is unavailable."""
    try:
        import ctypes

        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        hip.hipGetDeviceCount.restype = ctypes.c_int
        count = ctypes.c_int()
        result = hip.hipGetDeviceCount(ctypes.byref(count))
        return count.value if result == 0 else 0
    except Exception as exc:
        logger.debug("HIP device-count fallback failed: %s", exc)
        return 0


def _patch_torch_device_count() -> None:
    import torch

    try:
        torch_count = torch.cuda.device_count()
    except Exception as exc:
        logger.debug("torch device_count failed: %s", exc)
        torch_count = 0

    if torch_count != 0:
        return

    hip_count = _hip_device_count()
    if hip_count <= 0:
        return

    def device_count() -> int:
        return hip_count

    torch.cuda.device_count = device_count
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "device_count"):
        accelerator.device_count = device_count
    logger.info("Patched torch device_count to %d via HIP", hip_count)


_patch_torch_device_count()
'''
    return source.replace(marker, injection, 1), "applied"


def _backup_and_write(path: Path, content: str) -> None:
    backup = path.with_name(path.name + ".wsl2-amdsmi.bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(content)
            temp_path = Path(temporary.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def clear_pycache(vllm_package: Path) -> None:
    """Remove stale bytecode for files changed by this script."""
    for relative_path in ("platforms/__pycache__", "v1/worker/__pycache__"):
        cache = vllm_package / relative_path
        if cache.is_dir():
            shutil.rmtree(cache)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-root",
        type=Path,
        help="vLLM checkout root or installed vllm package directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show whether the expected patch locations are present without writing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        vllm_package = find_vllm_package(args.vllm_root)
        platform_path = vllm_package / "platforms" / "__init__.py"
        rocm_path = vllm_package / "platforms" / "rocm.py"

        platform_source = platform_path.read_text(encoding="utf-8")
        rocm_source = rocm_path.read_text(encoding="utf-8")

        patched_platform, platform_status = patch_platform_detection(platform_source)
        patched_rocm, arch_status = patch_gcn_arch_fallback(rocm_source)
        patched_rocm, device_count_status = patch_torch_device_count(patched_rocm)

        print(f"vLLM package: {vllm_package}")
        print(f"Platform detection: {platform_status}")
        print(f"GCN architecture fallback: {arch_status}")
        print(f"torch device count fallback: {device_count_status}")
        print(f"Default GCN architecture: {DEFAULT_GCN_ARCH}")

        if args.dry_run:
            print("Dry run: no files changed.")
            return 0

        if patched_platform != platform_source:
            _backup_and_write(platform_path, patched_platform)
        if patched_rocm != rocm_source:
            _backup_and_write(rocm_path, patched_rocm)
        clear_pycache(vllm_package)
        print("Patches applied. Original files are saved with .wsl2-amdsmi.bak suffixes.")
        return 0
    except PatchError as exc:
        print(f"Patch failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Could not update vLLM files: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
