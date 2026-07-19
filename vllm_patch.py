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
ROCM_DEVICE_COUNT_MARKER = "WSL2_AMDSMI_HIP_DEVICE_COUNT_PATCH"
DEVICE_NAME_MARKER = "WSL2_AMDSMI_DEVICE_NAME_FALLBACK"
LEGACY_DEVICE_COUNT_MARKER = "WSL2_HIP_DEVICE_COUNT_PATCH"


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


def _definition_start(definition: ast.FunctionDef) -> int:
    """Return the first source line occupied by a definition or decorator."""
    return min(
        [definition.lineno] + [decorator.lineno for decorator in definition.decorator_list]
    )


def _find_class_method(source: str, class_name: str, method_name: str) -> ast.FunctionDef:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PatchError(f"Could not parse target source: {exc}") from exc

    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise PatchError(
            f"Expected exactly one class named {class_name!r}; found {len(classes)}."
        )
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if len(methods) != 1:
        raise PatchError(
            f"Expected exactly one {class_name}.{method_name}() method; found {len(methods)}."
        )
    return methods[0]


def _replace_lines(source: str, start: int, end: int, replacement: str) -> str:
    """Replace a 1-based, inclusive line range while preserving all other text."""
    lines = source.splitlines(keepends=True)
    replacement_lines = replacement.splitlines(keepends=True)
    if replacement_lines and not replacement_lines[-1].endswith(("\n", "\r")):
        replacement_lines[-1] += "\n"
    return "".join(lines[: start - 1] + replacement_lines + lines[end:])


def patch_platform_detection(source: str) -> tuple[str, str]:
    """Use HIP only when the upstream amdsmi platform check fails."""
    function = _find_top_level_function(source, "rocm_platform_plugin")
    original = ast.get_source_segment(source, function) or ""
    if PLATFORM_MARKER in original:
        return source, "already applied"

    replacement = f'''def rocm_platform_plugin() -> str | None:
    # {PLATFORM_MARKER}
    """Detect ROCm with amdsmi, falling back to HIP for WSL2."""
    is_rocm = False
    logger.debug("Checking if ROCm platform is available.")
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            is_rocm = len(amdsmi.amdsmi_get_processor_handles()) > 0
            if is_rocm:
                logger.debug("Confirmed ROCm platform is available via amdsmi.")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as amdsmi_exc:
        logger.debug("amdsmi platform check failed: %s; trying HIP.", amdsmi_exc)
        try:
            import ctypes

            hip = ctypes.CDLL("libamdhip64.so")
            hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
            hip.hipGetDeviceCount.restype = ctypes.c_int
            count = ctypes.c_int()
            result = hip.hipGetDeviceCount(ctypes.byref(count))
            is_rocm = result == 0 and count.value > 0
            if is_rocm:
                logger.debug("Confirmed ROCm platform is available via HIP.")
            else:
                logger.debug(
                    "ROCm unavailable via HIP (result=%d, count=%d).", result, count.value
                )
        except Exception as hip_exc:
            logger.debug("ROCm unavailable via HIP: %s", hip_exc)
    return "vllm.platforms.rocm.RocmPlatform" if is_rocm else None
'''
    return (
        _replace_lines(source, _definition_start(function), function.end_lineno, replacement),
        "applied",
    )


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
    """Avoid amdsmi altogether when resolving the WSL2 GCN architecture."""
    function = _find_top_level_function(source, "_get_gcn_arch")
    original = ast.get_source_segment(source, function) or ""
    if ARCH_MARKER in original:
        return source, "already applied"

    if "amdsmi" not in original:
        raise PatchError("The _get_gcn_arch() implementation is not amdsmi-based.")

    replacement = f'''def _get_gcn_arch() -> str:
    # {ARCH_MARKER}
    """Return the configured WSL2 architecture without initializing amdsmi."""
    arch = os.environ.get("PYTORCH_ROCM_ARCH") or "{DEFAULT_GCN_ARCH}"
    logger.info("Using %s for GCN arch in the WSL2 amdsmi fallback", arch)
    return arch
'''
    source = _replace_lines(
        source, _definition_start(function), function.end_lineno, replacement
    )
    return _ensure_os_import(source), "applied"


def _remove_legacy_device_count_patch(source: str) -> str:
    """Remove the pre-v0.25 monkey patch when upgrading an existing target."""
    if LEGACY_DEVICE_COUNT_MARKER not in source:
        return source

    start = source.index(f"# {LEGACY_DEVICE_COUNT_MARKER}")
    end_marker = "_ON_GFX1X ="
    end = source.find(end_marker, start)
    if end < 0:
        raise PatchError("Could not remove the legacy torch device-count patch.")
    return source[:start] + source[end:]


def patch_rocm_device_count(source: str) -> tuple[str, str]:
    """Patch v0.25's ROCm-specific, amdsmi-backed device counter."""
    source = _remove_legacy_device_count_patch(source)
    function = _find_top_level_function(source, "_rocm_device_count_stateless")
    original = ast.get_source_segment(source, function) or ""
    if ROCM_DEVICE_COUNT_MARKER in original:
        return source, "already applied"

    replacement = f'''def _hip_device_count() -> int:
    """Return HIP-visible GPU count, including in WSL2 without amdsmi."""
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


@lru_cache(maxsize=8)
def _rocm_device_count_stateless(cuda_visible_devices: str | None = None) -> int:
    # {ROCM_DEVICE_COUNT_MARKER}
    """Count ROCm devices, using HIP if the amdsmi counter is unusable."""
    # Keep the parameter as part of the cache key, matching the upstream API.
    del cuda_visible_devices
    import torch.cuda

    if not torch.cuda._is_compiled():
        return 0

    try:
        amdsmi_count = (
            torch.cuda._device_count_amdsmi()
            if hasattr(torch.cuda, "_device_count_amdsmi")
            else 0
        )
    except Exception as exc:
        logger.debug("amdsmi device-count query failed: %s", exc)
        amdsmi_count = 0

    if amdsmi_count > 0:
        return amdsmi_count
    return _hip_device_count()
'''
    return (
        _replace_lines(
            source, _definition_start(function), function.end_lineno, replacement
        ),
        "applied",
    )


def patch_device_name_fallback(source: str) -> tuple[str, str]:
    """Fall back to torch properties when WSL2 cannot initialize amdsmi."""
    method = _find_class_method(source, "RocmPlatform", "get_device_name")
    original = ast.get_source_segment(source, method) or ""
    if DEVICE_NAME_MARKER in original:
        return source, "already applied"

    replacement = f'''    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        # {DEVICE_NAME_MARKER}
        try:
            amdsmi_init()
            try:
                physical_device_id = cls.device_id_to_physical_device_id(device_id)
                handle = amdsmi_get_processor_handles()[physical_device_id]
                asic_info = amdsmi_get_gpu_asic_info(handle)
                device_id_from_asic: str = asic_info["device_id"]
                return _ROCM_DEVICE_ID_NAME_MAP.get(
                    device_id_from_asic, asic_info["market_name"]
                )
            finally:
                amdsmi_shut_down()
        except Exception as exc:
            logger.debug("Failed to get device name via amdsmi: %s", exc)
            return torch.cuda.get_device_properties(device_id).name
'''
    return (
        _replace_lines(source, _definition_start(method), method.end_lineno, replacement),
        "applied",
    )


def patch_torch_device_count(source: str) -> tuple[str, str]:
    """Compatibility alias for callers of the pre-v0.25 patch function."""
    return patch_rocm_device_count(source)


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
        patched_rocm, device_count_status = patch_rocm_device_count(patched_rocm)
        patched_rocm, device_name_status = patch_device_name_fallback(patched_rocm)

        print(f"vLLM package: {vllm_package}")
        print(f"Platform detection: {platform_status}")
        print(f"GCN architecture fallback: {arch_status}")
        print(f"ROCm device count fallback: {device_count_status}")
        print(f"Device name fallback: {device_name_status}")
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
