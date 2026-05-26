"""Python-side build driver for the distribution C kernel.

:func:`ensure_distribution_kernel` compiles ``distribution.cpp`` (or returns
the cached build) and yields the absolute path of the resulting shared
library, so callers can ``ctypes.cdll.LoadLibrary`` it without a separate
build step.

Default flags: ``-std=c++17 -fopenmp -O2 -fPIC -shared -I<jaxlib>/include``.
``CXX`` and ``CXXFLAGS`` env vars override the compiler and flags.

Run as a script (``python -m c_kernels.build``) to pre-build without importing
the rest of the package.
"""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys
import threading

_KERNEL_DIR = pathlib.Path(__file__).parent
_SRC_PATH = _KERNEL_DIR / "distribution.cpp"
_LIB_PATH = _KERNEL_DIR / "distribution.so"

_DEFAULT_CXXFLAGS = ("-std=c++17", "-fopenmp", "-O2", "-fPIC", "-shared")

_BUILD_LOCK = threading.Lock()


def _jaxlib_include_dir() -> pathlib.Path:
  import jaxlib  # pytype: disable=import-error

  return pathlib.Path(jaxlib.__file__).resolve().parent / "include"


def _needs_rebuild() -> bool:
  if not _LIB_PATH.exists():
    return True
  if not _SRC_PATH.exists():
    return False
  return _SRC_PATH.stat().st_mtime > _LIB_PATH.stat().st_mtime


def _build() -> None:
  cxx = os.environ.get("CXX", "g++")
  flags_env = os.environ.get("CXXFLAGS")
  cxxflags = shlex.split(flags_env) if flags_env else list(_DEFAULT_CXXFLAGS)
  cmd = [
      cxx,
      *cxxflags,
      f"-I{_jaxlib_include_dir()}",
      str(_SRC_PATH),
      "-o",
      str(_LIB_PATH),
  ]
  print(
      f"[c_kernels.build] compiling: {' '.join(shlex.quote(c) for c in cmd)}",
      file=sys.stderr,
  )
  subprocess.run(cmd, check=True, cwd=_KERNEL_DIR)


def ensure_distribution_kernel(force: bool = False) -> str:
  """Compile :file:`distribution.cpp` on demand and return the .so path.

  Args:
    force: rebuild even if the cached library is up to date.

  Returns:
    Absolute path to ``distribution.so``.

  Raises:
    FileNotFoundError: if pre-built kernel not found in test environment.
  """
  if "TEST_SRCDIR" in os.environ:
    runfiles_dir = pathlib.Path(os.environ["TEST_SRCDIR"])
    lib_path = (
        runfiles_dir
        / "google3/third_party/py/jaxite/jaxite_ec/c_kernels/distribution.so"
    )
    if lib_path.exists():
      return str(lib_path)
    raise FileNotFoundError(
        f"Pre-built kernel not found in runfiles at {lib_path}"
    )

  if "UNITTEST_ON_FORGE" in os.environ:
    if _LIB_PATH.exists():
      return str(_LIB_PATH)
    raise FileNotFoundError(f"Pre-built kernel not found at {_LIB_PATH}")

  with _BUILD_LOCK:
    if force or _needs_rebuild():
      _build()
  return str(_LIB_PATH)


if __name__ == "__main__":
  path = ensure_distribution_kernel(force="--force" in sys.argv[1:])
  print(path)
