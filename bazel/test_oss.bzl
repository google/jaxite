"""Testing util helpers"""

load("@rules_python//python:defs.bzl", "py_test")

# This file is a shim to have the same Google-internal and external API
# for running jaxite on TPU and GPU tests. The internal API is not
# suitable for OSS release as it depends on all manner of internal
# infrastructural details. So any targets that use these rule will
# default to CPU tests, which would add wasteful overhead for anyone
# running all tests, so multiple copies are just skipped.

def tpu_test(name, **kwargs):
    """A shim that only generates CPU tests.

    Strict opt-in to supported kwargs, since many are not supported by the
    internal version.

    Args:
      name: the name of the CPU test.
      **kwargs: other args passed along to generated rules.
    """
    kwargs.setdefault("size", "large")
    kwargs.setdefault("srcs_version", "PY3")
    kwargs.setdefault("python_version", "PY3")
    if not kwargs.get("main"):
        main = kwargs.get("srcs", ["%s.py" % name])[0]
        kwargs.setdefault("main", main)
    py_test(
        name = name,
        srcs = kwargs["srcs"],
        size = kwargs["size"],
        main = kwargs["main"],
        srcs_version = kwargs["srcs_version"],
        python_version = kwargs["python_version"],
        deps = kwargs.get("deps", []),
    )

def multichip_tpu_test(name, **kwargs):
    """A shim that only generates CPU tests.

    Args:
      name: the name of the CPU test.
      **kwargs: other args passed along to generated rules.
    """
    return tpu_test(name, **kwargs)

def cpu_tpu_test(name, **kwargs):
    """A shim that only generates CPU tests.

    Args:
      name: the name of the CPU test.
      **kwargs: other args passed along to generated rules.
    """
    py_test(name = name, **kwargs)

def gpu_tpu_test(name, **kwargs):
    """A shim that only generates CPU tests.

    Args:
      name: the name of the CPU test.
      **kwargs: other args passed along to generated rules.
    """
    py_test(name = name, **kwargs)

def cpu_gpu_tpu_test(name, **kwargs):
    """A shim that only generates CPU tests.

    Args:
      name: the name of the CPU test.
      **kwargs: other args passed along to generated rules.
    """
    py_test(name = name, **kwargs)

def cpu_gpu_tpu_py_benchmark(name, **kwargs):
    """A shim that only generates CPU tests.

    Args:
      name: the base name for the test rules, appended with
        `_<platform>_benchmark`, e.g., `_cpu_benchmark`
      **kwargs: other args passed along to generated rules.
    """
    py_test(name = name + "_cpu_benchmark", **kwargs)
