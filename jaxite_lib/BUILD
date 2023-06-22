# The base Jaxite library

load("@rules_python//python:defs.bzl", "py_library")
load("@jaxite//bazel:test_oss.bzl", "cpu_gpu_tpu_test", "cpu_tpu_test")

package(
    default_applicable_licenses = ["@jaxite//:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":encoding",
        ":lwe",
        ":parameters",
        ":rlwe",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite//jaxite_bool:bool_params",
    ],
)

py_library(
    name = "matrix_utils",
    srcs = ["matrix_utils.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "matrix_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = [
        "matrix_utils_test.py",
    ],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3ONLY",
    deps = [
        ":matrix_utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

py_library(
    name = "decomposition",
    srcs = ["decomposition.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

# TODO(b/245554747): fix for GPU
cpu_tpu_test(
    name = "decomposition_test",
    size = "small",
    timeout = "moderate",
    srcs = [
        "decomposition_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":decomposition",
        "@com_google_absl_py//absl/testing:absltest",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

py_library(
    name = "encoding",
    srcs = ["encoding.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":types",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "encoding_test",
    size = "small",
    timeout = "moderate",
    srcs = [
        "encoding_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":encoding",
        ":types",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

py_library(
    name = "lwe",
    srcs = ["lwe.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":encoding",
        ":matrix_utils",
        ":parameters",
        ":random_source",
        ":types",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "lwe_test",
    size = "small",
    timeout = "moderate",
    srcs = [
        "lwe_test.py",
    ],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3ONLY",
    deps = [
        ":encoding",
        ":lwe",
        ":parameters",
        ":random_source",
        ":test_utils",
        ":types",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

py_library(
    name = "rlwe",
    srcs = ["rlwe.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":encoding",
        ":lwe",
        ":matrix_utils",
        ":parameters",
        ":random_source",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "rlwe_test",
    size = "small",
    timeout = "moderate",
    srcs = [
        "rlwe_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":parameters",
        ":random_source",
        ":rlwe",
        ":test_utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

py_library(
    name = "bootstrap",
    srcs = ["bootstrap.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":decomposition",
        ":key_switch",
        ":lwe",
        ":matrix_utils",
        ":parameters",
        ":random_source",
        ":rgsw",
        ":rlwe",
        ":types",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "bootstrap_test",
    size = "large",
    srcs = [
        "bootstrap_test.py",
    ],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3ONLY",
    deps = [
        ":bootstrap",
        ":decomposition",
        ":encoding",
        ":key_switch",
        ":lwe",
        ":matrix_utils",
        ":parameters",
        ":random_source",
        ":rgsw",
        ":rlwe",
        ":test_polynomial",
        ":test_utils",
        ":types",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "blind_rotate_test",
    size = "large",
    srcs = [
        "blind_rotate_test.py",
    ],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3ONLY",
    deps = [
        ":bootstrap",
        ":decomposition",
        ":encoding",
        ":key_switch",
        ":lwe",
        ":matrix_utils",
        ":parameters",
        ":random_source",
        ":rgsw",
        ":rlwe",
        ":test_polynomial",
        ":types",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

py_library(
    name = "test_polynomial",
    srcs = ["test_polynomial.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":encoding",
        ":parameters",
        ":random_source",
        ":rlwe",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "test_polynomial_test",
    size = "small",
    timeout = "moderate",
    srcs = [
        "test_polynomial_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":decomposition",
        ":encoding",
        ":parameters",
        ":random_source",
        ":rlwe",
        ":test_polynomial",
        ":types",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

py_library(
    name = "key_switch",
    srcs = ["key_switch.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":decomposition",
        ":encoding",
        ":lwe",
        ":parameters",
        ":random_source",
        ":types",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "key_switch_test",
    size = "large",
    srcs = [
        "key_switch_test.py",
    ],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3ONLY",
    deps = [
        ":decomposition",
        ":encoding",
        ":key_switch",
        ":lwe",
        ":parameters",
        ":random_source",
        ":rlwe",
        ":test_utils",
        ":types",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "random_source_test",
    srcs = ["random_source_test.py"],
    deps = [
        ":random_source",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

py_library(
    name = "rgsw",
    srcs = ["rgsw.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":decomposition",
        ":encoding",
        ":matrix_utils",
        ":parameters",
        ":random_source",
        ":rlwe",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "rgsw_test",
    size = "small",
    timeout = "moderate",
    srcs = [
        "rgsw_test.py",
    ],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3ONLY",
    deps = [
        ":decomposition",
        ":parameters",
        ":random_source",
        ":rgsw",
        ":test_utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

py_library(
    name = "parameters",
    srcs = ["parameters.py"],
    srcs_version = "PY3ONLY",
)

py_library(
    name = "random_source",
    srcs = ["random_source.py"],
    deps = [
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)
