# The Jaxite Boolean API

load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:defs.bzl", "py_test")
load("@jaxite//bazel:test_oss.bzl", "gpu_tpu_test", "tpu_test")

package(
    default_applicable_licenses = ["@jaxite//:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "bool_encoding",
    srcs = ["bool_encoding.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "@jaxite//jaxite_lib:encoding",
        "@jaxite//jaxite_lib:types",
    ],
)

py_library(
    name = "type_converters",
    srcs = ["type_converters.py"],
    srcs_version = "PY3ONLY",
    deps = [],
)

py_library(
    name = "bool_params",
    srcs = ["bool_params.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":bool_encoding",
        ":lut",
        "@jaxite//jaxite_lib:decomposition",
        "@jaxite//jaxite_lib:encoding",
        "@jaxite//jaxite_lib:lwe",
        "@jaxite//jaxite_lib:parameters",
        "@jaxite//jaxite_lib:random_source",
        "@jaxite//jaxite_lib:rlwe",
        "@jaxite//jaxite_lib:types",
    ],
)

py_library(
    name = "jaxite_bool",
    srcs = ["jaxite_bool.py"],
    srcs_version = "PY3",
    deps = [
        ":bool_encoding",
        ":bool_params",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite//jaxite_lib:bootstrap",
        "@jaxite//jaxite_lib:encoding",
        "@jaxite//jaxite_lib:key_switch",
        "@jaxite//jaxite_lib:lwe",
        "@jaxite//jaxite_lib:random_source",
        "@jaxite//jaxite_lib:rgsw",
        "@jaxite//jaxite_lib:rlwe",
        "@jaxite//jaxite_lib:types",
    ],
)

py_library(
    name = "lut",
    srcs = ["lut.py"],
    srcs_version = "PY3",
    deps = [
        ":bool_encoding",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite//jaxite_lib:parameters",
        "@jaxite//jaxite_lib:rlwe",
        "@jaxite//jaxite_lib:test_polynomial",
        "@jaxite//jaxite_lib:types",
    ],
)

py_test(
    name = "lut_test",
    srcs = [
        "lut_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bool_encoding",
        ":lut",
        "@com_google_absl_py//absl/testing:absltest",
        "@jaxite//jaxite_lib:parameters",
    ],
)

gpu_tpu_test(
    name = "jaxite_bool_test",
    size = "large",
    srcs = [
        "jaxite_bool_test.py",
    ],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    deps = [
        ":bool_params",
        ":jaxite_bool",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite//jaxite_lib:decomposition",
        "@jaxite//jaxite_lib:encoding",
        "@jaxite//jaxite_lib:parameters",
        "@jaxite//jaxite_lib:random_source",
        "@jaxite//jaxite_lib:test_utils",
    ],
)

gpu_tpu_test(
    name = "jaxite_bool_multigate_test",
    size = "large",
    srcs = [
        "jaxite_bool_multigate_test.py",
    ],
    python_version = "PY3",
    shard_count = 20,
    srcs_version = "PY3",
    deps = [
        ":bool_params",
        ":jaxite_bool",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite//jaxite_lib:decomposition",
        "@jaxite//jaxite_lib:encoding",
        "@jaxite//jaxite_lib:parameters",
        "@jaxite//jaxite_lib:random_source",
        "@jaxite//jaxite_lib:test_utils",
    ],
)

tpu_test(
    name = "pmap_test",
    size = "large",
    srcs = ["pmap_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bool_params",
        ":jaxite_bool",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite//jaxite_lib:parameters",
    ],
)
