# An FHE cryptosystem built in JAX

load("@jaxite//bazel:test_oss.bzl", "cpu_gpu_tpu_test", "gpu_tpu_test", "multichip_tpu_test", "tpu_test")
load("@rules_license//rules:license.bzl", "license")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["@jaxite//:license"],
    default_visibility = ["//visibility:public"],
)

package_group(
    name = "internal",
    packages = [],
)

license(
    name = "license",
    package_name = "jaxite",
)

licenses(["notice"])

exports_files(["LICENSE"])

# a single-source build dependency that gives the whole (non-test) jaxite
# source tree; note we chose the style of putting all test rules below, because
# glob does not recurse into subdirectories with BUILD files in them.
py_library(
    name = "jaxite",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "**/*_test.py",
            "**/test_util.py",
        ],
    ),
    visibility = [":internal"],
    deps = [
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        # copybara: jax:pallas_lib
        # copybara: jax:pallas_tpu
    ],
)

# Test rules are below, though the source files are in subdirectories.
py_library(
    name = "test_utils",
    srcs = ["jaxite/jaxite_lib/test_utils.py"],
    deps = [
        ":jaxite",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "matrix_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/matrix_utils_test.py"],
    shard_count = 3,
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

tpu_test(
    name = "polymul_kernel_test",
    size = "large",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/polymul_kernel_test.py"],
    shard_count = 3,
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "decomposition_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/decomposition_test.py"],
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "encoding_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/encoding_test.py"],
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "lwe_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/lwe_test.py"],
    shard_count = 50,
    deps = [
        ":jaxite",
        ":test_utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "rlwe_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/rlwe_test.py"],
    deps = [
        ":jaxite",
        ":test_utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "bootstrap_test",
    size = "large",
    srcs = ["jaxite/jaxite_lib/bootstrap_test.py"],
    shard_count = 50,
    deps = [
        ":jaxite",
        ":test_utils",
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
    srcs = ["jaxite/jaxite_lib/blind_rotate_test.py"],
    shard_count = 10,
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "test_polynomial_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/test_polynomial_test.py"],
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "key_switch_test",
    size = "large",
    srcs = ["jaxite/jaxite_lib/key_switch_test.py"],
    shard_count = 50,
    deps = [
        ":jaxite",
        ":test_utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "random_source_test",
    srcs = ["jaxite/jaxite_lib/random_source_test.py"],
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
    ],
)

cpu_gpu_tpu_test(
    name = "rgsw_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/rgsw_test.py"],
    shard_count = 10,
    deps = [
        ":jaxite",
        ":test_utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jaxite_deps_hypothesis//:pkg",
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        "@jaxite_deps_numpy//:pkg",
    ],
)

py_test(
    name = "lut_test",
    srcs = ["jaxite/jaxite_bool/lut_test.py"],
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

gpu_tpu_test(
    name = "jaxite_bool_test",
    size = "large",
    srcs = ["jaxite/jaxite_bool/jaxite_bool_test.py"],
    shard_count = 50,
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

gpu_tpu_test(
    name = "jaxite_bool_multigate_test",
    size = "large",
    srcs = ["jaxite/jaxite_bool/jaxite_bool_multigate_test.py"],
    shard_count = 20,
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

multichip_tpu_test(
    name = "pmap_test",
    size = "large",
    srcs = ["jaxite/jaxite_bool/pmap_test.py"],
    tags = ["manual"],
    deps = [
        ":jaxite",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)
