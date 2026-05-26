# An FHE cryptosystem built in JAX

load("@jaxite//bazel:test_oss.bzl", "cpu_gpu_tpu_test", "gpu_tpu_test", "multichip_tpu_test", "tpu_test")
load("@rules_license//rules:license.bzl", "license")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = [":license"],
    default_visibility = [
        "//visibility:public",
    ],
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

py_library(
    name = "jaxite_ckks",
    srcs = glob(
        ["jaxite/jaxite_ckks/*.py"],
        exclude = ["**/*_test.py"],
    ),
    deps = [
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

py_test(
    name = "encode_test",
    size = "small",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/encode_test.py"],
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

py_test(
    name = "encrypt_test",
    size = "small",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/encrypt_test.py"],
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

py_library(
    name = "jaxite",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "**/*_test.py",
            "**/test_util.py",
            "jaxite_ckks/*",
        ],
    ),
    data = [
        "jaxite_ec/configurations.toml",
        # "@jaxite//jaxite_ec/c_kernels:distribution.so",
    ],
    visibility = [":internal"],
    deps = [
        ":jaxite_ckks",
        # copybara: xprof_analysis_client  # buildcleaner: keep
        # copybara: xprof_session  # buildcleaner: keep
        "@jaxite_deps//gmpy2",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        # copybara: jax/experimental:pallas_lib
        # copybara: jax/experimental:pallas_tpu
        "@jaxite//jaxite_ec/c_kernels:build",
        "@jaxite_deps//numpy",
        # copybara: pandas
        # copybara: toml
    ],
)

# Test rules are below, though the source files are in subdirectories.
py_library(
    name = "test_utils",
    srcs = ["jaxite/jaxite_lib/test_utils.py"],
    deps = [
        ":jaxite",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
    ],
)

# Test rules are below, though the source files are in subdirectories.
tpu_test(
    name = "matrix_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/matrix_utils_test.py"],
    shard_count = 3,
    deps = [
        ":jaxite",
        # copybara: xprof_analysis_client  # buildcleaner: keep
        # copybara: xprof_session  # buildcleaner: keep
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
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
        # copybara: xprof_analysis_client  # buildcleaner: keep
        # copybara: xprof_session  # buildcleaner: keep
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

tpu_test(
    name = "ec_finite_field_test",
    srcs = ["jaxite_ec/finite_field_test.py"],
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

tpu_test(
    name = "ec_finite_field_perf_test",
    srcs = ["jaxite_ec/finite_field_perf_test.py"],
    deps = [
        ":jaxite",
        # copybara: xprof_analysis_client  # buildcleaner: keep
        # copybara: xprof_session  # buildcleaner: keep
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
        # copybara: toml
    ],
)

tpu_test(
    name = "elliptic_curve_test",
    srcs = ["jaxite_ec/elliptic_curve_test.py"],
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
        # copybara: toml
    ],
)

tpu_test(
    name = "elliptic_curve_perf_test",
    srcs = ["jaxite_ec/elliptic_curve_perf_test.py"],
    deps = [
        ":jaxite",
        # copybara: xprof_analysis_client  # buildcleaner: keep
        # copybara: xprof_session  # buildcleaner: keep
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
        # copybara: toml
    ],
)

tpu_test(
    name = "multiscalar_multiplication_test",
    srcs = ["jaxite_ec/multiscalar_multiplication_test.py"],
    data = glob(["jaxite_ec/data/t1024/*.csv"]),
    deps = [
        ":jaxite",
        # copybara: xprof_analysis_client  # buildcleaner: keep
        # copybara: xprof_session  # buildcleaner: keep
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
        # copybara: toml
    ],
)

tpu_test(
    name = "multiscalar_multiplication_perf_test",
    srcs = ["jaxite_ec/multiscalar_multiplication_perf_test.py"],
    deps = [
        ":jaxite",
        # copybara: xprof_analysis_client  # buildcleaner: keep
        # copybara: xprof_session  # buildcleaner: keep
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

tpu_test(
    name = "number_theory_transform_test",
    srcs = ["jaxite_ec/number_theory_transform_test.py"],
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

tpu_test(
    name = "number_theory_transform_perf_test",
    size = "large",
    timeout = "eternal",
    srcs = ["jaxite_ec/number_theory_transform_perf_test.py"],
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

tpu_test(
    name = "jaxite_word_ntt_test",
    size = "large",
    timeout = "eternal",
    srcs = ["jaxite/jaxite_word/ntt_test.py"],
    shard_count = 3,
    deps = [
        ":jaxite",
        # copybara: xprof_analysis_client  # buildcleaner: keep
        # copybara: xprof_session  # buildcleaner: keep
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

tpu_test(
    name = "jaxite_word_sub_test",
    size = "large",
    timeout = "eternal",
    srcs = ["jaxite/jaxite_word/sub_test.py"],
    shard_count = 3,
    deps = [
        ":jaxite",
        # copybara: xprof_analysis_client  # buildcleaner: keep
        # copybara: xprof_session  # buildcleaner: keep
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

tpu_test(
    name = "add_test",
    size = "large",
    timeout = "eternal",
    srcs = ["jaxite/jaxite_word/add_test.py"],
    shard_count = 3,
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

cpu_gpu_tpu_test(
    name = "decomposition_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/decomposition_test.py"],
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

cpu_gpu_tpu_test(
    name = "encoding_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/encoding_test.py"],
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
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
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
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
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
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
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

cpu_gpu_tpu_test(
    name = "blind_rotate_test",
    size = "large",
    srcs = ["jaxite/jaxite_lib/blind_rotate_test.py"],
    shard_count = 10,
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

cpu_gpu_tpu_test(
    name = "test_polynomial_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_lib/test_polynomial_test.py"],
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
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
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
    ],
)

cpu_gpu_tpu_test(
    name = "random_source_test",
    srcs = ["jaxite/jaxite_lib/random_source_test.py"],
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
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
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

py_test(
    name = "lut_test",
    srcs = ["jaxite/jaxite_bool/lut_test.py"],
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
    ],
)

tpu_test(
    name = "jaxite_bool_test",
    size = "large",
    srcs = ["jaxite/jaxite_bool/jaxite_bool_test.py"],
    shard_count = 50,
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

gpu_tpu_test(
    name = "jaxite_bool_multigate_test",
    size = "large",
    srcs = ["jaxite/jaxite_bool/jaxite_bool_multigate_test.py"],
    shard_count = 20,
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

multichip_tpu_test(
    name = "pmap_test",
    size = "large",
    srcs = ["jaxite/jaxite_bool/pmap_test.py"],
    tags = ["manual"],
    deps = [
        ":jaxite",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

py_test(
    name = "rns_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_ckks/rns_test.py"],
    deps = [
        ":jaxite_ckks",
        ":test_utils",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
        "@jaxite_deps//parameterized",
    ],
)

py_test(
    name = "rns_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_ckks/rns_utils_test.py"],
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
    ],
)

cpu_gpu_tpu_test(
    name = "basis_conversion_test",
    size = "small",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/basis_conversion_test.py"],
    shard_count = 10,
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

cpu_gpu_tpu_test(
    name = "barrett_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_ckks/barrett_test.py"],
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

tpu_test(
    name = "ntt_test",
    size = "small",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/ntt_test.py"],
    shard_count = 3,
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

tpu_test(
    name = "ntt_openfhe_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_ckks/ntt_openfhe_test.py"],
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

py_test(
    name = "math_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_ckks/math_test.py"],
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//numpy",
    ],
)
