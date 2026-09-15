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
        exclude = [
            "**/*_test.py",
        ],
    ),
    deps = [
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

py_library(
    name = "jaxite_cggi",
    srcs = glob(
        ["jaxite/jaxite_cggi/*.py"],
        exclude = [
            "**/*_test.py",
            "jaxite/jaxite_cggi/test_utils.py",
        ],
    ),
    deps = [
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        # copybara: jax/experimental:pallas_lib
        # copybara: jax/experimental:pallas_tpu
    ],
)

tpu_test(
    name = "cross_equivalence_test",
    size = "large",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/cross_equivalence_test.py"],
    data = ["jaxite_ckks/cross_equivalence_test_data.json"],
    shard_count = 3,
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

py_library(
    name = "jaxite_bool",
    srcs = glob(
        ["jaxite/jaxite_bool/*.py"],
        exclude = ["**/*_test.py"],
    ),
    deps = [
        ":jaxite_cggi",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
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
        # copybara: xprof_analysis_client  # buildcleaner: keep
        # copybara: xprof_session  # buildcleaner: keep
        "@abseil-py//absl/testing:absltest",
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

py_library(
    name = "test_utils",
    srcs = ["jaxite/jaxite_cggi/test_utils.py"],
    deps = [
        ":jaxite_bool",
        ":jaxite_cggi",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
    ],
)

# Test rules are below, though the source files are in subdirectories.
tpu_test(
    name = "matrix_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_cggi/matrix_utils_test.py"],
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
    srcs = ["jaxite/jaxite_cggi/polymul_kernel_test.py"],
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

cpu_gpu_tpu_test(
    name = "decomposition_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_cggi/decomposition_test.py"],
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
    srcs = ["jaxite/jaxite_cggi/encoding_test.py"],
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
    srcs = ["jaxite/jaxite_cggi/lwe_test.py"],
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
    srcs = ["jaxite/jaxite_cggi/rlwe_test.py"],
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
    srcs = ["jaxite/jaxite_cggi/bootstrap_test.py"],
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
    srcs = ["jaxite/jaxite_cggi/blind_rotate_test.py"],
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
    srcs = ["jaxite/jaxite_cggi/test_polynomial_test.py"],
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
    srcs = ["jaxite/jaxite_cggi/key_switch_test.py"],
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
    srcs = ["jaxite/jaxite_cggi/random_source_test.py"],
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
    srcs = ["jaxite/jaxite_cggi/rgsw_test.py"],
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
    name = "bat_utils_test",
    size = "small",
    srcs = ["jaxite/jaxite_ckks/bat_utils_test.py"],
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
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
    name = "mul_test",
    size = "large",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/mul_test.py"],
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

py_test(
    name = "math_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jaxite/jaxite_ckks/math_test.py"],
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

py_test(
    name = "key_switching_key_test",
    size = "small",
    srcs = ["jaxite/jaxite_ckks/key_switching_key_test.py"],
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

py_test(
    name = "key_gen_test",
    size = "small",
    srcs = ["jaxite/jaxite_ckks/key_gen_test.py"],
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
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
        "@jaxite_deps//numpy",
    ],
)

tpu_test(
    name = "add_kernel_test",
    size = "small",
    srcs = ["jaxite/jaxite_ckks/add_test.py"],
    main = "jaxite/jaxite_ckks/add_test.py",
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

cpu_gpu_tpu_test(
    name = "blind_rotate_ckks_test",
    size = "small",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/blind_rotate_test.py"],
    main = "jaxite/jaxite_ckks/blind_rotate_test.py",
    shard_count = 5,
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
    name = "rescale_test",
    size = "medium",
    srcs = ["jaxite/jaxite_ckks/rescale_test.py"],
    shard_count = 6,
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
    name = "conjugate_test",
    size = "small",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/conjugate_test.py"],
    main = "jaxite/jaxite_ckks/conjugate_test.py",
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
    name = "rotate_test",
    size = "small",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/rotate_test.py"],
    main = "jaxite/jaxite_ckks/rotate_test.py",
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

cpu_gpu_tpu_test(
    name = "key_switching_test",
    size = "small",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/key_switching_test.py"],
    main = "jaxite/jaxite_ckks/key_switching_test.py",
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
    name = "blind_rotate_utils_test",
    size = "small",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/blind_rotate_utils_test.py"],
    main = "jaxite/jaxite_ckks/blind_rotate_utils_test.py",
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

cpu_gpu_tpu_test(
    name = "bootstrapping_utils_test",
    size = "small",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/bootstrapping_utils_test.py"],
    main = "jaxite/jaxite_ckks/bootstrapping_utils_test.py",
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)

cpu_gpu_tpu_test(
    name = "bootstrapping_test",
    size = "medium",
    timeout = "long",
    srcs = ["jaxite/jaxite_ckks/bootstrapping_test.py"],
    main = "jaxite/jaxite_ckks/bootstrapping_test.py",
    shard_count = 6,
    deps = [
        ":jaxite_ckks",
        "@abseil-py//absl/testing:absltest",
        "@jaxite_deps//hypothesis",
        "@jaxite_deps//jax",
        "@jaxite_deps//jaxlib",
        "@jaxite_deps//numpy",
    ],
)
