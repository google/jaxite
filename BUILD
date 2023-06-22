# An FHE cryptosystem built in JAX

load("@rules_python//python:defs.bzl", "py_library")
load("@rules_license//rules:license.bzl", "license")

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

# a single-source build dependency that gives the whole (non-test) jaxite
# source tree
py_library(
    name = "jaxite",
    srcs = glob(
        [
            "jaxite_lib/*.py",
            "jaxite_bool/*.py",
        ],
        exclude = [
            "**/*_test.py",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "@jaxite//jaxite_bool",
        "@jaxite//jaxite_bool:bool_encoding",
        "@jaxite//jaxite_bool:bool_params",
        "@jaxite//jaxite_bool:lut",
        "@jaxite//jaxite_bool:type_converters",
        "@jaxite//jaxite_lib:bootstrap",
        "@jaxite//jaxite_lib:decomposition",
        "@jaxite//jaxite_lib:encoding",
        "@jaxite//jaxite_lib:key_switch",
        "@jaxite//jaxite_lib:lwe",
        "@jaxite//jaxite_lib:matrix_utils",
        "@jaxite//jaxite_lib:parameters",
        "@jaxite//jaxite_lib:random_source",
        "@jaxite//jaxite_lib:rgsw",
        "@jaxite//jaxite_lib:rlwe",
        "@jaxite//jaxite_lib:test_polynomial",
        "@jaxite//jaxite_lib:test_utils",
        "@jaxite//jaxite_lib:types",
    ],
)

exports_files(["LICENSE"])
