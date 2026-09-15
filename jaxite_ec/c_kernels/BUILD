load("@rules_python//python:defs.bzl", "py_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")

package(
    default_applicable_licenses = ["@jaxite//:license"],
    default_visibility = ["//visibility:public"],
)

# cc_binary distribution.so

py_library(
    name = "build",
    srcs = ["build.py"],
    visibility = ["//visibility:public"],
)
