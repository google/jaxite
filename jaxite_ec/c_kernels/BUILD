# load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    default_applicable_licenses = ["@jaxite//:license"],
    default_visibility = ["//visibility:public"],
)

# cc_binary distribution.so

py_library(
    name = "build",
    srcs = ["build.py"],
    type_checking = True,
    visibility = ["//visibility:public"],
)
