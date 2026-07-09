"""Shared pytest configuration for lasusie tests.

Force CPU (the experimental Metal backend is flaky for second-order autodiff) and
enable float64, which the numerical-equivalence tests rely on.
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
