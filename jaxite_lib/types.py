"""A module containing basic types for TFHE."""

from typing import Any

import jax.numpy as jnp


LweCleartext = int
LwePlaintext = jnp.uint32
LweCiphertext = Any
