"""
spherepar.benchmark
===================

Stage 1 synthetic data generation pipeline for pmConv.

Sub-modules
-----------
surface : SurfaceFactory and Surface classes
signals : isotropic and anisotropic Gaussian signal generators
"""

from spherepar.benchmark.surface import Surface, SurfaceFactory
from spherepar.benchmark.signals import (
    isotropic_gaussian,
    anisotropic_gaussian,
)

__all__ = [
    "Surface",
    "SurfaceFactory",
    "isotropic_gaussian",
    "anisotropic_gaussian",
]