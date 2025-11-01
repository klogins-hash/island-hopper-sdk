"""
Strands Models - Provider-agnostic AI model implementations.
"""

from .scaleway import ScalewayModel, create_scaleway_model

__all__ = ["ScalewayModel", "create_scaleway_model"]
