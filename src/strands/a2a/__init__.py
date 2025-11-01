"""
Strands A2A - NATS-based agent-to-agent communication.
"""

from .scaleway import ScalewayA2AExecutor, create_scaleway_a2a_executor, A2AContext

__all__ = ["ScalewayA2AExecutor", "create_scaleway_a2a_executor", "A2AContext"]
