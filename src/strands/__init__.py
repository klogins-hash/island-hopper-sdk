"""
Island Hopper SDK

Provider-agnostic Strands SDK optimized for Scaleway.
"""

from .models.scaleway import ScalewayModel, create_scaleway_model
from .session.scaleway import ScalewaySessionRepository, create_scaleway_session_repository
from .telemetry.scaleway import ScalewayTelemetry, configure_scaleway_telemetry, AgentRequestTracer
from .hooks.scaleway import CostTrackingHook, SecurityHook, ResourceQuotaHook
from .hooks.scaleway import create_cost_tracking_hook, create_security_hook, create_resource_quota_hook
from .a2a.scaleway import ScalewayA2AExecutor, create_scaleway_a2a_executor, A2AContext

__version__ = "0.1.0"
__author__ = "Island Hopper Team"

__all__ = [
    # Models
    "ScalewayModel",
    "create_scaleway_model",
    
    # Sessions
    "ScalewaySessionRepository",
    "create_scaleway_session_repository",
    
    # Telemetry
    "ScalewayTelemetry",
    "configure_scaleway_telemetry",
    "AgentRequestTracer",
    
    # Hooks
    "CostTrackingHook",
    "SecurityHook",
    "ResourceQuotaHook",
    "create_cost_tracking_hook",
    "create_security_hook",
    "create_resource_quota_hook",
    
    # A2A Communication
    "ScalewayA2AExecutor",
    "create_scaleway_a2a_executor",
    "A2AContext",
]
