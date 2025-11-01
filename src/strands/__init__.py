"""
Island Hopper SDK

Provider-agnostic Strands SDK optimized for Scaleway.
"""

# Import core classes that will be implemented
from .models.scaleway import ScalewayModel, create_scaleway_model
from .session.scaleway import ScalewaySessionRepository, create_scaleway_session_repository
from .telemetry.scaleway import ScalewayTelemetry, configure_scaleway_telemetry, AgentRequestTracer
from .hooks.scaleway import CostTrackingHook, SecurityHook, ResourceQuotaHook
from .hooks.scaleway import create_cost_tracking_hook, create_security_hook, create_resource_quota_hook
from .a2a.scaleway import ScalewayA2AExecutor, create_scaleway_a2a_executor, A2AContext

# Basic Agent class (simplified for standalone use)
class Agent:
    """Basic Agent class for Island Hopper SDK"""
    
    def __init__(self, model=None, tools=None, hooks=None, system_prompt=None):
        self.model = model
        self.tools = tools or []
        self.hooks = hooks or []
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.tool_executor = None
    
    async def __call__(self, message):
        """Call the agent with a message"""
        if hasattr(self.model, 'generate'):
            response = await self.model.generate(message)
            return AgentResponse(message=response, model_used=self.model)
        else:
            return AgentResponse(message="Model not configured", model_used=None)
    
    def get_function_schemas(self):
        """Get function schemas for tools"""
        return [tool.get_schema() for tool in self.tools] if self.tools else []


class AgentResponse:
    """Agent response wrapper"""
    
    def __init__(self, message, model_used=None):
        self.message = message
        self.model_used = model_used
        self.tool_calls = []

__version__ = "0.1.0"
__author__ = "Island Hopper Team"

__all__ = [
    # Core Agent
    "Agent",
    "AgentResponse",
    
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
