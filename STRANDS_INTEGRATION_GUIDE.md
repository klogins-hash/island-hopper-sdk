# 🔗 Strands SDK Integration Guide

## 📋 Overview

This guide explains how to integrate the **Island Hopper SDK** with the existing **Strands SDK** to create Scaleway-optimized AI agents. The Island Hopper SDK provides Scaleway-specific extensions while maintaining compatibility with the core Strands framework.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Integration Architecture                │
├─────────────────────────────────────────────────────────────┤
│  🎯 Strands Core SDK                                        │
│  • Base Agent class                                         │
│  • Provider interfaces                                      │
│  • Tool system                                              │
│  • Session management                                       │
├─────────────────────────────────────────────────────────────┤
│  🏝️ Island Hopper Extensions                               │
│  • ScalewayModel (provider-agnostic routing)               │
│  • ScalewaySessionRepository (PostgreSQL)                  │
│  • ScalewayTelemetry (Cockpit integration)                 │
│  • Scaleway Tools (Object Storage, Database)               │
│  • A2A Communication (NATS-based)                          │
├─────────────────────────────────────────────────────────────┤
│  ☁️ Scaleway Services                                       │
│  • Kapsys Kubernetes                                        │
│  • PostgreSQL Database                                      │
│  • Object Storage (S3-compatible)                          │
│  • Cockpit Monitoring                                       │
│  • NATS Messaging                                          │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Integration

### 1. Install Dependencies

```bash
# Core Strands SDK (existing)
pip install strands-sdk

# Island Hopper extensions
pip install island-hopper-sdk

# Scaleway-specific dependencies
pip install psycopg2-binary boto3 nats-py
```

### 2. Basic Integration

```python
from strands import Agent as BaseAgent
from island_hopper_sdk import ScalewayModel, ScalewaySessionRepository
from island_hopper_tools import ObjectStorageTool, DatabaseTool

# Create Scaleway-optimized agent
class ScalewayAgent(BaseAgent):
    """Agent with Scaleway optimizations"""
    
    def __init__(self, **kwargs):
        # Initialize with Scaleway model
        model = ScalewayModel(
            primary_provider="openrouter",
            primary_model="groq/llama-4-scout",
            fallback_provider="anthropic", 
            fallback_model="claude-4.5-sonnet"
        )
        
        # Use Scaleway session repository
        session_repo = ScalewaySessionRepository(
            connection_string=os.getenv("SCALEWAY_DATABASE_URL")
        )
        
        super().__init__(
            model=model,
            session_repository=session_repo,
            **kwargs
        )
        
        # Add Scaleway tools
        self.tools = [
            ObjectStorageTool(),
            DatabaseTool()
        ]
```

## 🔧 Configuration

### Environment Variables

```bash
# AI Provider Keys (existing Strands)
export OPENROUTER_API_KEY="sk-or-v1-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Scaleway Configuration (Island Hopper extensions)
export SCALEWAY_ACCESS_KEY="SCW..."
export SCALEWAY_SECRET_KEY="..."
export SCALEWAY_PROJECT_ID="..."
export SCALEWAY_REGION="fr-par"

# Database (Scaleway PostgreSQL)
export SCALEWAY_DATABASE_URL="postgresql://user:pass@host:5432/db"

# Object Storage (Scaleway S3)
export SCALEWAY_STORAGE_ENDPOINT="s3.fr-par.scw.cloud"
export SCALEWAY_STORAGE_BUCKET="island-hopper-data"

# Messaging (NATS)
export NATS_URL="nats://nats.example.com:4222"

# Monitoring (Scaleway Cockpit)
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otel.fr-par.scw.cloud"
```

### Provider Configuration

```python
# Custom provider configuration
provider_config = {
    "openrouter": {
        "name": "OpenRouter",
        "endpoint": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "models": ["groq/llama-4-scout", "anthropic/claude-4.5-sonnet"],
        "priority": 1
    },
    "anthropic": {
        "name": "Anthropic", 
        "endpoint": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
        "models": ["claude-4.5-sonnet", "claude-3-5-sonnet"],
        "priority": 2
    },
    "scaleway-llm": {  # Future Scaleway LLM service
        "name": "Scaleway LLM",
        "endpoint": "https://api.scaleway.com/llm",
        "api_key_env": "SCALEWAY_ACCESS_KEY", 
        "models": ["scaleway/mistral-7b", "scaleway/llama-3-70b"],
        "priority": 3
    }
}

model_config = {
    "primary_provider": "openrouter",
    "primary_model": "groq/llama-4-scout",
    "fallback_provider": "anthropic",
    "fallback_model": "claude-4.5-sonnet",
    "routing_strategy": "balanced"
}

# Create model with custom configuration
model = ScalewayModel(
    provider_config=provider_config,
    model_config=model_config
)
```

## 📊 Session Management Integration

### Existing Strands Sessions

```python
# Standard Strands session
from strands.sessions import MemorySessionRepository

# Scaleway-enhanced session  
from island_hopper_sdk import ScalewaySessionRepository

# Migration path
class HybridSessionRepository:
    """Hybrid session repository for gradual migration"""
    
    def __init__(self, use_scaleway=True):
        if use_scaleway:
            self.primary = ScalewaySessionRepository()
            self.fallback = MemorySessionRepository()
        else:
            self.primary = MemorySessionRepository()
            self.fallback = None
    
    async def get_session(self, session_id):
        try:
            return await self.primary.get_session(session_id)
        except Exception as e:
            if self.fallback:
                return await self.fallback.get_session(session_id)
            raise e
    
    async def save_session(self, session):
        try:
            await self.primary.save_session(session)
            if self.fallback:
                await self.fallback.save_session(session)  # Backup
        except Exception as e:
            if self.fallback:
                await self.fallback.save_session(session)
            else:
                raise e
```

## 🛠️ Tool Integration

### Extending Existing Tools

```python
from strands.tools import BaseTool
from island_hopper_tools import ScalewayTool

class EnhancedScalewayTool(ScalewayTool):
    """Enhanced tool with both Strands and Island Hopper features"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add Strands-specific features
        self.strands_compatible = True
    
    async def execute(self, **kwargs):
        # Execute with Scaleway optimizations
        result = await super().execute(**kwargs)
        
        # Add Strands-specific logging
        await self._log_to_strands(result)
        
        return result
    
    async def _log_to_strands(self, result):
        """Log to Strands monitoring system"""
        # Integration with existing Strands monitoring
        pass
```

### Tool Registry Integration

```python
from strands.tools import ToolRegistry as StrandsToolRegistry
from island_hopper_tools import ScalewayToolRegistry

class UnifiedToolRegistry:
    """Unified registry for both Strands and Island Hopper tools"""
    
    def __init__(self):
        self.strands_registry = StrandsToolRegistry()
        self.scaleway_registry = ScalewayToolRegistry()
    
    def register_tool(self, tool):
        """Register tool in appropriate registry"""
        if isinstance(tool, ScalewayTool):
            self.scaleway_registry.register_tool(tool)
        else:
            self.strands_registry.register_tool(tool)
    
    def get_all_tools(self):
        """Get all available tools"""
        return {
            **self.strands_registry.list_tools(),
            **self.scaleway_registry.list_tools()
        }
```

## 📡 A2A Communication Integration

### NATS-based Agent Communication

```python
from strands.a2a import BaseA2AExecutor
from island_hopper_sdk import ScalewayA2AExecutor

class HybridA2AExecutor(BaseA2AExecutor):
    """Hybrid A2A executor with NATS support"""
    
    def __init__(self, use_nats=True, **kwargs):
        super().__init__(**kwargs)
        
        if use_nats:
            self.nats_executor = ScalewayA2AExecutor(**kwargs)
        else:
            self.nats_executor = None
    
    async def send_message(self, recipient, message_type, payload):
        """Send message using NATS if available"""
        if self.nats_executor:
            return await self.nats_executor.send_message(
                recipient, message_type, payload
            )
        else:
            # Fallback to original Strands A2A
            return await super().send_message(
                recipient, message_type, payload
            )
```

## 📈 Monitoring & Telemetry

### Scaleway Cockpit Integration

```python
from strands.telemetry import BaseTelemetry
from island_hopper_sdk import ScalewayTelemetry

class UnifiedTelemetry(BaseTelemetry):
    """Unified telemetry with Scaleway Cockpit"""
    
    def __init__(self, enable_cockpit=True, **kwargs):
        super().__init__(**kwargs)
        
        if enable_cockpit:
            self.cockpit_telemetry = ScalewayTelemetry()
        else:
            self.cockpit_telemetry = None
    
    async def track_request(self, request_data):
        """Track request in both systems"""
        # Track in original Strands
        await super().track_request(request_data)
        
        # Track in Scaleway Cockpit
        if self.cockpit_telemetry:
            await self.cockpit_telemetry.track_request(request_data)
    
    async def track_cost(self, cost_data):
        """Track costs with Scaleway integration"""
        await super().track_cost(cost_data)
        
        if self.cockpit_telemetry:
            # Send cost data to Scaleway Cockpit
            await self.cockpit_telemetry.send_metrics({
                "agent_costs": cost_data,
                "scaleway_costs": await self._get_scaleway_costs()
            })
```

## 🔄 Migration Path

### Phase 1: Parallel Integration
```python
# Run both systems in parallel
class MigrationAgent(BaseAgent):
    def __init__(self, migration_mode=True):
        if migration_mode:
            # Use Island Hopper for new features
            self.model = ScalewayModel()
            self.session_repo = ScalewaySessionRepository()
        else:
            # Use original Strands
            self.model = OpenAIModel()  # Original
            self.session_repo = MemorySessionRepository()  # Original
```

### Phase 2: Gradual Migration
```python
# Gradually migrate components
class GradualMigrationAgent(BaseAgent):
    def __init__(self):
        # Start with Scaleway model
        self.model = ScalewayModel()
        
        # Keep original sessions temporarily
        self.session_repo = MemorySessionRepository()
        
        # Add Scaleway tools progressively
        self.tools = [
            OriginalTool(),  # Keep existing
            ScalewayTool()   # Add new
        ]
```

### Phase 3: Full Migration
```python
# Fully migrated to Island Hopper
class ProductionAgent(ScalewayAgent):
    """Complete migration to Island Hopper SDK"""
    
    def __init__(self):
        super().__init__()
        # All components are Scaleway-optimized
```

## 🧪 Testing Integration

### Integration Tests

```python
import pytest
from unittest.mock import Mock

class TestStrandsIntegration:
    """Test integration between Strands and Island Hopper"""
    
    @pytest.fixture
    def agent(self):
        return ScalewayAgent(
            model=ScalewayModel(),
            session_repo=ScalewaySessionRepository()
        )
    
    async def test_session_compatibility(self, agent):
        """Test that sessions work with both systems"""
        session_id = "test-session"
        
        # Create session using Island Hopper
        session = await agent.session_repo.create_session(session_id)
        
        # Verify it's compatible with Strands interface
        assert hasattr(session, 'session_id')
        assert hasattr(session, 'messages')
    
    async def test_tool_compatibility(self, agent):
        """Test that tools work with both systems"""
        # Test Scaleway tools with Strands interface
        results = await agent.execute_tools([
            {"name": "object_storage", "action": "list_files"}
        ])
        
        assert isinstance(results, list)
        assert len(results) > 0
```

## 📚 Best Practices

### 1. **Gradual Migration**
- Start with model migration (easiest)
- Progress to session management
- Finally migrate tools and A2A communication

### 2. **Backward Compatibility**
- Maintain Strands interfaces during migration
- Use adapter patterns for smooth transitions
- Test thoroughly before removing old components

### 3. **Configuration Management**
- Use environment variables for provider selection
- Implement feature flags for gradual rollout
- Maintain separate configs for each environment

### 4. **Monitoring & Observability**
- Track metrics from both systems during migration
- Implement unified dashboards
- Monitor costs and performance

### 5. **Error Handling**
- Implement graceful fallbacks
- Log integration issues for debugging
- Provide clear migration error messages

## 🔗 Additional Resources

- [Strands SDK Documentation](https://docs.strands.ai/)
- [Island Hopper SDK](https://github.com/klogins-hash/island-hopper-sdk)
- [Scaleway Documentation](https://www.scaleway.com/en/docs/)
- [Migration Examples](https://github.com/klogins-hash/island-hopper-samples)

---

**Need help with integration?** Check out our [examples](../examples/) or open an issue on GitHub! 🚀
