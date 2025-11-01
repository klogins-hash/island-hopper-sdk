# 🔄 Strands to Scaleway Translation Reference

## 📋 Overview

This document serves as a **RAG-searchable translation reference** for adapting existing Strands SDK code to work with Scaleway services. It provides mapping patterns, configuration examples, and migration guidance without duplicating the core Strands functionality.

## 🎯 Purpose

- **Reference Guide**: Look up how to adapt specific Strands patterns for Scaleway
- **RAG-Optimized**: Searchable by Strands class names, methods, and patterns
- **Translation Layer**: Maps existing concepts to Scaleway implementations
- **Migration Helper**: Step-by-step adaptation instructions

---

## 🏗️ Core Architecture Mapping

### Strands Core → Scaleway Integration

| Strands Component | Scaleway Integration | Implementation Pattern |
|-------------------|---------------------|----------------------|
| `strands.Agent` | Scaleway-optimized Agent | Add Scaleway model + session repo |
| `strands.Model` | Provider-agnostic routing | Use OpenRouter/Anthropic via Scaleway |
| `strands.Session` | PostgreSQL session storage | Scaleway Database + Cockpit |
| `strands.Tools` | Scaleway-native tools | Object Storage, Database, NATS |
| `strands.Telemetry` | Cockpit integration | OpenTelemetry + Scaleway metrics |
| `strands.A2A` | NATS-based communication | JetStream + persistence |

---

## 🔧 Model Layer Translation

### OpenAI Model → Scaleway Provider Routing

#### Original Strands Pattern
```python
from strands.models import OpenAIModel

model = OpenAIModel(
    api_key="sk-...",
    model="gpt-4",
    base_url="https://api.openai.com/v1"
)
```

#### Scaleway Translation
```python
# Option 1: Direct replacement with provider routing
from island_hopper_sdk import ScalewayModel, create_scaleway_model

model = create_scaleway_model(
    primary_provider="openrouter",
    primary_model="groq/llama-4-scout",
    fallback_provider="anthropic",
    fallback_model="claude-4.5-sonnet"
)

# Option 2: Keep existing pattern, add Scaleway backend
model = ScalewayModel(
    provider_config={
        "openrouter": {
            "endpoint": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "models": ["groq/llama-4-scout", "anthropic/claude-4.5-sonnet"]
        }
    },
    model_config={
        "primary_provider": "openrouter",
        "primary_model": "groq/llama-4-scout"
    }
)
```

#### Environment Configuration
```bash
# Original: OpenAI only
export OPENAI_API_KEY="sk-..."

# Scaleway: Multiple providers
export OPENROUTER_API_KEY="sk-or-v1-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."  # Still supported
```

#### Key Differences
- **Provider Agnostic**: No locked to single provider
- **Automatic Failover**: Built-in fallback handling
- **Cost Optimization**: Routing strategies for cost/quality/speed
- **Scaleway Native**: Optimized for Scaleway infrastructure

---

## 🗄️ Session Management Translation

### Memory Sessions → Scaleway PostgreSQL

#### Original Strands Pattern
```python
from strands.sessions import MemorySessionRepository

sessions = MemorySessionRepository()
await sessions.save_session(session_id, messages)
```

#### Scaleway Translation
```python
# Option 1: Direct replacement
from island_hopper_sdk import ScalewaySessionRepository

sessions = ScalewaySessionRepository(
    connection_string="postgresql://user:pass@scaleway-db:5432/strands"
)

# Option 2: Hybrid approach (gradual migration)
class HybridSessionRepository:
    def __init__(self, use_scaleway=True):
        self.scaleway = ScalewaySessionRepository() if use_scaleway else None
        self.memory = MemorySessionRepository()  # Fallback
    
    async def save_session(self, session_id, data):
        if self.scaleway:
            try:
                return await self.scaleway.save_session(session_id, data)
            except Exception:
                pass  # Fallback to memory
        return await self.memory.save_session(session_id, data)
```

#### Database Schema Translation
```sql
-- Original: In-memory sessions
-- No schema required

-- Scaleway: PostgreSQL sessions
CREATE TABLE strands_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    messages JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_session_id ON strands_sessions(session_id);
CREATE INDEX idx_sessions_created_at ON strands_sessions(created_at);
```

#### Configuration Translation
```python
# Original: No configuration
sessions = MemorySessionRepository()

# Scaleway: Database configuration
sessions = ScalewaySessionRepository(
    connection_string=os.getenv("SCALEWAY_DATABASE_URL"),
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600
)
```

---

## 🛠️ Tool System Translation

### Base Tools → Scaleway-Native Tools

#### Original Strands Pattern
```python
from strands.tools import BaseTool

class CustomTool(BaseTool):
    async def execute(self, **kwargs):
        # Custom implementation
        return result
```

#### Scaleway Translation
```python
from island_hopper_tools import ScalewayTool, ObjectStorageTool, DatabaseTool

# Option 1: Use existing Scaleway tools
tools = [
    ObjectStorageTool(
        access_key=os.getenv("SCALEWAY_ACCESS_KEY"),
        bucket="agent-data"
    ),
    DatabaseTool(
        connection_string=os.getenv("SCALEWAY_DATABASE_URL")
    )
]

# Option 2: Extend Scaleway tool for custom functionality
class ScalewayCustomTool(ScalewayTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = ObjectStorageTool()
        self.database = DatabaseTool()
    
    async def execute(self, operation, **kwargs):
        if operation == "store_data":
            # Use Scaleway Object Storage
            return await self.storage.upload_file(**kwargs)
        elif operation == "query_data":
            # Use Scaleway Database
            return await self.database.execute_query(**kwargs)
```

#### Tool Registration Translation
```python
# Original: Simple registration
from strands.tools import ToolRegistry

registry = ToolRegistry()
registry.register_tool(CustomTool())

# Scaleway: Dynamic registration with validation
from island_hopper_tools import ScalewayToolRegistry

registry = ScalewayToolRegistry(
    auto_discover=True,
    enable_validation=True,
    search_paths=["./tools", "./scaleway_tools"]
)

# Automatic discovery of Scaleway tools
tools = registry.list_tools()
# ["ObjectStorageTool", "DatabaseTool", "NATSTool", ...]
```

---

## 📊 Telemetry Translation

### Basic Logging → Scaleway Cockpit Integration

#### Original Strands Pattern
```python
from strands.telemetry import BaseTelemetry

telemetry = BaseTelemetry()
await telemetry.track_request({"model": "gpt-4", "tokens": 100})
```

#### Scaleway Translation
```python
from island_hopper_sdk import ScalewayTelemetry, configure_scaleway_telemetry

# Option 1: Direct Cockpit integration
telemetry = ScalewayTelemetry(
    endpoint="https://otel.fr-par.scw.cloud",
    service_name="strands-agent",
    enable_cockpit=True
)

# Option 2: Enhanced with cost tracking
telemetry = configure_scaleway_telemetry(
    service_name="strands-agent",
    enable_cost_tracking=True,
    enable_cockpit=True,
    budget_limit=100.0  # USD
)

await telemetry.track_request({
    "model": "groq/llama-4-scout",
    "provider": "openrouter",
    "tokens": 100,
    "cost": 0.005,
    "session_id": "user-123"
})
```

#### Metrics Translation
```python
# Original: Basic metrics
telemetry.log_metric("request_count", 1)

# Scaleway: Rich metrics with Cockpit
telemetry.log_metric({
    "name": "request_count",
    "value": 1,
    "labels": {
        "provider": "openrouter",
        "model": "groq/llama-4-scout",
        "region": "fr-par"
    },
    "cockpit_dimension": "agent_metrics"
})
```

---

## 📡 A2A Communication Translation

### Basic Messaging → NATS with JetStream

#### Original Strands Pattern
```python
from strands.a2a import BaseA2AExecutor

a2a = BaseA2AExecutor()
await a2a.send_message("agent-2", "request", {"data": "payload"})
```

#### Scaleway Translation
```python
from island_hopper_sdk import ScalewayA2AExecutor, create_scaleway_a2a_executor

# Option 1: NATS-based communication
a2a = ScalewayA2AExecutor(
    agent_id="agent-1",
    nats_url="nats://scaleway-nats:4222",
    enable_jetstream=True,
    enable_persistence=True
)

# Option 2: Factory function with defaults
a2a = create_scaleway_a2a_executor(
    agent_id="agent-1",
    cluster_name="island-hopper"
)

await a2a.send_message(
    recipient_id="agent-2",
    message_type="request",
    payload={"data": "payload"},
    timeout=30.0,
    persistent=True  # Stored in JetStream
)
```

#### Message Patterns Translation
```python
# Original: Simple request/response
response = await a2a.request("agent-2", {"query": "data"})

# Scaleway: Enhanced with patterns and context
from island_hopper_sdk import A2AContext, MessageType

context = A2AContext(
    conversation_id="conv-123",
    user_id="user-456",
    session_id="session-789"
)

response = await a2a.send_message(
    recipient_id="agent-2",
    message_type=MessageType.REQUEST,
    payload={"query": "data"},
    context=context,
    priority="high",
    retry_attempts=3
)
```

---

## 🔧 Configuration Translation

### Static Config → Dynamic Scaleway Configuration

#### Original Strands Pattern
```python
# Hardcoded configuration
model = OpenAIModel(api_key="sk-...", model="gpt-4")
sessions = MemorySessionRepository()
```

#### Scaleway Translation
```python
# Environment-driven configuration
import os
from island_hopper_sdk import create_scaleway_model, ScalewaySessionRepository

# Model configuration from environment
model = create_scaleway_model(
    primary_provider=os.getenv("PRIMARY_PROVIDER", "openrouter"),
    primary_model=os.getenv("PRIMARY_MODEL", "groq/llama-4-scout"),
    fallback_provider=os.getenv("FALLBACK_PROVIDER", "anthropic"),
    fallback_model=os.getenv("FALLBACK_MODEL", "claude-4.5-sonnet")
)

# Session configuration from environment
sessions = ScalewaySessionRepository(
    connection_string=os.getenv("SCALEWAY_DATABASE_URL"),
    pool_size=int(os.getenv("DB_POOL_SIZE", "10"))
)
```

#### Configuration File Translation
```yaml
# Original: config.yaml
model:
  api_key: "sk-..."
  model: "gpt-4"

# Scaleway: scaleway-config.yaml
scaleway:
  model:
    primary_provider: "openrouter"
    primary_model: "groq/llama-4-scout"
    fallback_provider: "anthropic"
    fallback_model: "claude-4.5-sonnet"
    routing_strategy: "balanced"
  
  database:
    connection_string: "${SCALEWAY_DATABASE_URL}"
    pool_size: 10
    enable_backup: true
  
  object_storage:
    endpoint: "s3.fr-par.scw.cloud"
    bucket: "strands-data"
    access_key: "${SCALEWAY_ACCESS_KEY}"
  
  nats:
    url: "nats://scaleway-nats:4222"
    cluster_name: "island-hopper"
    enable_jetstream: true
```

---

## 🚀 Deployment Translation

### Local Development → Scaleway Infrastructure

#### Original Strands Pattern
```python
# Local development
agent = Agent(model=OpenAIModel(), sessions=MemorySessionRepository())
```

#### Scaleway Translation
```python
# Production-ready with Scaleway
from island_hopper_sdk import ScalewayModel, ScalewaySessionRepository
from island_hopper_tools import ObjectStorageTool, DatabaseTool

# Production configuration
model = ScalewayModel(
    provider_config="production",  # Loaded from config
    model_config="production"
)

sessions = ScalewaySessionRepository(
    connection_string=os.getenv("SCALEWAY_DATABASE_URL"),
    enable_ssl=True,
    backup_enabled=True
)

tools = [
    ObjectStorageTool(
        bucket=os.getenv("SCALEWAY_BUCKET"),
        encryption_enabled=True
    ),
    DatabaseTool(
        connection_string=os.getenv("SCALEWAY_DATABASE_URL"),
        readonly_replicas=True
    )
]

agent = Agent(
    model=model,
    session_repository=sessions,
    tools=tools,
    enable_telemetry=True,
    enable_monitoring=True
)
```

#### Docker Translation
```dockerfile
# Original: Simple Dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]

# Scaleway: Multi-stage with optimizations
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim as production
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["python", "-m", "strands.server", "--host", "0.0.0.0"]
```

---

## 📈 Performance Optimization Translation

### Basic Caching → Scaleway-Optimized Performance

#### Original Strands Pattern
```python
# Simple in-memory caching
cache = {}

def get_cached_result(key):
    return cache.get(key)
```

#### Scaleway Translation
```python
# Scaleway Redis + application caching
import redis
from island_hopper_sdk import ScalewayCache

cache = ScalewayCache(
    redis_url=os.getenv("SCALEWAY_REDIS_URL"),
    ttl=3600,
    enable_compression=True
)

async def get_cached_result(key):
    # Try Redis first (Scaleway)
    result = await cache.get(key)
    if result:
        return result
    
    # Fallback to computation
    result = await compute_expensive_operation(key)
    await cache.set(key, result)
    return result
```

---

## 🔍 Search Patterns for RAG

### Common Search Queries

| Search Term | Translation Pattern |
|-------------|-------------------|
| "OpenAIModel" | ScalewayModel with provider routing |
| "MemorySessionRepository" | ScalewaySessionRepository with PostgreSQL |
| "BaseTool" | ScalewayTool with Object Storage/Database |
| "BaseTelemetry" | ScalewayTelemetry with Cockpit integration |
| "BaseA2AExecutor" | ScalewayA2AExecutor with NATS/JetStream |
| "docker deploy" | Scaleway Kapsys deployment |
| "configuration" | Environment-driven Scaleway config |
| "monitoring" | Cockpit + OpenTelemetry integration |

### Migration Search Patterns

```python
# Find these patterns in existing code:
# 1. Model initialization
model = OpenAIModel(...)  # → ScalewayModel(...)

# 2. Session creation  
sessions = MemorySessionRepository()  # → ScalewaySessionRepository(...)

# 3. Tool definition
class CustomTool(BaseTool):  # → class CustomTool(ScalewayTool):

# 4. Telemetry setup
telemetry = BaseTelemetry()  # → ScalewayTelemetry(...)

# 5. A2A communication
a2a = BaseA2AExecutor()  # → ScalewayA2AExecutor(...)
```

---

## 📚 Quick Reference Cheat Sheet

### Essential Imports
```python
# Core replacements
from island_hopper_sdk import ScalewayModel, create_scaleway_model
from island_hopper_sdk import ScalewaySessionRepository  
from island_hopper_sdk import ScalewayTelemetry, configure_scaleway_telemetry
from island_hopper_sdk import ScalewayA2AExecutor, create_scaleway_a2a_executor

# Tools
from island_hopper_tools import ScalewayTool, ObjectStorageTool, DatabaseTool
from island_hopper_tools import ScalewayToolRegistry

# Environment
import os
# Required: OPENROUTER_API_KEY, ANTHROPIC_API_KEY, SCALEWAY_ACCESS_KEY
# Required: SCALEWAY_SECRET_KEY, SCALEWAY_DATABASE_URL, NATS_URL
```

### One-Liner Replacements
```python
# Model
model = create_scaleway_model()  # Replaces OpenAIModel configuration

# Sessions  
sessions = ScalewaySessionRepository()  # Replaces MemorySessionRepository

# Tools
tools = [ObjectStorageTool(), DatabaseTool()]  # Replaces custom tool setup

# A2A
a2a = create_scaleway_a2a_executor("agent-id")  # Replaces BaseA2AExecutor

# Telemetry
telemetry = configure_scaleway_telemetry("service-name")  # Enhanced monitoring
```

---

## 🔗 Additional Resources

### Documentation Links
- [Scaleway Documentation](https://www.scaleway.com/en/docs/)
- [Strands SDK Reference](https://docs.strands.ai/)
- [Island Hopper Examples](https://github.com/klogins-hash/island-hopper-samples)
- [Migration Guide](./MIGRATION_GUIDE.md)

### Configuration Templates
- [Development Config](./configs/development.yaml)
- [Production Config](./configs/production.yaml)  
- [Docker Examples](./docker/examples/)
- [Kubernetes Manifests](../infrastructure/kubernetes/)

---

**This reference is designed to be RAG-searchable. Look up any Strands class or pattern to find its Scaleway equivalent!** 🚀
