# 🚀 Strands to Scaleway Migration Cheat Sheet

## ⚡ Quick Migration Patterns

### 1. Model Migration (5 minutes)
```python
# BEFORE
from strands.models import OpenAIModel
model = OpenAIModel(api_key="sk-...", model="gpt-4")

# AFTER  
from island_hopper_sdk import create_scaleway_model
model = create_scaleway_model(
    primary_provider="openrouter",
    primary_model="groq/llama-4-scout",
    fallback_provider="anthropic",
    fallback_model="claude-4.5-sonnet"
)
```

### 2. Session Migration (5 minutes)
```python
# BEFORE
from strands.sessions import MemorySessionRepository
sessions = MemorySessionRepository()

# AFTER
from island_hopper_sdk import ScalewaySessionRepository
sessions = ScalewaySessionRepository(
    connection_string=os.getenv("SCALEWAY_DATABASE_URL")
)
```

### 3. Tools Migration (10 minutes)
```python
# BEFORE
from strands.tools import BaseTool
class MyTool(BaseTool): pass

# AFTER
from island_hopper_tools import ScalewayTool, ObjectStorageTool, DatabaseTool
tools = [ObjectStorageTool(), DatabaseTool()]
```

### 4. A2A Migration (5 minutes)
```python
# BEFORE
from strands.a2a import BaseA2AExecutor
a2a = BaseA2AExecutor()

# AFTER
from island_hopper_sdk import create_scaleway_a2a_executor
a2a = create_scaleway_a2a_executor("my-agent-id")
```

## 🌍 Environment Setup

```bash
# Add these to your .env file
export OPENROUTER_API_KEY="sk-or-v1-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export SCALEWAY_ACCESS_KEY="SCW..."
export SCALEWAY_SECRET_KEY="..."
export SCALEWAY_DATABASE_URL="postgresql://..."
export NATS_URL="nats://..."
```

## 📦 Install Dependencies

```bash
pip install island-hopper-sdk island-hopper-tools
```

## ✅ Verification Test

```python
import asyncio
from island_hopper_sdk import create_scaleway_model, ScalewaySessionRepository

async def test_migration():
    # Test model
    model = create_scaleway_model()
    print("✅ Model works")
    
    # Test sessions (if DB configured)
    try:
        sessions = ScalewaySessionRepository()
        print("✅ Sessions work")
    except Exception as e:
        print(f"⚠️  Sessions need DB setup: {e}")
    
    print("🎉 Migration ready!")

asyncio.run(test_migration())
```

---

**That's it! Your Strands code is now Scaleway-optimized!** 🏝️
