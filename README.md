# 🏝️ Island Hopper SDK

**Provider-agnostic Strands SDK optimized for Scaleway**

## 🎯 Overview

Island Hopper extends the Strands Agents framework with Scaleway-native integrations while maintaining provider flexibility. Bring your own API keys and use any OpenAI-compatible provider.

## ✨ Key Features

- 🔄 **Provider-Agnostic**: Works with OpenAI, Anthropic, OpenRouter, or any OpenAI-compatible API
- 🏗️ **Scaleway-Native**: PostgreSQL sessions, Cockpit telemetry, NATS messaging
- 💰 **Cost Tracking**: Built-in cost monitoring and budget enforcement
- 🛠️ **Production Hooks**: Security, quotas, and monitoring hooks
- 🚀 **Agent Swarms**: Multi-agent coordination via NATS events

## 🚀 Quick Start

```python
from strands import Agent
from strands.models.scaleway import ScalewayModel

# Use any provider via API keys
agent = Agent(
    model=ScalewayModel(
        primary_provider="openrouter",
        primary_model="llama-4-scout",
        api_keys={
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY")  # optional fallback
        }
    )
)

result = agent("What is the capital of France?")
print(result.message)
```

## 📦 Installation

```bash
pip install island-hopper-sdk
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your Agent    │───▶│  ScalewayModel   │───▶│  Any Provider  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Scaleway Hooks  │    │ Scaleway Session │    │ Scaleway Telemetry│
│ (Cost/Security) │    │ (PostgreSQL)     │    │ (Cockpit)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration](docs/configuration.md)
- [Provider Setup](docs/providers.md)
- [Sessions & State](docs/sessions.md)
- [Cost Tracking](docs/cost-tracking.md)
- [Hooks & Monitoring](docs/hooks.md)
- [Multi-Agent Patterns](docs/multi-agent.md)

## 🤝 Contributing

We welcome contributions! See [Contributing Guide](CONTRIBUTING.md).

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) file.

## 🔗 Links

- [GitHub](https://github.com/klogins-hash/island-hopper-sdk)
- [Island Hopper Project](https://github.com/klogins-hash/island-hopper)
- [Strands Agents](https://github.com/strands-agents/sdk-python)

---

**Island Hopper - Bring your own API keys. Use any provider. No lock-in.** 🏝️
