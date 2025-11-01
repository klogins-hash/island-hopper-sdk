"""
Provider-agnostic Scaleway Model

Works with any OpenAI-compatible provider via API keys.
Flexible routing with configurable primary and fallback providers.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Optional imports with fallbacks
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class RoutingStrategy(Enum):
    """Strategies for routing requests to providers"""
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"
    SPEED_FIRST = "speed_first"


@dataclass
class ProviderConfig:
    """Configuration for a single provider"""
    name: str
    endpoint: str
    api_key_env: str
    models: List[str]
    priority: int = 1  # 1=highest priority
    max_rpm: Optional[int] = None
    max_tpm: Optional[int] = None


@dataclass
class ModelConfig:
    """Configuration for model routing"""
    primary_provider: str
    primary_model: str
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED


class ModelRouter:
    """Routes requests to appropriate provider based on configuration"""
    
    def __init__(self, providers: Dict[str, ProviderConfig], model_config: ModelConfig):
        self.providers = providers
        self.model_config = model_config
        self.logger = logging.getLogger(__name__)
        
        # Load API keys from environment
        self.api_keys = self._load_api_keys()
        
        # Validate configuration
        self._validate_config()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables"""
        keys = {}
        for provider_name, provider in self.providers.items():
            key = os.getenv(provider.api_key_env)
            if not key:
                raise ValueError(f"Missing API key for {provider_name}: set {provider.api_key_env}")
            keys[provider_name] = key
        return keys
    
    def _validate_config(self):
        """Validate the provider and model configuration"""
        # Check primary provider exists
        if self.model_config.primary_provider not in self.providers:
            raise ValueError(f"Primary provider {self.model_config.primary_provider} not configured")
        
        # Check fallback provider exists if specified
        if (self.model_config.fallback_provider and 
            self.model_config.fallback_provider not in self.providers):
            raise ValueError(f"Fallback provider {self.model_config.fallback_provider} not configured")
    
    def get_primary_client(self) -> OpenAI:
        """Get OpenAI client for primary provider"""
        provider = self.providers[self.model_config.primary_provider]
        return OpenAI(
            api_key=self.api_keys[self.model_config.primary_provider],
            base_url=provider.endpoint
        )
    
    def get_fallback_client(self) -> Optional[OpenAI]:
        """Get OpenAI client for fallback provider"""
        if not self.model_config.fallback_provider:
            return None
        
        provider = self.providers[self.model_config.fallback_provider]
        return OpenAI(
            api_key=self.api_keys[self.model_config.fallback_provider],
            base_url=provider.endpoint
        )
    
    def get_model_for_provider(self, provider_name: str) -> str:
        """Get the model to use for a specific provider"""
        if provider_name == self.model_config.primary_provider:
            return self.model_config.primary_model
        elif provider_name == self.model_config.fallback_provider:
            return self.model_config.fallback_model or self.model_config.primary_model
        else:
            raise ValueError(f"Unknown provider: {provider_name}")


class ScalewayModel:
    """
    Provider-agnostic model with flexible routing.
    
    Supports any OpenAI-compatible provider:
    - OpenAI (GPT-4, etc.)
    - Anthropic (Claude)
    - OpenRouter (100+ models)
    - Groq (via OpenRouter)
    - Custom endpoints
    
    Features:
    - Configurable primary and fallback providers
    - Cost optimization strategies
    - Automatic failover
    - Rate limiting awareness
    """
    
    def __init__(
        self,
        provider_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize ScalewayModel with provider configuration.
        
        Args:
            provider_config: Dictionary of provider configurations
            model_config: Dictionary of model routing configuration
            **kwargs: Additional arguments
        """
        # Load default configuration if none provided
        provider_config = provider_config or self._get_default_providers()
        model_config = model_config or self._get_default_model_config()
        
        # Initialize configuration
        self.model_config = ModelConfig(**model_config)
        self.router = ProviderRouter(provider_config, self.model_config)
        
        # Initialize OpenAI clients for each provider
        self.clients = {}
        for provider_name in provider_config.keys():
            self.clients[provider_name] = self._create_client_for_provider(provider_name)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ScalewayModel initialized with primary provider: {self.model_config.primary_provider}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the configured provider routing"""
        try:
            # Get the best provider for this request
            provider_name = self.router.get_best_provider()
            model_name = self.get_model_for_provider(provider_name)
            
            # Get the appropriate client
            client = self.clients[provider_name]
            
            # Make the API call
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            result = response.choices[0].message.content
            
            # Log the usage for routing optimization
            self.router.log_usage(provider_name, response.usage)
            
            return result
            
        except Exception as e:
            # Try fallback provider if available
            if self.model_config.fallback_provider and provider_name != self.model_config.fallback_provider:
                self.logger.warning(f"Primary provider {provider_name} failed, trying fallback: {e}")
                try:
                    fallback_client = self.clients[self.model_config.fallback_provider]
                    fallback_model = self.get_model_for_provider(self.model_config.fallback_provider)
                    
                    response = fallback_client.chat.completions.create(
                        model=fallback_model,
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                    
                    return response.choices[0].message.content
                    
                except Exception as fallback_error:
                    self.logger.error(f"Fallback provider also failed: {fallback_error}")
                    raise Exception(f"All providers failed. Primary: {e}, Fallback: {fallback_error}")
            else:
                raise e
    
    def switch_provider(self, provider_name: str, model_name: str):
        """Switch to a specific provider and model"""
        if provider_name not in self.clients:
            raise ValueError(f"Provider {provider_name} not configured")
        
        self.model_config.primary_provider = provider_name
        self.model_config.primary_model = model_name
        
        self.logger.info(f"Switched to provider: {provider_name}, model: {model_name}")
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider configuration"""
        return {
            "primary_provider": self.model_config.primary_provider,
            "primary_model": self.model_config.primary_model,
            "fallback_provider": self.model_config.fallback_provider,
            "fallback_model": self.model_config.fallback_model,
            "routing_strategy": self.model_config.routing_strategy.value,
            "available_providers": list(self.clients.keys())
        }
    
    def _get_default_providers(self) -> Dict[str, Any]:
        """Get default provider configuration"""
        return {
            "openrouter": {
                "name": "OpenRouter",
                "endpoint": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "models": ["groq/llama-4-scout", "anthropic/claude-4.5-sonnet", "openai/gpt-4"],
                "priority": 1
            },
            "anthropic": {
                "name": "Anthropic",
                "endpoint": "https://api.anthropic.com",
                "api_key_env": "ANTHROPIC_API_KEY",
                "models": ["claude-4.5-sonnet", "claude-3-5-sonnet"],
                "priority": 2
            },
            "openai": {
                "name": "OpenAI",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "priority": 3
            }
        }
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            "primary_provider": "openrouter",
            "primary_model": "groq/llama-4-scout",
            "fallback_provider": "anthropic",
            "fallback_model": "claude-4.5-sonnet",
            "routing_strategy": "balanced"
        }
    
    def _create_client_for_provider(self, provider_name: str) -> OpenAI:
        """Create OpenAI client for a specific provider"""
        provider = self.router.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not configured")
        
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found for provider {provider_name}. Set {provider.api_key_env} environment variable.")
        
        return OpenAI(
            api_key=api_key,
            base_url=provider.endpoint
        )
    
    def get_model_for_provider(self, provider_name: str) -> str:
        """Get the model to use for a specific provider"""
        if provider_name == self.model_config.primary_provider:
            return self.model_config.primary_model
        elif provider_name == self.model_config.fallback_provider:
            return self.model_config.fallback_model or self.model_config.primary_model
        else:
            raise ValueError(f"Unknown provider: {provider_name}")


def create_scaleway_model(
    primary_provider: str = "openrouter",
    primary_model: str = "groq/llama-4-scout",
    fallback_provider: Optional[str] = "anthropic",
    fallback_model: Optional[str] = "claude-4.5-sonnet",
    **kwargs
) -> ScalewayModel:
    """
    Convenience function to create a ScalewayModel with common configuration.
    
    Args:
        primary_provider: Primary provider name
        primary_model: Primary model name
        fallback_provider: Fallback provider name
        fallback_model: Fallback model name
        **kwargs: Additional arguments
        
    Returns:
        Configured ScalewayModel instance
    """
    model_config = {
        "primary_provider": primary_provider,
        "primary_model": primary_model,
        "fallback_provider": fallback_provider,
        "fallback_model": fallback_model,
        "routing_strategy": "balanced"
    }
    
    return ScalewayModel(model_config=model_config, **kwargs)
