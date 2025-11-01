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

import openai
from openai import OpenAI
import httpx

from strands.models.openai import OpenAIModel


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


class ScalewayModel(OpenAIModel):
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
            **kwargs: Additional arguments passed to OpenAIModel
        """
        # Load default configuration if none provided
        provider_config = provider_config or self._get_default_providers()
        model_config = model_config or self._get_default_model_config()
        
        # Parse configurations
        providers = self._parse_providers(provider_config)
        model_cfg = self._parse_model_config(model_config)
        
        # Initialize router
        self.router = ModelRouter(providers, model_cfg)
        
        # Get primary client for initialization
        primary_client = self.router.get_primary_client()
        self.fallback_client = self.router.get_fallback_client()
        
        # Initialize with primary provider
        super().__init__(
            client=primary_client,
            model=self.router.get_model_for_provider(model_cfg.primary_provider),
            **kwargs
        )
        
        # Store configuration for failover
        self.model_config = model_cfg
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"ScalewayModel initialized with primary provider: {model_cfg.primary_provider}"
        )
    
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
                "endpoint": "https://api.anthropic.com/v1",
                "api_key_env": "ANTHROPIC_API_KEY",
                "models": ["claude-4.5-sonnet", "claude-3-5-sonnet"],
                "priority": 2
            },
            "openai": {
                "name": "OpenAI",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "models": ["gpt-4", "gpt-4-turbo"],
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
            "routing_strategy": "cost_optimized"
        }
    
    def _parse_providers(self, provider_config: Dict[str, Any]) -> Dict[str, ProviderConfig]:
        """Parse provider configuration dictionary"""
        providers = {}
        for key, config in provider_config.items():
            providers[key] = ProviderConfig(
                name=config["name"],
                endpoint=config["endpoint"],
                api_key_env=config["api_key_env"],
                models=config["models"],
                priority=config.get("priority", 1),
                max_rpm=config.get("max_rpm"),
                max_tpm=config.get("max_tpm")
            )
        return providers
    
    def _parse_model_config(self, model_config: Dict[str, Any]) -> ModelConfig:
        """Parse model configuration dictionary"""
        return ModelConfig(
            primary_provider=model_config["primary_provider"],
            primary_model=model_config["primary_model"],
            fallback_provider=model_config.get("fallback_provider"),
            fallback_model=model_config.get("fallback_model"),
            routing_strategy=RoutingStrategy(model_config.get("routing_strategy", "balanced"))
        )
    
    async def _call_with_fallback(self, *args, **kwargs):
        """Make API call with automatic fallback to secondary provider"""
        try:
            # Try primary provider first
            return await super()._call(*args, **kwargs)
        except Exception as primary_error:
            if self.fallback_client:
                self.logger.warning(
                    f"Primary provider {self.model_config.primary_provider} failed: {primary_error}. "
                    f"Trying fallback {self.model_config.fallback_provider}"
                )
                
                # Temporarily switch to fallback client
                original_client = self.client
                original_model = self.model
                
                self.client = self.fallback_client
                self.model = self.router.get_model_for_provider(self.model_config.fallback_provider)
                
                try:
                    result = await super()._call(*args, **kwargs)
                    self.logger.info(f"Fallback provider {self.model_config.fallback_provider} succeeded")
                    return result
                except Exception as fallback_error:
                    self.logger.error(f"Fallback provider also failed: {fallback_error}")
                    raise Exception(f"Both providers failed. Primary: {primary_error}, Fallback: {fallback_error}")
                finally:
                    # Restore original client
                    self.client = original_client
                    self.model = original_model
            else:
                # No fallback configured
                raise primary_error
    
    # Override the call method to use fallback
    async def _call(self, *args, **kwargs):
        """Override to add fallback logic"""
        return await self._call_with_fallback(*args, **kwargs)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider configuration"""
        return {
            "primary_provider": self.model_config.primary_provider,
            "primary_model": self.model_config.primary_model,
            "fallback_provider": self.model_config.fallback_provider,
            "fallback_model": self.model_config.fallback_model,
            "routing_strategy": self.model_config.routing_strategy.value,
            "available_providers": list(self.router.providers.keys())
        }
    
    def switch_provider(self, new_primary: str, new_model: Optional[str] = None):
        """
        Switch to a different provider at runtime.
        
        Args:
            new_primary: Name of the new primary provider
            new_model: Optional new model name
        """
        if new_primary not in self.router.providers:
            raise ValueError(f"Provider {new_primary} not configured")
        
        # Update configuration
        old_primary = self.model_config.primary_provider
        self.model_config.primary_provider = new_primary
        if new_model:
            self.model_config.primary_model = new_model
        
        # Update client
        self.client = self.router.get_primary_client()
        self.model = self.router.get_model_for_provider(new_primary)
        
        self.logger.info(f"Switched from {old_primary} to {new_primary}")


# Convenience function for easy initialization
def create_scaleway_model(
    primary_provider: str = "openrouter",
    primary_model: str = "groq/llama-4-scout",
    fallback_provider: Optional[str] = None,
    fallback_model: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None,
    **kwargs
) -> ScalewayModel:
    """
    Convenience function to create a ScalewayModel with common configurations.
    
    Args:
        primary_provider: Name of primary provider
        primary_model: Model to use with primary provider
        fallback_provider: Optional fallback provider
        fallback_model: Optional model for fallback
        api_keys: Optional API keys (will use environment if not provided)
        **kwargs: Additional arguments for ScalewayModel
    
    Returns:
        Configured ScalewayModel instance
    """
    # Set environment variables if API keys provided
    if api_keys:
        for provider, key in api_keys.items():
            provider_config = {
                "openrouter": "OPENROUTER_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY"
            }
            if provider in provider_config:
                os.environ[provider_config[provider]] = key
    
    model_config = {
        "primary_provider": primary_provider,
        "primary_model": primary_model,
        "fallback_provider": fallback_provider,
        "fallback_model": fallback_model,
        "routing_strategy": "cost_optimized"
    }
    
    return ScalewayModel(model_config=model_config, **kwargs)
