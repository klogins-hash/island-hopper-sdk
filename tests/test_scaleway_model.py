"""
Tests for ScalewayModel
"""

import pytest
import os
from unittest.mock import Mock, patch

from strands.models.scaleway import ScalewayModel, create_scaleway_model, RoutingStrategy


class TestScalewayModel:
    """Test cases for ScalewayModel"""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration"""
        # Mock environment variables
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key-1',
            'ANTHROPIC_API_KEY': 'test-key-2',
            'OPENAI_API_KEY': 'test-key-3'
        }):
            model = ScalewayModel()
            
            assert model.router.model_config.primary_provider == "openrouter"
            assert model.router.model_config.primary_model == "groq/llama-4-scout"
            assert model.router.model_config.fallback_provider == "anthropic"
            assert model.router.model_config.routing_strategy == RoutingStrategy.COST_OPTIMIZED
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration"""
        provider_config = {
            "openai": {
                "name": "OpenAI",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "models": ["gpt-4"],
                "priority": 1
            }
        }
        
        model_config = {
            "primary_provider": "openai",
            "primary_model": "gpt-4",
            "routing_strategy": "quality_first"
        }
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            model = ScalewayModel(
                provider_config=provider_config,
                model_config=model_config
            )
            
            assert model.router.model_config.primary_provider == "openai"
            assert model.router.model_config.primary_model == "gpt-4"
            assert model.router.model_config.routing_strategy == RoutingStrategy.QUALITY_FIRST
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises error"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing API key"):
                ScalewayModel()
    
    def test_get_provider_info(self):
        """Test getting provider information"""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key-1',
            'ANTHROPIC_API_KEY': 'test-key-2'
        }):
            model = ScalewayModel()
            info = model.get_provider_info()
            
            assert "primary_provider" in info
            assert "primary_model" in info
            assert "fallback_provider" in info
            assert "routing_strategy" in info
            assert "available_providers" in info
    
    def test_switch_provider(self):
        """Test switching providers at runtime"""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key-1',
            'ANTHROPIC_API_KEY': 'test-key-2',
            'OPENAI_API_KEY': 'test-key-3'
        }):
            model = ScalewayModel()
            
            # Switch to OpenAI
            model.switch_provider("openai", "gpt-4")
            
            assert model.router.model_config.primary_provider == "openai"
            assert model.router.model_config.primary_model == "gpt-4"
    
    def test_switch_to_unknown_provider_raises_error(self):
        """Test switching to unknown provider raises error"""
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
            model = ScalewayModel()
            
            with pytest.raises(ValueError, match="Provider unknown not configured"):
                model.switch_provider("unknown")


class TestCreateScalewayModel:
    """Test cases for create_scaleway_model convenience function"""
    
    def test_create_with_defaults(self):
        """Test creating model with default parameters"""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key-1',
            'ANTHROPIC_API_KEY': 'test-key-2'
        }):
            model = create_scaleway_model()
            
            assert isinstance(model, ScalewayModel)
            assert model.router.model_config.primary_provider == "openrouter"
    
    def test_create_with_api_keys(self):
        """Test creating model with provided API keys"""
        api_keys = {
            "openrouter": "custom-key-1",
            "anthropic": "custom-key-2"
        }
        
        with patch.dict(os.environ, {}, clear=True):
            model = create_scaleway_model(api_keys=api_keys)
            
            assert isinstance(model, ScalewayModel)
            assert os.getenv('OPENROUTER_API_KEY') == "custom-key-1"
            assert os.getenv('ANTHROPIC_API_KEY') == "custom-key-2"
    
    def test_create_with_custom_provider(self):
        """Test creating model with custom provider"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            model = create_scaleway_model(
                primary_provider="openai",
                primary_model="gpt-4"
            )
            
            assert model.router.model_config.primary_provider == "openai"
            assert model.router.model_config.primary_model == "gpt-4"


if __name__ == "__main__":
    pytest.main([__file__])
