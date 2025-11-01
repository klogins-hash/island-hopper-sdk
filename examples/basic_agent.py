"""
Basic Agent Example

Demonstrates using Island Hopper SDK with provider-agnostic routing.
"""

import os
import asyncio
import logging

from strands import Agent
from strands.models.scaleway import create_scaleway_model
from strands.session.scaleway import create_scaleway_session_repository
from strands.telemetry.scaleway import configure_scaleway_telemetry
from strands.hooks.scaleway import create_cost_tracking_hook, create_security_hook


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function"""
    
    # Configure telemetry (optional - requires Scaleway Cockpit setup)
    try:
        telemetry = configure_scaleway_telemetry(
            service_name="basic-agent-example",
            environment="development",
            enable_tracing=False,  # Disable for local testing
            enable_metrics=False   # Disable for local testing
        )
        logger.info("Telemetry configured")
    except Exception as e:
        logger.warning(f"Telemetry setup failed: {e}")
        telemetry = None
    
    # Create model with provider-agnostic routing
    try:
        model = create_scaleway_model(
            primary_provider="openrouter",
            primary_model="groq/llama-4-scout",
            fallback_provider="anthropic",
            fallback_model="claude-4.5-sonnet"
        )
        logger.info("Model created with provider-agnostic routing")
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        logger.error("Make sure to set OPENROUTER_API_KEY and ANTHROPIC_API_KEY environment variables")
        return
    
    # Create session repository (optional - requires PostgreSQL)
    try:
        session_repo = create_scaleway_session_repository()
        logger.info("Session repository configured")
    except Exception as e:
        logger.warning(f"Session repository setup failed: {e}")
        logger.warning("Make sure PostgreSQL is running and SCALEWAY_DATABASE_URL is set")
        session_repo = None
    
    # Create hooks
    cost_hook = create_cost_tracking_hook(
        budget_limit=10.0,  # $10 monthly budget
        alert_thresholds=[1.0, 5.0, 10.0]
    )
    
    security_hook = create_security_hook(
        max_requests_per_minute=30,
        enable_content_filter=True
    )
    
    # Create agent with all components
    agent = Agent(
        model=model,
        session_repository=session_repo,
        hooks=[cost_hook, security_hook],
        system_prompt="You are a helpful AI assistant. Be concise and accurate."
    )
    
    logger.info("Agent created successfully")
    
    # Test the agent
    try:
        logger.info("Testing agent with basic question...")
        response = await agent("What is the capital of France?")
        logger.info(f"Agent response: {response.message}")
        
        # Get provider info
        provider_info = model.get_provider_info()
        logger.info(f"Provider configuration: {provider_info}")
        
        # Get cost summary
        cost_summary = cost_hook.get_cost_summary()
        logger.info(f"Cost summary: {cost_summary}")
        
        # Get security summary
        security_summary = security_hook.get_security_summary()
        logger.info(f"Security summary: {security_summary}")
        
        if session_repo:
            # Test session operations
            session_id = session_repo.create_session(
                agent_id="basic-agent",
                data={"test": "data"},
                ttl=3600
            )
            logger.info(f"Created session: {session_id}")
            
            session_data = session_repo.get_session(session_id)
            logger.info(f"Retrieved session: {session_data}")
            
            # Update session
            session_repo.update_session(
                session_id=session_id,
                data={"updated": "value"}
            )
            logger.info("Session updated")
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Agent test failed: {e}")
    
    # Cleanup
    if session_repo:
        session_repo.close()
        logger.info("Session repository closed")


if __name__ == "__main__":
    # Check required environment variables
    required_vars = ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set the following environment variables:")
        for var in missing_vars:
            logger.error(f"  export {var}=your_api_key_here")
        exit(1)
    
    # Run the example
    asyncio.run(main())
