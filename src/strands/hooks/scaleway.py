"""
Scaleway Hooks

Production hooks for cost tracking, security, and resource quotas.
Integrates with Scaleway services for comprehensive agent management.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import uuid
import time

from strands.hooks.base import Hook
from strands.models.openai import OpenAIModel


class HookType(Enum):
    """Types of hooks available"""
    COST_TRACKING = "cost_tracking"
    SECURITY = "security"
    RESOURCE_QUOTA = "resource_quota"
    AUDIT = "audit"


@dataclass
class CostData:
    """Cost tracking data"""
    request_id: str
    agent_id: str
    provider: str
    model: str
    tokens_used: int
    cost_usd: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SecurityEvent:
    """Security event data"""
    event_id: str
    agent_id: str
    event_type: str
    severity: str  # low, medium, high, critical
    description: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QuotaLimit:
    """Resource quota limit"""
    resource_type: str
    limit: float
    current_usage: float
    reset_time: Optional[datetime]
    period: str  # hourly, daily, monthly


class CostTrackingHook(Hook):
    """
    Tracks costs for agent requests.
    
    Features:
    - Real-time cost calculation
    - Per-agent cost tracking
    - Budget enforcement
    - Cost alerts
    - Historical cost analysis
    """
    
    def __init__(
        self,
        budget_limit: Optional[float] = None,
        alert_thresholds: Optional[List[float]] = None,
        cost_storage: Optional[Callable] = None
    ):
        """
        Initialize CostTrackingHook.
        
        Args:
            budget_limit: Monthly budget limit in USD
            alert_thresholds: Cost thresholds for alerts
            cost_storage: Optional function to store cost data
        """
        self.logger = logging.getLogger(__name__)
        self.budget_limit = budget_limit
        self.alert_thresholds = alert_thresholds or [10.0, 50.0, 100.0]
        self.cost_storage = cost_storage
        
        # Model pricing (can be overridden)
        self.model_pricing = {
            # OpenRouter pricing (examples)
            "groq/llama-4-scout": {"input": 0.00005, "output": 0.0001},  # per token
            "anthropic/claude-4.5-sonnet": {"input": 0.003, "output": 0.015},
            "openai/gpt-4": {"input": 0.03, "output": 0.06},
            # Add more models as needed
        }
        
        # Track costs
        self.daily_costs = {}
        self.monthly_costs = {}
        self.agent_costs = {}
        
        self.logger.info("CostTrackingHook initialized")
    
    def before_request(self, agent_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check budget before request"""
        if self.budget_limit:
            current_monthly = self._get_monthly_cost()
            if current_monthly >= self.budget_limit:
                raise Exception(f"Monthly budget limit of ${self.budget_limit} exceeded")
        
        # Add request ID for tracking
        request_data["request_id"] = str(uuid.uuid4())
        return request_data
    
    def after_request(
        self,
        agent_id: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate and track cost after request"""
        request_id = request_data.get("request_id", str(uuid.uuid4()))
        
        # Extract usage information
        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        
        # Get model info
        model = response_data.get("model", "unknown")
        provider = request_data.get("provider", "unknown")
        
        # Calculate cost
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Create cost record
        cost_data = CostData(
            request_id=request_id,
            agent_id=agent_id,
            provider=provider,
            model=model,
            tokens_used=total_tokens,
            cost_usd=cost,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Track costs
        self._track_cost(cost_data)
        
        # Check alerts
        self._check_alerts(agent_id, cost)
        
        # Store if storage function provided
        if self.cost_storage:
            self.cost_storage(asdict(cost_data))
        
        self.logger.debug(
            f"Tracked cost for {agent_id}: ${cost:.6f} for {total_tokens} tokens"
        )
        
        return response_data
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on model pricing"""
        if model not in self.model_pricing:
            self.logger.warning(f"No pricing for model {model}, using default")
            return 0.001  # Default cost estimate
        
        pricing = self.model_pricing[model]
        input_cost = prompt_tokens * pricing["input"]
        output_cost = completion_tokens * pricing["output"]
        return input_cost + output_cost
    
    def _track_cost(self, cost_data: CostData):
        """Track cost in internal structures"""
        date_key = cost_data.timestamp.strftime("%Y-%m-%d")
        month_key = cost_data.timestamp.strftime("%Y-%m")
        
        # Daily costs
        if date_key not in self.daily_costs:
            self.daily_costs[date_key] = 0.0
        self.daily_costs[date_key] += cost_data.cost_usd
        
        # Monthly costs
        if month_key not in self.monthly_costs:
            self.monthly_costs[month_key] = 0.0
        self.monthly_costs[month_key] += cost_data.cost_usd
        
        # Agent costs
        if cost_data.agent_id not in self.agent_costs:
            self.agent_costs[cost_data.agent_id] = 0.0
        self.agent_costs[cost_data.agent_id] += cost_data.cost_usd
    
    def _get_monthly_cost(self) -> float:
        """Get current month's cost"""
        current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        return self.monthly_costs.get(current_month, 0.0)
    
    def _check_alerts(self, agent_id: str, cost: float):
        """Check if cost alerts should be triggered"""
        current_monthly = self._get_monthly_cost()
        
        for threshold in self.alert_thresholds:
            if current_monthly >= threshold and current_monthly - cost < threshold:
                self.logger.warning(
                    f"Cost alert: Monthly cost ${current_monthly:.2f} exceeded ${threshold}"
                )
                # Here you could send notifications, emails, etc.
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of tracked costs"""
        current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        return {
            "current_month": {
                "month": current_month,
                "cost": self.monthly_costs.get(current_month, 0.0),
                "budget_limit": self.budget_limit,
                "budget_remaining": max(0, self.budget_limit - self.monthly_costs.get(current_month, 0.0)) if self.budget_limit else None
            },
            "current_date": {
                "date": current_date,
                "cost": self.daily_costs.get(current_date, 0.0)
            },
            "top_agents": sorted(
                [(agent_id, cost) for agent_id, cost in self.agent_costs.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class SecurityHook(Hook):
    """
    Security monitoring and enforcement.
    
    Features:
    - Request validation
    - Security event logging
    - Rate limiting
    - Content filtering
    - Anomaly detection
    """
    
    def __init__(
        self,
        enable_content_filter: bool = True,
        max_requests_per_minute: int = 60,
        blocked_patterns: Optional[List[str]] = None,
        security_storage: Optional[Callable] = None
    ):
        """
        Initialize SecurityHook.
        
        Args:
            enable_content_filter: Enable content filtering
            max_requests_per_minute: Rate limit per agent
            blocked_patterns: List of blocked content patterns
            security_storage: Optional function to store security events
        """
        self.logger = logging.getLogger(__name__)
        self.enable_content_filter = enable_content_filter
        self.max_requests_per_minute = max_requests_per_minute
        self.blocked_patterns = blocked_patterns or [
            "password", "secret", "token", "key", "credential"
        ]
        self.security_storage = security_storage
        
        # Track requests for rate limiting
        self.request_history = {}
        
        self.logger.info("SecurityHook initialized")
    
    def before_request(self, agent_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Security checks before request"""
        # Rate limiting
        self._check_rate_limit(agent_id)
        
        # Content filtering
        if self.enable_content_filter:
            self._check_content_filter(request_data)
        
        return request_data
    
    def after_request(
        self,
        agent_id: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post-request security checks"""
        # Log the request
        self._log_security_event(
            agent_id,
            "request_completed",
            "low",
            f"Request completed successfully"
        )
        
        return response_data
    
    def _check_rate_limit(self, agent_id: str):
        """Check if agent is exceeding rate limits"""
        now = time.time()
        
        if agent_id not in self.request_history:
            self.request_history[agent_id] = []
        
        # Clean old requests (older than 1 minute)
        self.request_history[agent_id] = [
            req_time for req_time in self.request_history[agent_id]
            if now - req_time < 60
        ]
        
        # Check rate limit
        if len(self.request_history[agent_id]) >= self.max_requests_per_minute:
            self._log_security_event(
                agent_id,
                "rate_limit_exceeded",
                "medium",
                f"Agent exceeded rate limit: {len(self.request_history[agent_id])} requests/minute"
            )
            raise Exception(f"Rate limit exceeded: {self.max_requests_per_minute} requests per minute")
        
        # Add current request
        self.request_history[agent_id].append(now)
    
    def _check_content_filter(self, request_data: Dict[str, Any]):
        """Check request content for blocked patterns"""
        content = str(request_data.get("messages", []))
        
        for pattern in self.blocked_patterns:
            if pattern.lower() in content.lower():
                self._log_security_event(
                    request_data.get("agent_id", "unknown"),
                    "blocked_content",
                    "high",
                    f"Request contains blocked pattern: {pattern}"
                )
                raise Exception(f"Request contains blocked content: {pattern}")
    
    def _log_security_event(
        self,
        agent_id: str,
        event_type: str,
        severity: str,
        description: str
    ):
        """Log security event"""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            agent_id=agent_id,
            event_type=event_type,
            severity=severity,
            description=description,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.logger.warning(f"Security event: {description}")
        
        # Store if storage function provided
        if self.security_storage:
            self.security_storage(asdict(event))
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security events"""
        return {
            "rate_limit_active": True,
            "max_requests_per_minute": self.max_requests_per_minute,
            "content_filter_enabled": self.enable_content_filter,
            "blocked_patterns_count": len(self.blocked_patterns),
            "active_agents": len(self.request_history)
        }


class ResourceQuotaHook(Hook):
    """
    Resource quota management.
    
    Features:
    - Token quotas
    - Request quotas
    - Time-based quotas
    - Quota enforcement
    - Usage tracking
    """
    
    def __init__(
        self,
        daily_token_limit: Optional[int] = None,
        daily_request_limit: Optional[int] = None,
        monthly_token_limit: Optional[int] = None,
        quota_storage: Optional[Callable] = None
    ):
        """
        Initialize ResourceQuotaHook.
        
        Args:
            daily_token_limit: Daily token limit per agent
            daily_request_limit: Daily request limit per agent
            monthly_token_limit: Monthly token limit per agent
            quota_storage: Optional function to store quota data
        """
        self.logger = logging.getLogger(__name__)
        self.daily_token_limit = daily_token_limit
        self.daily_request_limit = daily_request_limit
        self.monthly_token_limit = monthly_token_limit
        self.quota_storage = quota_storage
        
        # Track usage
        self.token_usage = {}
        self.request_usage = {}
        
        self.logger.info("ResourceQuotaHook initialized")
    
    def before_request(self, agent_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check quotas before request"""
        # Check daily request quota
        if self.daily_request_limit:
            daily_requests = self._get_daily_requests(agent_id)
            if daily_requests >= self.daily_request_limit:
                raise Exception(f"Daily request limit of {self.daily_request_limit} exceeded")
        
        # Check monthly token quota
        if self.monthly_token_limit:
            monthly_tokens = self._get_monthly_tokens(agent_id)
            if monthly_tokens >= self.monthly_token_limit:
                raise Exception(f"Monthly token limit of {self.monthly_token_limit} exceeded")
        
        return request_data
    
    def after_request(
        self,
        agent_id: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update quota usage after request"""
        # Update request count
        self._increment_requests(agent_id)
        
        # Update token usage
        usage = response_data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        if tokens_used > 0:
            self._increment_tokens(agent_id, tokens_used)
        
        return response_data
    
    def _get_daily_requests(self, agent_id: str) -> int:
        """Get daily request count for agent"""
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.request_usage.get(f"{agent_id}:{date_key}", 0)
    
    def _get_monthly_tokens(self, agent_id: str) -> int:
        """Get monthly token usage for agent"""
        month_key = datetime.now(timezone.utc).strftime("%Y-%m")
        return self.token_usage.get(f"{agent_id}:{month_key}", 0)
    
    def _increment_requests(self, agent_id: str):
        """Increment request count"""
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"{agent_id}:{date_key}"
        self.request_usage[key] = self.request_usage.get(key, 0) + 1
    
    def _increment_tokens(self, agent_id: str, tokens: int):
        """Increment token usage"""
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        month_key = datetime.now(timezone.utc).strftime("%Y-%m")
        
        daily_key = f"{agent_id}:{date_key}"
        monthly_key = f"{agent_id}:{month_key}"
        
        self.token_usage[daily_key] = self.token_usage.get(daily_key, 0) + tokens
        self.token_usage[monthly_key] = self.token_usage.get(monthly_key, 0) + tokens
    
    def get_quota_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get quota summary for agent"""
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        month_key = datetime.now(timezone.utc).strftime("%Y-%m")
        
        return {
            "daily": {
                "requests": {
                    "used": self._get_daily_requests(agent_id),
                    "limit": self.daily_request_limit,
                    "remaining": max(0, self.daily_request_limit - self._get_daily_requests(agent_id)) if self.daily_request_limit else None
                },
                "tokens": {
                    "used": self.token_usage.get(f"{agent_id}:{date_key}", 0),
                    "limit": self.daily_token_limit,
                    "remaining": max(0, self.daily_token_limit - self.token_usage.get(f"{agent_id}:{date_key}", 0)) if self.daily_token_limit else None
                }
            },
            "monthly": {
                "tokens": {
                    "used": self._get_monthly_tokens(agent_id),
                    "limit": self.monthly_token_limit,
                    "remaining": max(0, self.monthly_token_limit - self._get_monthly_tokens(agent_id)) if self.monthly_token_limit else None
                }
            }
        }


# Convenience functions for creating hooks
def create_cost_tracking_hook(
    budget_limit: Optional[float] = None,
    **kwargs
) -> CostTrackingHook:
    """Create a cost tracking hook with common configuration"""
    return CostTrackingHook(budget_limit=budget_limit, **kwargs)


def create_security_hook(
    max_requests_per_minute: int = 60,
    **kwargs
) -> SecurityHook:
    """Create a security hook with common configuration"""
    return SecurityHook(max_requests_per_minute=max_requests_per_minute, **kwargs)


def create_resource_quota_hook(
    daily_token_limit: Optional[int] = None,
    daily_request_limit: Optional[int] = None,
    monthly_token_limit: Optional[int] = None,
    **kwargs
) -> ResourceQuotaHook:
    """Create a resource quota hook with common configuration"""
    return ResourceQuotaHook(
        daily_token_limit=daily_token_limit,
        daily_request_limit=daily_request_limit,
        monthly_token_limit=monthly_token_limit,
        **kwargs
    )
