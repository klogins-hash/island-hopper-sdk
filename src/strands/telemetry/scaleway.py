"""
Scaleway Telemetry Configuration

OpenTelemetry setup for Scaleway Cockpit integration.
Provides metrics, traces, and logs for Strands agents.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import uuid

from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.context import Context


@dataclass
class TelemetryConfig:
    """Configuration for Scaleway telemetry"""
    service_name: str
    service_version: str
    environment: str
    otel_endpoint: Optional[str] = None
    otel_headers: Optional[Dict[str, str]] = None
    enable_tracing: bool = True
    enable_metrics: bool = True
    sampling_rate: float = 1.0
    metric_export_interval: int = 30  # seconds


class ScalewayTelemetry:
    """
    Scaleway Cockpit telemetry integration.
    
    Features:
    - Distributed tracing with OpenTelemetry
    - Custom metrics for agent operations
    - Automatic instrumentation
    - Cost tracking integration
    - Performance monitoring
    """
    
    def __init__(self, config: TelemetryConfig):
        """
        Initialize ScalewayTelemetry.
        
        Args:
            config: Telemetry configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenTelemetry
        self._setup_tracing()
        self._setup_metrics()
        
        # Get tracer and meter
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Initialize metrics
        self._init_metrics()
        
        self.logger.info("ScalewayTelemetry initialized")
    
    def _setup_tracing(self):
        """Setup distributed tracing"""
        if not self.config.enable_tracing:
            return
        
        # Configure OTLP exporter
        endpoint = self.config.otel_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "https://otel-agent.scw.cloud:4317"
        )
        
        headers = self.config.otel_headers or {
            "Authorization": f"Bearer {os.getenv('SCALEWAY_API_KEY')}"
        }
        
        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers
        )
        
        # Configure tracer provider
        resource = Resource(attributes={
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "deployment.environment": self.config.environment,
            "service.instance.id": str(uuid.uuid4())
        })
        
        tracer_provider = TracerProvider(resource=resource)
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
    
    def _setup_metrics(self):
        """Setup metrics collection"""
        if not self.config.enable_metrics:
            return
        
        # Configure OTLP exporter
        endpoint = self.config.otel_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "https://otel-agent.scw.cloud:4317"
        )
        
        headers = self.config.otel_headers or {
            "Authorization": f"Bearer {os.getenv('SCALEWAY_API_KEY')}"
        }
        
        exporter = OTLPMetricExporter(
            endpoint=endpoint,
            headers=headers
        )
        
        # Configure metric reader
        metric_reader = PeriodicExportingMetricReader(
            exporter=exporter,
            export_interval_millis=self.config.metric_export_interval * 1000
        )
        
        # Configure meter provider
        resource = Resource(attributes={
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "deployment.environment": self.config.environment,
            "service.instance.id": str(uuid.uuid4())
        })
        
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        
        # Set global meter provider
        metrics.set_meter_provider(meter_provider)
    
    def _init_metrics(self):
        """Initialize custom metrics"""
        if not self.config.enable_metrics:
            return
        
        # Agent operation metrics
        self.agent_requests_total = self.meter.create_counter(
            "agent_requests_total",
            description="Total number of agent requests"
        )
        
        self.agent_request_duration = self.meter.create_histogram(
            "agent_request_duration_seconds",
            description="Duration of agent requests"
        )
        
        self.agent_errors_total = self.meter.create_counter(
            "agent_errors_total",
            description="Total number of agent errors"
        )
        
        # Model-specific metrics
        self.model_requests_total = self.meter.create_counter(
            "model_requests_total",
            description="Total number of model requests",
            unit="1"
        )
        
        self.model_tokens_used = self.meter.create_counter(
            "model_tokens_used",
            description="Total number of tokens used",
            unit="1"
        )
        
        self.model_cost_usd = self.meter.create_counter(
            "model_cost_usd",
            description="Total cost in USD",
            unit="USD"
        )
        
        # Session metrics
        self.session_operations_total = self.meter.create_counter(
            "session_operations_total",
            description="Total number of session operations"
        )
        
        self.active_sessions = self.meter.create_up_down_counter(
            "active_sessions",
            description="Number of active sessions"
        )
        
        # Tool metrics
        self.tool_executions_total = self.meter.create_counter(
            "tool_executions_total",
            description="Total number of tool executions"
        )
        
        self.tool_duration = self.meter.create_histogram(
            "tool_execution_duration_seconds",
            description="Duration of tool executions"
        )
    
    def trace_agent_request(
        self,
        agent_id: str,
        request_id: str,
        provider: str,
        model: str
    ):
        """Create a span for agent request"""
        if not self.config.enable_tracing:
            return Context()
        
        span = self.tracer.start_span("agent_request")
        span.set_attribute("agent.id", agent_id)
        span.set_attribute("request.id", request_id)
        span.set_attribute("model.provider", provider)
        span.set_attribute("model.name", model)
        span.set_attribute("service.name", self.config.service_name)
        
        return trace.set_span_in_context(span)
    
    def record_agent_request(
        self,
        agent_id: str,
        provider: str,
        model: str,
        duration: float,
        success: bool = True,
        error_message: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None
    ):
        """Record metrics for agent request"""
        if not self.config.enable_metrics:
            return
        
        # Request metrics
        self.agent_requests_total.add(
            1,
            {"agent.id": agent_id, "model.provider": provider, "model.name": model}
        )
        
        self.agent_request_duration.record(
            duration,
            {"agent.id": agent_id, "model.provider": provider, "model.name": model}
        )
        
        if not success:
            self.agent_errors_total.add(
                1,
                {"agent.id": agent_id, "model.provider": provider, "error.type": error_message or "unknown"}
            )
        
        # Model metrics
        self.model_requests_total.add(
            1,
            {"provider": provider, "model": model}
        )
        
        if tokens_used:
            self.model_tokens_used.add(
                tokens_used,
                {"provider": provider, "model": model}
            )
        
        if cost_usd:
            self.model_cost_usd.add(
                cost_usd,
                {"provider": provider, "model": model}
            )
    
    def record_session_operation(
        self,
        operation: str,  # create, read, update, delete
        agent_id: str,
        success: bool = True
    ):
        """Record session operation metrics"""
        if not self.config.enable_metrics:
            return
        
        self.session_operations_total.add(
            1,
            {"operation": operation, "agent.id": agent_id, "success": str(success)}
        )
        
        if operation == "create":
            self.active_sessions.add(1, {"agent.id": agent_id})
        elif operation == "delete":
            self.active_sessions.add(-1, {"agent.id": agent_id})
    
    def record_tool_execution(
        self,
        tool_name: str,
        agent_id: str,
        duration: float,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Record tool execution metrics"""
        if not self.config.enable_metrics:
            return
        
        self.tool_executions_total.add(
            1,
            {"tool.name": tool_name, "agent.id": agent_id, "success": str(success)}
        )
        
        self.tool_duration.record(
            duration,
            {"tool.name": tool_name, "agent.id": agent_id}
        )
    
    def create_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Create a custom span"""
        if not self.config.enable_tracing:
            return Context()
        
        span = self.tracer.start_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        return trace.set_span_in_context(span)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics (for debugging)"""
        return {
            "service_name": self.config.service_name,
            "environment": self.config.environment,
            "tracing_enabled": self.config.enable_tracing,
            "metrics_enabled": self.config.enable_metrics,
            "otel_endpoint": self.config.otel_endpoint
        }


def configure_scaleway_telemetry(
    service_name: str,
    service_version: str = "0.1.0",
    environment: str = "production",
    otel_endpoint: Optional[str] = None,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    **kwargs
) -> ScalewayTelemetry:
    """
    Configure Scaleway telemetry with sensible defaults.
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        environment: Deployment environment
        otel_endpoint: OTLP endpoint for OpenTelemetry
        enable_tracing: Enable distributed tracing
        enable_metrics: Enable metrics collection
        **kwargs: Additional configuration
    
    Returns:
        Configured ScalewayTelemetry instance
    """
    config = TelemetryConfig(
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        otel_endpoint=otel_endpoint,
        enable_tracing=enable_tracing,
        enable_metrics=enable_metrics,
        **kwargs
    )
    
    return ScalewayTelemetry(config)


# Context manager for tracing
class AgentRequestTracer:
    """Context manager for tracing agent requests"""
    
    def __init__(
        self,
        telemetry: ScalewayTelemetry,
        agent_id: str,
        request_id: str,
        provider: str,
        model: str
    ):
        self.telemetry = telemetry
        self.agent_id = agent_id
        self.request_id = request_id
        self.provider = provider
        self.model = model
        self.start_time = None
        self.context = None
        self.span = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.context = self.telemetry.trace_agent_request(
            self.agent_id,
            self.request_id,
            self.provider,
            self.model
        )
        
        # Get the span from context for additional attributes
        self.span = trace.get_current_span(self.context)
        
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None
        
        # Record metrics
        self.telemetry.record_agent_request(
            self.agent_id,
            self.provider,
            self.model,
            duration,
            success,
            error_message
        )
        
        # Update span
        if self.span:
            if success:
                self.span.set_status(Status(StatusCode.OK))
            else:
                self.span.set_status(Status(StatusCode.ERROR, error_message))
                self.span.set_attribute("error.message", error_message)
            
            self.span.end()


# Decorator for automatic tracing
def trace_agent_call(telemetry: ScalewayTelemetry):
    """Decorator to automatically trace agent calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract agent info from kwargs or instance
            agent_id = kwargs.get('agent_id', getattr(args[0], 'agent_id', 'unknown'))
            request_id = kwargs.get('request_id', str(uuid.uuid4()))
            
            # Get model info
            model_config = getattr(args[0], 'model_config', {})
            provider = model_config.get('primary_provider', 'unknown')
            model = model_config.get('primary_model', 'unknown')
            
            with AgentRequestTracer(telemetry, agent_id, request_id, provider, model):
                return func(*args, **kwargs)
        return wrapper
    return decorator
