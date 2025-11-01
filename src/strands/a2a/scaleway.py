"""
Scaleway A2A (Agent-to-Agent) Communication

NATS-based event-driven communication for multi-agent systems.
Provides scalable, reliable messaging between agents.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import uuid
import pickle
import base64

import nats
from nats.aio.client import Client as NATSClient
from nats.aio.subscription import Subscription
from nats.js import JetStreamContext


class MessageType(Enum):
    """Types of messages between agents"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"


@dataclass
class A2AMessage:
    """Agent-to-agent message structure"""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    priority: int = 5  # 1-10, 10 is highest
    ttl: Optional[int] = None  # Time to live in seconds


class ScalewayA2AExecutor:
    """
    NATS-based agent-to-agent communication executor.
    
    Features:
    - Event-driven messaging
    - Request/response patterns
    - Broadcast messaging
    - Message filtering
    - Automatic reconnection
    - JetStream persistence
    - Load balancing
    """
    
    def __init__(
        self,
        nats_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        enable_jetstream: bool = True,
        max_reconnect_attempts: int = 10,
        reconnect_time: float = 2.0,
        **kwargs
    ):
        """
        Initialize ScalewayA2AExecutor.
        
        Args:
            nats_url: NATS server URL
            agent_id: Unique ID for this agent
            enable_jetstream: Enable JetStream for persistence
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_time: Time between reconnection attempts
            **kwargs: Additional configuration
        """
        self.logger = logging.getLogger(__name__)
        self.nats_url = nats_url or os.getenv(
            "NATS_URL",
            "nats://localhost:4222"
        )
        self.agent_id = agent_id or str(uuid.uuid4())
        self.enable_jetstream = enable_jetstream
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_time = reconnect_time
        
        # NATS client and JetStream
        self.nc: Optional[NATSClient] = None
        self.js: Optional[JetStreamContext] = None
        
        # Message handlers
        self.handlers: Dict[str, Callable] = {}
        self.subscriptions: List[Subscription] = []
        
        # Connection state
        self.connected = False
        self.connection_lock = asyncio.Lock()
        
        self.logger.info(f"ScalewayA2AExecutor initialized for agent {self.agent_id}")
    
    async def connect(self):
        """Connect to NATS server"""
        async with self.connection_lock:
            if self.connected:
                return
            
            try:
                # Create NATS client
                self.nc = NATSClient()
                
                # Configure connection options
                options = {
                    "servers": self.nats_url,
                    "max_reconnect_attempts": self.max_reconnect_attempts,
                    "reconnect_time_wait": self.reconnect_time,
                    "name": f"agent-{self.agent_id}",
                    "error_cb": self._error_callback,
                    "closed_cb": self._closed_callback,
                    "reconnected_cb": self._reconnected_callback
                }
                
                # Connect to NATS
                await self.nc.connect(**options)
                
                # Initialize JetStream if enabled
                if self.enable_jetstream:
                    self.js = self.nc.jetstream()
                    await self._ensure_jetstream_streams()
                
                self.connected = True
                self.logger.info(f"Connected to NATS at {self.nats_url}")
                
                # Setup default subscriptions
                await self._setup_default_subscriptions()
                
            except Exception as e:
                self.logger.error(f"Failed to connect to NATS: {e}")
                raise
    
    async def disconnect(self):
        """Disconnect from NATS server"""
        async with self.connection_lock:
            if not self.connected:
                return
            
            try:
                # Unsubscribe from all subscriptions
                for sub in self.subscriptions:
                    await sub.unsubscribe()
                self.subscriptions.clear()
                
                # Close connection
                if self.nc:
                    await self.nc.close()
                
                self.connected = False
                self.logger.info("Disconnected from NATS")
                
            except Exception as e:
                self.logger.error(f"Error disconnecting from NATS: {e}")
    
    async def _ensure_jetstream_streams(self):
        """Ensure JetStream streams exist"""
        if not self.js:
            return
        
        try:
            # Agent messages stream
            await self.js.add_stream(
                name="agent_messages",
                subjects=["agent.*.*.*"],  # agent.{sender}.{recipient}.{type}
                description="Agent-to-agent messages",
                retention="workqueue",
                max_age=86400,  # 24 hours
                max_bytes=1024 * 1024 * 1024,  # 1GB
                replicas=1
            )
            
            # Agent events stream
            await self.js.add_stream(
                name="agent_events",
                subjects=["events.*.*"],  # events.{agent}.{type}
                description="Agent events and status updates",
                retention="workqueue",
                max_age=3600,  # 1 hour
                max_bytes=512 * 1024 * 1024,  # 512MB
                replicas=1
            )
            
            # Broadcast stream
            await self.js.add_stream(
                name="agent_broadcast",
                subjects=["broadcast.*"],  # broadcast.{type}
                description="Broadcast messages",
                retention="workqueue",
                max_age=1800,  # 30 minutes
                max_bytes=256 * 1024 * 1024,  # 256MB
                replicas=1
            )
            
            self.logger.info("JetStream streams ensured")
            
        except Exception as e:
            self.logger.error(f"Error ensuring JetStream streams: {e}")
    
    async def _setup_default_subscriptions(self):
        """Setup default message subscriptions"""
        # Subscribe to direct messages
        await self.subscribe_to_agent(self.agent_id, self._handle_direct_message)
        
        # Subscribe to broadcasts
        await self.subscribe_to_broadcast(self._handle_broadcast)
        
        # Subscribe to agent events
        await self.subscribe_to_events(self.agent_id, self._handle_agent_events)
    
    async def _handle_direct_message(self, message: A2AMessage):
        """Handle direct message to this agent"""
        handler = self.handlers.get("direct_message")
        if handler:
            await handler(message)
        else:
            self.logger.warning(f"No handler for direct message from {message.sender_id}")
    
    async def _handle_broadcast(self, message: A2AMessage):
        """Handle broadcast message"""
        handler = self.handlers.get("broadcast")
        if handler:
            await handler(message)
    
    async def _handle_agent_events(self, message: A2AMessage):
        """Handle agent-specific events"""
        handler = self.handlers.get("agent_event")
        if handler:
            await handler(message)
    
    async def send_message(
        self,
        recipient_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        priority: int = 5,
        ttl: Optional[int] = None
    ) -> str:
        """
        Send a message to another agent.
        
        Args:
            recipient_id: ID of the recipient agent
            message_type: Type of message
            payload: Message payload
            correlation_id: Optional correlation ID
            priority: Message priority (1-10)
            ttl: Time to live in seconds
        
        Returns:
            Message ID
        """
        if not self.connected:
            await self.connect()
        
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(timezone.utc),
            correlation_id=correlation_id,
            priority=priority,
            ttl=ttl
        )
        
        # Send message
        subject = f"agent.{self.agent_id}.{recipient_id}.{message_type.value}"
        await self._publish_message(subject, message)
        
        self.logger.debug(f"Sent message {message.message_id} to {recipient_id}")
        return message.message_id
    
    async def broadcast_message(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 5,
        ttl: Optional[int] = None
    ) -> str:
        """
        Send a broadcast message to all agents.
        
        Args:
            message_type: Type of message
            payload: Message payload
            priority: Message priority (1-10)
            ttl: Time to live in seconds
        
        Returns:
            Message ID
        """
        if not self.connected:
            await self.connect()
        
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(timezone.utc),
            priority=priority,
            ttl=ttl
        )
        
        # Send broadcast
        subject = f"broadcast.{message_type.value}"
        await self._publish_message(subject, message)
        
        self.logger.debug(f"Broadcast message {message.message_id}")
        return message.message_id
    
    async def send_event(
        self,
        event_type: str,
        payload: Dict[str, Any]
    ) -> str:
        """
        Send an event about this agent.
        
        Args:
            event_type: Type of event
            payload: Event payload
        
        Returns:
            Message ID
        """
        if not self.connected:
            await self.connect()
        
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=None,
            message_type=MessageType.EVENT,
            payload=payload,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Send event
        subject = f"events.{self.agent_id}.{event_type}"
        await self._publish_message(subject, message)
        
        self.logger.debug(f"Sent event {message.message_id}: {event_type}")
        return message.message_id
    
    async def request_response(
        self,
        recipient_id: str,
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Send a request and wait for response.
        
        Args:
            recipient_id: ID of the recipient agent
            payload: Request payload
            timeout: Timeout in seconds
        
        Returns:
            Response payload or None if timeout
        """
        correlation_id = str(uuid.uuid4())
        
        # Send request
        await self.send_message(
            recipient_id=recipient_id,
            message_type=MessageType.REQUEST,
            payload=payload,
            correlation_id=correlation_id
        )
        
        # Wait for response
        response = await self._wait_for_response(correlation_id, timeout)
        return response
    
    async def _wait_for_response(
        self,
        correlation_id: str,
        timeout: float
    ) -> Optional[Dict[str, Any]]:
        """Wait for response with specific correlation ID"""
        # Create inbox for response
        inbox = f"agent.{self.agent_id}.inbox.{correlation_id}"
        
        # Subscribe to inbox
        response_future = asyncio.Future()
        
        async def response_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                message = A2AMessage(**data)
                response_future.set_result(message.payload)
            except Exception as e:
                response_future.set_exception(e)
        
        sub = await self.nc.subscribe(inbox, cb=response_handler)
        
        try:
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for response to {correlation_id}")
            return None
        finally:
            await sub.unsubscribe()
    
    async def _publish_message(self, subject: str, message: A2AMessage):
        """Publish message to NATS"""
        # Serialize message
        data = json.dumps(asdict(message), default=str).encode()
        
        # Publish with headers
        headers = {
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "priority": str(message.priority)
        }
        
        if message.ttl:
            headers["ttl"] = str(message.ttl)
        
        if self.js:
            # Use JetStream for persistence
            await self.js.publish(subject, data, headers=headers)
        else:
            # Use regular NATS
            await self.nc.publish(subject, data, headers=headers)
    
    async def subscribe_to_agent(
        self,
        agent_id: str,
        handler: Callable[[A2AMessage], None]
    ):
        """Subscribe to messages for a specific agent"""
        if not self.connected:
            await self.connect()
        
        subject = f"agent.*.{agent_id}.*"
        
        async def message_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                message = A2AMessage(**data)
                await handler(message)
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
        
        sub = await self.nc.subscribe(subject, cb=message_handler)
        self.subscriptions.append(sub)
        
        self.logger.info(f"Subscribed to messages for agent {agent_id}")
    
    async def subscribe_to_broadcast(
        self,
        handler: Callable[[A2AMessage], None]
    ):
        """Subscribe to broadcast messages"""
        if not self.connected:
            await self.connect()
        
        subject = "broadcast.*"
        
        async def message_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                message = A2AMessage(**data)
                await handler(message)
            except Exception as e:
                self.logger.error(f"Error processing broadcast: {e}")
        
        sub = await self.nc.subscribe(subject, cb=message_handler)
        self.subscriptions.append(sub)
        
        self.logger.info("Subscribed to broadcast messages")
    
    async def subscribe_to_events(
        self,
        agent_id: str,
        handler: Callable[[A2AMessage], None]
    ):
        """Subscribe to events for a specific agent"""
        if not self.connected:
            await self.connect()
        
        subject = f"events.{agent_id}.*"
        
        async def message_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                message = A2AMessage(**data)
                await handler(message)
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
        
        sub = await self.nc.subscribe(subject, cb=message_handler)
        self.subscriptions.append(sub)
        
        self.logger.info(f"Subscribed to events for agent {agent_id}")
    
    def register_handler(self, message_type: str, handler: Callable[[A2AMessage], None]):
        """Register a message handler"""
        self.handlers[message_type] = handler
    
    async def send_heartbeat(self):
        """Send heartbeat message"""
        await self.send_event("heartbeat", {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def send_status_update(self, status: Dict[str, Any]):
        """Send status update"""
        await self.send_event("status_update", {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    # Connection callbacks
    async def _error_callback(self, e):
        """Handle connection error"""
        self.logger.error(f"NATS connection error: {e}")
        self.connected = False
    
    async def _closed_callback(self):
        """Handle connection closed"""
        self.logger.info("NATS connection closed")
        self.connected = False
    
    async def _reconnected_callback(self):
        """Handle reconnection"""
        self.logger.info("NATS connection reestablished")
        self.connected = True


# Convenience function for creating A2A executor
def create_scaleway_a2a_executor(
    nats_url: Optional[str] = None,
    agent_id: Optional[str] = None,
    **kwargs
) -> ScalewayA2AExecutor:
    """
    Convenience function to create a ScalewayA2AExecutor.
    
    Args:
        nats_url: NATS server URL
        agent_id: Unique ID for this agent
        **kwargs: Additional configuration
    
    Returns:
        Configured ScalewayA2AExecutor instance
    """
    return ScalewayA2AExecutor(
        nats_url=nats_url,
        agent_id=agent_id,
        **kwargs
    )


# Context manager for A2A communication
class A2AContext:
    """Context manager for A2A communication"""
    
    def __init__(self, executor: ScalewayA2AExecutor):
        self.executor = executor
    
    async def __aenter__(self):
        await self.executor.connect()
        return self.executor
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.executor.disconnect()
