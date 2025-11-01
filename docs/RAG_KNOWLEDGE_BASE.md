# 🧠 Island Hopper RAG Knowledge Base

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Scaleway Integrations](#scaleway-integrations)
4. [Development Patterns](#development-patterns)
5. [Deployment Strategies](#deployment-strategies)
6. [Performance Optimization](#performance-optimization)
7. [Security & Compliance](#security--compliance)
8. [Monitoring & Observability](#monitoring--observability)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Best Practices](#best-practices)
11. [Migration Patterns](#migration-patterns)
12. [Advanced Configurations](#advanced-configurations)
13. [API Reference](#api-reference)
14. [Examples & Use Cases](#examples--use-cases)
15. [Frequently Asked Questions](#frequently-asked-questions)

---

## 🏗️ Architecture Overview

### System Architecture

The Island Hopper ecosystem is built on a layered architecture that extends the Strands SDK with Scaleway-optimized components. This design ensures maximum compatibility while providing cloud-native enhancements.

#### Core Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  • AI Agents & Workflows                                     │
│  • Business Logic                                           │
│  • User Interfaces                                          │
├─────────────────────────────────────────────────────────────┤
│                    Island Hopper SDK                        │
│  • Provider-Agnostic Models                                 │
│  • Scaleway-Optimized Sessions                              │
│  • Enhanced Tool System                                      │
│  • NATS-based A2A Communication                             │
├─────────────────────────────────────────────────────────────┤
│                    Strands SDK Core                         │
│  • Base Agent Framework                                     │
│  • Tool Abstraction Layer                                   │
│  • Session Management Interface                             │
│  • Telemetry Hooks                                           │
├─────────────────────────────────────────────────────────────┤
│                    Scaleway Infrastructure                  │
│  • Kapsys Kubernetes Cluster                                │
│  • PostgreSQL Databases                                     │
│  • Object Storage (S3-compatible)                           │
│  • NATS Messaging                                           │
│  • Cockpit Monitoring                                       │
└─────────────────────────────────────────────────────────────┘
```

#### Design Principles

1. **Provider Agnosticism**: Support multiple AI providers without code changes
2. **Scaleway Native**: Leverage Scaleway services for optimal performance
3. **Backward Compatibility**: Maintain full compatibility with Strands SDK
4. **Production Ready**: Built for enterprise-scale deployments
5. **Developer Experience**: Simple APIs with powerful capabilities

#### Component Interaction Flow

```
User Request → Agent → ScalewayModel → Provider Routing → AI Provider
                ↓
            Tools → Scaleway Services → Database/Object Storage
                ↓
            Sessions → ScalewaySessionRepository → PostgreSQL
                ↓
            A2A → ScalewayA2AExecutor → NATS → Other Agents
                ↓
            Telemetry → ScalewayTelemetry → Cockpit/Monitoring
```

### Integration Patterns

#### Strands SDK Integration

Island Hopper extends rather than replaces the Strands SDK:

```python
# Core Strands SDK (unchanged)
from strands import Agent, BaseTool, BaseA2AExecutor

# Island Hopper extensions
from island_hopper_sdk import ScalewayModel, ScalewaySessionRepository
from island_hopper_tools import ScalewayTool, ObjectStorageTool

# Seamless integration
class EnhancedAgent(Agent):
    def __init__(self):
        super().__init__(
            model=ScalewayModel(),  # Enhanced model routing
            session_repository=ScalewaySessionRepository(),  # Scaleway sessions
            tools=[ObjectStorageTool()]  # Scaleway-native tools
        )
```

#### Provider Abstraction Layer

The provider abstraction allows seamless switching between AI providers:

```python
# Configuration-driven provider selection
model = ScalewayModel(
    provider_config={
        "openrouter": {"endpoint": "...", "models": ["groq/llama-4-scout"]},
        "anthropic": {"endpoint": "...", "models": ["claude-4.5-sonnet"]},
        "scaleway": {"endpoint": "...", "models": ["scaleway/mistral-7b"]}  # Future
    },
    routing_strategy="cost_optimized"  # Automatic provider selection
)
```

---

## 🧩 Core Components

### ScalewayModel

The ScalewayModel provides provider-agnostic AI model access with intelligent routing and failover capabilities.

#### Core Features

- **Multi-Provider Support**: OpenRouter, Anthropic, OpenAI, and future Scaleway models
- **Automatic Failover**: Seamless switching between providers on failures
- **Cost Optimization**: Intelligent routing based on cost, quality, or speed
- **Rate Limiting**: Built-in awareness of provider rate limits
- **Circuit Breaker**: Prevents cascading failures

#### Configuration Options

```python
# Basic configuration
model = create_scaleway_model(
    primary_provider="openrouter",
    primary_model="groq/llama-4-scout",
    fallback_provider="anthropic",
    fallback_model="claude-4.5-sonnet"
)

# Advanced configuration
model = ScalewayModel(
    provider_config={
        "openrouter": {
            "name": "OpenRouter",
            "endpoint": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "models": ["groq/llama-4-scout", "anthropic/claude-4.5-sonnet"],
            "priority": 1,
            "max_rpm": 1000,
            "max_tpm": 10000
        },
        "anthropic": {
            "name": "Anthropic",
            "endpoint": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
            "models": ["claude-4.5-sonnet", "claude-3-5-sonnet"],
            "priority": 2,
            "max_rpm": 500,
            "max_tpm": 5000
        }
    },
    model_config={
        "primary_provider": "openrouter",
        "primary_model": "groq/llama-4-scout",
        "fallback_provider": "anthropic",
        "fallback_model": "claude-4.5-sonnet",
        "routing_strategy": "balanced"
    }
)
```

#### Routing Strategies

1. **Cost Optimized**: Selects lowest-cost provider that meets requirements
2. **Quality First**: Prioritizes highest quality models
3. **Balanced**: Balances cost, quality, and speed
4. **Speed First**: Prioritizes fastest response times

#### Usage Patterns

```python
# Simple generation
response = await model.generate("What is the capital of France?")

# Advanced generation with parameters
response = await model.generate(
    "Explain quantum computing",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9
)

# Provider switching at runtime
model.switch_provider("anthropic", "claude-4.5-sonnet")

# Get provider information
info = model.get_provider_info()
print(f"Current provider: {info['primary_provider']}")
```

#### Error Handling

```python
try:
    response = await model.generate(prompt)
except ProviderError as e:
    logger.error(f"Provider error: {e}")
    # Automatic fallback to secondary provider
except RateLimitError as e:
    logger.warning(f"Rate limit exceeded: {e}")
    # Automatic rate limit handling
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    # Requires manual intervention
```

### ScalewaySessionRepository

Scaleway-optimized session management using PostgreSQL with enhanced features for production deployments.

#### Core Features

- **PostgreSQL Backend**: Persistent, scalable session storage
- **Connection Pooling**: Optimized database connections
- **Automatic Backup**: Integrated with Scaleway backup systems
- **Session Analytics**: Built-in session metrics and insights
- **Multi-Region Support**: Geo-distributed session storage

#### Database Schema

```sql
-- Core sessions table
CREATE TABLE strands_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    agent_id VARCHAR(255),
    messages JSONB NOT NULL,
    metadata JSONB,
    context JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    cost_total DECIMAL(10, 4) DEFAULT 0.0000
);

-- Indexes for performance
CREATE INDEX idx_sessions_session_id ON strands_sessions(session_id);
CREATE INDEX idx_sessions_user_id ON strands_sessions(user_id);
CREATE INDEX idx_sessions_agent_id ON strands_sessions(agent_id);
CREATE INDEX idx_sessions_created_at ON strands_sessions(created_at);
CREATE INDEX idx_sessions_last_activity ON strands_sessions(last_activity);
CREATE INDEX idx_sessions_expires_at ON strands_sessions(expires_at);

-- Session analytics table
CREATE TABLE strand_session_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) REFERENCES strands_sessions(session_id),
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_agent TEXT,
    ip_address INET,
    region VARCHAR(50)
);

-- Session costs tracking
CREATE TABLE strand_session_costs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) REFERENCES strands_sessions(session_id),
    provider VARCHAR(50),
    model VARCHAR(100),
    tokens_used INTEGER,
    cost_amount DECIMAL(10, 4),
    cost_currency VARCHAR(3) DEFAULT 'USD',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Configuration Options

```python
# Basic configuration
sessions = ScalewaySessionRepository(
    connection_string="postgresql://user:pass@scaleway-db:5432/strands"
)

# Advanced configuration
sessions = ScalewaySessionRepository(
    connection_string=os.getenv("SCALEWAY_DATABASE_URL"),
    pool_size=20,
    max_overflow=30,
    pool_timeout=60,
    pool_recycle=3600,
    pool_pre_ping=True,
    ssl_mode="require",
    backup_enabled=True,
    analytics_enabled=True,
    cost_tracking_enabled=True,
    session_ttl=86400,  # 24 hours
    cleanup_interval=3600  # 1 hour
)
```

#### Session Management Patterns

```python
# Create session
session_id = await sessions.create_session(
    user_id="user-123",
    agent_id="agent-456",
    metadata={"source": "web", "version": "1.0"}
)

# Add message to session
await sessions.add_message(
    session_id=session_id,
    role="user",
    content="Hello, how can you help me?",
    metadata={"timestamp": datetime.now().isoformat()}
)

# Get session with messages
session = await sessions.get_session(session_id)
print(f"Session has {len(session.messages)} messages")

# Update session metadata
await sessions.update_metadata(
    session_id=session_id,
    metadata={"last_intent": "help_request", "priority": "high"}
)

# Session analytics
analytics = await sessions.get_session_analytics(session_id)
print(f"Session cost: ${analytics['total_cost']:.4f}")

# Cleanup expired sessions
cleaned_count = await sessions.cleanup_expired_sessions()
print(f"Cleaned up {cleaned_count} expired sessions")
```

#### Advanced Features

```python
# Session search and filtering
sessions = await sessions.search_sessions(
    user_id="user-123",
    date_from=datetime.now() - timedelta(days=7),
    date_to=datetime.now(),
    limit=10,
    offset=0
)

# Session aggregation
stats = await sessions.get_session_statistics(
    group_by="user_id",
    date_from=datetime.now() - timedelta(days=30)
)

# Batch operations
batch_results = await sessions.batch_update_sessions(
    session_ids=["sess-1", "sess-2", "sess-3"],
    updates={"metadata": {"archived": True}}
)
```

### ScalewayTool

Base class for Scaleway-native tools with enhanced capabilities and integration patterns.

#### Core Features

- **Scaleway Service Integration**: Native integration with Object Storage, Database, and other services
- **Automatic Authentication**: Leverages Scaleway IAM and access keys
- **Resource Management**: Built-in resource lifecycle management
- **Error Handling**: Comprehensive error handling and retry logic
- **Monitoring**: Automatic metrics and logging

#### Tool Architecture

```python
class ScalewayTool(BaseTool):
    """Base class for Scaleway-native tools"""
    
    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "fr-par",
        project_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.access_key = access_key or os.getenv("SCALEWAY_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("SCALEWAY_SECRET_KEY")
        self.region = region
        self.project_id = project_id or os.getenv("SCALEWAY_PROJECT_ID")
        
        # Initialize Scaleway clients
        self._init_clients()
        
        # Metrics and monitoring
        self.metrics = ToolMetrics()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _init_clients(self):
        """Initialize Scaleway service clients"""
        # Implemented by subclasses
        pass
    
    async def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute tool operation with monitoring"""
        start_time = time.time()
        
        try:
            # Execute operation
            result = await self._execute_operation(operation, **kwargs)
            
            # Record metrics
            execution_time = time.time() - start_time
            self.metrics.record_execution(operation, execution_time, True)
            
            return ToolResult(
                success=True,
                data=result,
                operation=operation,
                execution_time=execution_time
            )
            
        except Exception as e:
            # Record error metrics
            execution_time = time.time() - start_time
            self.metrics.record_execution(operation, execution_time, False)
            
            self.logger.error(f"Tool execution failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                operation=operation,
                execution_time=execution_time
            )
    
    async def _execute_operation(self, operation: str, **kwargs):
        """Execute specific tool operation - implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_operation")
```

#### Tool Registration and Discovery

```python
# Dynamic tool registration
registry = ScalewayToolRegistry(
    auto_discover=True,
    search_paths=["./tools", "./scaleway_tools"],
    enable_validation=True
)

# Register individual tools
registry.register_tool(ObjectStorageTool())
registry.register_tool(DatabaseTool())

# Get available tools
available_tools = registry.list_tools()
print(f"Available tools: {available_tools}")

# Get tool schemas for function calling
schemas = registry.get_function_schemas()

# Execute tool by name
result = await registry.execute_tool(
    tool_name="ObjectStorageTool",
    operation="upload_file",
    bucket="my-bucket",
    file_path="data/file.txt",
    content="Hello, World!"
)
```

---

## ☁️ Scaleway Integrations

### Object Storage Integration

Scaleway Object Storage provides S3-compatible storage for files, documents, and agent artifacts.

#### Core Features

- **S3-Compatible API**: Full compatibility with S3 tools and libraries
- **Lifecycle Management**: Automatic file expiration and archival
- **Versioning**: File version tracking and rollback
- **Encryption**: Server-side and client-side encryption
- **CDN Integration**: Global content delivery network

#### ObjectStorageTool Implementation

```python
class ObjectStorageTool(ScalewayTool):
    """Scaleway Object Storage tool for file operations"""
    
    def __init__(self, default_bucket: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.default_bucket = default_bucket
        self.s3_client = self._create_s3_client()
    
    def _create_s3_client(self):
        """Create S3 client for Scaleway Object Storage"""
        import boto3
        
        return boto3.client(
            's3',
            endpoint_url=f"https://s3.{self.region}.scw.cloud",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )
    
    async def _execute_operation(self, operation: str, **kwargs):
        """Execute Object Storage operations"""
        
        if operation == "upload_file":
            return await self._upload_file(**kwargs)
        elif operation == "download_file":
            return await self._download_file(**kwargs)
        elif operation == "list_files":
            return await self._list_files(**kwargs)
        elif operation == "delete_file":
            return await self._delete_file(**kwargs)
        elif operation == "create_bucket":
            return await self._create_bucket(**kwargs)
        elif operation == "get_presigned_url":
            return await self._get_presigned_url(**kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _upload_file(self, bucket: str, file_path: str, content: str, **kwargs):
        """Upload file to Object Storage"""
        bucket = bucket or self.default_bucket
        
        # Upload file
        self.s3_client.put_object(
            Bucket=bucket,
            Key=file_path,
            Body=content.encode('utf-8'),
            **kwargs
        )
        
        # Get file metadata
        metadata = self.s3_client.head_object(Bucket=bucket, Key=file_path)
        
        return {
            "bucket": bucket,
            "file_path": file_path,
            "size": metadata['ContentLength'],
            "etag": metadata['ETag'],
            "last_modified": metadata['LastModified'].isoformat(),
            "url": f"https://s3.{self.region}.scw.cloud/{bucket}/{file_path}"
        }
    
    async def _download_file(self, bucket: str, file_path: str, **kwargs):
        """Download file from Object Storage"""
        bucket = bucket or self.default_bucket
        
        response = self.s3_client.get_object(Bucket=bucket, Key=file_path)
        content = response['Body'].read().decode('utf-8')
        
        return {
            "bucket": bucket,
            "file_path": file_path,
            "content": content,
            "size": response['ContentLength'],
            "last_modified": response['LastModified'].isoformat(),
            "metadata": response.get('Metadata', {})
        }
    
    async def _list_files(self, bucket: str, prefix: str = "", **kwargs):
        """List files in bucket"""
        bucket = bucket or self.default_bucket
        
        response = self.s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            **kwargs
        )
        
        files = []
        for obj in response.get('Contents', []):
            files.append({
                "key": obj['Key'],
                "size": obj['Size'],
                "last_modified": obj['LastModified'].isoformat(),
                "etag": obj['ETag'],
                "storage_class": obj.get('StorageClass', 'STANDARD')
            })
        
        return {
            "bucket": bucket,
            "prefix": prefix,
            "files": files,
            "total_count": len(files),
            "total_size": sum(f['size'] for f in files),
            "is_truncated": response.get('IsTruncated', False)
        }
    
    async def _get_presigned_url(self, bucket: str, file_path: str, expiration: int = 3600, **kwargs):
        """Generate presigned URL for file access"""
        bucket = bucket or self.default_bucket
        
        url = self.s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': file_path},
            ExpiresIn=expiration
        )
        
        return {
            "bucket": bucket,
            "file_path": file_path,
            "url": url,
            "expires_in": expiration,
            "expires_at": (datetime.now() + timedelta(seconds=expiration)).isoformat()
        }
```

#### Usage Patterns

```python
# Initialize tool
storage = ObjectStorageTool(
    default_bucket="agent-data",
    region="fr-par"
)

# Upload file
result = await storage.execute(
    operation="upload_file",
    bucket="agent-data",
    file_path="reports/daily_report.json",
    content=json.dumps({"date": "2024-01-15", "metrics": {...}}),
    Metadata={"type": "report", "generated_by": "agent"}
)

# Download file
result = await storage.execute(
    operation="download_file",
    bucket="agent-data",
    file_path="reports/daily_report.json"
)

# List files with prefix
result = await storage.execute(
    operation="list_files",
    bucket="agent-data",
    prefix="reports/",
    MaxKeys=100
)

# Generate presigned URL for sharing
result = await storage.execute(
    operation="get_presigned_url",
    bucket="agent-data",
    file_path="reports/daily_report.json",
    expiration=3600  # 1 hour
)
```

#### Advanced Features

```python
# Batch operations
batch_results = await storage.batch_upload_files(
    bucket="agent-data",
    files={
        "data1.json": {"content": "...", "metadata": {...}},
        "data2.json": {"content": "...", "metadata": {...}},
        "data3.json": {"content": "...", "metadata": {...}}
    }
)

# Lifecycle management
await storage.set_lifecycle_rule(
    bucket="agent-data",
    rule={
        "id": "expire-old-reports",
        "status": "Enabled",
        "filter": {"prefix": "reports/"},
        "transitions": [
            {"days": 30, "storage_class": "GLACIER"},
            {"days": 90, "storage_class": "DEEP_ARCHIVE"}
        ],
        "expiration": {"days": 365}
    }
)

# Bucket analytics
analytics = await storage.get_bucket_analytics("agent-data")
print(f"Bucket size: {analytics['total_size']} bytes")
print(f"File count: {analytics['file_count']}")
```

### Database Integration

Scaleway PostgreSQL integration for persistent data storage, analytics, and agent state management.

#### Core Features

- **PostgreSQL Backend**: Full PostgreSQL compatibility
- **Connection Pooling**: Optimized database connections
- **Query Builder**: Safe SQL query construction
- **Analytics**: Built-in query performance analytics
- **Backup Integration**: Automatic backup and point-in-time recovery

#### DatabaseTool Implementation

```python
class DatabaseTool(ScalewayTool):
    """Scaleway PostgreSQL database tool"""
    
    def __init__(self, connection_string: str, **kwargs):
        super().__init__(**kwargs)
        self.connection_string = connection_string
        self.pool = self._create_connection_pool()
    
    def _create_connection_pool(self):
        """Create PostgreSQL connection pool"""
        import asyncpg
        
        return asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
    
    async def _execute_operation(self, operation: str, **kwargs):
        """Execute database operations"""
        
        if operation == "execute_query":
            return await self._execute_query(**kwargs)
        elif operation == "execute_batch":
            return await self._execute_batch(**kwargs)
        elif operation == "get_table_info":
            return await self._get_table_info(**kwargs)
        elif operation == "export_to_csv":
            return await self._export_to_csv(**kwargs)
        elif operation == "import_from_csv":
            return await self._import_from_csv(**kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _execute_query(self, query: str, parameters: Optional[List] = None, **kwargs):
        """Execute SQL query with safety checks"""
        
        async with self.pool.acquire() as conn:
            # Safety check for destructive operations
            if self._is_destructive_query(query):
                self.logger.warning(f"Executing destructive query: {query[:100]}...")
            
            # Execute query
            if parameters:
                result = await conn.fetch(query, *parameters)
            else:
                result = await conn.fetch(query)
            
            # Convert to dict format
            rows = [dict(row) for row in result]
            
            return {
                "query": query,
                "parameters": parameters,
                "row_count": len(rows),
                "rows": rows,
                "execution_time": kwargs.get("_execution_time", 0)
            }
    
    async def _get_table_info(self, table_name: str, **kwargs):
        """Get detailed table information"""
        
        async with self.pool.acquire() as conn:
            # Get column information
            columns_query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position
            """
            
            columns = await conn.fetch(columns_query, table_name)
            
            # Get table statistics
            stats_query = """
                SELECT 
                    n_live_tup as row_count,
                    n_tup_ins as total_inserts,
                    n_tup_upd as total_updates,
                    n_tup_del as total_deletes
                FROM pg_stat_user_tables
                WHERE schemaname = 'public' AND relname = $1
            """
            
            stats = await conn.fetchrow(stats_query, table_name)
            
            return {
                "table_name": table_name,
                "columns": [dict(col) for col in columns],
                "statistics": dict(stats) if stats else {},
                "indexes": await self._get_table_indexes(table_name, conn)
            }
    
    async def _export_to_csv(self, query: str, file_path: str, **kwargs):
        """Export query results to CSV"""
        
        import csv
        import io
        
        # Execute query
        result = await self._execute_query(query)
        
        # Create CSV
        output = io.StringIO()
        if result['rows']:
            writer = csv.DictWriter(output, fieldnames=result['rows'][0].keys())
            writer.writeheader()
            writer.writerows(result['rows'])
        
        # Save to object storage
        storage = ObjectStorageTool()
        await storage.execute(
            operation="upload_file",
            file_path=file_path,
            content=output.getvalue()
        )
        
        return {
            "query": query,
            "file_path": file_path,
            "row_count": result['row_count'],
            "file_size": len(output.getvalue())
        }
    
    def _is_destructive_query(self, query: str) -> bool:
        """Check if query is destructive (DROP, DELETE, etc.)"""
        destructive_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']
        query_upper = query.upper().strip()
        return any(query_upper.startswith(keyword) for keyword in destructive_keywords)
```

#### Usage Patterns

```python
# Initialize database tool
db = DatabaseTool(
    connection_string="postgresql://user:pass@scaleway-db:5432/agents"
)

# Execute query
result = await db.execute(
    operation="execute_query",
    query="SELECT * FROM agent_logs WHERE created_at > $1 ORDER BY created_at DESC",
    parameters=["2024-01-01"]
)

# Get table information
table_info = await db.execute(
    operation="get_table_info",
    table_name="agent_logs"
)

# Export to CSV
export_result = await db.execute(
    operation="export_to_csv",
    query="SELECT * FROM agent_sessions WHERE status = 'active'",
    file_path="exports/active_sessions.csv"
)

# Batch insert
batch_data = [
    ("agent-1", "task-1", "completed", 100),
    ("agent-2", "task-2", "running", 50),
    ("agent-3", "task-3", "pending", 0)
]

await db.execute(
    operation="execute_batch",
    query="INSERT INTO agent_tasks (agent_id, task_id, status, progress) VALUES ($1, $2, $3, $4)",
    parameters=batch_data
)
```

### NATS Messaging Integration

NATS-based agent-to-agent communication with JetStream persistence and high-performance messaging.

#### Core Features

- **Lightweight Messaging**: High-performance message passing
- **JetStream Persistence**: Durable message storage
- **Request-Reply Pattern**: Built-in request/response handling
- **Publish-Subscribe**: Flexible pub/sub messaging
- **Queue Groups**: Load balancing across consumers

#### ScalewayA2AExecutor Implementation

```python
class ScalewayA2AExecutor(ScalewayTool):
    """NATS-based agent-to-agent communication"""
    
    def __init__(self, agent_id: str, nats_url: str, **kwargs):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.nats_url = nats_url
        self.nc = None
        self.js = None
        self.message_handlers = {}
    
    async def connect(self):
        """Connect to NATS server"""
        import nats
        
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()
        
        # Create streams for persistence
        await self._setup_streams()
        
        # Subscribe to agent inbox
        await self._setup_subscriptions()
        
        self.logger.info(f"Connected to NATS as agent {self.agent_id}")
    
    async def send_message(
        self,
        recipient_id: str,
        message_type: str,
        payload: Dict[str, Any],
        **kwargs
    ):
        """Send message to another agent"""
        
        message = {
            "sender_id": self.agent_id,
            "recipient_id": recipient_id,
            "message_type": message_type,
            "payload": payload,
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid.uuid4()),
            "correlation_id": kwargs.get("correlation_id")
        }
        
        # Publish to recipient's inbox
        subject = f"agents.{recipient_id}.inbox"
        
        if kwargs.get("persistent", False):
            # Use JetStream for persistence
            await self.js.publish(subject, json.dumps(message).encode())
        else:
            # Regular NATS message
            await self.nc.publish(subject, json.dumps(message).encode())
        
        return {
            "message_id": message["message_id"],
            "recipient_id": recipient_id,
            "sent_at": message["timestamp"]
        }
    
    async def request_response(
        self,
        recipient_id: str,
        payload: Dict[str, Any],
        timeout: float = 30.0,
        **kwargs
    ):
        """Send request and wait for response"""
        
        # Create temporary inbox for response
        inbox = f"agents.{self.agent_id}.requests.{str(uuid.uuid4())}"
        
        # Send request message
        request_message = {
            "sender_id": self.agent_id,
            "recipient_id": recipient_id,
            "message_type": "request",
            "payload": payload,
            "reply_to": inbox,
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid.uuid4())
        }
        
        # Send request
        await self.nc.publish(
            f"agents.{recipient_id}.inbox",
            json.dumps(request_message).encode()
        )
        
        # Wait for response
        response = await self.nc.request(inbox, b"", timeout=timeout)
        response_data = json.loads(response.data.decode())
        
        return response_data
    
    async def broadcast_message(
        self,
        message_type: str,
        payload: Dict[str, Any],
        **kwargs
    ):
        """Broadcast message to all agents"""
        
        message = {
            "sender_id": self.agent_id,
            "message_type": message_type,
            "payload": payload,
            "timestamp": datetime.now().isoformat(),
            "broadcast": True
        }
        
        # Publish to broadcast topic
        await self.nc.publish(
            "agents.broadcast",
            json.dumps(message).encode()
        )
        
        return {"broadcast_id": message["timestamp"], "recipients": "all"}
    
    async def _setup_streams(self):
        """Setup JetStream streams for persistence"""
        
        # Agent messages stream
        await self.js.add_stream(
            name="agent_messages",
            subjects=["agents.*.inbox"],
            retention="work_queue",
            max_msgs_per_subject=1000,
            max_age=86400  # 24 hours
        )
        
        # Broadcast stream
        await self.js.add_stream(
            name="agent_broadcasts",
            subjects=["agents.broadcast"],
            retention="limits",
            max_msgs=10000,
            max_age=3600  # 1 hour
        )
    
    async def _setup_subscriptions(self):
        """Setup message subscriptions"""
        
        # Subscribe to agent inbox
        await self.nc.subscribe(
            f"agents.{self.agent_id}.inbox",
            cb=self._handle_inbox_message
        )
        
        # Subscribe to broadcasts
        await self.nc.subscribe(
            "agents.broadcast",
            cb=self._handle_broadcast_message
        )
    
    async def _handle_inbox_message(self, msg):
        """Handle incoming message"""
        try:
            message_data = json.loads(msg.data.decode())
            message_type = message_data.get("message_type")
            
            # Route to appropriate handler
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](message_data)
            else:
                await self._handle_default_message(message_data)
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    def register_handler(self, message_type: str, handler):
        """Register message handler"""
        self.message_handlers[message_type] = handler
```

#### Usage Patterns

```python
# Initialize A2A executor
a2a = ScalewayA2AExecutor(
    agent_id="data-analyst",
    nats_url="nats://scaleway-nats:4222"
)

await a2a.connect()

# Register message handlers
async def handle_task_request(message):
    task = message["payload"]["task"]
    result = await process_task(task)
    
    await a2a.send_message(
        recipient_id=message["sender_id"],
        message_type="response",
        payload={"result": result, "task_id": message["payload"]["task_id"]}
    )

a2a.register_handler("task_request", handle_task_request)

# Send message to another agent
await a2a.send_message(
    recipient_id="report-generator",
    message_type="generate_report",
    payload={"data": {...}, "format": "pdf"}
)

# Request-response pattern
response = await a2a.request_response(
    recipient_id="database-agent",
    payload={"query": "SELECT * FROM metrics"},
    timeout=10.0
)

# Broadcast to all agents
await a2a.broadcast_message(
    message_type="system_shutdown",
    payload={"reason": "maintenance", "eta": "2 hours"}
)
```

---

## 🛠️ Development Patterns

### Agent Development Patterns

#### Basic Agent Structure

```python
from strands import Agent
from island_hopper_sdk import ScalewayModel, ScalewaySessionRepository
from island_hopper_tools import ObjectStorageTool, DatabaseTool

class ScalewayOptimizedAgent(Agent):
    """Base agent with Scaleway optimizations"""
    
    def __init__(
        self,
        agent_id: str,
        model_config: Optional[Dict] = None,
        enable_persistence: bool = True,
        enable_tools: bool = True,
        **kwargs
    ):
        self.agent_id = agent_id
        
        # Initialize Scaleway model
        self.model = create_scaleway_model(**(model_config or {}))
        
        # Initialize session repository
        if enable_persistence:
            self.session_repository = ScalewaySessionRepository()
        else:
            self.session_repository = None
        
        # Initialize tools
        self.tools = []
        if enable_tools:
            self.tools = [
                ObjectStorageTool(),
                DatabaseTool()
            ]
        
        # Initialize A2A communication
        self.a2a = ScalewayA2AExecutor(
            agent_id=agent_id,
            nats_url=os.getenv("NATS_URL")
        )
        
        # Initialize telemetry
        self.telemetry = configure_scaleway_telemetry(
            service_name=f"agent-{agent_id}"
        )
        
        super().__init__(
            model=self.model,
            session_repository=self.session_repository,
            tools=self.tools,
            **kwargs
        )
    
    async def start(self):
        """Start agent services"""
        await self.a2a.connect()
        await self._register_message_handlers()
        self.logger.info(f"Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop agent services"""
        await self.a2a.disconnect()
        if self.session_repository:
            await self.session_repository.cleanup()
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    async def _register_message_handlers(self):
        """Register message handlers for A2A communication"""
        self.a2a.register_handler("task_request", self.handle_task_request)
        self.a2a.register_handler("status_request", self.handle_status_request)
    
    async def handle_task_request(self, message):
        """Handle incoming task requests"""
        try:
            task = message["payload"]["task"]
            task_id = message["payload"]["task_id"]
            
            # Execute task
            result = await self.execute_task(task)
            
            # Send response
            await self.a2a.send_message(
                recipient_id=message["sender_id"],
                message_type="task_response",
                payload={
                    "task_id": task_id,
                    "result": result,
                    "status": "completed"
                }
            )
            
        except Exception as e:
            # Send error response
            await self.a2a.send_message(
                recipient_id=message["sender_id"],
                message_type="task_response",
                payload={
                    "task_id": message["payload"]["task_id"],
                    "error": str(e),
                    "status": "failed"
                }
            )
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent task - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute_task")
```

#### Specialized Agent Examples

```python
class DataAnalysisAgent(ScalewayOptimizedAgent):
    """Specialized agent for data analysis tasks"""
    
    def __init__(self, **kwargs):
        super().__init__(
            agent_id="data-analyst",
            model_config={
                "primary_provider": "openrouter",
                "primary_model": "groq/llama-4-scout"
            },
            **kwargs
        )
        
        # Add data analysis tools
        self.tools.extend([
            DatabaseTool(),
            ObjectStorageTool(),
            # Add custom data analysis tools
        ])
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data analysis task"""
        
        task_type = task.get("type")
        
        if task_type == "query_analysis":
            return await self._analyze_query(task["query"])
        elif task_type == "report_generation":
            return await self._generate_report(task["parameters"])
        elif task_type == "data_export":
            return await self._export_data(task["export_config"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze SQL query and generate insights"""
        
        # Get database tool
        db_tool = self.get_tool("DatabaseTool")
        
        # Execute query
        result = await db_tool.execute(
            operation="execute_query",
            query=query
        )
        
        # Generate analysis using AI
        analysis_prompt = f"""
        Analyze this query result and provide insights:
        
        Query: {query}
        Results: {result['rows'][:10]}  # First 10 rows
        
        Provide:
        1. Summary of findings
        2. Key trends or patterns
        3. Recommendations
        4. Potential issues or anomalies
        """
        
        analysis = await self.model.generate(analysis_prompt)
        
        return {
            "query": query,
            "row_count": result["row_count"],
            "analysis": analysis,
            "sample_data": result["rows"][:5]
        }

class ReportGenerationAgent(ScalewayOptimizedAgent):
    """Specialized agent for report generation"""
    
    def __init__(self, **kwargs):
        super().__init__(
            agent_id="report-generator",
            model_config={
                "primary_provider": "anthropic",
                "primary_model": "claude-4.5-sonnet"
            },
            **kwargs
        )
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute report generation task"""
        
        report_type = task.get("type")
        data = task.get("data", {})
        
        if report_type == "performance_report":
            return await self._generate_performance_report(data)
        elif report_type == "cost_analysis":
            return await self._generate_cost_analysis(data)
        elif report_type == "usage_summary":
            return await self._generate_usage_summary(data)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    async def _generate_performance_report(self, data: Dict) -> Dict[str, Any]:
        """Generate performance report"""
        
        # Generate report content
        report_prompt = f"""
        Generate a comprehensive performance report based on this data:
        
        {json.dumps(data, indent=2)}
        
        Include:
        1. Executive Summary
        2. Key Performance Indicators
        3. Trend Analysis
        4. Recommendations
        5. Action Items
        
        Format as structured JSON.
        """
        
        report_content = await self.model.generate(report_prompt)
        
        # Parse and validate report structure
        try:
            report = json.loads(report_content)
        except json.JSONDecodeError:
            # Fallback structure
            report = {
                "executive_summary": report_content,
                "kpis": {},
                "trends": [],
                "recommendations": [],
                "action_items": []
            }
        
        # Save report to object storage
        storage = self.get_tool("ObjectStorageTool")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/performance_{timestamp}.json"
        
        await storage.execute(
            operation="upload_file",
            file_path=report_path,
            content=json.dumps(report, indent=2),
            Metadata={
                "type": "performance_report",
                "generated_by": self.agent_id,
                "timestamp": timestamp
            }
        )
        
        return {
            "report_id": f"perf_{timestamp}",
            "report_path": report_path,
            "summary": report.get("executive_summary", ""),
            "generated_at": timestamp
        }
```

### Tool Development Patterns

#### Custom Scaleway Tool

```python
class CustomScalewayTool(ScalewayTool):
    """Template for custom Scaleway tools"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize tool-specific resources
        self._init_custom_resources()
    
    def _init_custom_resources(self):
        """Initialize tool-specific resources"""
        # Example: Initialize API clients, connections, etc.
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for function calling"""
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "Custom Scaleway tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation to execute",
                        "enum": self.get_supported_operations()
                    }
                },
                "required": ["operation"]
            }
        }
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations"""
        return ["custom_operation_1", "custom_operation_2"]
    
    async def _execute_operation(self, operation: str, **kwargs):
        """Execute tool operation"""
        
        if operation == "custom_operation_1":
            return await self._custom_operation_1(**kwargs)
        elif operation == "custom_operation_2":
            return await self._custom_operation_2(**kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _custom_operation_1(self, **kwargs):
        """Implement custom operation 1"""
        # Custom implementation
        return {"result": "success", "data": {...}}
    
    async def _custom_operation_2(self, **kwargs):
        """Implement custom operation 2"""
        # Custom implementation
        return {"result": "success", "data": {...}}
```

#### Tool Composition Pattern

```python
class CompositeScalewayTool(ScalewayTool):
    """Tool that combines multiple Scaleway services"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize component tools
        self.storage = ObjectStorageTool(**kwargs)
        self.database = DatabaseTool(
            connection_string=os.getenv("SCALEWAY_DATABASE_URL"),
            **kwargs
        )
        self.nats = ScalewayA2AExecutor(
            agent_id="composite-tool",
            nats_url=os.getenv("NATS_URL"),
            **kwargs
        )
    
    async def _execute_operation(self, operation: str, **kwargs):
        """Execute composite operations"""
        
        if operation == "process_and_store":
            return await self._process_and_store(**kwargs)
        elif operation == "analyze_and_notify":
            return await self._analyze_and_notify(**kwargs)
        elif operation == "backup_and_archive":
            return await self._backup_and_archive(**kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _process_and_store(self, data: Dict, storage_path: str, **kwargs):
        """Process data and store in multiple locations"""
        
        # Step 1: Store raw data in object storage
        await self.storage.execute(
            operation="upload_file",
            file_path=f"raw/{storage_path}",
            content=json.dumps(data),
            Metadata={"stage": "raw", "processed_at": datetime.now().isoformat()}
        )
        
        # Step 2: Process data and store in database
        processed_data = await self._process_data(data)
        await self.database.execute(
            operation="execute_query",
            query="INSERT INTO processed_data (data, metadata) VALUES ($1, $2)",
            parameters=[json.dumps(processed_data), {"source": storage_path}]
        )
        
        # Step 3: Store processed data in object storage
        await self.storage.execute(
            operation="upload_file",
            file_path=f"processed/{storage_path}",
            content=json.dumps(processed_data),
            Metadata={"stage": "processed", "processed_at": datetime.now().isoformat()}
        )
        
        return {
            "status": "completed",
            "raw_path": f"raw/{storage_path}",
            "processed_path": f"processed/{storage_path}",
            "database_id": processed_data.get("id")
        }
    
    async def _process_data(self, data: Dict) -> Dict:
        """Process data - implement custom logic"""
        # Example processing logic
        processed = {
            "id": str(uuid.uuid4()),
            "original_data": data,
            "processed_at": datetime.now().isoformat(),
            "summary": f"Processed {len(data)} items"
        }
        return processed
```

---

## 🚀 Deployment Strategies

### Local Development

#### Docker Compose Setup

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: island_hopper_dev
      POSTGRES_USER: developer
      POSTGRES_PASSWORD: dev_password_123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    networks:
      - island-hopper-dev

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - island-hopper-dev

  # NATS Messaging
  nats:
    image: nats:2.10-alpine
    ports:
      - "4222:4222"
      - "8222:8222"  # Monitoring
    volumes:
      - nats_data:/data
    networks:
      - island-hopper-dev
    command: ["--store_dir", "/data", "--jetstream", "--http_port", "8222"]

  # MinIO (S3-compatible storage)
  minio:
    image: minio/minio:latest
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    ports:
      - "9000:9000"
      - "9001:9001"  # Console
    volumes:
      - minio_data:/data
    networks:
      - island-hopper-dev
    command: server /data --console-address ":9001"

  # Agent Service
  agent:
    build:
      context: .
      dockerfile: Dockerfile.dev
    environment:
      # Database
      DATABASE_URL: postgresql://developer:dev_password_123@postgres:5432/island_hopper_dev
      
      # Object Storage
      STORAGE_ENDPOINT: http://minio:9000
      STORAGE_ACCESS_KEY: minioadmin
      STORAGE_SECRET_KEY: minioadmin123
      STORAGE_BUCKET: island-hopper-dev
      
      # NATS
      NATS_URL: nats://nats:4222
      
      # AI Providers
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - agent_logs:/app/logs
    networks:
      - island-hopper-dev
    depends_on:
      - postgres
      - redis
      - nats
      - minio

volumes:
  postgres_data:
  redis_data:
  nats_data:
  minio_data:
  agent_logs:

networks:
  island-hopper-dev:
    driver: bridge
```

#### Development Environment Setup

```bash
#!/bin/bash
# scripts/setup-dev.sh

echo "🏝️  Setting up Island Hopper Development Environment"

# Create environment file
cat > .env.dev << EOF
# AI Provider Keys
OPENROUTER_API_KEY=your_openrouter_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Scaleway Configuration (optional for development)
SCALEWAY_ACCESS_KEY=your_scaleway_access_key
SCALEWAY_SECRET_KEY=your_scaleway_secret_key
SCALEWAY_PROJECT_ID=your_project_id

# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development
EOF

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Initialize database
echo "🗄️  Initializing database..."
docker-compose -f docker-compose.dev.yml exec postgres psql -U developer -d island_hopper_dev -f /docker-entrypoint-initdb.d/init-db.sql

# Create MinIO buckets
echo "📦 Creating storage buckets..."
docker-compose -f docker-compose.dev.yml exec agent python -c "
from minio import Minio
import os

client = Minio('minio:9000', access_key='minioadmin', secret_key='minioadmin123', secure=False)

buckets = ['island-hopper-dev', 'agent-data', 'reports', 'logs']
for bucket in buckets:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f'Created bucket: {bucket}')
"

echo "✅ Development environment ready!"
echo "🌐 Agent API: http://localhost:8000"
echo "🗄️  Database: postgresql://developer:dev_password_123@localhost:5432/island_hopper_dev"
echo "📦 Storage Console: http://localhost:9001"
echo "📊 NATS Monitoring: http://localhost:8222"
```

### Production Deployment

#### Kubernetes Configuration

```yaml
# k8s/production/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: island-hopper
  labels:
    name: island-hopper
    environment: production

---
# k8s/production/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: island-hopper-config
  namespace: island-hopper
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DEBUG: "false"
  NATS_CLUSTER_NAME: "island-hopper-prod"
  STORAGE_BUCKET: "island-hopper-prod"
  
  # Model configuration
  PRIMARY_PROVIDER: "openrouter"
  PRIMARY_MODEL: "groq/llama-4-scout"
  FALLBACK_PROVIDER: "anthropic"
  FALLBACK_MODEL: "claude-4.5-sonnet"
  ROUTING_STRATEGY: "balanced"

---
# k8s/production/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: island-hopper-secrets
  namespace: island-hopper
type: Opaque
data:
  # Base64 encoded values
  DATABASE_URL: <base64-encoded-database-url>
  OPENROUTER_API_KEY: <base64-encoded-openrouter-key>
  ANTHROPIC_API_KEY: <base64-encoded-anthropic-key>
  OPENAI_API_KEY: <base64-encoded-openai-key>
  SCALEWAY_ACCESS_KEY: <base64-encoded-scaleway-access-key>
  SCALEWAY_SECRET_KEY: <base64-encoded-scaleway-secret-key>
  NATS_AUTH_TOKEN: <base64-encoded-nats-auth-token>

---
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: island-hopper-agent
  namespace: island-hopper
  labels:
    app: island-hopper-agent
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: island-hopper-agent
  template:
    metadata:
      labels:
        app: island-hopper-agent
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: agent
        image: rg.fr-par.scw.cloud/island-hopper/agent:latest
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: island-hopper-secrets
              key: DATABASE_URL
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: island-hopper-secrets
              key: OPENROUTER_API_KEY
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: island-hopper-secrets
              key: ANTHROPIC_API_KEY
        envFrom:
        - configMapRef:
            name: island-hopper-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
      imagePullSecrets:
      - name: scaleway-registry-secret
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000

---
# k8s/production/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: island-hopper-agent-service
  namespace: island-hopper
  labels:
    app: island-hopper-agent
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: island-hopper-agent

---
# k8s/production/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: island-hopper-ingress
  namespace: island-hopper
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.island-hopper.com
    secretName: island-hopper-tls
  rules:
  - host: api.island-hopper.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: island-hopper-agent-service
            port:
              number: 80

---
# k8s/production/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: island-hopper-hpa
  namespace: island-hopper
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: island-hopper-agent
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

#### Deployment Scripts

```bash
#!/bin/bash
# scripts/deploy-production.sh

set -e

echo "🚀 Deploying Island Hopper to Production"

# Configuration
REGISTRY="rg.fr-par.scw.cloud/island-hopper"
IMAGE_TAG="${1:-latest}"
NAMESPACE="island-hopper"

# Build and push image
echo "📦 Building and pushing Docker image..."
docker build -t $REGISTRY/agent:$IMAGE_TAG .
docker push $REGISTRY/agent:$IMAGE_TAG

# Apply Kubernetes manifests
echo "☸️  Applying Kubernetes manifests..."
kubectl apply -f k8s/production/namespace.yaml
kubectl apply -f k8s/production/configmap.yaml
kubectl apply -f k8s/production/secret.yaml
kubectl apply -f k8s/production/deployment.yaml
kubectl apply -f k8s/production/service.yaml
kubectl apply -f k8s/production/ingress.yaml
kubectl apply -f k8s/production/hpa.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/island-hopper-agent -n $NAMESPACE --timeout=300s

# Run health checks
echo "🏥 Running health checks..."
kubectl get pods -n $NAMESPACE
kubectl logs -l app=island-hopper-agent -n $NAMESPACE --tail=20

# Test API endpoint
echo "🔍 Testing API endpoint..."
EXTERNAL_IP=$(kubectl get ingress island-hopper-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if curl -f -s "http://$EXTERNAL_IP/health" > /dev/null; then
    echo "✅ Deployment successful!"
    echo "🌐 API available at: http://$EXTERNAL_IP"
else
    echo "❌ Health check failed"
    exit 1
fi

echo "🎉 Production deployment complete!"
```

### Multi-Environment Deployment

#### Environment Configuration

```yaml
# environments/staging/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: island-hopper-staging

resources:
- ../../k8s/base/

patchesStrategicMerge:
- deployment-patch.yaml
- configmap-patch.yaml

images:
- name: island-hopper/agent
  newTag: staging

replicas:
- name: island-hopper-agent
  count: 2

---
# environments/staging/deployment-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: island-hopper-agent
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: agent
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "250m"

---
# environments/staging/configmap-patch.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: island-hopper-config
data:
  ENVIRONMENT: "staging"
  LOG_LEVEL: "DEBUG"
  PRIMARY_MODEL: "groq/llama-4-scout"  # Use cost-effective model for staging
```

#### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy Island Hopper

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: rg.fr-par.scw.cloud/island-hopper

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Scaleway Container Registry
      uses: docker/login-action@v3
      with:
        registry: rg.fr-par.scw.cloud
        username: nologin
        password: ${{ secrets.SCW_SECRET_KEY }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/agent
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
    
    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
    
    - name: Configure kubectl
      run: |
        mkdir -p $HOME/.kube
        echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > $HOME/.kube/config
    
    - name: Deploy to staging
      run: |
        kubectl apply -k environments/staging/
        kubectl rollout status deployment/island-hopper-agent -n island-hopper-staging
    
    - name: Run integration tests
      run: |
        ./scripts/run-integration-tests.sh staging

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
    
    - name: Configure kubectl
      run: |
        mkdir -p $HOME/.kube
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > $HOME/.kube/config
    
    - name: Deploy to production
      run: |
        kubectl apply -k environments/production/
        kubectl rollout status deployment/island-hopper-agent -n island-hopper
    
    - name: Run smoke tests
      run: |
        ./scripts/run-smoke-tests.sh production
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        text: "🚀 Island Hopper deployed to production"
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## ⚡ Performance Optimization

### Model Optimization

#### Provider Selection Strategies

```python
class OptimizedScalewayModel(ScalewayModel):
    """Performance-optimized model with intelligent routing"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_cache = {}
        self.cost_tracker = CostTracker()
        self.latency_monitor = LatencyMonitor()
    
    async def generate_with_optimization(
        self, 
        prompt: str, 
        optimization_goal: str = "balanced",
        **kwargs
    ):
        """Generate with performance optimization"""
        
        # Analyze prompt characteristics
        prompt_analysis = self._analyze_prompt(prompt)
        
        # Select optimal provider based on goal
        if optimization_goal == "speed":
            provider = self._select_fastest_provider(prompt_analysis)
        elif optimization_goal == "cost":
            provider = self._select_cheapest_provider(prompt_analysis)
        elif optimization_goal == "quality":
            provider = self._select_highest_quality_provider(prompt_analysis)
        else:  # balanced
            provider = self._select_balanced_provider(prompt_analysis)
        
        # Generate with monitoring
        start_time = time.time()
        result = await self._generate_with_provider(provider, prompt, **kwargs)
        latency = time.time() - start_time
        
        # Record performance metrics
        self._record_performance_metrics(provider, prompt_analysis, latency, result)
        
        return result
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt characteristics for optimization"""
        
        analysis = {
            "length": len(prompt),
            "complexity": self._calculate_complexity(prompt),
            "domain": self._detect_domain(prompt),
            "requires_reasoning": self._requires_reasoning(prompt),
            "creative_vs_analytical": self._assess_creative_analytical_balance(prompt)
        }
        
        return analysis
    
    def _select_fastest_provider(self, analysis: Dict) -> str:
        """Select provider optimized for speed"""
        
        # For simple prompts, use fastest model
        if analysis["complexity"] < 0.3:
            return "openrouter"  # Groq models are fastest
        
        # For complex prompts, balance speed and capability
        return "anthropic"  # Claude balances speed and quality
    
    def _select_cheapest_provider(self, analysis: Dict) -> str:
        """Select provider optimized for cost"""
        
        # Use cost-effective models for simple tasks
        if analysis["complexity"] < 0.5:
            return "openrouter"  # Groq is cost-effective
        
        # Use higher-quality models only when necessary
        return "anthropic"  # Better value for complex tasks
    
    def _record_performance_metrics(
        self, 
        provider: str, 
        analysis: Dict, 
        latency: float, 
        result: str
    ):
        """Record performance metrics for optimization"""
        
        metrics = {
            "provider": provider,
            "prompt_length": analysis["length"],
            "complexity": analysis["complexity"],
            "latency": latency,
            "result_length": len(result),
            "timestamp": datetime.now().isoformat()
        }
        
        self.performance_cache[provider] = metrics
        self.latency_monitor.record_latency(provider, latency)
        self.cost_tracker.record_usage(provider, analysis["length"], len(result))
```

#### Caching Strategies

```python
class ModelResponseCache:
    """Intelligent caching for model responses"""
    
    def __init__(self, redis_client, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
        self.similarity_threshold = 0.8
    
    async def get_cached_response(self, prompt: str, **kwargs) -> Optional[str]:
        """Get cached response if available"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, **kwargs)
        
        # Try exact match first
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Try semantic similarity search
        similar_response = await self._find_similar_cached_response(prompt)
        if similar_response:
            return similar_response
        
        return None
    
    async def cache_response(self, prompt: str, response: str, **kwargs):
        """Cache model response"""
        
        cache_key = self._generate_cache_key(prompt, **kwargs)
        cache_data = {
            "prompt": prompt,
            "response": response,
            "parameters": kwargs,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.redis.setex(
            cache_key, 
            self.ttl, 
            json.dumps(cache_data)
        )
        
        # Add to similarity index
        await self._add_to_similarity_index(prompt, response)
    
    def _generate_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate deterministic cache key"""
        
        import hashlib
        
        # Include prompt and relevant parameters
        key_data = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def _find_similar_cached_response(self, prompt: str) -> Optional[str]:
        """Find semantically similar cached responses"""
        
        # Generate embedding for prompt
        prompt_embedding = await self._generate_embedding(prompt)
        
        # Search for similar prompts in cache
        similar_prompts = await self._search_similar_prompts(prompt_embedding)
        
        # Return response if similarity threshold is met
        for cached_prompt, similarity, response in similar_prompts:
            if similarity >= self.similarity_threshold:
                return response
        
        return None
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding for similarity search"""
        
        # Use a fast embedding model or service
        # This is a placeholder - implement with actual embedding service
        import numpy as np
        
        # Simple hash-based embedding for demonstration
        hash_val = hash(text)
        embedding = np.array([hash_val % 1000] * 384)  # 384-dimensional embedding
        return embedding.tolist()
```

### Database Optimization

#### Connection Pool Management

```python
class OptimizedDatabaseTool(DatabaseTool):
    """Database tool with performance optimizations"""
    
    def __init__(self, connection_string: str, **kwargs):
        super().__init__(connection_string, **kwargs)
        
        # Enhanced connection pool configuration
        self.pool_config = {
            "min_size": kwargs.get("min_connections", 5),
            "max_size": kwargs.get("max_connections", 20),
            "max_queries": kwargs.get("max_queries_per_connection", 50000),
            "max_inactive_connection_lifetime": kwargs.get("connection_lifetime", 300),
            "timeout": kwargs.get("connection_timeout", 60),
            "command_timeout": kwargs.get("command_timeout", 30)
        }
        
        # Query optimization
        self.query_cache = QueryCache(ttl=300)
        self.slow_query_threshold = kwargs.get("slow_query_threshold", 1.0)
        
        # Performance monitoring
        self.performance_monitor = DatabasePerformanceMonitor()
    
    async def execute_with_optimization(
        self, 
        query: str, 
        parameters: Optional[List] = None,
        use_cache: bool = True,
        **kwargs
    ):
        """Execute query with performance optimizations"""
        
        # Check query cache first
        if use_cache and self._is_cacheable_query(query):
            cache_key = self._generate_query_cache_key(query, parameters)
            cached_result = await self.query_cache.get(cache_key)
            if cached_result:
                return cached_result
        
        # Analyze and optimize query
        optimized_query = await self._optimize_query(query)
        
        # Execute with monitoring
        start_time = time.time()
        result = await self._execute_query(optimized_query, parameters, **kwargs)
        execution_time = time.time() - start_time
        
        # Record performance metrics
        await self.performance_monitor.record_query_execution(
            query, execution_time, len(result.get("rows", []))
        )
        
        # Cache result if appropriate
        if use_cache and self._is_cacheable_query(query):
            await self.query_cache.set(cache_key, result, ttl=300)
        
        # Alert on slow queries
        if execution_time > self.slow_query_threshold:
            await self._alert_slow_query(query, execution_time)
        
        return result
    
    async def _optimize_query(self, query: str) -> str:
        """Optimize SQL query for better performance"""
        
        # Add query hints for PostgreSQL
        optimizations = [
            # Enable parallel query execution
            "SET max_parallel_workers_per_gather = 4;",
            
            # Optimize memory usage
            "SET work_mem = '256MB';",
            "SET shared_buffers = '1GB';",
            
            # Optimize planning
            "SET random_page_cost = 1.1;",
            "SET effective_io_concurrency = 200;"
        ]
        
        # Prepend optimizations to query
        optimized_query = "\n".join(optimizations) + "\n" + query
        
        return optimized_query
    
    def _is_cacheable_query(self, query: str) -> bool:
        """Check if query is safe to cache"""
        
        # Only cache SELECT queries
        if not query.strip().upper().startswith("SELECT"):
            return False
        
        # Don't cache queries with time-sensitive functions
        time_sensitive_functions = ["NOW()", "CURRENT_TIMESTAMP", "clock_timestamp()"]
        query_upper = query.upper()
        
        for func in time_sensitive_functions:
            if func in query_upper:
                return False
        
        return True
    
    async def _alert_slow_query(self, query: str, execution_time: float):
        """Alert on slow query execution"""
        
        alert_data = {
            "query": query[:200] + "..." if len(query) > 200 else query,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "threshold": self.slow_query_threshold
        }
        
        # Send to monitoring system
        await self._send_alert("slow_query", alert_data)
```

#### Batch Processing Optimization

```python
class BatchProcessor:
    """Optimized batch processing for database operations"""
    
    def __init__(self, db_tool: DatabaseTool, batch_size: int = 1000):
        self.db_tool = db_tool
        self.batch_size = batch_size
        self.progress_callback = None
    
    async def batch_insert(
        self, 
        table_name: str, 
        data: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Optimized batch insert with progress tracking"""
        
        self.progress_callback = progress_callback
        total_records = len(data)
        processed_records = 0
        errors = []
        
        # Generate insert statement
        columns = list(data[0].keys())
        placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
        insert_query = f"""
            INSERT INTO {table_name} ({", ".join(columns)})
            VALUES ({placeholders})
        """
        
        # Process in batches
        start_time = time.time()
        
        for i in range(0, total_records, self.batch_size):
            batch = data[i:i + self.batch_size]
            
            try:
                # Prepare batch parameters
                batch_params = []
                for row in batch:
                    batch_params.append([row[col] for col in columns])
                
                # Execute batch insert
                await self.db_tool.execute_batch(insert_query, batch_params)
                
                processed_records += len(batch)
                
                # Report progress
                if self.progress_callback:
                    progress = processed_records / total_records
                    await self.progress_callback(progress, processed_records, total_records)
                
            except Exception as e:
                errors.append({
                    "batch_start": i,
                    "batch_end": i + len(batch),
                    "error": str(e)
                })
        
        execution_time = time.time() - start_time
        
        return {
            "total_records": total_records,
            "processed_records": processed_records,
            "failed_records": total_records - processed_records,
            "errors": errors,
            "execution_time": execution_time,
            "records_per_second": processed_records / execution_time if execution_time > 0 else 0
        }
    
    async def batch_update(
        self, 
        table_name: str, 
        updates: List[Dict[str, Any]],
        key_column: str = "id"
    ) -> Dict[str, Any]:
        """Optimized batch update operations"""
        
        # Build CASE statements for batch update
        update_statements = []
        parameters = []
        
        for i, update in enumerate(updates):
            key_value = update[key_column]
            update_data = {k: v for k, v in update.items() if k != key_column}
            
            for column, value in update_data.items():
                update_statements.append(
                    f"WHEN ${len(parameters)+1} THEN ${len(parameters)+2}"
                )
                parameters.extend([key_value, value])
        
        # Construct batch update query
        set_clause = f"{column} = CASE {key_column} " + " ".join(update_statements) + f" ELSE {column} END"
        
        update_query = f"""
            UPDATE {table_name}
            SET {set_clause}
            WHERE {key_column} IN ANY($1)
        """
        
        # Extract key values for WHERE clause
        key_values = [update[key_column] for update in updates]
        
        # Execute batch update
        start_time = time.time()
        result = await self.db_tool.execute_query(update_query, [key_values] + parameters)
        execution_time = time.time() - start_time
        
        return {
            "updated_records": result["row_count"],
            "execution_time": execution_time,
            "records_per_second": len(updates) / execution_time if execution_time > 0 else 0
        }
```

### Storage Optimization

#### Object Storage Performance

```python
class OptimizedObjectStorageTool(ObjectStorageTool):
    """Object storage tool with performance optimizations"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Performance configurations
        self.multipart_threshold = kwargs.get("multipart_threshold", 64 * 1024 * 1024)  # 64MB
        self.max_concurrent_uploads = kwargs.get("max_concurrent_uploads", 10)
        self.compression_enabled = kwargs.get("compression_enabled", True)
        self.encryption_enabled = kwargs.get("encryption_enabled", True)
        
        # Connection pooling
        self.max_pool_connections = kwargs.get("max_pool_connections", 50)
        
        # Caching
        self.download_cache = LRUCache(maxsize=100)
        self.metadata_cache = LRUCache(maxsize=1000)
    
    async def upload_optimized(
        self, 
        bucket: str, 
        file_path: str, 
        content: Union[str, bytes],
        **kwargs
    ):
        """Optimized file upload with compression and multipart"""
        
        # Prepare content
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
        
        original_size = len(content_bytes)
        
        # Apply compression if enabled
        if self.compression_enabled and original_size > 1024:  # Only compress files > 1KB
            compressed_content = await self._compress_content(content_bytes)
            if len(compressed_content) < original_size * 0.8:  # Only use compression if it helps
                content_bytes = compressed_content
                kwargs['ContentEncoding'] = 'gzip'
        
        # Use multipart upload for large files
        if len(content_bytes) > self.multipart_threshold:
            result = await self._multipart_upload(bucket, file_path, content_bytes, **kwargs)
        else:
            result = await self._single_upload(bucket, file_path, content_bytes, **kwargs)
        
        # Add performance metadata
        result['compression_ratio'] = len(content_bytes) / original_size
        result['upload_method'] = 'multipart' if len(content_bytes) > self.multipart_threshold else 'single'
        
        return result
    
    async def _multipart_upload(
        self, 
        bucket: str, 
        file_path: str, 
        content: bytes,
        **kwargs
    ):
        """Multipart upload for large files"""
        
        # Calculate optimal part size (5MB to 5GB)
        part_size = max(5 * 1024 * 1024, min(len(content) // 1000, 5 * 1024 * 1024 * 1024))
        num_parts = (len(content) + part_size - 1) // part_size
        
        # Initiate multipart upload
        upload_id = self.s3_client.create_multipart_upload(
            Bucket=bucket,
            Key=file_path,
            **kwargs
        )['UploadId']
        
        # Upload parts concurrently
        parts = []
        semaphore = asyncio.Semaphore(self.max_concurrent_uploads)
        
        async def upload_part(part_number, start, end):
            async with semaphore:
                part_data = content[start:end]
                
                response = self.s3_client.upload_part(
                    Bucket=bucket,
                    Key=file_path,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=part_data
                )
                
                return {
                    'PartNumber': part_number,
                    'ETag': response['ETag']
                }
        
        # Create upload tasks
        tasks = []
        for i in range(num_parts):
            start = i * part_size
            end = min((i + 1) * part_size, len(content))
            task = upload_part(i + 1, start, end)
            tasks.append(task)
        
        # Wait for all parts to upload
        parts = await asyncio.gather(*tasks)
        
        # Complete multipart upload
        result = self.s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=file_path,
            UploadId=upload_id,
            MultipartUpload={'Parts': sorted(parts, key=lambda x: x['PartNumber'])}
        )
        
        return {
            "bucket": bucket,
            "file_path": file_path,
            "upload_id": upload_id,
            "parts": num_parts,
            "part_size": part_size,
            "etag": result['ETag'],
            "location": result['Location']
        }
    
    async def _compress_content(self, content: bytes) -> bytes:
        """Compress content using gzip"""
        
        import gzip
        import io
        
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb') as gz_file:
            gz_file.write(content)
        
        return buffer.getvalue()
    
    async def batch_upload(
        self, 
        bucket: str, 
        files: Dict[str, Union[str, bytes]],
        **kwargs
    ):
        """Upload multiple files concurrently"""
        
        semaphore = asyncio.Semaphore(self.max_concurrent_uploads)
        
        async def upload_single(file_path, content):
            async with semaphore:
                return await self.upload_optimized(bucket, file_path, content, **kwargs)
        
        # Create upload tasks
        tasks = [
            upload_single(file_path, content) 
            for file_path, content in files.items()
        ]
        
        # Wait for all uploads to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_uploads = []
        failed_uploads = []
        
        for i, (file_path, _) in enumerate(files.items()):
            result = results[i]
            if isinstance(result, Exception):
                failed_uploads.append({"file_path": file_path, "error": str(result)})
            else:
                successful_uploads.append(result)
        
        return {
            "total_files": len(files),
            "successful_uploads": len(successful_uploads),
            "failed_uploads": len(failed_uploads),
            "results": successful_uploads,
            "errors": failed_uploads
        }
```

---

This comprehensive RAG knowledge base provides detailed documentation for every aspect of the Island Hopper ecosystem. It's structured to be easily searchable and contains practical examples, configuration details, and best practices for developers working with the system.

The documentation continues with additional sections covering security, monitoring, troubleshooting, and advanced configurations. Would you like me to continue with the remaining sections?
