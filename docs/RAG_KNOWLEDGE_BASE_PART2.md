# 🧠 Island Hopper RAG Knowledge Base - Part 2

## 🔒 Security & Compliance

### Security Architecture

Island Hopper implements a defense-in-depth security model with multiple layers of protection:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Security                     │
│  • Input Validation & Sanitization                          │
│  • Authentication & Authorization                           │
│  • Rate Limiting & Throttling                               │
│  • Audit Logging                                            │
├─────────────────────────────────────────────────────────────┤
│                    Network Security                         │
│  • TLS/SSL Encryption                                        │
│  • Network Policies (Kubernetes)                            │
│  • VPC Isolation                                             │
│  • DDoS Protection                                          │
├─────────────────────────────────────────────────────────────┤
│                    Data Security                            │
│  • Encryption at Rest (AES-256)                             │
│  • Encryption in Transit (TLS 1.3)                         │
│  • Key Management (Scaleway KMS)                            │
│  • Data Masking & Anonymization                             │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure Security                  │
│  • Container Security                                        │
│  • Pod Security Standards                                   │
│  • RBAC & IAM Policies                                      │
│  • Security Scanning & Vulnerability Management             │
└─────────────────────────────────────────────────────────────┘
```

### Authentication & Authorization

#### Scaleway IAM Integration

```python
class ScalewaySecurityManager:
    """Manages authentication and authorization with Scaleway IAM"""
    
    def __init__(self, project_id: str, access_key: str, secret_key: str):
        self.project_id = project_id
        self.access_key = access_key
        self.secret_key = secret_key
        self.iam_client = self._create_iam_client()
        self.session_manager = SecureSessionManager()
    
    def _create_iam_client(self):
        """Create Scaleway IAM client"""
        import scaleway
        
        client = scaleway.Client(
            access_key=self.access_key,
            secret_key=self.secret_key,
            default_project_id=self.project_id
        )
        
        return client.iam.v1
    
    async def authenticate_agent(self, agent_id: str, credentials: Dict) -> AuthResult:
        """Authenticate agent using Scaleway IAM"""
        
        try:
            # Validate credentials with Scaleway IAM
            user_info = await self.iam_client.get_user(credentials['user_id'])
            
            # Check agent permissions
            permissions = await self._check_agent_permissions(agent_id, user_info)
            
            if permissions['allowed']:
                # Create secure session
                session_token = await self.session_manager.create_session(
                    agent_id=agent_id,
                    user_id=user_info['id'],
                    permissions=permissions['scopes'],
                    ttl=3600  # 1 hour
                )
                
                return AuthResult(
                    success=True,
                    session_token=session_token,
                    permissions=permissions['scopes'],
                    expires_at=datetime.now() + timedelta(hours=1)
                )
            else:
                return AuthResult(
                    success=False,
                    error="Insufficient permissions",
                    required_permissions=permissions['required']
                )
                
        except Exception as e:
            return AuthResult(success=False, error=str(e))
    
    async def authorize_operation(
        self, 
        session_token: str, 
        operation: str, 
        resource: str
    ) -> AuthzResult:
        """Authorize operation based on session permissions"""
        
        # Validate session token
        session = await self.session_manager.validate_session(session_token)
        if not session:
            return AuthzResult(success=False, error="Invalid or expired session")
        
        # Check operation permissions
        required_permission = f"{operation}:{resource}"
        
        if self._has_permission(session['permissions'], required_permission):
            return AuthzResult(
                success=True,
                agent_id=session['agent_id'],
                operation=operation,
                resource=resource
            )
        else:
            return AuthzResult(
                success=False,
                error="Operation not permitted",
                required_permission=required_permission
            )
    
    def _has_permission(self, permissions: List[str], required: str) -> bool:
        """Check if required permission is granted"""
        
        # Direct permission match
        if required in permissions:
            return True
        
        # Wildcard permission match
        for permission in permissions:
            if permission.endswith('*'):
                prefix = permission[:-1]
                if required.startswith(prefix):
                    return True
        
        return False

@dataclass
class AuthResult:
    success: bool
    session_token: Optional[str] = None
    permissions: Optional[List[str]] = None
    expires_at: Optional[datetime] = None
    error: Optional[str] = None
    required_permissions: Optional[List[str]] = None

@dataclass
class AuthzResult:
    success: bool
    agent_id: Optional[str] = None
    operation: Optional[str] = None
    resource: Optional[str] = None
    error: Optional[str] = None
    required_permission: Optional[str] = None
```

#### API Security

```python
class SecureAPIServer:
    """Secure API server with authentication and rate limiting"""
    
    def __init__(self, security_manager: ScalewaySecurityManager):
        self.security_manager = security_manager
        self.rate_limiter = RateLimiter()
        self.audit_logger = AuditLogger()
    
    async def handle_request(self, request: HTTPRequest) -> HTTPResponse:
        """Handle incoming API request with security checks"""
        
        client_ip = request.remote_addr
        endpoint = request.path
        method = request.method
        
        # Log request start
        request_id = str(uuid.uuid4())
        await self.audit_logger.log_request_start(
            request_id=request_id,
            client_ip=client_ip,
            endpoint=endpoint,
            method=method
        )
        
        try:
            # Rate limiting check
            if not await self.rate_limiter.check_rate_limit(client_ip):
                await self.audit_logger.log_security_event(
                    event_type="rate_limit_exceeded",
                    client_ip=client_ip,
                    endpoint=endpoint
                )
                return HTTPResponse(429, {"error": "Rate limit exceeded"})
            
            # Authentication check
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                await self.audit_logger.log_security_event(
                    event_type="missing_auth",
                    client_ip=client_ip,
                    endpoint=endpoint
                )
                return HTTPResponse(401, {"error": "Authentication required"})
            
            # Validate session token
            session_token = auth_header.replace("Bearer ", "")
            auth_result = await self.security_manager.authenticate_token(session_token)
            
            if not auth_result.success:
                await self.audit_logger.log_security_event(
                    event_type="auth_failed",
                    client_ip=client_ip,
                    endpoint=endpoint,
                    error=auth_result.error
                )
                return HTTPResponse(401, {"error": auth_result.error})
            
            # Authorization check
            operation = f"{method}:{endpoint}"
            authz_result = await self.security_manager.authorize_operation(
                session_token=session_token,
                operation=operation,
                resource=endpoint
            )
            
            if not authz_result.success:
                await self.audit_logger.log_security_event(
                    event_type="authz_failed",
                    client_ip=client_ip,
                    endpoint=endpoint,
                    operation=operation,
                    error=authz_result.error
                )
                return HTTPResponse(403, {"error": authz_result.error})
            
            # Process request
            response = await self._process_authenticated_request(
                request, 
                auth_result.agent_id
            )
            
            # Log successful request
            await self.audit_logger.log_request_success(
                request_id=request_id,
                agent_id=auth_result.agent_id,
                response_status=response.status_code
            )
            
            return response
            
        except Exception as e:
            # Log error
            await self.audit_logger.log_request_error(
                request_id=request_id,
                error=str(e)
            )
            
            return HTTPResponse(500, {"error": "Internal server error"})

class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self):
        self.redis_client = redis.Redis()
        self.limits = {
            "default": {"requests": 100, "window": 60},  # 100 requests per minute
            "authenticated": {"requests": 1000, "window": 60},  # 1000 requests per minute
            "premium": {"requests": 10000, "window": 60}  # 10000 requests per minute
        }
    
    async def check_rate_limit(self, client_ip: str, user_tier: str = "default") -> bool:
        """Check if client has exceeded rate limit"""
        
        limit_config = self.limits.get(user_tier, self.limits["default"])
        key = f"rate_limit:{client_ip}:{user_tier}"
        
        # Use sliding window counter
        current_time = int(time.time())
        window_start = current_time - limit_config["window"]
        
        # Remove old entries
        await self.redis_client.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_requests = await self.redis_client.zcard(key)
        
        if current_requests >= limit_config["requests"]:
            return False
        
        # Add current request
        await self.redis_client.zadd(key, {str(current_time): current_time})
        await self.redis_client.expire(key, limit_config["window"])
        
        return True
```

### Data Encryption

#### Encryption at Rest

```python
class EncryptionManager:
    """Manages encryption for data at rest and in transit"""
    
    def __init__(self, kms_key_id: str):
        self.kms_key_id = kms_key_id
        self.kms_client = self._create_kms_client()
        self.encryption_cache = {}
    
    def _create_kms_client(self):
        """Create Scaleway KMS client"""
        import scaleway
        
        client = scaleway.Client(
            access_key=os.getenv("SCALEWAY_ACCESS_KEY"),
            secret_key=os.getenv("SCALEWAY_SECRET_KEY")
        )
        
        return client.kms.v1
    
    async def encrypt_sensitive_data(self, data: str, context: Optional[Dict] = None) -> EncryptedData:
        """Encrypt sensitive data using KMS"""
        
        try:
            # Generate data key
            data_key_response = await self.kms_client.create_encrypted_data_key(
                key_id=self.kms_key_id,
                context=context or {}
            )
            
            encrypted_data_key = data_key_response.encrypted_data_key
            plaintext_data_key = data_key_response.plaintext_data_key
            
            # Encrypt data with data key
            encrypted_data = self._encrypt_with_data_key(data, plaintext_data_key)
            
            # Securely delete plaintext data key
            del plaintext_data_key
            
            return EncryptedData(
                encrypted_data=encrypted_data,
                encrypted_key=encrypted_data_key,
                key_id=self.kms_key_id,
                encryption_context=context or {},
                algorithm="AES-256-GCM"
            )
            
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt data: {e}")
    
    async def decrypt_sensitive_data(self, encrypted_data: EncryptedData) -> str:
        """Decrypt sensitive data using KMS"""
        
        try:
            # Decrypt data key
            data_key_response = await self.kms_client.decrypt_data_key(
                encrypted_data_key=encrypted_data.encrypted_key,
                key_id=encrypted_data.key_id,
                context=encrypted_data.encryption_context
            )
            
            plaintext_data_key = data_key_response.plaintext_data_key
            
            # Decrypt data with data key
            decrypted_data = self._decrypt_with_data_key(
                encrypted_data.encrypted_data,
                plaintext_data_key
            )
            
            # Securely delete plaintext data key
            del plaintext_data_key
            
            return decrypted_data
            
        except Exception as e:
            raise DecryptionError(f"Failed to decrypt data: {e}")
    
    def _encrypt_with_data_key(self, data: str, data_key: bytes) -> str:
        """Encrypt data using AES-256-GCM"""
        
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        aesgcm = AESGCM(data_key)
        nonce = os.urandom(12)  # 96-bit nonce
        
        encrypted_data = aesgcm.encrypt(
            nonce,
            data.encode('utf-8'),
            None  # No additional authenticated data
        )
        
        # Combine nonce and encrypted data
        combined = nonce + encrypted_data
        
        # Return base64 encoded result
        import base64
        return base64.b64encode(combined).decode('utf-8')
    
    def _decrypt_with_data_key(self, encrypted_data: str, data_key: bytes) -> str:
        """Decrypt data using AES-256-GCM"""
        
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        import base64
        
        # Decode base64
        combined = base64.b64decode(encrypted_data)
        
        # Extract nonce and encrypted data
        nonce = combined[:12]
        ciphertext = combined[12:]
        
        # Decrypt
        aesgcm = AESGCM(data_key)
        decrypted_data = aesgcm.decrypt(nonce, ciphertext, None)
        
        return decrypted_data.decode('utf-8')

@dataclass
class EncryptedData:
    encrypted_data: str
    encrypted_key: str
    key_id: str
    encryption_context: Dict[str, str]
    algorithm: str
```

#### Secure Session Management

```python
class SecureSessionManager:
    """Manages secure sessions with encryption and validation"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis()
        self.encryption_manager = EncryptionManager(os.getenv("KMS_KEY_ID"))
        self.session_ttl = 3600  # 1 hour
        self.max_sessions_per_user = 10
    
    async def create_session(
        self, 
        agent_id: str, 
        user_id: str, 
        permissions: List[str],
        metadata: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> str:
        """Create secure session with encryption"""
        
        session_id = str(uuid.uuid4())
        session_ttl = ttl or self.session_ttl
        
        # Check session limit per user
        user_sessions = await self._get_user_sessions(user_id)
        if len(user_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            await self._remove_oldest_session(user_id)
        
        # Create session data
        session_data = {
            "session_id": session_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "permissions": permissions,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=session_ttl)).isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        
        # Encrypt session data
        encrypted_session = await self.encryption_manager.encrypt_sensitive_data(
            json.dumps(session_data),
            context={"session_id": session_id, "agent_id": agent_id}
        )
        
        # Store in Redis
        session_key = f"session:{session_id}"
        user_sessions_key = f"user_sessions:{user_id}"
        
        pipe = self.redis_client.pipeline()
        pipe.setex(session_key, session_ttl, encrypted_session.encrypted_data)
        pipe.sadd(user_sessions_key, session_id)
        pipe.expire(user_sessions_key, session_ttl)
        await pipe.execute()
        
        # Generate session token (JWT)
        session_token = await self._generate_session_token(session_id, session_data)
        
        return session_token
    
    async def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate and decrypt session token"""
        
        try:
            # Decode JWT token
            payload = await self._decode_session_token(session_token)
            session_id = payload["session_id"]
            
            # Get encrypted session from Redis
            session_key = f"session:{session_id}"
            encrypted_session_data = await self.redis_client.get(session_key)
            
            if not encrypted_session_data:
                return None  # Session not found or expired
            
            # Decrypt session data
            encrypted_data = EncryptedData(
                encrypted_data=encrypted_session_data.decode('utf-8'),
                encrypted_key=payload["encrypted_key"],
                key_id=payload["key_id"],
                encryption_context=payload["context"],
                algorithm=payload["algorithm"]
            )
            
            session_json = await self.encryption_manager.decrypt_sensitive_data(encrypted_data)
            session_data = json.loads(session_json)
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now() > expires_at:
                await self._remove_session(session_id, session_data["user_id"])
                return None
            
            # Update last accessed time
            await self._update_last_accessed(session_id)
            
            return session_data
            
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return None
    
    async def revoke_session(self, session_token: str) -> bool:
        """Revoke a specific session"""
        
        session_data = await self.validate_session(session_token)
        if not session_data:
            return False
        
        await self._remove_session(session_data["session_id"], session_data["user_id"])
        return True
    
    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user"""
        
        user_sessions_key = f"user_sessions:{user_id}"
        session_ids = await self.redis_client.smembers(user_sessions_key)
        
        revoked_count = 0
        for session_id in session_ids:
            await self._remove_session(session_id.decode('utf-8'), user_id)
            revoked_count += 1
        
        return revoked_count
    
    async def _generate_session_token(self, session_id: str, session_data: Dict) -> str:
        """Generate JWT session token"""
        
        import jwt
        
        payload = {
            "session_id": session_id,
            "agent_id": session_data["agent_id"],
            "user_id": session_data["user_id"],
            "iat": int(time.time()),
            "exp": int(time.time()) + self.session_ttl
        }
        
        # Include encryption metadata for session validation
        payload.update({
            "encrypted_key": "placeholder",  # Will be replaced with actual encrypted key
            "key_id": self.encryption_manager.kms_key_id,
            "context": {"session_id": session_id},
            "algorithm": "AES-256-GCM"
        })
        
        token = jwt.encode(payload, os.getenv("JWT_SECRET_KEY"), algorithm="HS256")
        return token
    
    async def _decode_session_token(self, token: str) -> Dict:
        """Decode JWT session token"""
        
        import jwt
        
        payload = jwt.decode(
            token, 
            os.getenv("JWT_SECRET_KEY"), 
            algorithms=["HS256"]
        )
        
        return payload
```

### Compliance & Auditing

#### Audit Logging System

```python
class AuditLogger:
    """Comprehensive audit logging for compliance"""
    
    def __init__(self, storage_backend: str = "database"):
        self.storage_backend = storage_backend
        self.log_buffer = []
        self.buffer_size = 100
        self.flush_interval = 60  # seconds
        
        # Initialize storage backend
        if storage_backend == "database":
            self.db_client = self._create_database_client()
        elif storage_backend == "object_storage":
            self.storage_client = self._create_storage_client()
        
        # Start background flush task
        asyncio.create_task(self._periodic_flush())
    
    async def log_security_event(
        self,
        event_type: str,
        client_ip: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        operation: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log security-related events"""
        
        audit_event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "security",
            "security_event_type": event_type,
            "client_ip": client_ip,
            "user_id": user_id,
            "agent_id": agent_id,
            "endpoint": endpoint,
            "operation": operation,
            "error": error,
            "metadata": metadata or {},
            "severity": self._get_severity_level(event_type)
        }
        
        await self._add_to_buffer(audit_event)
    
    async def log_data_access(
        self,
        user_id: str,
        agent_id: str,
        resource_type: str,
        resource_id: str,
        operation: str,
        access_granted: bool,
        metadata: Optional[Dict] = None
    ):
        """Log data access events for GDPR compliance"""
        
        audit_event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "data_access",
            "user_id": user_id,
            "agent_id": agent_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "operation": operation,
            "access_granted": access_granted,
            "metadata": metadata or {},
            "compliance_frameworks": ["GDPR", "SOC2"]
        }
        
        await self._add_to_buffer(audit_event)
    
    async def log_configuration_change(
        self,
        user_id: str,
        component: str,
        configuration: Dict,
        old_values: Optional[Dict] = None,
        change_reason: Optional[str] = None
    ):
        """Log configuration changes for SOX compliance"""
        
        audit_event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "configuration_change",
            "user_id": user_id,
            "component": component,
            "configuration": configuration,
            "old_values": old_values,
            "change_reason": change_reason,
            "compliance_frameworks": ["SOX", "ISO27001"]
        }
        
        await self._add_to_buffer(audit_event)
    
    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        frameworks: List[str] = None
    ) -> Dict:
        """Generate compliance report for specified period"""
        
        frameworks = frameworks or ["GDPR", "SOC2", "SOX", "ISO27001"]
        
        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "generated_at": datetime.now().isoformat(),
            "frameworks": frameworks,
            "summary": {},
            "detailed_events": {}
        }
        
        for framework in frameworks:
            framework_events = await self._get_events_by_framework(
                framework, start_date, end_date
            )
            
            report["summary"][framework] = {
                "total_events": len(framework_events),
                "security_events": len([e for e in framework_events if e["event_type"] == "security"]),
                "data_access_events": len([e for e in framework_events if e["event_type"] == "data_access"]),
                "configuration_changes": len([e for e in framework_events if e["event_type"] == "configuration_change"])
            }
            
            report["detailed_events"][framework] = framework_events
        
        return report
    
    def _get_severity_level(self, event_type: str) -> str:
        """Determine severity level for security events"""
        
        high_severity_events = [
            "auth_failed", "authz_failed", "rate_limit_exceeded",
            "sql_injection_attempt", "xss_attempt", "data_breach"
        ]
        
        medium_severity_events = [
            "missing_auth", "invalid_token", "suspicious_activity"
        ]
        
        if event_type in high_severity_events:
            return "HIGH"
        elif event_type in medium_severity_events:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def _add_to_buffer(self, event: Dict):
        """Add event to buffer for batch processing"""
        
        self.log_buffer.append(event)
        
        if len(self.log_buffer) >= self.buffer_size:
            await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Flush audit log buffer to storage"""
        
        if not self.log_buffer:
            return
        
        if self.storage_backend == "database":
            await self._flush_to_database()
        elif self.storage_backend == "object_storage":
            await self._flush_to_storage()
        
        self.log_buffer.clear()
    
    async def _flush_to_database(self):
        """Flush audit logs to database"""
        
        try:
            # Batch insert audit events
            insert_query = """
                INSERT INTO audit_logs (
                    timestamp, event_type, user_id, agent_id, 
                    client_ip, endpoint, operation, metadata, severity
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            batch_params = []
            for event in self.log_buffer:
                batch_params.append((
                    event["timestamp"],
                    event["event_type"],
                    event.get("user_id"),
                    event.get("agent_id"),
                    event.get("client_ip"),
                    event.get("endpoint"),
                    event.get("operation"),
                    json.dumps(event["metadata"]),
                    event.get("severity", "LOW")
                ))
            
            await self.db_client.execute_batch(insert_query, batch_params)
            
        except Exception as e:
            logger.error(f"Failed to flush audit logs to database: {e}")
    
    async def _flush_to_storage(self):
        """Flush audit logs to object storage"""
        
        try:
            # Create daily log file
            date_str = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_logs/{date_str}/audit_{timestamp}.jsonl"
            
            # Convert events to JSONL format
            log_content = "\n".join(json.dumps(event) for event in self.log_buffer)
            
            # Upload to object storage
            await self.storage_client.execute(
                operation="upload_file",
                file_path=filename,
                content=log_content,
                Metadata={
                    "event_count": str(len(self.log_buffer)),
                    "generated_at": datetime.now().isoformat(),
                    "compliance": "true"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to flush audit logs to storage: {e}")
```

---

## 📊 Monitoring & Observability

### Comprehensive Monitoring Stack

Island Hopper provides enterprise-grade monitoring with multiple layers of observability:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Monitoring                   │
│  • Custom Metrics & Instrumentation                         │
│  • Distributed Tracing                                      │
│  • Error Tracking & Alerting                               │
│  • Performance Profiling                                   │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure Monitoring                │
│  • Resource Utilization (CPU, Memory, Disk)                │
│  • Network Performance                                      │
│  • Container Health                                         │
│  • Service Availability                                     │
├─────────────────────────────────────────────────────────────┤
│                    Business Metrics                         │
│  • Agent Performance                                        │
│  • Cost Tracking                                            │
│  • Usage Analytics                                         │
│  • SLA Monitoring                                           │
├─────────────────────────────────────────────────────────────┤
│                    Scaleway Cockpit Integration             │
│  • Native Metrics Collection                                │
│  • Log Aggregation                                          │
│  • Alert Management                                         │
│  • Dashboard Visualization                                 │
└─────────────────────────────────────────────────────────────┘
```

### OpenTelemetry Integration

```python
class IslandHopperTelemetry:
    """Comprehensive telemetry with OpenTelemetry and Scaleway Cockpit"""
    
    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        enable_cockpit: bool = True,
        cockpit_endpoint: Optional[str] = None
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.enable_cockpit = enable_cockpit
        self.cockpit_endpoint = cockpit_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        
        # Initialize OpenTelemetry
        self._initialize_opentelemetry()
        
        # Initialize metrics
        self.metrics = self._initialize_metrics()
        
        # Initialize tracing
        self.tracer = trace.get_tracer(__name__)
        
        # Initialize logging
        self.logger = self._initialize_logger()
    
    def _initialize_opentelemetry(self):
        """Initialize OpenTelemetry with appropriate exporters"""
        
        from opentelemetry import trace, metrics
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        
        # Configure trace provider
        trace_provider = TracerProvider()
        if self.enable_cockpit:
            span_exporter = OTLPSpanExporter(endpoint=self.cockpit_endpoint)
            span_processor = BatchSpanProcessor(span_exporter)
            trace_provider.add_span_processor(span_processor)
        
        trace.set_tracer_provider(trace_provider)
        
        # Configure metrics provider
        metric_reader = PeriodicExportingMetricReader(
            exporter=OTLPMetricExporter(endpoint=self.cockpit_endpoint),
            export_interval_millis=30000  # 30 seconds
        )
        
        metrics_provider = MeterProvider(metric_readers=[metric_reader])
        metrics.set_meter_provider(metrics_provider)
    
    def _initialize_metrics(self) -> Dict:
        """Initialize custom metrics"""
        
        from opentelemetry import metrics
        
        meter = metrics.get_meter(self.service_name)
        
        return {
            # Request metrics
            "request_counter": meter.create_counter(
                "island_hopper_requests_total",
                description="Total number of requests"
            ),
            "request_duration": meter.create_histogram(
                "island_hopper_request_duration_seconds",
                description="Request duration in seconds"
            ),
            "request_size": meter.create_histogram(
                "island_hopper_request_size_bytes",
                description="Request size in bytes"
            ),
            "response_size": meter.create_histogram(
                "island_hopper_response_size_bytes",
                description="Response size in bytes"
            ),
            
            # Agent metrics
            "agent_operations": meter.create_counter(
                "island_hopper_agent_operations_total",
                description="Total number of agent operations"
            ),
            "agent_errors": meter.create_counter(
                "island_hopper_agent_errors_total",
                description="Total number of agent errors"
            ),
            "active_sessions": meter.create_up_down_counter(
                "island_hopper_active_sessions",
                description="Number of active sessions"
            ),
            
            # Model metrics
            "model_requests": meter.create_counter(
                "island_hopper_model_requests_total",
                description="Total number of model requests"
            ),
            "model_tokens": meter.create_counter(
                "island_hopper_model_tokens_total",
                description="Total number of model tokens"
            ),
            "model_costs": meter.create_counter(
                "island_hopper_model_costs_usd",
                description="Total model costs in USD"
            ),
            
            # Tool metrics
            "tool_executions": meter.create_counter(
                "island_hopper_tool_executions_total",
                description="Total number of tool executions"
            ),
            "tool_duration": meter.create_histogram(
                "island_hopper_tool_duration_seconds",
                description="Tool execution duration in seconds"
            ),
            
            # System metrics
            "resource_usage": meter.create_observable_gauge(
                "island_hopper_resource_usage",
                description="Resource usage metrics",
                callbacks=[self._get_resource_metrics]
            )
        }
    
    async def track_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: int = 0,
        response_size: int = 0,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """Track HTTP request metrics"""
        
        attributes = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code),
            "status_class": self._get_status_class(status_code)
        }
        
        if user_id:
            attributes["user_id"] = user_id
        if agent_id:
            attributes["agent_id"] = agent_id
        
        # Record metrics
        self.metrics["request_counter"].add(1, attributes)
        self.metrics["request_duration"].record(duration, attributes)
        
        if request_size > 0:
            self.metrics["request_size"].record(request_size, attributes)
        if response_size > 0:
            self.metrics["response_size"].record(response_size, attributes)
    
    async def track_agent_operation(
        self,
        agent_id: str,
        operation: str,
        success: bool,
        duration: float,
        error: Optional[str] = None
    ):
        """Track agent operation metrics"""
        
        attributes = {
            "agent_id": agent_id,
            "operation": operation,
            "success": str(success)
        }
        
        if error:
            attributes["error_type"] = self._classify_error(error)
        
        self.metrics["agent_operations"].add(1, attributes)
        self.metrics["agent_operations_duration"].record(duration, attributes)
        
        if not success:
            self.metrics["agent_errors"].add(1, attributes)
    
    async def track_model_usage(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        cost: float,
        duration: float,
        success: bool = True
    ):
        """Track AI model usage metrics"""
        
        attributes = {
            "provider": provider,
            "model": model,
            "success": str(success)
        }
        
        self.metrics["model_requests"].add(1, attributes)
        self.metrics["model_tokens"].add(tokens_used, attributes)
        self.metrics["model_costs"].add(cost, attributes)
        self.metrics["model_duration"].record(duration, attributes)
    
    def create_span(self, name: str, **attributes) -> trace.Span:
        """Create a new trace span"""
        
        span = self.tracer.start_span(name)
        
        # Set standard attributes
        span.set_attribute("service.name", self.service_name)
        span.set_attribute("service.version", self.service_version)
        
        # Set custom attributes
        for key, value in attributes.items():
            span.set_attribute(key, value)
        
        return span
    
    def _get_resource_metrics(self, options):
        """Callback for observable resource metrics"""
        
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": cpu_percent,
            "memory_percent": memory.percent,
            "memory_bytes": memory.used,
            "disk_percent": disk.percent,
            "disk_bytes": disk.used
        }
    
    def _get_status_class(self, status_code: int) -> str:
        """Get HTTP status class from status code"""
        
        if 200 <= status_code < 300:
            return "2xx"
        elif 300 <= status_code < 400:
            return "3xx"
        elif 400 <= status_code < 500:
            return "4xx"
        elif 500 <= status_code < 600:
            return "5xx"
        else:
            return "unknown"
    
    def _classify_error(self, error: str) -> str:
        """Classify error type for metrics"""
        
        error_lower = error.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "rate limit" in error_lower:
            return "rate_limit"
        elif "authentication" in error_lower or "auth" in error_lower:
            return "authentication"
        elif "permission" in error_lower or "authorization" in error_lower:
            return "authorization"
        elif "network" in error_lower or "connection" in error_lower:
            return "network"
        else:
            return "unknown"
```

### Custom Metrics & Dashboards

```python
class BusinessMetricsCollector:
    """Collects business-specific metrics for Island Hopper"""
    
    def __init__(self, telemetry: IslandHopperTelemetry):
        self.telemetry = telemetry
        self.metrics = self._initialize_business_metrics()
    
    def _initialize_business_metrics(self) -> Dict:
        """Initialize business-specific metrics"""
        
        from opentelemetry import metrics
        
        meter = metrics.get_meter("island_hopper_business")
        
        return {
            # User engagement metrics
            "active_users": meter.create_up_down_counter(
                "island_hopper_active_users",
                description="Number of active users"
            ),
            "user_sessions": meter.create_counter(
                "island_hopper_user_sessions_total",
                description="Total number of user sessions"
            ),
            "session_duration": meter.create_histogram(
                "island_hopper_session_duration_seconds",
                description="User session duration in seconds"
            ),
            
            # Agent performance metrics
            "agent_success_rate": meter.create_gauge(
                "island_hopper_agent_success_rate",
                description="Agent operation success rate"
            ),
            "agent_utilization": meter.create_gauge(
                "island_hopper_agent_utilization",
                description="Agent resource utilization"
            ),
            "agent_response_time": meter.create_histogram(
                "island_hopper_agent_response_time_seconds",
                description="Agent response time in seconds"
            ),
            
            # Cost metrics
            "daily_costs": meter.create_counter(
                "island_hopper_daily_costs_usd",
                description="Daily costs in USD"
            ),
            "cost_per_user": meter.create_gauge(
                "island_hopper_cost_per_user_usd",
                description="Cost per user in USD"
            ),
            "roi_metrics": meter.create_gauge(
                "island_hopper_roi_ratio",
                description="Return on investment ratio"
            ),
            
            # Quality metrics
            "user_satisfaction": meter.create_gauge(
                "island_hopper_user_satisfaction_score",
                description="User satisfaction score"
            ),
            "task_completion_rate": meter.create_gauge(
                "island_hopper_task_completion_rate",
                description="Task completion rate"
            ),
            "error_rate": meter.create_gauge(
                "island_hopper_error_rate",
                description="Overall error rate"
            )
        }
    
    async def track_user_activity(
        self,
        user_id: str,
        session_id: str,
        activity_type: str,
        duration: Optional[float] = None
    ):
        """Track user activity metrics"""
        
        attributes = {
            "user_id": user_id,
            "activity_type": activity_type
        }
        
        self.metrics["user_sessions"].add(1, attributes)
        
        if duration:
            self.metrics["session_duration"].record(duration, attributes)
    
    async def track_agent_performance(
        self,
        agent_id: str,
        task_type: str,
        success: bool,
        response_time: float,
        resource_usage: float
    ):
        """Track agent performance metrics"""
        
        attributes = {
            "agent_id": agent_id,
            "task_type": task_type
        }
        
        self.metrics["agent_response_time"].record(response_time, attributes)
        self.metrics["agent_utilization"].record(resource_usage, attributes)
        
        # Update success rate (simplified - in production, use rolling average)
        success_rate = 1.0 if success else 0.0
        self.metrics["agent_success_rate"].set(success_rate, attributes)
    
    async def track_costs(
        self,
        cost_breakdown: Dict[str, float],
        user_count: int,
        revenue: Optional[float] = None
    ):
        """Track cost and ROI metrics"""
        
        total_cost = sum(cost_breakdown.values())
        
        # Track daily costs
        for category, cost in cost_breakdown.items():
            self.metrics["daily_costs"].add(cost, {"category": category})
        
        # Calculate cost per user
        if user_count > 0:
            cost_per_user = total_cost / user_count
            self.metrics["cost_per_user"].set(cost_per_user)
        
        # Calculate ROI if revenue provided
        if revenue:
            roi = (revenue - total_cost) / total_cost if total_cost > 0 else 0
            self.metrics["roi_metrics"].set(roi)
    
    async def generate_business_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Generate comprehensive business metrics report"""
        
        # This would typically query the metrics backend
        # For demonstration, we'll return a structure
        
        report = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "generated_at": datetime.now().isoformat(),
            "metrics": {
                "user_engagement": {
                    "active_users": 1250,
                    "total_sessions": 15420,
                    "avg_session_duration": 180.5,  # seconds
                    "retention_rate": 0.78
                },
                "agent_performance": {
                    "total_operations": 45680,
                    "success_rate": 0.94,
                    "avg_response_time": 2.3,  # seconds
                    "utilization": 0.67
                },
                "costs": {
                    "total_costs": 1250.50,
                    "cost_breakdown": {
                        "ai_models": 850.25,
                        "compute": 280.15,
                        "storage": 75.10,
                        "network": 45.00
                    },
                    "cost_per_user": 1.00,
                    "cost_reduction": 0.15  # 15% reduction from previous period
                },
                "quality": {
                    "user_satisfaction": 4.2,  # out of 5
                    "task_completion_rate": 0.91,
                    "error_rate": 0.06,
                    "support_tickets": 23
                }
            },
            "trends": {
                "user_growth": 0.12,  # 12% growth
                "cost_efficiency": 0.08,  # 8% improvement
                "performance_improvement": 0.15  # 15% faster response times
            }
        }
        
        return report
```

### Alerting System

```python
class AlertManager:
    """Intelligent alerting system for Island Hopper"""
    
    def __init__(self, notification_channels: List[str]):
        self.notification_channels = notification_channels
        self.alert_rules = self._load_alert_rules()
        self.alert_history = deque(maxlen=1000)
        self.suppression_rules = {}
    
    def _load_alert_rules(self) -> List[AlertRule]:
        """Load alert rules from configuration"""
        
        return [
            # High error rate alert
            AlertRule(
                name="high_error_rate",
                condition="error_rate > 0.1",  # 10% error rate
                duration=300,  # 5 minutes
                severity="high",
                message="Error rate is {{error_rate}}% (threshold: 10%)",
                channels=["slack", "email"]
            ),
            
            # High response time alert
            AlertRule(
                name="high_response_time",
                condition="avg_response_time > 5.0",  # 5 seconds
                duration=180,  # 3 minutes
                severity="medium",
                message="Average response time is {{avg_response_time}}s (threshold: 5s)",
                channels=["slack"]
            ),
            
            # Cost overrun alert
            AlertRule(
                name="cost_overrun",
                condition="daily_cost > 100",  # $100 per day
                duration=86400,  # 24 hours
                severity="high",
                message="Daily cost is ${{daily_cost}} (threshold: $100)",
                channels=["slack", "email", "pagerduty"]
            ),
            
            # Resource exhaustion alert
            AlertRule(
                name="resource_exhaustion",
                condition="cpu_usage > 0.9 or memory_usage > 0.9",
                duration=60,  # 1 minute
                severity="critical",
                message="Resource usage critical: CPU={{cpu_usage}}%, Memory={{memory_usage}}%",
                channels=["slack", "email", "pagerduty"]
            ),
            
            # Service downtime alert
            AlertRule(
                name="service_down",
                condition="availability < 0.99",  # 99% availability
                duration=120,  # 2 minutes
                severity="critical",
                message="Service availability is {{availability}}% (threshold: 99%)",
                channels=["slack", "email", "pagerduty"]
            )
        ]
    
    async def evaluate_metrics(self, metrics: Dict[str, Any]):
        """Evaluate metrics against alert rules"""
        
        for rule in self.alert_rules:
            try:
                # Check if rule conditions are met
                if self._evaluate_rule_condition(rule.condition, metrics):
                    # Check if alert should be fired (duration threshold)
                    if await self._should_fire_alert(rule, metrics):
                        await self._fire_alert(rule, metrics)
                        
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
    
    def _evaluate_rule_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert rule condition against current metrics"""
        
        # Simple condition evaluation (in production, use a proper expression parser)
        
        # Replace placeholders with actual values
        for key, value in metrics.items():
            condition = condition.replace(key, str(value))
        
        # Evaluate the condition
        try:
            return eval(condition)
        except:
            return False
    
    async def _should_fire_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """Check if alert should be fired based on duration and suppression"""
        
        # Check if alert is suppressed
        if self._is_alert_suppressed(rule.name):
            return False
        
        # Check duration threshold (simplified - in production, use time series)
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.rule_name == rule.name 
            and (datetime.now() - alert.timestamp).total_seconds() < rule.duration
        ]
        
        return len(recent_alerts) >= 1  # Fire if condition persists
    
    async def _fire_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Fire alert to configured channels"""
        
        # Format alert message
        message = self._format_alert_message(rule.message, metrics)
        
        alert = Alert(
            id=str(uuid.uuid4()),
            rule_name=rule.name,
            severity=rule.severity,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics
        )
        
        # Send to notification channels
        for channel in rule.channels:
            try:
                if channel == "slack":
                    await self._send_slack_alert(alert)
                elif channel == "email":
                    await self._send_email_alert(alert)
                elif channel == "pagerduty":
                    await self._send_pagerduty_alert(alert)
                    
            except Exception as e:
                logger.error(f"Failed to send alert to {channel}: {e}")
        
        # Record alert
        self.alert_history.append(alert)
        
        # Apply suppression
        self._apply_alert_suppression(rule.name, rule.severity)
    
    def _format_alert_message(self, template: str, metrics: Dict[str, Any]) -> str:
        """Format alert message with metric values"""
        
        message = template
        
        # Replace placeholders with formatted values
        replacements = {
            "error_rate": f"{metrics.get('error_rate', 0) * 100:.2f}",
            "avg_response_time": f"{metrics.get('avg_response_time', 0):.2f}",
            "daily_cost": f"{metrics.get('daily_cost', 0):.2f}",
            "cpu_usage": f"{metrics.get('cpu_usage', 0) * 100:.1f}",
            "memory_usage": f"{metrics.get('memory_usage', 0) * 100:.1f}",
            "availability": f"{metrics.get('availability', 0) * 100:.2f}"
        }
        
        for placeholder, value in replacements.items():
            message = message.replace(f"{{{{{placeholder}}}}}", value)
        
        return message
    
    async def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        
        import aiohttp
        
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            return
        
        color_map = {
            "low": "good",
            "medium": "warning", 
            "high": "danger",
            "critical": "#ff0000"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, "warning"),
                "title": f"🚨 Island Hopper Alert: {alert.rule_name}",
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.upper(), "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True}
                ],
                "footer": "Island Hopper Monitoring",
                "ts": int(alert.timestamp.timestamp())
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=payload)
    
    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if not all([smtp_server, smtp_user, smtp_password]):
            return
        
        # Compose email
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = os.getenv("ALERT_EMAIL_RECIPIENTS", "")
        msg['Subject'] = f"[{alert.severity.upper()}] Island Hopper Alert: {alert.rule_name}"
        
        body = f"""
        Alert Details:
        
        Rule: {alert.rule_name}
        Severity: {alert.severity.upper()}
        Time: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}
        
        Message: {alert.message}
        
        Current Metrics:
        {json.dumps(alert.metrics, indent=2)}
        
        ---
        Island Hopper Monitoring System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()

@dataclass
class AlertRule:
    name: str
    condition: str
    duration: int  # seconds
    severity: str
    message: str
    channels: List[str]

@dataclass
class Alert:
    id: str
    rule_name: str
    severity: str
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]
```

---

This comprehensive RAG knowledge base continues with detailed coverage of security, monitoring, and observability. The documentation is designed to be searchable and provides practical implementations for every aspect of the Island Hopper ecosystem.

Would you like me to continue with the remaining sections including troubleshooting, best practices, and advanced configurations?
