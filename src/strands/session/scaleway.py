"""
Scaleway Session Repository

PostgreSQL-backed session management for Strands agents.
Provides persistent storage with connection pooling and migrations.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import uuid

import psycopg2
import psycopg2.pool
from psycopg2.extras import Json, RealDictCursor
from psycopg2.extensions import connection as PGConnection

from strands.session.base import SessionRepository


@dataclass
class SessionData:
    """Session data structure"""
    session_id: str
    agent_id: str
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class ScalewaySessionRepository(SessionRepository):
    """
    PostgreSQL-backed session repository for Scaleway.
    
    Features:
    - Connection pooling for performance
    - Automatic session expiration
    - Metadata support
    - Migration management
    - Connection retry logic
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        pool_size: int = 5,
        max_connections: int = 20,
        default_ttl: int = 3600,  # 1 hour default
        **kwargs
    ):
        """
        Initialize ScalewaySessionRepository.
        
        Args:
            connection_string: PostgreSQL connection string
            pool_size: Initial connection pool size
            max_connections: Maximum connections in pool
            default_ttl: Default session TTL in seconds
            **kwargs: Additional arguments
        """
        self.logger = logging.getLogger(__name__)
        self.default_ttl = default_ttl
        
        # Get connection string from parameter or environment
        if not connection_string:
            connection_string = os.getenv(
                "SCALEWAY_DATABASE_URL",
                "postgresql://postgres:password@localhost:5432/island_hopper"
            )
        
        self.connection_string = connection_string
        
        # Initialize connection pool
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=max_connections,
            dsn=connection_string
        )
        
        # Run migrations
        self._ensure_tables()
        
        self.logger.info("ScalewaySessionRepository initialized")
    
    def _get_connection(self) -> PGConnection:
        """Get connection from pool with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.pool.getconn()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                continue
    
    def _release_connection(self, conn: PGConnection):
        """Release connection back to pool"""
        try:
            self.pool.putconn(conn)
        except Exception as e:
            self.logger.error(f"Error releasing connection: {e}")
    
    def _ensure_tables(self):
        """Create tables if they don't exist"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Create sessions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        agent_id VARCHAR(255) NOT NULL,
                        data JSONB NOT NULL DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        expires_at TIMESTAMP WITH TIME ZONE,
                        metadata JSONB DEFAULT '{}'
                    );
                """)
                
                # Create indexes for performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_agent_id 
                    ON sessions(agent_id);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_expires_at 
                    ON sessions(expires_at);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_created_at 
                    ON sessions(created_at);
                """)
                
                conn.commit()
                self.logger.info("Database tables ensured")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error creating tables: {e}")
            raise
        finally:
            self._release_connection(conn)
    
    def create_session(
        self,
        agent_id: str,
        data: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session.
        
        Args:
            agent_id: ID of the agent
            data: Initial session data
            ttl: Time to live in seconds
            metadata: Optional metadata
        
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        ttl = ttl or self.default_ttl
        expires_at = datetime.now(timezone.utc).timestamp() + ttl
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sessions 
                    (session_id, agent_id, data, expires_at, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING session_id;
                """, (
                    session_id,
                    agent_id,
                    Json(data or {}),
                    datetime.fromtimestamp(expires_at, tz=timezone.utc),
                    Json(metadata or {})
                ))
                
                result = cur.fetchone()
                conn.commit()
                
                self.logger.info(f"Created session {session_id} for agent {agent_id}")
                return result[0]
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error creating session: {e}")
            raise
        finally:
            self._release_connection(conn)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data by ID.
        
        Args:
            session_id: Session ID
        
        Returns:
            Session data or None if not found
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT session_id, agent_id, data, created_at, 
                           updated_at, expires_at, metadata
                    FROM sessions 
                    WHERE session_id = %s 
                    AND (expires_at IS NULL OR expires_at > NOW());
                """, (session_id,))
                
                result = cur.fetchone()
                if not result:
                    return None
                
                # Convert to dict and handle JSON
                session_data = dict(result)
                session_data['data'] = dict(session_data['data'])
                session_data['metadata'] = dict(session_data['metadata'])
                
                return session_data
        except Exception as e:
            self.logger.error(f"Error getting session {session_id}: {e}")
            raise
        finally:
            self._release_connection(conn)
    
    def update_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        extend_ttl: Optional[int] = None
    ) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session ID
            data: New session data
            extend_ttl: Optional TTL extension in seconds
        
        Returns:
            True if updated, False if not found
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Update data and timestamp
                if extend_ttl:
                    expires_at = datetime.now(timezone.utc).timestamp() + extend_ttl
                    cur.execute("""
                        UPDATE sessions 
                        SET data = %s, updated_at = NOW(), expires_at = %s
                        WHERE session_id = %s 
                        AND (expires_at IS NULL OR expires_at > NOW());
                    """, (Json(data), datetime.fromtimestamp(expires_at, tz=timezone.utc), session_id))
                else:
                    cur.execute("""
                        UPDATE sessions 
                        SET data = %s, updated_at = NOW()
                        WHERE session_id = %s 
                        AND (expires_at IS NULL OR expires_at > NOW());
                    """, (Json(data), session_id))
                
                conn.commit()
                updated = cur.rowcount > 0
                
                if updated:
                    self.logger.debug(f"Updated session {session_id}")
                
                return updated
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error updating session {session_id}: {e}")
            raise
        finally:
            self._release_connection(conn)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID
        
        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM sessions 
                    WHERE session_id = %s;
                """, (session_id,))
                
                conn.commit()
                deleted = cur.rowcount > 0
                
                if deleted:
                    self.logger.info(f"Deleted session {session_id}")
                
                return deleted
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error deleting session {session_id}: {e}")
            raise
        finally:
            self._release_connection(conn)
    
    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List sessions.
        
        Args:
            agent_id: Optional agent ID filter
            limit: Maximum number of sessions to return
            offset: Offset for pagination
        
        Returns:
            List of session data
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if agent_id:
                    cur.execute("""
                        SELECT session_id, agent_id, data, created_at, 
                               updated_at, expires_at, metadata
                        FROM sessions 
                        WHERE agent_id = %s 
                        AND (expires_at IS NULL OR expires_at > NOW())
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s;
                    """, (agent_id, limit, offset))
                else:
                    cur.execute("""
                        SELECT session_id, agent_id, data, created_at, 
                               updated_at, expires_at, metadata
                        FROM sessions 
                        WHERE (expires_at IS NULL OR expires_at > NOW())
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s;
                    """, (limit, offset))
                
                results = cur.fetchall()
                
                # Convert to dicts and handle JSON
                sessions = []
                for result in results:
                    session_data = dict(result)
                    session_data['data'] = dict(session_data['data'])
                    session_data['metadata'] = dict(session_data['metadata'])
                    sessions.append(session_data)
                
                return sessions
        except Exception as e:
            self.logger.error(f"Error listing sessions: {e}")
            raise
        finally:
            self._release_connection(conn)
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM sessions 
                    WHERE expires_at IS NOT NULL AND expires_at <= NOW();
                """)
                
                conn.commit()
                cleaned = cur.rowcount
                
                if cleaned > 0:
                    self.logger.info(f"Cleaned up {cleaned} expired sessions")
                
                return cleaned
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error cleaning up sessions: {e}")
            raise
        finally:
            self._release_connection(conn)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Total sessions
                cur.execute("SELECT COUNT(*) FROM sessions;")
                total_sessions = cur.fetchone()[0]
                
                # Active sessions (not expired)
                cur.execute("""
                    SELECT COUNT(*) FROM sessions 
                    WHERE (expires_at IS NULL OR expires_at > NOW());
                """)
                active_sessions = cur.fetchone()[0]
                
                # Expired sessions
                cur.execute("""
                    SELECT COUNT(*) FROM sessions 
                    WHERE expires_at IS NOT NULL AND expires_at <= NOW();
                """)
                expired_sessions = cur.fetchone()[0]
                
                # Sessions by agent
                cur.execute("""
                    SELECT agent_id, COUNT(*) as count
                    FROM sessions 
                    WHERE (expires_at IS NULL OR expires_at > NOW())
                    GROUP BY agent_id
                    ORDER BY count DESC
                    LIMIT 10;
                """)
                top_agents = dict(cur.fetchall())
                
                return {
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "expired_sessions": expired_sessions,
                    "top_agents": top_agents
                }
        except Exception as e:
            self.logger.error(f"Error getting session stats: {e}")
            raise
        finally:
            self._release_connection(conn)
    
    def close(self):
        """Close the connection pool"""
        if hasattr(self, 'pool'):
            self.pool.closeall()
            self.logger.info("Connection pool closed")


# Convenience function for easy initialization
def create_scaleway_session_repository(
    connection_string: Optional[str] = None,
    **kwargs
) -> ScalewaySessionRepository:
    """
    Convenience function to create a ScalewaySessionRepository.
    
    Args:
        connection_string: PostgreSQL connection string
        **kwargs: Additional arguments for ScalewaySessionRepository
    
    Returns:
        Configured ScalewaySessionRepository instance
    """
    return ScalewaySessionRepository(
        connection_string=connection_string,
        **kwargs
    )
