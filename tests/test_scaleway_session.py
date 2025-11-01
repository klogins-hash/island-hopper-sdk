"""
Tests for ScalewaySessionRepository
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import uuid
from datetime import datetime, timezone

from strands.session.scaleway import ScalewaySessionRepository, create_scaleway_session_repository


class TestScalewaySessionRepository:
    """Test cases for ScalewaySessionRepository"""
    
    @pytest.fixture
    def mock_connection_pool(self):
        """Mock connection pool"""
        mock_pool = Mock()
        mock_conn = Mock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool.putconn.return_value = None
        mock_pool.closeall.return_value = None
        return mock_pool, mock_conn
    
    @pytest.fixture
    def session_repo(self, mock_connection_pool):
        """Create session repository with mocked connection"""
        mock_pool, mock_conn = mock_connection_pool
        
        with patch('strands.session.scaleway.psycopg2.pool.ThreadedConnectionPool') as mock_pool_class:
            mock_pool_class.return_value = mock_pool
            
            repo = ScalewaySessionRepository(
                connection_string="postgresql://test:test@localhost/test"
            )
            
            repo.pool = mock_pool
            return repo
    
    def test_init_with_connection_string(self):
        """Test initialization with connection string"""
        with patch('strands.session.scaleway.psycopg2.pool.ThreadedConnectionPool') as mock_pool:
            repo = ScalewaySessionRepository(
                connection_string="postgresql://test:test@localhost/test"
            )
            
            mock_pool.assert_called_once()
            assert repo.connection_string == "postgresql://test:test@localhost/test"
    
    def test_init_with_environment_variable(self):
        """Test initialization with environment variable"""
        with patch.dict(os.environ, {
            'SCALEWAY_DATABASE_URL': 'postgresql://env:test@localhost/env'
        }):
            with patch('strands.session.scaleway.psycopg2.pool.ThreadedConnectionPool') as mock_pool:
                repo = ScalewaySessionRepository()
                
                assert repo.connection_string == "postgresql://env:test@localhost/env"
    
    def test_create_session(self, session_repo):
        """Test creating a session"""
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = [str(uuid.uuid4())]
        mock_conn = session_repo.pool.getconn.return_value
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        session_id = session_repo.create_session(
            agent_id="test-agent",
            data={"key": "value"},
            ttl=3600
        )
        
        assert session_id is not None
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
    
    def test_get_session_found(self, session_repo):
        """Test getting a session that exists"""
        session_id = str(uuid.uuid4())
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {
            'session_id': session_id,
            'agent_id': 'test-agent',
            'data': {'key': 'value'},
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'expires_at': None,
            'metadata': {}
        }
        mock_conn = session_repo.pool.getconn.return_value
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        result = session_repo.get_session(session_id)
        
        assert result is not None
        assert result['session_id'] == session_id
        assert result['agent_id'] == 'test-agent'
        assert result['data'] == {'key': 'value'}
    
    def test_get_session_not_found(self, session_repo):
        """Test getting a session that doesn't exist"""
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_conn = session_repo.pool.getconn.return_value
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        result = session_repo.get_session("nonexistent")
        
        assert result is None
    
    def test_update_session(self, session_repo):
        """Test updating a session"""
        mock_cursor = Mock()
        mock_cursor.rowcount = 1
        mock_conn = session_repo.pool.getconn.return_value
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        updated = session_repo.update_session(
            session_id="test-session",
            data={"new_key": "new_value"}
        )
        
        assert updated is True
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
    
    def test_delete_session(self, session_repo):
        """Test deleting a session"""
        mock_cursor = Mock()
        mock_cursor.rowcount = 1
        mock_conn = session_repo.pool.getconn.return_value
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        deleted = session_repo.delete_session("test-session")
        
        assert deleted is True
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
    
    def test_list_sessions(self, session_repo):
        """Test listing sessions"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {
                'session_id': str(uuid.uuid4()),
                'agent_id': 'test-agent-1',
                'data': {'key': 'value1'},
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'expires_at': None,
                'metadata': {}
            },
            {
                'session_id': str(uuid.uuid4()),
                'agent_id': 'test-agent-2',
                'data': {'key': 'value2'},
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'expires_at': None,
                'metadata': {}
            }
        ]
        mock_conn = session_repo.pool.getconn.return_value
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        sessions = session_repo.list_sessions()
        
        assert len(sessions) == 2
        assert sessions[0]['agent_id'] == 'test-agent-1'
        assert sessions[1]['agent_id'] == 'test-agent-2'
    
    def test_list_sessions_with_agent_filter(self, session_repo):
        """Test listing sessions with agent filter"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {
                'session_id': str(uuid.uuid4()),
                'agent_id': 'specific-agent',
                'data': {'key': 'value'},
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'expires_at': None,
                'metadata': {}
            }
        ]
        mock_conn = session_repo.pool.getconn.return_value
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        sessions = session_repo.list_sessions(agent_id="specific-agent")
        
        assert len(sessions) == 1
        assert sessions[0]['agent_id'] == 'specific-agent'
    
    def test_cleanup_expired_sessions(self, session_repo):
        """Test cleaning up expired sessions"""
        mock_cursor = Mock()
        mock_cursor.rowcount = 5
        mock_conn = session_repo.pool.getconn.return_value
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        cleaned = session_repo.cleanup_expired_sessions()
        
        assert cleaned == 5
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
    
    def test_get_session_stats(self, session_repo):
        """Test getting session statistics"""
        mock_cursor = Mock()
        mock_cursor.side_effect = [
            Mock(fetchone=lambda: [100]),  # total sessions
            Mock(fetchone=lambda: [80]),   # active sessions
            Mock(fetchone=lambda: [20]),   # expired sessions
            Mock(fetchall=lambda: [('agent1', 10), ('agent2', 5)])  # top agents
        ]
        mock_conn = session_repo.pool.getconn.return_value
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        stats = session_repo.get_session_stats()
        
        assert stats['total_sessions'] == 100
        assert stats['active_sessions'] == 80
        assert stats['expired_sessions'] == 20
        assert stats['top_agents'] == {'agent1': 10, 'agent2': 5}
    
    def test_close(self, session_repo):
        """Test closing the connection pool"""
        session_repo.close()
        session_repo.pool.closeall.assert_called_once()


class TestCreateScalewaySessionRepository:
    """Test cases for create_scaleway_session_repository convenience function"""
    
    def test_create_with_connection_string(self):
        """Test creating repository with connection string"""
        with patch('strands.session.scaleway.psycopg2.pool.ThreadedConnectionPool') as mock_pool:
            repo = create_scaleway_session_repository(
                connection_string="postgresql://test:test@localhost/test"
            )
            
            assert isinstance(repo, ScalewaySessionRepository)
            assert repo.connection_string == "postgresql://test:test@localhost/test"
    
    def test_create_with_defaults(self):
        """Test creating repository with defaults"""
        with patch('strands.session.scaleway.psycopg2.pool.ThreadedConnectionPool') as mock_pool:
            repo = create_scaleway_session_repository()
            
            assert isinstance(repo, ScalewaySessionRepository)


if __name__ == "__main__":
    pytest.main([__file__])
