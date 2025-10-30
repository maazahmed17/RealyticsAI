"""
Database Connection Management
"""

import asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from ..config.settings import get_database_url
from ..models.base import Base

# Database engines
sync_engine = None
async_engine = None
async_session = None


async def init_database():
    """Initialize database connections"""
    global sync_engine, async_engine, async_session
    
    database_url = get_database_url()
    
    # Sync engine for migrations
    sync_engine = create_engine(database_url)
    
    # Async engine for API operations
    if database_url.startswith("sqlite"):
        async_database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        async_engine = create_async_engine(async_database_url)
    else:
        # For PostgreSQL
        async_database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        async_engine = create_async_engine(async_database_url)
    
    # Create async session factory
    async_session = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    # Create tables if they do not exist (dev convenience; migrations can manage in prod)
    try:
        Base.metadata.create_all(sync_engine)
    except Exception as e:
        print(f"⚠️  Failed to auto-create tables: {e}")
    
    print(f"✅ Database initialized: {database_url}")


async def get_db_session():
    """Get async database session"""
    if not async_session:
        await init_database()
    
    async with async_session() as session:
        yield session
