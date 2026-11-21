import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Explicitly load .env from the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# Retrieve DATABASE_URL with a sensible local SQLite fallback for out-of-the-box ease of use
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./retention_system.db")

# Cloud PostgreSQL compatibility fix:
# Providers like Supabase, Neon, or Heroku often supply "postgres://" instead of "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configure engine parameters depending on the database backend
connect_args = {}
engine_kwargs = {
    "echo": False,
}

if DATABASE_URL.startswith("sqlite"):
    # SQLite requires check_same_thread=False for multi-threaded FastAPI execution
    connect_args["check_same_thread"] = False
    engine_kwargs["connect_args"] = connect_args
else:
    # Production PostgreSQL (Neon, Supabase, RDS)
    # pool_pre_ping checks connection vitality before using it from the pool (prevents stale SSL drops)
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20
    engine_kwargs["pool_recycle"] = 300

# Create SQLAlchemy Engine
engine = create_engine(DATABASE_URL, **engine_kwargs)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base for models
Base = declarative_base()


def get_db():
    """
    FastAPI dependency yielding an independent database session per request.
    Guarantees session cleanup upon completion or exception.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initializes all database tables registered under Base.metadata.
    Safe to execute multiple times (idempotent).
    """
    # Import models here to register them with Base before metadata creation
    from app import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully.")

