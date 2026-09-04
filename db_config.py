import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use environment variables if set, otherwise use defaults
DB_USER     = os.getenv("DB_USER",     "qrc_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "P4ssw0rd")
DB_HOST     = os.getenv("DB_HOST",     "147.232.204.254")
DB_NAME     = os.getenv("DB_NAME",     "quantumReservoir")

# pymysql is pure-Python (no C build needed); fallback to mysqldb if preferred
DB_DRIVER = os.getenv("DB_DRIVER", "mysql+pymysql")

CONNECTION_URL = f"{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

engine = create_engine(
    CONNECTION_URL,
    pool_recycle=3600,   # recycle connections before MySQL wait_timeout
    pool_pre_ping=True,  # verify connection is alive before using it
)

Session = sessionmaker(bind=engine, autocommit=False)


def get_session():
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    return Session()
