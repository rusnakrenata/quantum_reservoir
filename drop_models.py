"""
drop_models.py
--------------
Drops all QRC tables from the database (in dependency order).
Run this to reset the schema during development.

Usage:
    python drop_models.py            # asks for confirmation
    python drop_models.py --force    # skips confirmation prompt
"""

import sys
from db_config import engine
from models import Base


def drop_all(force: bool = False) -> None:
    tables = list(Base.metadata.tables.keys())
    if not tables:
        print("No tables found in metadata.")
        return

    print("Tables to be dropped:")
    for t in tables:
        print(f"  - {t}")

    if not force:
        answer = input("\nType 'yes' to confirm: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return

    Base.metadata.drop_all(engine)
    print("All QRC tables dropped successfully.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    drop_all(force=force)
