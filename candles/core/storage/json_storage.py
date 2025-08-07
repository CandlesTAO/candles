import argparse
import json
import os
from datetime import datetime
from decimal import Decimal
from logging import getLogger
from pathlib import Path
from typing import Any, Optional

from .base import BaseStorage

logger = getLogger(__name__)

DEFAULT_PATH = Path("~/.candles/data")

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime and Decimal objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


def _read_json(file_path: Path) -> Optional[dict]:
    """Read JSON file and return deserialized data."""
    try:
        with file_path.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        logger.error(f"Error loading/decoding {file_path.as_posix()} file.")
        return None


class BaseJsonStorage(BaseStorage):
    def __init__(self, config=None, clean_up: bool = False):
        self.config = config or self.get_config()

        self.path = (
            Path(self.config.json_path).expanduser()
            if getattr(self.config, "json_path", None)
            else DEFAULT_PATH.expanduser()
        )
        logger.info(f"Initializing storage with path: {self.path.absolute()}")
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Successfully created/verified directory: {self.path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to create directory {self.path.absolute()}: {str(e)}")
            raise

        if clean_up:
            self._cleanup()

    @classmethod
    def add_args(cls, parser: "argparse.ArgumentParser"):
        """Add Json storage-specific arguments to parser."""
        parser.add_argument(
            "--json_path",
            type=str,
            default=os.getenv("JSON_PATH", DEFAULT_PATH),
            help="Path to save pool configuration JSON files",
        )

    def _cleanup(self):
        """Remove JSON files older than the oldest prediction."""
        file = self.path / "predictions.json"
        if file.exists():
            file.unlink()

    def save_data(self, data: Any, prefix: str = "predictions") -> None:
        """Save data for specific block."""
        file_name = f"{prefix}.json"
        data_file = self.path / file_name
        logger.info(f"Attempting to save data to: {data_file.absolute()}")

        try:
            # Ensure the directory exists
            self.path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory exists: {self.path.absolute()}")

            # Write the data
            with data_file.open("w") as f:
                json.dump(data, f, indent=4, cls=CustomJSONEncoder)

            logger.info(f"Successfully saved data to {data_file.absolute()}")

            # Verify the file was written
            if data_file.exists():
                logger.info(f"File exists after write: {data_file.absolute()}")
                logger.info(f"File size: {data_file.stat().st_size} bytes")
            else:
                logger.error(f"File does not exist after write: {data_file.absolute()}")

        except Exception as e:
            logger.error(f"Error saving data to {data_file.absolute()}: {str(e)}")
            logger.error(f"Current working directory: {Path.cwd().absolute()}")
            logger.error(f"Directory permissions: {oct(self.path.stat().st_mode)[-3:]}")
            raise

    def load_data(self, prefix: str = "predictions") -> Optional[Any]:
        data_file = self.path / f"{prefix}.json"
        return _read_json(data_file) if data_file.exists() else None


