import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_file_logger(
    log_file: Path,
    level: int = logging.INFO,
    max_size: int = 10**6,
    backup_count: int = 3,
) -> None:
    """
    Configures a global logger that writes to a file with log rotation.

    Args:
        log_file (Path): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).
        max_size (int): Maximum size of the log file in bytes before rotation.
        backup_count (int): Number of backup log files to keep.
    """
    logger = logging.getLogger()  # Root logger
    logger.setLevel(level)

    # Avoid adding duplicate handlers if the function is called multiple times
    if any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        return

    # File handler with rotation
    handler = RotatingFileHandler(log_file, maxBytes=max_size, backupCount=backup_count)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)


def setup_worker_logger(worker_id: int, log_dir: Path) -> None:
    """Sets up logging for a worker process using a file handler."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logging if multiple initializations occur
    logger.handlers = []

    # Define log file name for this worker
    log_file = log_dir / f"worker_{worker_id}.log"

    # File handler with rotation
    handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=3)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)

    logging.getLogger("curobo").setLevel(logging.WARNING)


# Example Usage
if __name__ == "__main__":
    log_path = Path("application.log")
    setup_file_logger(log_path)

    # Logging from anywhere in the application
    logging.info("This is an informational message.")
    logging.error("This is an error message.")
