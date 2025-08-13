"""
Logging setup utilities for OrganoidReader.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from organoidreader.config.config_manager import LoggingConfig


def setup_logging(logging_config: LoggingConfig, log_dir: Optional[Path] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        logging_config: Logging configuration object
        log_dir: Directory for log files (default: current directory)
    """
    if log_dir is None:
        log_dir = Path(".")
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, logging_config.level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(logging_config.format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, logging_config.level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if logging_config.log_to_file:
        log_file_path = log_dir / logging_config.log_file
        
        # Use RotatingFileHandler for log rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=logging_config.max_log_size_mb * 1024 * 1024,
            backupCount=logging_config.backup_count
        )
        file_handler.setLevel(getattr(logging, logging_config.level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.debug(f"Log level: {logging_config.level}")
    logger.debug(f"Log to file: {logging_config.log_to_file}")
    if logging_config.log_to_file:
        logger.debug(f"Log file: {log_dir / logging_config.log_file}")