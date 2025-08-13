"""
Main entry point for OrganoidReader application.
"""

import sys
import logging
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from organoidreader.config.config_manager import get_config, get_config_manager
from organoidreader.utils.logging_setup import setup_logging


def main():
    """Main entry point for the application."""
    try:
        # Load configuration
        config_manager = get_config_manager()
        config = config_manager.load_config()
        
        # Setup logging
        setup_logging(config.logging)
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting {config.app_name} v{config.version}")
        
        # Create necessary directories
        config_manager.create_directories(config)
        
        # Validate configuration
        issues = config_manager.validate_config(config)
        if issues:
            logger.warning(f"Configuration issues found: {issues}")
        
        # Import and start GUI (when implemented)
        # from organoidreader.gui.main_window import MainWindow
        # app = MainWindow(config)
        # app.run()
        
        print(f"OrganoidReader v{config.version} initialized successfully!")
        print(f"Configuration loaded from: {config_manager.config_path}")
        print("GUI implementation coming soon...")
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()