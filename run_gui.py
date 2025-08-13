#!/usr/bin/env python3
"""
OrganoidReader GUI Launcher

This script launches the OrganoidReader GUI application.
"""

import sys
import logging
from pathlib import Path

# Add the organoidreader package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('organoidreader.log')
        ]
    )

def main():
    """Main entry point for the GUI application."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting OrganoidReader GUI")
        
        # Check Python version
        if sys.version_info < (3, 7):
            print("Error: Python 3.7 or later is required")
            sys.exit(1)
        
        # Import and run GUI
        from organoidreader.gui.main_window import run_gui
        
        # Run the application
        exit_code = run_gui()
        
        logger.info("OrganoidReader GUI closed")
        sys.exit(exit_code)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"Error starting OrganoidReader: {e}")
        logging.error(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()