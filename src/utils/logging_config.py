import logging
from pathlib import Path
import time
import os

def setup_logging(script_name):
    """
    Set up logging configuration for DVC pipeline scripts.
    
    Args:
        script_name: Name of the script (used in the log file name)
        
    Returns:
        Logger instance configured for the script
    """
    # Create log directory at the project root if it doesn't exist
    root_path = Path(__file__).parent.parent.parent
    log_dir = root_path / "log"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Get DVC run ID from environment if available, otherwise use timestamp
    dvc_run_id = os.environ.get('DVC_RUN_ID', time.strftime("%Y%m%d-%H%M%S"))
    
    # Use a common log file for all scripts in the same DVC run
    log_file_path = log_dir / f"pipeline_run_{dvc_run_id}.log"
    
    # Configure logging
    logger = logging.getLogger(script_name)
    
    # Only configure handlers if they haven't been configured yet
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Add a banner to clearly mark the start of this script's logs
        banner = f"\n{'*' * 80}\n* STARTING SCRIPT: {script_name}\n* {'*' * 78}"
        logger.info(banner)
        
        logger.info(f"Logging for {script_name} initialized. Log file: {log_file_path}")
    
    return logger