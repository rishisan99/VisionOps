import logging
import os
import sys
from datetime import datetime

# Create logs directory
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path with timestamp
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Define log format
LOG_FORMAT = "[ %(asctime)s ] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"

# Create a root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler (for persistence)
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Stream handler (for Docker/stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Avoid duplicate handlers (important if this is imported multiple times)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# For convenience, expose the logging module-style interface
logging = logger
