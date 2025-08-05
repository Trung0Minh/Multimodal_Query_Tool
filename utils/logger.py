import os
import sys
from loguru import logger
from config.settings import settings

# logger configuration
logger.remove() # remove default config

log_path = os.path.join(settings.LOG_DIR, "file_{time}.log")
logger.add(
    log_path,
    rotation="10 MB",
    compression="zip",
    level=settings.LOG_LEVEL, # log level from settings
    colorize=True,
    format="{time} {level} {message}",
    enqueue=True
)
logger.add(
    sys.stderr, # output to console
    level=settings.LOG_LEVEL,
    colorize=True,
    format="<green>{time}</green> <level>{level}</level> <bold>{message}</bold>"
)