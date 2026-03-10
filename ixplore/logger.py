import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Package-level logger
logger: logging.Logger = logging.getLogger("ixplore")
logger.setLevel(logging.INFO) 

