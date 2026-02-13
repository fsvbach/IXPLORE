import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Package-level logger
logger = logging.getLogger("ixplore")
logger.setLevel(logging.INFO) 

