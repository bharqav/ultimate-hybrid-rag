import logging


def setup_logging(level="INFO"):
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("UltimateRAG")
