import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

def setup_logger():
        logger = logging.getLogger('income_fore')
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        base_dir = Path(__file__).resolve().parent.parent
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / "incomes.log"
        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

main_logger = setup_logger()
