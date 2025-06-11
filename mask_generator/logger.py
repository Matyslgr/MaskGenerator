##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## logging_setup
##

import os
import sys
import logging

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    "\033[94m",  # Bleu clair
        logging.INFO:     "\033[92m",  # Vert
        logging.WARNING:  "\033[93m",  # Jaune
        logging.ERROR:    "\033[91m",  # Rouge
        logging.CRITICAL: "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        original_levelname = record.levelname

        formatted = super().format(record)

        color = self.COLORS.get(record.levelno, self.RESET)

        formatted = formatted.replace(original_levelname, f"{color}{original_levelname}{self.RESET}")

        return formatted

def _configure_third_party_loggers():
    """Configure third-party loggers to reduce verbosity."""
    logging.getLogger("git").setLevel(logging.WARNING)

def setup_logging(name: str = None, level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """Set up logging with a color formatter."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColorFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False

    _configure_third_party_loggers()

    return logger
