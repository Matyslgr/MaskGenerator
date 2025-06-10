##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## logging_setup
##

import logging
import sys

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
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.name = f"{color}{record.name}{self.RESET}"
        return super().format(record)

class LoggerManager:
    def __init__(self, name: str = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = ColorFormatter(
                "%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.propagate = False

        # Configure third-party loggers here (exemple avec git)
        self._configure_third_party_loggers()

    def _configure_third_party_loggers(self):
        logging.getLogger("git").setLevel(logging.WARNING)

    def get_logger(self):
        return self.logger
