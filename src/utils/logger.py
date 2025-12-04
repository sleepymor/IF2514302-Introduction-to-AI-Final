import logging


class Logger:
    """Logger utility for game and algorithm logging."""

    def __init__(self, name="app"):
        # grey = "\x1b[38;20m"
        # yellow = "\x1b[33;20m"
        # reset = "\x1b[0m"
        self.logger = logging.getLogger(name)

        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s"
            )

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)