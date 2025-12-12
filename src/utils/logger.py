# FIXED: src/utils/logger.py
import logging
import os


class Logger:
    """Logger utility for game and algorithm logging with benchmark mode support."""
    
    # Class-level flag to control logging across all instances
    _benchmark_mode = False
    _min_level = logging.INFO
    _handlers_disabled = False

    def __init__(self, name="app"):
        self.logger = logging.getLogger(name)
        self.name = name
        
        # Set logger to always propagate to root
        self.logger.propagate = True
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set initial level
        self._update_level()

    @classmethod
    def set_benchmark_mode(cls, enabled: bool = True):
        """Enable/disable benchmark mode (suppresses all logs)."""
        cls._benchmark_mode = enabled
        
        if enabled:
            # DISABLE ALL LOGGING
            logging.disable(logging.CRITICAL)
            cls._handlers_disabled = True
        else:
            # RE-ENABLE LOGGING
            logging.disable(logging.NOTSET)
            cls._handlers_disabled = False
            cls._min_level = logging.INFO
        
        # Update all existing loggers
        for logger_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            if enabled:
                logger.setLevel(logging.CRITICAL)
            else:
                logger.setLevel(logging.INFO)

    def _update_level(self):
        """Update logger level based on mode."""
        if self._benchmark_mode:
            self.logger.setLevel(logging.CRITICAL)
        else:
            self.logger.setLevel(self._min_level)

    def info(self, message):
        if not self._benchmark_mode:
            self.logger.info(message)

    def warning(self, message):
        if not self._benchmark_mode:
            self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)  # Always log errors

    def debug(self, message):
        if not self._benchmark_mode:
            self.logger.debug(message)