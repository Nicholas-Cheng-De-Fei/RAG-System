import logging
import logging.handlers
import os
import sys

from utils.utils import get_envvar

LOG_DATE_FMT = "%d-%m-%Y %H:%M:%S"

log_dir = get_envvar("LOG_DIR")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

log = logging.getLogger(get_envvar("LOG_NAME"))
log.propagate = False
log.setLevel(get_envvar("LOG_LEVEL"))

msg_formatter = logging.Formatter(
    fmt="%(asctime)s|%(levelname)s|%(message)s", datefmt=LOG_DATE_FMT
)


def create_time_rotating_file_handler(log_level, filename, formatter):
    handler = logging.handlers.TimedRotatingFileHandler(
        f"{log_dir}/{filename}.log", when="midnight", backupCount=30
    )
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    return handler


class DebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG


# debug_handler
debug_handler = create_time_rotating_file_handler(logging.DEBUG, "debug", msg_formatter)
debug_handler.addFilter(DebugFilter())

# error_handler
error_handler = create_time_rotating_file_handler(
    logging.WARNING, "error", msg_formatter
)

# info_handler
info_handler = create_time_rotating_file_handler(logging.INFO, "info", msg_formatter)

print_handler = logging.StreamHandler(sys.stdout)
print_handler.setLevel(logging.INFO)
print_handler.setFormatter(msg_formatter)
log.addHandler(print_handler)

log.addHandler(debug_handler)
log.addHandler(error_handler)
log.addHandler(info_handler)
