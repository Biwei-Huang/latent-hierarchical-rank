import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/logs",
    format="%(levelname)-5s %(funcName)-30s %(message)s",
    filemode="w",
)
LOGGER = logging.getLogger()
