import logging
import sys
import coloredlogs  # type: ignore[import-untyped]

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
coloredlogs.install(level="INFO")
