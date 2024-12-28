import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_file_path="../logs/app.log", log_level=logging.INFO):
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Configure file logging (detailed logs)
    logging.basicConfig(
        filename=log_file_path,
        filemode='a',  # Append to log file
        level=logging.DEBUG,  # File logs capture all levels
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Add console logging (minimal output)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)  # Adjust to INFO for minimal terminal output
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    # Add rotating file handler
    rotating_handler = RotatingFileHandler(log_file_path, maxBytes=10_000_000, backupCount=5)
    rotating_handler.setFormatter(formatter)
    logging.getLogger().addHandler(rotating_handler)

    logging.info("Logging setup complete!")
