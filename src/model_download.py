import kagglehub
import shutil
import os
import sys
import logging


def _init_logger():
    logger = logging.getLogger("model_download")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def download_trained_model():
    """Downloads trained model from Kaggle."""
    try:
        path = kagglehub.model_download("soo907/roberta_model/pyTorch/default")

        destination = "src/roberta_model"
        os.makedirs(destination, exist_ok=True)
        shutil.copytree(path, destination, dirs_exist_ok=True)

        _logger.info(f"The model was successfully downloaded to {destination}.")

    except Exception as e:
        _logger.error("An unexpected error occurred while downloading the model: ", e)
        raise



_init_logger()
_logger = logging.getLogger("model_download")

if __name__ == "__main__":
    download_trained_model()