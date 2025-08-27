import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
import sys
import kagglehub

def _init_logger():
    logger = logging.getLogger("data_ingestion")
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


def load_params(file_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        _logger.debug("Successfully retrieved paramaters from %s", file_path)
        return params
    except FileNotFoundError:
        _logger.error("File not found: %s", file_path)
        raise
    except yaml.YAMLError as e:
        _logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        _logger.error("Unexpected error: %s", e)
        raise


def load_data(data: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data)
        _logger.debug("Successfully loaded data from %s", data)
        return df
    except FileNotFoundError:
        _logger.error("File not found: %s", data)
        raise
    except pd.errors.ParserError as e:
        _logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        _logger.error("An unexpected error occurred while loading the data: %s", e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data by removing missing values, duplicates and empty strings.
    It also renames columns and replaces strings with numbers in "Sentiment" column.
    """
    try:
        df.dropna(inplace=True)                     # Remove missing values
        df.drop_duplicates(inplace=True)            # Remove duplicates
        df = df[df["Comment"].str.strip() != ""]    # Remove empty comments
        df["Sentiment"] = df["Sentiment"].replace({"negative": 0, "neutral": 1, "positive": 2}) # Replace strings with numbers
        df.rename(columns={"Sentiment": "labels", "Comment":"text"}, inplace=True)  # Rename sentiment column
        _logger.debug("Successfully prepared the data.")
        return df
    except KeyError as e:
        _logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        _logger.error("An unexpected error occurred while preprocessing the data: %s", e)
        raise
        
def save_data(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, data_path: str) -> None:
    """Save train and test datasets. If the processed folder doesn't exist, create it."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_dataset.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_dataset.to_csv(os.path.join(data_path, "test.csv"), index=False)
        _logger.debug("Successfully saved train and test dataset to %s", data_path)
    except Exception as e:
        _logger.error("An unexpected error occurred while saving the data: %s", e)
        raise

def main():
    try:
        # Load parameters from params.yaml in the root directory
        params = load_params(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../params.yaml"))
        test_size = params["data_ingestion"]["test_size"]

        # Load data from the specified url
        data_path = kagglehub.dataset_download("atifaliak/youtube-comments-dataset")
        data_path += "/YoutubeCommentsDataSet.csv"
        df = load_data(data_path)

        # Preprocess data
        processed_df = preprocess_data(df)

        # Split into training and testing set
        train_dataset, test_dataset = train_test_split(processed_df, test_size=test_size, random_state=42, stratify=processed_df["labels"])

        # Save split datasets
        save_data(train_dataset, test_dataset, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"))

    except Exception as e:
        _logger.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")


    
_init_logger()
_logger = logging.getLogger("data_ingestion")

if __name__ == "__main__":
    main()