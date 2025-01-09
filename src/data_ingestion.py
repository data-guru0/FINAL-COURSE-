import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from utils.common_functions import read_yaml
from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import *

# Initialize logger and environment variables
logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        """
        Initialize the DataIngestion class with configuration values from YAML.

        :param config: Dictionary containing configuration values
        """
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]  # Default to 80-20 split

        # Ensure the local directory exists
        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"DataIngestion initialized with bucket: {self.bucket_name}, file: {self.file_name}")

    def download_csv_from_gcp(self):
        """Download the CSV file from the GCP bucket."""
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"File downloaded successfully to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error downloading file from GCP: {e}")
            raise CustomException(f"Failed to download {self.file_name} from bucket {self.bucket_name}", e)

    def split_data(self):
        """Split the raw CSV into train and test datasets."""
        try:
            logger.info("Starting data split process")

            # Read the raw CSV data
            data = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(
                data, test_size=1 - self.train_test_ratio, random_state=42
            )

            # Save the train and test CSV files
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error splitting or saving data: {e}")
            raise CustomException("Failed to split and save data", e)

    def run(self):
        """Orchestrates the entire data ingestion process."""
        try:
            logger.info("Starting data ingestion process")

            # Download CSV from GCP
            self.download_csv_from_gcp()

            # Split the data into train and test and save them
            self.split_data()

            logger.info("Data ingestion process completed successfully")

        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
        finally:
            logger.info("End of data ingestion process")


if __name__ == "__main__":
    try:
        config = read_yaml(CONFIG_PATH)

        # Initialize and run data ingestion
        data_ingestion = DataIngestion(config=config)
        data_ingestion.run()

    except CustomException as ce:
        logger.error(f"CustomException: {str(ce)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
