import os
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import load_data,read_yaml

logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)  # Load YAML configuration

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self, df):
        """
        Perform preprocessing on the DataFrame.
        """
        try:
            logger.info("Starting data preprocessing")

            # Drop unnecessary columns and duplicates
            logger.info("Dropping 'Booking_ID' column and duplicates")
            df.drop(columns=["Booking_ID"], inplace=True)
            df.drop_duplicates(inplace=True)

            # Extract columns from the configuration
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            # Apply Label Encoding for categorical columns
            logger.info("Applying Label Encoding")
            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label: code for label, code in zip(
                    label_encoder.classes_,
                    label_encoder.transform(label_encoder.classes_)
                )}

            logger.info("Label Encoding Mappings:")
            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            # Apply log transformation to highly skewed numerical columns
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            logger.info(f"Applying log transformation to columns with skewness > {skew_threshold}")
            skewness = df[num_cols].apply(lambda x: x.skew())
            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df

        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException("Data preprocessing failed", e)
    def balance_data(self, df):
        """
        Balance the dataset using SMOTE for the target class.
        """
        try:
            logger.info("Balancing the dataset using SMOTE")

            # Extract features and target column
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            # Apply SMOTE to handle class imbalance
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Combine resampled data into a DataFrame
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status'] = y_resampled

            logger.info("Dataset balanced successfully")
            return balanced_df

        except Exception as e:
            logger.error(f"Error during dataset balancing: {e}")
            raise CustomException("Dataset balancing failed", e)

    def select_features(self, df):
        """
        Select top features based on feature importance using Random Forest.
        """
        try:
            logger.info("Selecting top features using Random Forest")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values(by='importance', ascending=False)

            top_features_count = self.config["data_processing"]["num_features_to_select"]
            top_features = feature_importance_df['feature'].head(top_features_count).values

            top_df = df[top_features.tolist() + ['booking_status']]

            logger.info(f"Top features selected: {list(top_features)}")
            return top_df

        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            raise CustomException("Feature selection failed", e)
    
    def save_data(self, df, file_path):
        """
        Save the DataFrame to the specified file path.
        """
        try:
            logger.info(f"Saving data to {file_path}")
            
            # Save the DataFrame as a CSV file
            df.to_csv(file_path, index=False)
            
            logger.info(f"Data successfully saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise CustomException("Data saving failed", e)

    def process(self):
        """
        Main method to process the train and test datasets.
        """
        try:
            # Load datasets
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            # Preprocess datasets
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            # Balance datasets
            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            # Select top features
            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully")

        except Exception as e:
            logger.error(f"Unexpected error in data processing: {e}")
            raise CustomException("Data processing failed", e)

if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()