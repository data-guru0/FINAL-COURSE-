import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from scipy.stats import randint
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import load_data


# Initialize logger
logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.param_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading training data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading testing data from {self.test_path}")
            test_df = load_data(self.test_path)

            # Ensure columns are consistent
            common_columns = train_df.columns.intersection(test_df.columns)
            train_df = train_df[common_columns]
            test_df = test_df[common_columns]

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load data", e)

    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing LGBM model")
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])

            logger.info("Starting RandomizedSearchCV for hyperparameter tuning")
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.param_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )

            logger.info("Fitting RandomizedSearchCV to the training data")
            random_search.fit(X_train, y_train)

            logger.info("Hyperparameter tuning completed successfully")
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best Hyperparameters: {best_params}")
            return best_lgbm_model
        except Exception as e:
            logger.error(f"Error during RandomizedSearchCV: {e}")
            raise CustomException("Failed during RandomizedSearchCV", e)
        
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating the model on the test set")
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")

            return {
                "accuracy": accuracy,
                "recall": recall,
                "precision": precision,
                "f1": f1
            }
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException("Model evaluation failed", e)

    def save_model(self, model):
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info(f"Saving model to {self.model_output_path}")
            joblib.dump(model, self.model_output_path)
            logger.info("Model saved successfully!")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException("Model saving failed", e)

    def run(self):
        try:
            # Start MLflow run
            with mlflow.start_run():
                logger.info("Starting experiment with MLflow")

                # Log the dataset as an artifact (this can be either train or test dataset)
                logger.info("Logging the training dataset as an artifact")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")

                # Log the test dataset
                logger.info("Logging the testing dataset as an artifact")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                # Load data, train the model, evaluate and save it
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)

                mlflow.log_artifact(self.model_output_path)

                # Log model parameters and metrics
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Logging completed successfully!")

                return metrics

        except Exception as e:
            logger.error(f"Unexpected error in training pipeline: {e}")
            raise CustomException("Training pipeline failed", e)

if __name__ == "__main__":
    try:     
        trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
        metrics = trainer.run()

    except Exception as e:
        logger.error(f"Fatal error in training script: {e}")