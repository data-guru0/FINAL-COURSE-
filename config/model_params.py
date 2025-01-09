from scipy.stats import randint, uniform

# Hyperparameters for LightGBM model
LIGHTGBM_PARAMS = {
    'n_estimators': randint(100, 500),        # Randomly select between 100 and 500 trees
    'max_depth': randint(5, 50),              # Randomly select depth between 5 and 50
    'learning_rate': uniform(0.01, 0.2),      # Randomly select learning rate between 0.01 and 0.2
    'num_leaves': randint(20, 100),           # Randomly select number of leaves between 20 and 100
    'min_data_in_leaf': randint(10, 50),      # Randomly select min samples in a leaf between 10 and 50
    'subsample': uniform(0.5, 1.0),           # Randomly select subsample ratio between 0.5 and 1
    'colsample_bytree': uniform(0.5, 1.0),    # Randomly select column sample ratio between 0.5 and 1
    'boosting_type': ['gbdt', 'dart', 'goss'] # Randomly select boosting type (Gradient Boosting, DART, GOSS)
}

# RandomizedSearchCV parameters
RANDOM_SEARCH_PARAMS = {
    'n_iter': 4,                   # Number of iterations for random search
    'cv': 2,                       # 3-fold cross-validation
    'n_jobs': -1,                  # Use all available cores
    'verbose': 2,                  # Verbosity of output
    'random_state': 42,            # Random seed for reproducibility
    'scoring': 'accuracy'          # Use accuracy as the scoring metric
}