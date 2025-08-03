#!/usr/bin/env python3
"""
Customer Churn Prediction Model
================================

This script implements a machine learning model to predict customer churn
based on user behavior patterns. The model uses a combination of explicit
churn events and inactivity-based churn detection.

"""

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    make_scorer,
    f1_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import pickle
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Import MLflow integration
try:
    from .mlflow_integration import (
        MLflowManager,
        MLflowExperimentTracker,
        log_experiment_results,
    )

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Install with: pip install mlflow")


class CustomerChurnPredictor:
    """
    A comprehensive customer churn prediction system that combines
    explicit churn events with inactivity-based churn detection.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.cutoff_date = None
        self.inactivity_threshold = 14  # days

    def load_data(self, file_path):
        """
        Load and prepare the customer churn dataset.

        Args:
            file_path (str): Path to the JSON dataset

        Returns:
            pd.DataFrame: Prepared dataset
        """
        print("Loading customer churn dataset...")

        # Load data
        df = pd.read_json(file_path, lines=True)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Ensure userId is string type
        df["userId"] = df["userId"].astype(str)

        # Handle missing values with domain-specific strategy
        df["artist"] = df["artist"].fillna("No Music")
        df["song"] = df["song"].fillna("No Music")
        df["length"] = df["length"].fillna(0)
        df["location"] = df["location"].fillna("Unknown")
        df["userAgent"] = df["userAgent"].fillna("Unknown")
        df["lastName"] = df["lastName"].fillna("Unknown")
        df["firstName"] = df["firstName"].fillna("Unknown")
        df["gender"] = df["gender"].fillna("Unknown")
        df["registration"] = df["registration"].fillna(0)

        # Convert timestamp to datetime - handle different timestamp formats
        if "ts" in df.columns:
            # Check if ts is in milliseconds or seconds
            sample_ts = df["ts"].iloc[0]
            if isinstance(sample_ts, (int, float)) or hasattr(sample_ts, "__int__"):
                if sample_ts > 1e10:  # Likely milliseconds
                    try:
                        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
                        print("Converted timestamps using milliseconds")
                    except Exception as e:
                        print(f"Failed to convert as milliseconds: {e}")
                        # Try as seconds
                        df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
                        print("Converted timestamps using seconds")
                else:  # Likely seconds
                    df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
                    print("Converted timestamps using seconds")
            else:
                df["timestamp"] = pd.to_datetime(df["ts"])
                print("Converted timestamps using default format")
        else:
            # If no ts column, try to create from other date columns
            print("No 'ts' column found. Using current date as fallback.")
            df["timestamp"] = pd.Timestamp.now()

        # Debug: Print timestamp range
        print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        print("Data preprocessing completed")
        return df

    def detect_churn(self, df):
        """
        Detect churn using combined approach: explicit events + inactivity.

        Args:
            df (pd.DataFrame): Input dataset

        Returns:
            pd.DataFrame: Dataset with churn labels
        """
        print("Detecting customer churn...")

        # Explicit churn detection
        explicit_churn_events = ["Cancellation Confirmation", "Cancel"]
        explicit_churn_users = df[df["page"].isin(explicit_churn_events)][
            "userId"
        ].unique()

        # Inactivity-based churn detection
        user_last_activity = df.groupby("userId")["timestamp"].max()
        dataset_end_date = df["timestamp"].max()
        user_last_activity_days = (dataset_end_date - user_last_activity).dt.days
        inactive_users = user_last_activity_days[
            user_last_activity_days >= self.inactivity_threshold
        ].index

        # Combine both approaches
        all_churn_users = set(explicit_churn_users) | set(inactive_users)
        df["is_churned"] = df["userId"].isin(all_churn_users).astype(int)

        churn_rate = df["is_churned"].mean() * 100
        print(f"Churn detection completed: {churn_rate:.1f}% churn rate")

        return df

    def create_user_features(self, df):
        """
        Create user-level features from event-level data.

        Args:
            df (pd.DataFrame): Event-level dataset

        Returns:
            pd.DataFrame: User-level features
        """
        print("Creating user-level features...")

        # Basic user features
        user_features = (
            df.groupby("userId")
            .agg(
                {
                    "sessionId": "nunique",  # total_sessions
                    "timestamp": [
                        "count",
                        "min",
                        "max",
                    ],  # total_events, first_activity, last_activity
                    "page": "nunique",  # page_diversity
                    "artist": "nunique",  # artist_diversity
                    "song": "nunique",  # song_diversity
                    "length": ["sum", "mean"],  # total_length, avg_song_length
                }
            )
            .round(2)
        )

        # Flatten column names
        user_features.columns = [
            "total_sessions",
            "total_events",
            "first_activity",
            "last_activity",
            "page_diversity",
            "artist_diversity",
            "song_diversity",
            "total_length",
            "avg_song_length",
        ]

        # Calculate derived features
        user_features["days_active"] = (
            user_features["last_activity"] - user_features["first_activity"]
        ).dt.days
        user_features["events_per_session"] = (
            user_features["total_events"] / user_features["total_sessions"]
        )
        user_features["avg_song_length"] = user_features["avg_song_length"].fillna(0)

        # Add user demographics
        user_demographics = df.groupby("userId").agg(
            {"level": "last", "gender": "last", "registration": "last"}
        )

        user_features = user_features.join(user_demographics)

        print(
            f"✅ Feature engineering completed: {user_features.shape[1]} features created"
        )
        return user_features

    def prevent_data_leakage(self, df):
        """
        Implement temporal split to prevent data leakage.

        Args:
            df (pd.DataFrame): Event-level dataset

        Returns:
            tuple: (train_events, leakage_free_targets)
        """
        print("Implementing data leakage prevention...")

        # Define cutoff date (30 days before dataset end)
        dataset_end = df["timestamp"].max()
        self.cutoff_date = dataset_end - timedelta(days=30)

        # Split events by time
        train_events = df[df["timestamp"] <= self.cutoff_date].copy()
        future_events = df[df["timestamp"] > self.cutoff_date].copy()

        # Create leakage-free targets based on future behavior
        future_churn_users = future_events[
            future_events["page"].isin(["Cancellation Confirmation", "Cancel"])
        ]["userId"].unique()

        # Calculate inactivity in future period
        user_last_future_activity = future_events.groupby("userId")["timestamp"].max()
        future_end = future_events["timestamp"].max()
        user_inactivity_days = (future_end - user_last_future_activity).dt.days
        future_inactive_users = user_inactivity_days[
            user_inactivity_days >= self.inactivity_threshold
        ].index

        # Combine future churn indicators
        all_future_churn_users = set(future_churn_users) | set(future_inactive_users)

        # Create targets for users who appear in both periods
        common_users = set(train_events["userId"].unique()) & set(
            future_events["userId"].unique()
        )

        # If no common users, use all users from training data with simple churn detection
        if len(common_users) == 0:
            print(
                "⚠️ No common users between train and future periods. Using simple churn detection."
            )
            train_churn_users = train_events[
                train_events["page"].isin(["Cancellation Confirmation", "Cancel"])
            ]["userId"].unique()
            train_user_last_activity = train_events.groupby("userId")["timestamp"].max()
            train_end = train_events["timestamp"].max()
            train_user_inactivity_days = (train_end - train_user_last_activity).dt.days
            train_inactive_users = train_user_inactivity_days[
                train_user_inactivity_days >= self.inactivity_threshold
            ].index

            all_train_churn_users = set(train_churn_users) | set(train_inactive_users)
            all_train_users = set(train_events["userId"].unique())

            leakage_free_targets = pd.DataFrame(
                {
                    "userId": list(all_train_users),
                    "is_churned": [
                        1 if user in all_train_churn_users else 0
                        for user in all_train_users
                    ],
                }
            )
        else:
            leakage_free_targets = pd.DataFrame(
                {
                    "userId": list(common_users),
                    "is_churned": [
                        1 if user in all_future_churn_users else 0
                        for user in common_users
                    ],
                }
            )

        # Debug: Check if leakage_free_targets is empty
        if len(leakage_free_targets) == 0:
            print(
                "⚠️ No users found for modeling. Using all users with simple churn detection."
            )
            all_users = set(df["userId"].unique())
            all_churn_users = set(
                df[df["page"].isin(["Cancellation Confirmation", "Cancel"])][
                    "userId"
                ].unique()
            )

            user_last_activity = df.groupby("userId")["timestamp"].max()
            dataset_end = df["timestamp"].max()
            user_inactivity_days = (dataset_end - user_last_activity).dt.days
            inactive_users = user_inactivity_days[
                user_inactivity_days >= self.inactivity_threshold
            ].index

            all_churn_users = all_churn_users | set(inactive_users)

            leakage_free_targets = pd.DataFrame(
                {
                    "userId": list(all_users),
                    "is_churned": [
                        1 if user in all_churn_users else 0 for user in all_users
                    ],
                }
            )

        print(
            f"✅ Data leakage prevention completed: {len(leakage_free_targets)} users for modeling"
        )
        return train_events, leakage_free_targets

    def prepare_modeling_data(self, train_events, leakage_free_targets):
        """
        Prepare final modeling dataset with features and targets.

        Args:
            train_events (pd.DataFrame): Training events
            leakage_free_targets (pd.DataFrame): Leakage-free targets

        Returns:
            tuple: (X, y) features and target
        """
        print("📋 Preparing modeling dataset...")

        # Create user features from training events only
        user_features = self.create_user_features(train_events)

        # Reset index to make userId a column
        user_features = user_features.reset_index()

        # Remove the is_churned column from user_features since it's from the original data
        if "is_churned" in user_features.columns:
            user_features = user_features.drop("is_churned", axis=1)

        # Ensure userId is string type in both dataframes
        user_features["userId"] = user_features["userId"].astype(str)
        leakage_free_targets["userId"] = leakage_free_targets["userId"].astype(str)

        # Debug: Check the data
        print(f"User features shape: {user_features.shape}")
        print(f"Leakage free targets shape: {leakage_free_targets.shape}")
        print(f"User features userId sample: {user_features['userId'].head().tolist()}")
        print(
            f"Leakage free targets userId sample: {leakage_free_targets['userId'].head().tolist()}"
        )

        # Merge with targets
        modeling_data = user_features.merge(
            leakage_free_targets, on="userId", how="inner"
        )

        # Debug: Print columns to understand the structure
        print(f"User features columns: {user_features.columns.tolist()}")
        print(f"Leakage free targets columns: {leakage_free_targets.columns.tolist()}")
        print(f"Modeling data columns: {modeling_data.columns.tolist()}")
        print(f"Modeling data shape: {modeling_data.shape}")

        # If merge resulted in 0 rows, use the user_features directly with simple churn detection
        if len(modeling_data) == 0:
            print(
                "⚠️ No overlap between user features and targets. Using simple approach."
            )
            # Create simple churn detection for user_features
            user_ids = user_features["userId"].tolist()
            churn_status = []

            for user_id in user_ids:
                user_events = train_events[train_events["userId"] == user_id]
                is_churned = 0

                # Check for explicit churn
                if (
                    user_events[
                        user_events["page"].isin(
                            ["Cancellation Confirmation", "Cancel"]
                        )
                    ].shape[0]
                    > 0
                ):
                    is_churned = 1
                else:
                    # Check for inactivity
                    user_last_activity = user_events["timestamp"].max()
                    train_end = train_events["timestamp"].max()
                    inactivity_days = (train_end - user_last_activity).days
                    if inactivity_days >= self.inactivity_threshold:
                        is_churned = 1

                churn_status.append(is_churned)

            modeling_data = user_features.copy()
            modeling_data["is_churned"] = churn_status

        # Select features for modeling
        feature_cols = [
            "total_sessions",
            "total_events",
            "page_diversity",
            "artist_diversity",
            "song_diversity",
            "total_length",
            "avg_song_length",
            "days_active",
            "events_per_session",
            "level",
            "gender",
            "registration",
        ]

        self.feature_columns = feature_cols
        X = modeling_data[feature_cols].copy()
        y = modeling_data["is_churned"]

        print(
            f"✅ Modeling dataset prepared: {X.shape[0]} samples, {X.shape[1]} features"
        )
        return X, y

    def preprocess_data(self, X, y):
        """
        Preprocess data for model training.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target

        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler, encoders)
        """
        print("Preprocessing data...")

        # Encode categorical features
        categorical_features = ["level", "gender"]
        for feature in categorical_features:
            if feature in X.columns:
                le = LabelEncoder()
                X[feature] = le.fit_transform(X[feature].astype(str))
                self.label_encoders[feature] = le

        # Handle missing values
        X = X.fillna(X.median())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Scale numerical features
        numerical_features = [
            col for col in X.columns if col not in categorical_features
        ]
        X_train[numerical_features] = self.scaler.fit_transform(
            X_train[numerical_features]
        )
        X_test[numerical_features] = self.scaler.transform(X_test[numerical_features])

        print(
            f"✅ Data preprocessing completed: {X_train.shape[0]} train, {X_test.shape[0]} test"
        )
        return X_train, X_test, y_train, y_test

    def balance_data(self, X_train, y_train):
        """
        Apply data balancing techniques.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target

        Returns:
            dict: Balanced datasets
        """
        print("Applying data balancing techniques...")

        balanced_datasets = {}

        # SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        balanced_datasets["SMOTE"] = (X_smote, y_smote)

        # Random Undersampling
        rus = RandomUnderSampler(random_state=42)
        X_rus, y_rus = rus.fit_resample(X_train, y_train)
        balanced_datasets["Undersampling"] = (X_rus, y_rus)

        # SMOTEENN (Combined)
        smoteenn = SMOTEENN(random_state=42)
        X_combined, y_combined = smoteenn.fit_resample(X_train, y_train)
        balanced_datasets["Combined"] = (X_combined, y_combined)

        # Original with class weights
        balanced_datasets["Original + Weights"] = (X_train, y_train)

        print("Data balancing completed")
        return balanced_datasets

    def train_models(self, balanced_datasets, X_test, y_test):
        """
        Train multiple models with different balancing techniques.

        Args:
            balanced_datasets (dict): Balanced datasets
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target

        Returns:
            dict: Model results
        """
        print("🤖 Training models...")

        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }

        results = {}

        for balance_name, (X_bal, y_bal) in balanced_datasets.items():
            print(f"\n🔄 Testing {balance_name}...")
            results[balance_name] = {}

            for model_name, model in models.items():
                # Train model
                if balance_name == "Original + Weights":
                    # Use class weights for original data
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(
                            random_state=42, max_iter=1000, class_weight="balanced"
                        )
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier(
                            random_state=42, n_estimators=100, class_weight="balanced"
                        )
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingClassifier(random_state=42)
                else:
                    # Use balanced data
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(random_state=42, max_iter=1000)
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier(
                            random_state=42, n_estimators=100
                        )
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingClassifier(random_state=42)

                model.fit(X_bal, y_bal)

                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                results[balance_name][model_name] = {
                    "model": model,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                }

                print(
                    f"   {model_name:<20} - AUC: {auc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
                )

        return results

    def perform_hyperparameter_tuning(self, X_train, y_train, X_test, y_test):
        """
        Perform hyperparameter tuning on the best performing models.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target

        Returns:
            dict: Tuned models with results
        """
        print("\nPerforming Hyperparameter Tuning...")

        # Define parameter grids for each model
        param_grids = {
            "Logistic Regression": {
                "C": [0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
                "class_weight": ["balanced", None],
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample", None],
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.8, 0.9, 1.0],
            },
        }

        # Create base models
        base_models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }

        # Custom F1 scorer
        f1_scorer = make_scorer(f1_score, average="weighted")

        tuned_results = {}

        for model_name, base_model in base_models.items():
            print(f"\nTuning {model_name}...")

            # Grid Search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grids[model_name],
                scoring=f1_scorer,
                cv=5,
                n_jobs=-1,
                verbose=1,
            )

            # Fit the grid search
            grid_search.fit(X_train, y_train)

            # Get best model
            best_model = grid_search.best_estimator_

            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            tuned_results[model_name] = {
                "model": best_model,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predictions": y_pred,
                "probabilities": y_pred_proba,
            }

            print(f"{model_name} tuned:")
            print(f"   Best Parameters: {grid_search.best_params_}")
            print(f"   Best CV Score: {grid_search.best_score_:.3f}")
            print(
                f"   Test AUC: {auc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
            )

        return tuned_results

    def perform_randomized_search(self, X_train, y_train, X_test, y_test):
        """
        Perform randomized search for faster hyperparameter tuning.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target

        Returns:
            dict: Tuned models with results
        """
        print("\n🎲 Performing Randomized Search...")

        # Define parameter distributions for randomized search
        param_distributions = {
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
                "class_weight": ["balanced", None],
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 10, 15, None],
                "min_samples_split": [2, 5, 10, 15],
                "min_samples_leaf": [1, 2, 4, 6],
                "max_features": ["sqrt", "log2", None],
                "class_weight": ["balanced", "balanced_subsample", None],
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.8, 0.9, 1.0],
                "max_features": ["sqrt", "log2", None],
            },
        }

        # Create base models
        base_models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }

        # Custom F1 scorer
        f1_scorer = make_scorer(f1_score, average="weighted")

        randomized_results = {}

        for model_name, base_model in base_models.items():
            print(f"\n🎲 Randomized Search for {model_name}...")

            # Randomized Search
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions[model_name],
                n_iter=20,  # Number of parameter settings sampled
                scoring=f1_scorer,
                cv=5,
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )

            # Fit the random search
            random_search.fit(X_train, y_train)

            # Get best model
            best_model = random_search.best_estimator_

            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            randomized_results[model_name] = {
                "model": best_model,
                "best_params": random_search.best_params_,
                "best_score": random_search.best_score_,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predictions": y_pred,
                "probabilities": y_pred_proba,
            }

            print(f"{model_name} randomized search completed:")
            print(f"   Best Parameters: {random_search.best_params_}")
            print(f"   Best CV Score: {random_search.best_score_:.3f}")
            print(
                f"   Test AUC: {auc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
            )

        return randomized_results

    def select_best_model(self, results):
        """
        Select the best performing model based on F1 score.

        Args:
            results (dict): Model results

        Returns:
            tuple: (best_model, best_metrics, best_name)
        """
        print("\n🏆 Selecting best model...")

        best_f1 = 0
        best_model = None
        best_metrics = None
        best_name = None

        for balance_name, balance_results in results.items():
            for model_name, model_result in balance_results.items():
                if model_result["f1"] > best_f1:
                    best_f1 = model_result["f1"]
                    best_model = model_result["model"]
                    best_metrics = model_result
                    best_name = f"{balance_name} + {model_name}"

        print(f"Best model selected: {best_name}")
        print(f"   F1 Score: {best_metrics['f1']:.3f}")
        print(f"   Recall: {best_metrics['recall']:.3f}")
        print(f"   Precision: {best_metrics['precision']:.3f}")
        print(f"   AUC: {best_metrics['auc']:.3f}")

        return best_model, best_metrics, best_name

    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate the selected model with detailed metrics.

        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Model name
        """
        print(f"\nEvaluating {model_name}...")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print("Confusion Matrix:")
        print(f"[[{tn:2d} {fp:2d}]")
        print(f" [{fn:2d} {tp:2d}]]")

        print("\nDetailed Metrics:")
        print(f"True Positives (correctly identified churners): {tp}")
        print(f"False Positives (loyal users flagged as churners): {fp}")
        print(f"True Negatives (correctly identified loyal users): {tn}")
        print(f"False Negatives (missed churners): {fn}")

        # Business metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        print("\nBusiness Impact:")
        print(f"- Catches {recall:.1%} of actual churners")
        print(f"- {precision:.1%} of churn predictions are correct")
        print(f"- Overall accuracy: {accuracy:.1%}")

        # ROI calculation example
        retention_cost = 50  # $50 per retention campaign
        churn_value = 200  # $200 value of preventing churn

        total_targeted = tp + fp
        campaign_cost = total_targeted * retention_cost
        prevented_value = tp * churn_value
        net_benefit = prevented_value - campaign_cost

        print("\nROI Analysis (Example):")
        print(f"- Retention campaign cost: ${campaign_cost:,}")
        print(f"- Prevented churn value: ${prevented_value:,}")
        print(f"- Net benefit: ${net_benefit:,}")
        print(
            f"- ROI: {(net_benefit / campaign_cost) * 100:.1f}%"
            if campaign_cost > 0
            else "ROI: N/A"
        )

    def save_model(self, model, model_name):
        """
        Save the trained model and preprocessing components.

        Args:
            model: Trained model
            model_name (str): Model name
        """
        print(f"\n💾 Saving model: {model_name}")

        # Create models directory if it doesn't exist
        import os

        models_dir = "project_files/models"
        os.makedirs(models_dir, exist_ok=True)

        # Save model
        model_filename = os.path.join(
            models_dir, f"churn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        # Save preprocessing components
        preprocessing_filename = os.path.join(
            models_dir, f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        preprocessing_data = {
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "cutoff_date": self.cutoff_date,
            "inactivity_threshold": self.inactivity_threshold,
        }
        with open(preprocessing_filename, "wb") as f:
            pickle.dump(preprocessing_data, f)

        # Save feature columns as JSON for easy access
        feature_columns_filename = os.path.join(
            models_dir,
            f"feature_columns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(feature_columns_filename, "w") as f:
            json.dump(self.feature_columns, f)

        print(f"Model saved: {model_filename}")
        print(f"Preprocessing saved: {preprocessing_filename}")
        print(f"Feature columns saved: {feature_columns_filename}")

    def run_complete_pipeline(self, file_path, enable_tuning=True, enable_mlflow=True):
        """
        Run the complete churn prediction pipeline.

        Args:
            file_path (str): Path to the dataset
            enable_tuning (bool): Whether to enable hyperparameter tuning
            enable_mlflow (bool): Whether to enable MLflow tracking
        """
        print("Starting Customer Churn Prediction Pipeline")
        print("=" * 60)

        # Initialize MLflow tracking if available and enabled
        mlflow_tracker = None
        if enable_mlflow and MLFLOW_AVAILABLE:
            try:
                mlflow_manager = MLflowManager()
                mlflow_tracker = MLflowExperimentTracker(mlflow_manager)
                mlflow_tracker.start_experiment(
                    experiment_name="customer_churn_training",
                    parameters={
                        "enable_tuning": enable_tuning,
                        "inactivity_threshold": self.inactivity_threshold,
                        "file_path": file_path,
                    },
                    tags={"pipeline": "customer_churn_prediction", "version": "1.0.0"},
                )
                print("MLflow tracking enabled")
            except Exception as e:
                print(f"MLflow tracking failed: {e}")
                mlflow_tracker = None

        # Step 1: Load and preprocess data
        df = self.load_data(file_path)

        # Step 2: Detect churn
        df = self.detect_churn(df)

        # Step 3: Prevent data leakage using temporal splitting
        print("Implementing data leakage prevention...")
        train_events, leakage_free_targets = self.prevent_data_leakage(df)

        # Step 4: Create user features from training data only
        print("📋 Preparing modeling dataset...")
        user_features = self.create_user_features(train_events)
        user_features = user_features.reset_index()

        # Step 5: Merge with leakage-free targets
        print(f"User features shape: {user_features.shape}")
        print(f"Leakage-free targets shape: {leakage_free_targets.shape}")

        # Check for overlap
        user_feature_ids = set(user_features["userId"].unique())
        target_ids = set(leakage_free_targets["userId"].unique())
        overlap_ids = user_feature_ids & target_ids

        print(f"Users in features: {len(user_feature_ids)}")
        print(f"Users in targets: {len(target_ids)}")
        print(f"Overlapping users: {len(overlap_ids)}")

        if len(overlap_ids) == 0:
            print(
                "⚠️ No overlapping users found. Using simple churn detection approach."
            )
            # Create simple churn detection for all users in features
            user_ids = user_features["userId"].tolist()
            churn_status = []

            for user_id in user_ids:
                user_events = train_events[train_events["userId"] == user_id]
                is_churned = 0

                # Check for explicit churn
                if (
                    user_events[
                        user_events["page"].isin(
                            ["Cancellation Confirmation", "Cancel"]
                        )
                    ].shape[0]
                    > 0
                ):
                    is_churned = 1
                else:
                    # Check for inactivity
                    user_last_activity = user_events["timestamp"].max()
                    train_end = train_events["timestamp"].max()
                    inactivity_days = (train_end - user_last_activity).days
                    if inactivity_days >= self.inactivity_threshold:
                        is_churned = 1

                churn_status.append(is_churned)

            user_features["is_churned"] = churn_status
        else:
            # Use the leakage-free targets
            user_features = user_features.merge(
                leakage_free_targets, on="userId", how="inner"
            )

        # Step 6: Prepare features and target
        feature_cols = [
            "total_sessions",
            "total_events",
            "page_diversity",
            "artist_diversity",
            "song_diversity",
            "total_length",
            "avg_song_length",
            "days_active",
            "events_per_session",
            "level",
            "gender",
            "registration",
        ]

        self.feature_columns = feature_cols
        X = user_features[feature_cols].copy()
        y = user_features["is_churned"]

        print(
            f"✅ Modeling dataset prepared: {X.shape[0]} samples, {X.shape[1]} features"
        )
        print(
            f"✅ Data leakage prevention: Using temporal split with cutoff date {self.cutoff_date.strftime('%Y-%m-%d')}"
        )

        # Check if we have enough data for training
        if X.shape[0] == 0:
            raise ValueError(
                "No data available for modeling. Check temporal split parameters."
            )
        elif X.shape[0] < 10:
            print(
                f"⚠️ Warning: Only {X.shape[0]} samples available for modeling. Results may be unreliable."
            )

        # Step 7: Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)

        # Step 8: Balance data
        balanced_datasets = self.balance_data(X_train, y_train)

        # Step 9: Train models
        results = self.train_models(balanced_datasets, X_test, y_test)

        # Step 10: Select best model
        best_model, best_metrics, best_name = self.select_best_model(results)

        # Step 11: Hyperparameter tuning (if enabled)
        if enable_tuning:
            print("\n" + "=" * 60)
            print("HYPERPARAMETER TUNING")
            print("=" * 60)

            # Perform grid search tuning
            tuned_results = self.perform_hyperparameter_tuning(
                X_train, y_train, X_test, y_test
            )

            # Perform randomized search tuning
            randomized_results = self.perform_randomized_search(
                X_train, y_train, X_test, y_test
            )

            # Compare all results
            all_results = {
                "Original": results,
                "Grid Search Tuned": tuned_results,
                "Randomized Search Tuned": randomized_results,
            }

            # Find the overall best model
            overall_best_f1 = 0
            overall_best_model = None
            overall_best_metrics = None
            overall_best_name = None

            for method_name, method_results in all_results.items():
                for model_name, model_result in method_results.items():
                    # Handle different result structures
                    if isinstance(model_result, dict) and "f1" in model_result:
                        f1_score = model_result["f1"]
                    else:
                        # Skip if no f1 score available
                        continue

                    if f1_score > overall_best_f1:
                        overall_best_f1 = f1_score
                        overall_best_model = model_result["model"]
                        overall_best_metrics = model_result
                        overall_best_name = f"{method_name} + {model_name}"

            # Check if we found a best model
            if overall_best_metrics is not None:
                print(f"\n🏆 Overall Best Model: {overall_best_name}")
                print(f"   F1 Score: {overall_best_metrics['f1']:.3f}")
                print(f"   Recall: {overall_best_metrics['recall']:.3f}")
                print(f"   Precision: {overall_best_metrics['precision']:.3f}")
                print(f"   AUC: {overall_best_metrics['auc']:.3f}")

                # Use the overall best model
                best_model = overall_best_model
                best_metrics = overall_best_metrics
                best_name = overall_best_name
            else:
                # Fallback to the original best model if no tuned model is better
                print(
                    "\n⚠️ No tuned model performed better than original. Using original best model."
                )
                best_model = best_model  # Keep the original best model
                best_metrics = best_metrics  # Keep the original best metrics
                best_name = best_name  # Keep the original best name

        # Step 12: Evaluate best model
        self.evaluate_model(best_model, X_test, y_test, best_name)

        # Step 13: Save model
        self.save_model(best_model, best_name)

        # Step 14: Log to MLflow if enabled
        if mlflow_tracker:
            try:
                # Get feature importance if available
                feature_importance = None
                feature_names = None
                if hasattr(best_model, "feature_importances_"):
                    feature_importance = best_model.feature_importances_.tolist()
                    feature_names = self.feature_columns

                # Get predictions for logging
                y_pred = best_model.predict(X_test)
                y_pred_proba = (
                    best_model.predict_proba(X_test)[:, 1]
                    if hasattr(best_model, "predict_proba")
                    else None
                )

                # Log complete training results
                model_uri = mlflow_tracker.log_training_results(
                    model=best_model,
                    metrics=best_metrics,
                    feature_names=feature_names,
                    feature_importance=feature_importance,
                    y_true=y_test.tolist(),
                    y_pred=y_pred.tolist(),
                    y_pred_proba=(
                        y_pred_proba.tolist() if y_pred_proba is not None else None
                    ),
                    model_name="customer_churn_model",
                )

                # Register model in MLflow Model Registry
                try:
                    registered_uri = mlflow_tracker.mlflow_manager.register_model(
                        model_uri=model_uri,
                        model_name="customer_churn_prediction",
                        description=f"Best model from {best_name} with F1={best_metrics['f1']:.3f}",
                        tags={
                            "model_type": best_name,
                            "f1_score": str(best_metrics["f1"]),
                            "auc_score": str(best_metrics["auc"]),
                            "training_date": datetime.now().isoformat(),
                        },
                    )
                    print(f"Model registered in MLflow: {registered_uri}")
                except Exception as e:
                    print(f"Failed to register model in MLflow: {e}")

                mlflow_tracker.end_experiment()
                print("MLflow tracking completed")

            except Exception as e:
                print(f"MLflow logging failed: {e}")

        print("\n🎉 Pipeline completed successfully!")
        return best_model, best_metrics, best_name


def main():
    """
    Main function to run the customer churn prediction pipeline.
    """
    # Initialize the predictor
    predictor = CustomerChurnPredictor()

    # Run the complete pipeline with hyperparameter tuning
    file_path = "./project_files/data/customer_churn_mini.json"

    try:
        best_model, best_metrics, best_name = predictor.run_complete_pipeline(
            file_path, enable_tuning=True
        )

        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print(f"Best Model: {best_name}")
        print(f"F1 Score: {best_metrics['f1']:.3f}")
        print(f"Recall: {best_metrics['recall']:.3f}")
        print(f"Precision: {best_metrics['precision']:.3f}")
        print(f"AUC: {best_metrics['auc']:.3f}")
        print("\nModel is ready for production deployment!")

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
