"""
Run MLflow experiments for Customer Churn Prediction.
Simple experiment runner with different models and parameters.
"""

import sys
import os
sys.path.append("./project_files/src")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

from customer_churn_prediction import CustomerChurnPredictor
from .experiment_tracker import get_experiment_tracker


def run_mlflow_experiments():
    """Run comprehensive MLflow experiments."""
    print("🔬 Starting MLflow Experiments...")
    print("=================================")
    
    # Initialize predictor and load data
    predictor = CustomerChurnPredictor()
    df = predictor.load_data()
    
    # Prepare data
    print("📊 Preparing data...")
    df = predictor.detect_churn(df)
    train_events, leakage_free_targets = predictor.prevent_data_leakage(df)
    user_features = predictor.create_user_features(train_events)
    user_features = user_features.reset_index()
    
    # Merge with targets
    user_feature_ids = set(user_features['userId'].unique())
    target_ids = set(leakage_free_targets['userId'].unique())
    overlap_ids = user_feature_ids & target_ids
    
    if len(overlap_ids) == 0:
        print("⚠️ No overlapping users found. Using simple churn detection.")
        user_ids = user_features['userId'].tolist()
        churn_status = []
        
        for user_id in user_ids:
            user_events = train_events[train_events['userId'] == user_id]
            is_churned = 0
            
            if user_events[user_events['page'].isin(['Cancellation Confirmation', 'Cancel'])].shape[0] > 0:
                is_churned = 1
            else:
                user_last_activity = user_events['timestamp'].max()
                train_end = train_events['timestamp'].max()
                inactivity_days = (train_end - user_last_activity).days
                if inactivity_days >= predictor.inactivity_threshold:
                    is_churned = 1
            
            churn_status.append(is_churned)
        
        user_features['is_churned'] = churn_status
    else:
        user_features = user_features.merge(leakage_free_targets, on='userId', how='inner')
    
    # Prepare features
    feature_cols = [
        'total_sessions', 'total_events', 'page_diversity', 'artist_diversity',
        'song_diversity', 'total_length', 'avg_song_length', 'days_active',
        'events_per_session', 'level', 'gender', 'registration'
    ]
    
    X = user_features[feature_cols].copy()
    y = user_features['is_churned']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"✅ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Initialize experiment tracker
    tracker = get_experiment_tracker()
    
    # Define models and parameters
    models_config = [
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(random_state=42),
            'params': {
                'model_name': 'LogisticRegression',
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
        },
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'model_name': 'RandomForest',
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        {
            'name': 'GradientBoosting',
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'model_name': 'GradientBoosting',
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
        }
    ]
    
    # Run experiments
    results = []
    
    for config in models_config:
        print(f"\n🤖 Running experiment: {config['name']}")
        print("-" * 40)
        
        try:
            run_id, metrics = tracker.log_model_experiment(
                model=config['model'],
                model_name=config['name'],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                params=config['params']
            )
            
            results.append({
                'model_name': config['name'],
                'run_id': run_id,
                'metrics': metrics
            })
            
            print(f"✅ Experiment completed: {run_id}")
            print(f"📊 F1 Score: {metrics['f1_score']:.4f}")
            print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
            print(f"📊 ROC AUC: {metrics['roc_auc']:.4f}")
            
        except Exception as e:
            print(f"❌ Error in {config['name']} experiment: {e}")
    
    # Compare results
    if results:
        print("\n📊 Model Comparison")
        print("==================")
        
        comparison_df = tracker.compare_models(results)
        print(comparison_df[['model_name', 'f1_score', 'accuracy', 'roc_auc']].to_string(index=False))
        
        # Export results
        tracker.export_experiment_results()
        
        # Get best model
        best_model_info = tracker.get_best_model_info()
        if best_model_info:
            print(f"\n🏆 Best Model: {best_model_info['model_name']}")
            print(f"📈 Best F1 Score: {best_model_info['metrics']['f1_score']:.4f}")
            print(f"🆔 Run ID: {best_model_info['run_id']}")
    
    print("\n✅ All experiments completed!")
    print("🔬 View results in MLflow UI: http://localhost:5000")


if __name__ == "__main__":
    run_mlflow_experiments() 