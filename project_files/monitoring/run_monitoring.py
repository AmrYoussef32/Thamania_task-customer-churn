"""
Run Monitoring and Drift Detection for Customer Churn Prediction.
Demonstrates the monitoring system with sample data.
"""

import sys
import os
sys.path.append("./project_files/src")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from customer_churn_prediction import CustomerChurnPredictor
from .monitoring_orchestrator import MonitoringOrchestrator


def run_monitoring_demo():
    """Run monitoring demonstration with sample data."""
    print("🔍 Monitoring and Drift Detection Demo")
    print("=" * 50)
    
    # Initialize predictor and load data
    predictor = CustomerChurnPredictor()
    df = predictor.load_data()
    
    # Prepare reference data
    print("📊 Preparing reference data...")
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
    
    # Split data for reference and current
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train reference model
    print("🤖 Training reference model...")
    reference_model = RandomForestClassifier(random_state=42)
    reference_model.fit(X_train, y_train)
    
    # Get reference performance
    y_pred_ref = reference_model.predict(X_test)
    reference_f1 = f1_score(y_test, y_pred_ref, average='weighted')
    print(f"📈 Reference F1 Score: {reference_f1:.4f}")
    
    # Initialize monitoring orchestrator
    print("🔧 Setting up monitoring system...")
    orchestrator = MonitoringOrchestrator(
        reference_data=X_train,
        reference_performance=reference_f1
    )
    
    # Simulate current data with some drift
    print("\n🔄 Simulating current data with drift...")
    
    # Create drifted current data
    current_data = X_test.copy()
    
    # Simulate data drift by modifying some features
    np.random.seed(42)
    drift_factor = 0.3
    
    # Add some noise to simulate drift
    for col in ['total_sessions', 'total_events', 'total_length']:
        current_data[col] = current_data[col] * (1 + np.random.normal(0, drift_factor, len(current_data)))
        current_data[col] = current_data[col].abs()  # Ensure positive values
    
    # Get predictions on current data
    y_pred_current = reference_model.predict(current_data)
    y_pred_proba_current = reference_model.predict_proba(current_data)[:, 1]
    
    # Run monitoring check
    print("\n🔍 Running comprehensive monitoring check...")
    monitoring_results = orchestrator.run_monitoring_check(
        current_data=current_data,
        y_true=y_test.tolist(),
        y_pred=y_pred_current.tolist(),
        y_pred_proba=y_pred_proba_current.tolist(),
        metadata={'demo': True, 'drift_simulation': True}
    )
    
    # Export reports
    print("\n📊 Exporting monitoring reports...")
    orchestrator.export_monitoring_report()
    orchestrator.performance_monitor.export_performance_report()
    
    # Show monitoring summary
    print("\n📋 Monitoring Summary (Last 30 days)")
    print("-" * 40)
    summary = orchestrator.get_monitoring_summary(30)
    
    if 'error' not in summary:
        print(f"Total monitoring checks: {summary['total_checks']}")
        print(f"Data drift incidents: {summary['drift_incidents']['data_drift']}")
        print(f"Concept drift incidents: {summary['drift_incidents']['concept_drift']}")
        print(f"Performance degradation incidents: {summary['drift_incidents']['performance_degradation']}")
        print(f"Average F1 Score: {summary['average_performance']['f1_score']:.4f}")
        print(f"Average Accuracy: {summary['average_performance']['accuracy']:.4f}")
    else:
        print(f"Summary: {summary['error']}")
    
    print("\n✅ Monitoring demo completed!")
    print("📁 Check project_files/monitoring/ for detailed reports")


def run_scheduled_monitoring():
    """Run scheduled monitoring check."""
    print("🕐 Running Scheduled Monitoring Check")
    print("=" * 40)
    
    # This would typically be called by a scheduler (cron, Windows Task Scheduler, etc.)
    # For demo purposes, we'll simulate it
    
    # Load current data (in real scenario, this would be new data)
    predictor = CustomerChurnPredictor()
    df = predictor.load_data()
    
    # Prepare data (simplified for demo)
    df = predictor.detect_churn(df)
    user_features = predictor.create_user_features(df)
    user_features = user_features.reset_index()
    
    # Simple churn detection for demo
    user_ids = user_features['userId'].tolist()
    churn_status = []
    
    for user_id in user_ids:
        user_events = df[df['userId'] == user_id]
        is_churned = 0
        
        if user_events[user_events['page'].isin(['Cancellation Confirmation', 'Cancel'])].shape[0] > 0:
            is_churned = 1
        else:
            user_last_activity = user_events['timestamp'].max()
            train_end = df['timestamp'].max()
            inactivity_days = (train_end - user_last_activity).days
            if inactivity_days >= predictor.inactivity_threshold:
                is_churned = 1
        
        churn_status.append(is_churned)
    
    user_features['is_churned'] = churn_status
    
    # Prepare features
    feature_cols = [
        'total_sessions', 'total_events', 'page_diversity', 'artist_diversity',
        'song_diversity', 'total_length', 'avg_song_length', 'days_active',
        'events_per_session', 'level', 'gender', 'registration'
    ]
    
    X = user_features[feature_cols].copy()
    y = user_features['is_churned']
    
    # Load reference model (in real scenario, this would be the deployed model)
    try:
        import joblib
        reference_model = joblib.load("project_files/models/best_model.pkl")
    except:
        print("⚠️ Reference model not found. Using simple classifier for demo.")
        from sklearn.ensemble import RandomForestClassifier
        reference_model = RandomForestClassifier(random_state=42)
        reference_model.fit(X, y)
    
    # Get predictions
    y_pred = reference_model.predict(X)
    y_pred_proba = reference_model.predict_proba(X)[:, 1]
    
    # Initialize monitoring (in real scenario, this would be initialized once)
    # For demo, we'll create a simple reference
    reference_f1 = 0.75  # Example reference performance
    
    orchestrator = MonitoringOrchestrator(
        reference_data=X.sample(frac=0.7, random_state=42),  # Use 70% as reference
        reference_performance=reference_f1
    )
    
    # Run monitoring check
    monitoring_results = orchestrator.run_monitoring_check(
        current_data=X,
        y_true=y.tolist(),
        y_pred=y_pred.tolist(),
        y_pred_proba=y_pred_proba.tolist(),
        metadata={'scheduled_check': True, 'timestamp': pd.Timestamp.now().isoformat()}
    )
    
    print("✅ Scheduled monitoring check completed!")
    return monitoring_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run monitoring and drift detection")
    parser.add_argument("--mode", choices=["demo", "scheduled"], default="demo",
                       help="Monitoring mode: demo or scheduled")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_monitoring_demo()
    else:
        run_scheduled_monitoring() 