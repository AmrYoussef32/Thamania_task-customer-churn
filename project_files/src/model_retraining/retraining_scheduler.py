#!/usr/bin/env python3
"""
Model Retraining Scheduler

Simple scheduler to automate model retraining based on performance and drift detection.
"""

import sys
import os
import time
from datetime import datetime, timedelta
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model_monitoring import ModelMonitor
from model_retraining import ModelRetrainer


def run_scheduled_check():
    """Run scheduled retraining check."""
    try:
        monitor = ModelMonitor()
        retrainer = ModelRetrainer()
        
        # Check if retraining is needed
        retraining_needed, reasons = monitor.check_if_retraining_needed()
        
        if retraining_needed:
            print(f"Retraining needed: {reasons}")
            
            # Perform retraining
            result = retrainer.retrain_model()
            
            if result['success']:
                print("Scheduled retraining completed successfully")
            else:
                print("Scheduled retraining failed")
        else:
            print("No retraining needed - model performing well")
            
    except Exception as e:
        print(f"Error in scheduled check: {e}")


def run_weekly_retraining():
    """Run weekly retraining regardless of performance."""
    try:
        retrainer = ModelRetrainer()
        result = retrainer.force_retrain()
        
        if result['success']:
            print("Weekly retraining completed successfully")
        else:
            print("Weekly retraining failed")
            
    except Exception as e:
        print(f"Error in weekly retraining: {e}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Model Retraining Scheduler')
    parser.add_argument('command', choices=['scheduled', 'weekly', 'monitor'], 
                       help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'scheduled':
        print("Running scheduled retraining check...")
        run_scheduled_check()
        
    elif args.command == 'weekly':
        print("Running weekly retraining...")
        run_weekly_retraining()
        
    elif args.command == 'monitor':
        print("Running performance monitoring...")
        monitor = ModelMonitor()
        monitor.check_if_retraining_needed()
        
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
