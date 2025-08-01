"""
Model Tester for Fraud Detection Models
This module provides functionality to test saved fraud detection models
against new transaction data and compare their performance.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Any
from datetime import datetime


class ModelTester:
    """
    A class to test fraud detection models loaded from pickle files.
    
    Attributes:
        models_dir (str): Directory containing the saved models
        metadata (dict): Model metadata including feature names and paths
        models (dict): Dictionary of loaded models
        test_data (pd.DataFrame): Test dataset
        predictions (dict): Model predictions
        results (pd.DataFrame): Performance metrics for each model
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the ModelTester with models directory.
        
        Args:
            models_dir (str): Path to directory containing saved models
        """
        self.models_dir = models_dir
        self.metadata = None
        self.models = {}
        self.test_data = None
        self.predictions = {}
        self.results = None
        self.preprocessor = None
        
        # Load metadata
        self._load_metadata()
        
        # Load preprocessor
        self._load_preprocessor()
        
    def _load_metadata(self):
        """Load model metadata from pickle file."""
        metadata_path = os.path.join(self.models_dir, 'model_metadata.pkl')
        if os.path.exists(metadata_path):
            self.metadata = joblib.load(metadata_path)
            print(f"✓ Loaded metadata from {metadata_path}")
        else:
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    
    def _load_preprocessor(self):
        """Load the preprocessor used during model training."""
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"✓ Loaded preprocessor from {preprocessor_path}")
        else:
            print(f"⚠ Warning: Preprocessor not found at {preprocessor_path}")
    
    def load_models(self, model_names: List[str] = None):
        """
        Load specified models from pickle files.
        
        Args:
            model_names (List[str]): List of model names to load. If None, loads all models.
        """
        if model_names is None:
            model_names = list(self.metadata['model_paths'].keys())
        
        print(f"\nDEBUG: Available models in metadata:")
        for name, path in self.metadata['model_paths'].items():
            print(f"  - {name}: {path}")
        
        print(f"\nLoading {len(model_names)} models...")
        for name in model_names:
            if name in self.metadata['model_paths']:
                path = self.metadata['model_paths'][name]
                if os.path.exists(path):
                    try:
                        self.models[name] = joblib.load(path)
                        print(f"  ✓ Loaded {name} from {path}")
                    except Exception as e:
                        print(f"  ✗ Error loading {name} from {path}: {str(e)}")
                else:
                    print(f"  ✗ Model file not found: {path}")
            else:
                print(f"  ✗ Unknown model: {name}")
        
        print(f"\nSuccessfully loaded {len(self.models)} models")
        print(f"DEBUG: Models loaded into self.models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    
    def load_test_data(self, filepath: str = 'test_data.csv'):
        """
        Load test data from CSV file.
        
        Args:
            filepath (str): Path to test data CSV file
        """
        self.test_data = pd.read_csv(filepath)
        print(f"✓ Loaded test data from {filepath}")
        print(f"  Shape: {self.test_data.shape}")
        
        # Check for expected_result column
        if 'expected_result' in self.test_data.columns:
            fraud_count = (self.test_data['expected_result'] == 'fraud').sum()
            print(f"  Expected fraud cases: {fraud_count}/{len(self.test_data)}")
    
    def prepare_test_features(self) -> pd.DataFrame:
        """
        Prepare test data features (apply same feature engineering as training).
        
        Returns:
            pd.DataFrame: Feature matrix ready for prediction
        """
        df = self.test_data.copy()
        
        # Convert date columns
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
        
        # Temporal features
        df['hour'] = df['TransactionDate'].dt.hour
        df['day_of_week'] = df['TransactionDate'].dt.dayofweek
        df['month'] = df['TransactionDate'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time since previous transaction
        df['time_since_previous_hours'] = (
            (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds() / 3600
        ).clip(0, 24*30)
        
        # Amount-based features
        df['is_high_amount'] = (df['TransactionAmount'] > df['TransactionAmount'].quantile(0.8)).astype(int)
        df['is_low_amount'] = (df['TransactionAmount'] < df['TransactionAmount'].quantile(0.2)).astype(int)
        
        # Balance-related features
        df['amount_to_balance_ratio'] = df['TransactionAmount'] / (df['AccountBalance'] + 1)
        df['is_low_balance'] = (df['AccountBalance'] < 1000).astype(int)
        df['is_high_balance'] = (df['AccountBalance'] > 10000).astype(int)
        
        # Behavioral features
        df['multiple_login_attempts'] = (df['LoginAttempts'] > 1).astype(int)
        df['is_very_fast_transaction'] = (df['TransactionDuration'] < 30).astype(int)
        df['is_slow_transaction'] = (df['TransactionDuration'] > 180).astype(int)
        
        # Select only the features used in training
        feature_columns = self.metadata['feature_names']
        X_test = df[feature_columns]
        
        print(f"✓ Prepared test features: {X_test.shape}")
        return X_test
    
    def run_predictions(self):
        """Run predictions using all loaded models."""
        if self.test_data is None:
            raise ValueError("No test data loaded. Call load_test_data() first.")
        
        # Prepare features
        X_test = self.prepare_test_features()
        
        print(f"\nRunning predictions on {len(X_test)} test samples...")
        
        for model_name, model in self.models.items():
            try:
                # Check if this is a pipeline model (has predict method on the model directly)
                # or if it needs preprocessing first
                if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                    # This is a pipeline model (like Logistic Regression) - use raw features
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    # This is a standalone model that needs preprocessed features
                    if self.preprocessor is not None:
                        X_test_preprocessed = self.preprocessor.transform(X_test)
                        y_pred = model.predict(X_test_preprocessed)
                        y_pred_proba = model.predict_proba(X_test_preprocessed)[:, 1]
                    else:
                        # Fallback to raw features if no preprocessor
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                self.predictions[model_name] = {
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  ✓ {model_name}: {y_pred.sum()} fraud predictions")
                
            except Exception as e:
                print(f"  ✗ {model_name}: Error - {str(e)}")
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate all models against expected results.
        
        Returns:
            pd.DataFrame: Performance metrics for each model
        """
        if 'expected_result' not in self.test_data.columns:
            raise ValueError("Test data must contain 'expected_result' column")
        
        # Convert expected results to binary
        y_true = (self.test_data['expected_result'] == 'fraud').astype(int)
        
        results = []
        
        print(f"\nEvaluating models against expected results...")
        
        for model_name, pred_data in self.predictions.items():
            y_pred = pred_data['predictions']
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'True Positives': tp,
                'False Positives': fp,
                'True Negatives': tn,
                'False Negatives': fn,
                'Total Predictions': len(y_pred),
                'Fraud Predictions': y_pred.sum()
            })
        
        # Check if we have any results
        if not results:
            print("WARNING: No model predictions were generated. Check if models were loaded correctly.")
            self.results = pd.DataFrame()
        else:
            self.results = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
        return self.results
    
    def generate_detailed_report(self) -> str:
        """Generate a detailed performance report."""
        if self.results is None or self.results.empty:
            return "No results available. No models could be evaluated."
        
        report = []
        report.append("="*60)
        report.append("FRAUD DETECTION MODEL PERFORMANCE REPORT")
        report.append("="*60)
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Test Samples: {len(self.test_data)}")
        
        # Expected vs actual fraud cases
        expected_fraud = (self.test_data['expected_result'] == 'fraud').sum()
        report.append(f"Expected Fraud Cases: {expected_fraud}")
        report.append("")
        
        # Model rankings
        report.append("MODEL RANKINGS BY F1-SCORE:")
        report.append("-"*40)
        
        for idx, row in self.results.iterrows():
            rank = idx + 1
            report.append(f"{rank}. {row['Model']}")
            report.append(f"   F1-Score: {row['F1-Score']:.3f}")
            report.append(f"   Precision: {row['Precision']:.3f} (of {row['Fraud Predictions']} fraud predictions, {row['True Positives']} were correct)")
            report.append(f"   Recall: {row['Recall']:.3f} (detected {row['True Positives']} of {expected_fraud} actual fraud cases)")
            report.append(f"   Accuracy: {row['Accuracy']:.3f}")
            report.append("")
        
        # Best model
        best_model = self.results.iloc[0]
        report.append("="*60)
        report.append(f"BEST PERFORMING MODEL: {best_model['Model']}")
        report.append("="*60)
        report.append(f"F1-Score: {best_model['F1-Score']:.3f}")
        report.append(f"Correctly identified {best_model['True Positives']} of {expected_fraud} fraud cases")
        report.append(f"False alarms: {best_model['False Positives']} legitimate transactions flagged as fraud")
        report.append("")
        
        # Confusion matrices for all models
        report.append("="*60)
        report.append("CONFUSION MATRICES FOR ALL MODELS")
        report.append("="*60)
        
        for idx, row in self.results.iterrows():
            report.append(f"\n{row['Model']}:")
            report.append("                 Predicted")
            report.append("                 Normal  Fraud")
            report.append(f"Actual  Normal     {row['True Negatives']:4d}   {row['False Positives']:4d}")
            report.append(f"        Fraud      {row['False Negatives']:4d}   {row['True Positives']:4d}")
            report.append("")
        
        # Detailed predictions for each model
        report.append("="*60)
        report.append("DETAILED PREDICTIONS BY MODEL")
        report.append("="*60)
        
        # Show misclassifications for each model
        for model_name in self.predictions.keys():
            report.append(f"\n{model_name} - Misclassified Transactions:")
            report.append("-" * 40)
            
            predictions = self.predictions[model_name]['predictions']
            probabilities = self.predictions[model_name]['probabilities']
            misclassified_count = 0
            
            for i, (idx, row) in enumerate(self.test_data.iterrows()):
                pred = predictions[i]
                prob = probabilities[i]
                expected = row['expected_result']
                pred_label = 'fraud' if pred == 1 else 'normal'
                
                if pred_label != expected:
                    misclassified_count += 1
                    if misclassified_count <= 5:  # Show first 5 misclassifications
                        report.append(f"  ✗ {row['TransactionID']}: Expected={expected}, Predicted={pred_label} (prob={prob:.3f})")
                        report.append(f"     Amount=${row['TransactionAmount']:.2f}, Login={row['LoginAttempts']}, Channel={row['Channel']}")
            
            if misclassified_count == 0:
                report.append("  ✓ No misclassifications!")
            elif misclassified_count > 5:
                report.append(f"  ... and {misclassified_count - 5} more misclassifications")
            
            report.append(f"  Total misclassified: {misclassified_count}/{len(self.test_data)}")
        
        # Model summary table
        report.append("")
        report.append("="*60)
        report.append("MODEL PERFORMANCE SUMMARY TABLE")
        report.append("="*60)
        report.append("")
        report.append(self.results.to_string(index=False))
        
        return "\n".join(report)
    
    def save_results(self, filepath: str = 'model_test_results.csv'):
        """Save evaluation results to CSV."""
        if self.results is not None:
            self.results.to_csv(filepath, index=False)
            print(f"✓ Saved results to {filepath}")
    
    def save_all_predictions(self, output_dir: str = 'model_predictions'):
        """Save detailed predictions for all models to separate CSV files."""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for model_name, pred_data in self.predictions.items():
            # Create a dataframe with test data and predictions
            pred_df = self.test_data.copy()
            pred_df['predicted'] = pred_data['predictions']
            pred_df['predicted_label'] = pred_df['predicted'].map({0: 'normal', 1: 'fraud'})
            pred_df['probability_fraud'] = pred_data['probabilities']
            pred_df['correct'] = pred_df['expected_result'] == pred_df['predicted_label']
            
            # Save to CSV
            safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            filepath = os.path.join(output_dir, f'{safe_name}_predictions.csv')
            pred_df.to_csv(filepath, index=False)
            
        print(f"✓ Saved predictions for {len(self.predictions)} models to {output_dir}/")
    
    def run_complete_test(self, test_data_path: str = 'test_data.csv'):
        """
        Run a complete test cycle: load data, load models, predict, and evaluate.
        
        Args:
            test_data_path (str): Path to test data CSV file
        """
        print("RUNNING COMPLETE MODEL TEST -- PLEASE WAIT...")
        print("="*40)
        
        # Check if test data exists
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(
                f"Test data not found at {test_data_path}. "
                "Please generate test data using: python tests/test_data_generator.py"
            )
        
        # Load test data
        self.load_test_data(test_data_path)
        
        # Load all models
        self.load_models()
        
        # Run predictions
        self.run_predictions()
        
        # Evaluate models
        results = self.evaluate_models()
        
        # Generate report
        report = self.generate_detailed_report()
        print("\n" + report)
        
        # Save results
        self.save_results()
        
        return results


if __name__ == "__main__":
    """Main method to run model testing with default values."""
    # Create tester instance with default models directory
    tester = ModelTester(models_dir='models')
    
    # Set default test data path
    test_data_path = 'testdata/test_data.csv'
    
    try:
        # Run complete test cycle
        print("\n" + "="*60)
        print("FRAUD DETECTION MODEL TESTING SCRIPT")
        print("="*60)
        print(f"\nUsing test data from: {test_data_path}")
        print(f"Loading models from: {tester.models_dir}")
        
        # Run the test
        results = tester.run_complete_test(test_data_path=test_data_path)
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Display summary if results exist
        if not results.empty:
            print("\nModel Performance Summary:")
            print(results[['Model', 'F1-Score', 'Precision', 'Recall', 'Accuracy']].to_string(index=False))
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo generate test data, run:")
        print("  python tests/test_data_generator.py")
        
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()