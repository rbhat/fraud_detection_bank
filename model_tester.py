"""
Model Tester for Fraud Detection Models
This module provides functionality to test saved fraud detection models
against new transaction data and compare their performance.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Any


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
        
        # Load metadata
        self._load_metadata()
        
    def _load_metadata(self):
        """Load model metadata from pickle file."""
        metadata_path = os.path.join(self.models_dir, 'model_metadata.pkl')
        if os.path.exists(metadata_path):
            self.metadata = joblib.load(metadata_path)
            print(f"✓ Loaded metadata from {metadata_path}")
        else:
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    
    def load_models(self, model_names: List[str] = None):
        """
        Load specified models from pickle files.
        
        Args:
            model_names (List[str]): List of model names to load. If None, loads all models.
        """
        if model_names is None:
            model_names = list(self.metadata['model_paths'].keys())
        
        print(f"\nLoading {len(model_names)} models...")
        for name in model_names:
            if name in self.metadata['model_paths']:
                path = self.metadata['model_paths'][name]
                if os.path.exists(path):
                    self.models[name] = joblib.load(path)
                    print(f"  ✓ Loaded {name}")
                else:
                    print(f"  ✗ Model file not found: {path}")
            else:
                print(f"  ✗ Unknown model: {name}")
        
        print(f"\nSuccessfully loaded {len(self.models)} models")
    
    def create_test_data(self, n_samples: int = 10, fraud_rate: float = 0.3) -> pd.DataFrame:
        """
        Create synthetic test data with known labels.
        
        Args:
            n_samples (int): Number of test samples to generate
            fraud_rate (float): Proportion of fraud cases in test data
            
        Returns:
            pd.DataFrame: Test data with expected_result column
        """
        np.random.seed(42)
        n_fraud = int(n_samples * fraud_rate)
        n_normal = n_samples - n_fraud
        
        test_records = []
        
        # Generate normal transactions
        for i in range(n_normal):
            record = self._generate_transaction(is_fraud=False, index=i)
            test_records.append(record)
        
        # Generate fraud transactions
        for i in range(n_fraud):
            record = self._generate_transaction(is_fraud=True, index=i + n_normal)
            test_records.append(record)
        
        # Shuffle the records
        np.random.shuffle(test_records)
        
        # Create DataFrame
        test_df = pd.DataFrame(test_records)
        
        # Save to CSV
        test_df.to_csv('test_data.csv', index=False)
        print(f"✓ Created test data with {n_samples} samples ({n_fraud} fraud, {n_normal} normal)")
        print(f"✓ Saved to test_data.csv")
        
        return test_df
    
    def _generate_transaction(self, is_fraud: bool, index: int) -> dict:
        """Generate a single transaction record."""
        base_date = datetime(2023, 1, 1)
        
        if is_fraud:
            # Fraud patterns
            amount = np.random.lognormal(7, 1.5)  # Higher amounts
            login_attempts = np.random.choice([3, 4, 5], p=[0.4, 0.4, 0.2])
            duration = np.random.choice([15, 20, 25], p=[0.5, 0.3, 0.2])
            channel = np.random.choice(['Online', 'ATM'], p=[0.7, 0.3])
            age = np.random.randint(18, 45)
            balance = np.random.lognormal(6, 1)  # Lower balance
        else:
            # Normal patterns
            amount = np.random.lognormal(5, 1)  # Normal amounts
            login_attempts = np.random.choice([1, 2], p=[0.9, 0.1])
            duration = np.random.choice([60, 90, 120], p=[0.3, 0.4, 0.3])
            channel = np.random.choice(['Online', 'ATM', 'Branch'], p=[0.33, 0.33, 0.34])
            age = np.random.randint(25, 65)
            balance = np.random.lognormal(8, 1)  # Higher balance
        
        transaction_date = base_date.replace(
            month=np.random.randint(1, 13),
            day=np.random.randint(1, 28),
            hour=np.random.randint(0, 24),
            minute=np.random.randint(0, 60)
        )
        
        previous_date = transaction_date - pd.Timedelta(days=np.random.randint(1, 30))
        
        return {
            'TransactionID': f'TX_TEST_{index+1:03d}',
            'AccountID': f'AC{np.random.randint(10000, 99999)}',
            'TransactionAmount': round(amount, 2),
            'TransactionDate': transaction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'TransactionType': np.random.choice(['Debit', 'Credit'], p=[0.7, 0.3]),
            'Location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']),
            'DeviceID': f'D{np.random.randint(100000, 999999)}',
            'IP Address': f'{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}',
            'MerchantID': f'M{np.random.randint(100, 999)}',
            'Channel': channel,
            'CustomerAge': age,
            'CustomerOccupation': np.random.choice(['Engineer', 'Doctor', 'Student', 'Retired']),
            'TransactionDuration': duration,
            'LoginAttempts': login_attempts,
            'AccountBalance': round(balance, 2),
            'PreviousTransactionDate': previous_date.strftime('%Y-%m-%d %H:%M:%S'),
            'expected_result': 'fraud' if is_fraud else 'normal'
        }
    
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
                # Make predictions
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
        
        self.results = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
        return self.results
    
    def generate_detailed_report(self) -> str:
        """Generate a detailed performance report."""
        if self.results is None:
            raise ValueError("No results available. Run evaluate_models() first.")
        
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
        
        # Detailed predictions for best model
        report.append("")
        report.append("DETAILED PREDICTIONS (Best Model):")
        report.append("-"*40)
        
        best_predictions = self.predictions[best_model['Model']]
        for i, (idx, row) in enumerate(self.test_data.iterrows()):
            pred = best_predictions['predictions'][i]
            prob = best_predictions['probabilities'][i]
            expected = row['expected_result']
            pred_label = 'fraud' if pred == 1 else 'normal'
            
            status = "✓" if pred_label == expected else "✗"
            report.append(f"{status} Transaction {row['TransactionID']}: Expected={expected}, Predicted={pred_label} (prob={prob:.3f})")
        
        return "\n".join(report)
    
    def save_results(self, filepath: str = 'model_test_results.csv'):
        """Save evaluation results to CSV."""
        if self.results is not None:
            self.results.to_csv(filepath, index=False)
            print(f"✓ Saved results to {filepath}")
    
    def run_complete_test(self, n_test_samples: int = 10, fraud_rate: float = 0.3):
        """
        Run a complete test cycle: create data, load models, predict, and evaluate.
        
        Args:
            n_test_samples (int): Number of test samples to generate
            fraud_rate (float): Proportion of fraud cases in test data
        """
        print("RUNNING COMPLETE MODEL TEST")
        print("="*40)
        
        # Create test data
        self.create_test_data(n_samples=n_test_samples, fraud_rate=fraud_rate)
        
        # Load test data
        self.load_test_data()
        
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