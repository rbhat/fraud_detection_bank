"""
Test Data Generator for Fraud Detection Models
This module generates synthetic test transactions with known labels for model evaluation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataGenerator:
    """
    Generates synthetic test transactions with known fraud/normal labels.
    
    This class creates realistic test data that mimics the patterns found in
    actual fraud and normal transactions, allowing for controlled model evaluation.
    """
    
    def __init__(self, output_path: str = 'test_data.csv'):
        """
        Initialize the TestDataGenerator.
        
        Args:
            output_path (str): Path where test_data.csv will be saved
        """
        self.output_path = output_path
        # More realistic and varied fraud patterns
        # Not all fraud follows the exact same pattern - some fraudsters are sophisticated
        self.fraud_patterns = {
            'amount': {'mean': 6.5, 'std': 2},  # Varied amounts, some normal-looking
            'login_attempts': {'distribution': [1, 2, 3, 4, 5], 'probabilities': [0.15, 0.20, 0.30, 0.25, 0.10]},
            'duration': {'distribution': [10, 25, 45, 75, 120, 200], 'probabilities': [0.15, 0.25, 0.20, 0.15, 0.15, 0.10]},
            'channel': {'distribution': ['Online', 'ATM', 'Branch'], 'probabilities': [0.55, 0.35, 0.10]},
            'age': {'min': 18, 'max': 70},  # Wider age range
            'balance': {'mean': 7.5, 'std': 1.5}  # More varied balances
        }
        
        self.normal_patterns = {
            'amount': {'mean': 5.2, 'std': 1.3},  # Slightly overlapping with fraud
            'login_attempts': {'distribution': [1, 2, 3], 'probabilities': [0.85, 0.12, 0.03]},
            'duration': {'distribution': [30, 60, 90, 120, 180], 'probabilities': [0.15, 0.30, 0.30, 0.20, 0.05]},
            'channel': {'distribution': ['Online', 'ATM', 'Branch'], 'probabilities': [0.30, 0.35, 0.35]},
            'age': {'min': 20, 'max': 75},
            'balance': {'mean': 8.2, 'std': 1.2}
        }
    
    def generate_transaction(self, is_fraud: bool, index: int) -> dict:
        """
        Generate a single transaction record.
        
        Args:
            is_fraud (bool): Whether to generate a fraudulent transaction
            index (int): Transaction index for ID generation
            
        Returns:
            dict: Transaction record with all required fields
        """
        if is_fraud:
            # Create different types of fraud patterns
            fraud_type = np.random.choice(['aggressive', 'subtle', 'mixed'], p=[0.3, 0.4, 0.3])
            
            if fraud_type == 'aggressive':
                # Classic fraud: high amount, multiple attempts, fast
                patterns = self.fraud_patterns
            elif fraud_type == 'subtle':
                # Sophisticated fraud: looks more normal
                patterns = {
                    'amount': {'mean': 5.5, 'std': 1.2},
                    'login_attempts': {'distribution': [1, 2], 'probabilities': [0.7, 0.3]},
                    'duration': {'distribution': [40, 60, 80], 'probabilities': [0.3, 0.4, 0.3]},
                    'channel': self.fraud_patterns['channel'],
                    'age': {'min': 25, 'max': 55},
                    'balance': {'mean': 7.8, 'std': 1}
                }
            else:  # mixed
                # Mix of normal and suspicious characteristics
                patterns = self.fraud_patterns
        else:
            # Normal transactions with occasional anomalies
            if np.random.random() < 0.1:  # 10% of normal transactions have some unusual features
                patterns = self.normal_patterns.copy()
                # Add some noise - high amount or multiple login attempts
                if np.random.random() < 0.5:
                    patterns['amount'] = {'mean': 6.5, 'std': 1.5}
                else:
                    patterns['login_attempts'] = {'distribution': [2, 3], 'probabilities': [0.7, 0.3]}
            else:
                patterns = self.normal_patterns
        
        # Generate transaction features based on patterns
        amount = np.random.lognormal(patterns['amount']['mean'], patterns['amount']['std'])
        login_attempts = np.random.choice(
            patterns['login_attempts']['distribution'],
            p=patterns['login_attempts']['probabilities']
        )
        duration = np.random.choice(
            patterns['duration']['distribution'],
            p=patterns['duration']['probabilities']
        )
        channel = np.random.choice(
            patterns['channel']['distribution'],
            p=patterns['channel']['probabilities']
        )
        age = np.random.randint(patterns['age']['min'], patterns['age']['max'])
        balance = np.random.lognormal(patterns['balance']['mean'], patterns['balance']['std'])
        
        # Common features
        transaction_type = np.random.choice(['Debit', 'Credit'], p=[0.7, 0.3])
        customer_occupation = np.random.choice(['Engineer', 'Doctor', 'Student', 'Retired'])
        location = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])
        
        # Generate dates with more realistic patterns
        base_date = datetime(2023, 1, 1)
        
        # Fraud often happens at unusual hours
        if is_fraud and np.random.random() < 0.6:
            # 60% of fraud happens during off-hours
            hour = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], p=[0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1])
        else:
            # Normal distribution of hours with peak during business hours
            hour = np.random.choice(range(24), p=[
                0.02, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05,  # 0-7
                0.08, 0.09, 0.08, 0.06, 0.05, 0.06, 0.07, 0.07,  # 8-15
                0.06, 0.05, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02   # 16-23
            ])
        
        transaction_date = base_date.replace(
            month=np.random.randint(1, 13),
            day=np.random.randint(1, 28),
            hour=hour,
            minute=np.random.randint(0, 60)
        )
        
        # Time since previous transaction
        if is_fraud and np.random.random() < 0.4:
            # Some fraud happens in rapid succession
            days_since = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            days_since = np.random.randint(1, 30)
        
        previous_date = transaction_date - timedelta(days=int(days_since))
        
        return {
            'TransactionID': f'TX_TEST_{index+1:03d}',
            'AccountID': f'AC{np.random.randint(10000, 99999)}',
            'TransactionAmount': round(amount, 2),
            'TransactionDate': transaction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'TransactionType': transaction_type,
            'Location': location,
            'DeviceID': f'D{np.random.randint(100000, 999999)}',
            'IP Address': f'{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}',
            'MerchantID': f'M{np.random.randint(100, 999)}',
            'Channel': channel,
            'CustomerAge': age,
            'CustomerOccupation': customer_occupation,
            'TransactionDuration': duration,
            'LoginAttempts': login_attempts,
            'AccountBalance': round(balance, 2),
            'PreviousTransactionDate': previous_date.strftime('%Y-%m-%d %H:%M:%S'),
            'expected_result': 'fraud' if is_fraud else 'normal'
        }
    
    def generate_edge_case_transaction(self, index: int) -> dict:
        """
        Generate edge case transactions that are difficult to classify.
        These help test model robustness.
        """
        edge_case_type = np.random.choice([
            'high_amount_legitimate',
            'multiple_attempts_legitimate',
            'fast_legitimate',
            'normal_looking_fraud'
        ])
        
        if edge_case_type == 'high_amount_legitimate':
            # Large purchase but legitimate (e.g., electronics, furniture)
            amount = np.random.lognormal(7.5, 0.5)
            login_attempts = 1
            duration = np.random.randint(90, 180)
            channel = 'Online'
            balance = np.random.lognormal(9, 0.5)  # High balance
            is_fraud = False
            
        elif edge_case_type == 'multiple_attempts_legitimate':
            # User forgot password but legitimate
            amount = np.random.lognormal(5, 0.8)
            login_attempts = np.random.choice([3, 4])
            duration = np.random.randint(120, 300)  # Takes longer due to password issues
            channel = np.random.choice(['Online', 'ATM'])
            balance = np.random.lognormal(8, 1)
            is_fraud = False
            
        elif edge_case_type == 'fast_legitimate':
            # Quick routine transaction
            amount = np.random.lognormal(4, 0.5)
            login_attempts = 1
            duration = np.random.randint(15, 30)
            channel = 'ATM'  # ATM transactions are often quick
            balance = np.random.lognormal(8, 1)
            is_fraud = False
            
        else:  # normal_looking_fraud
            # Sophisticated fraud that mimics normal behavior
            amount = np.random.lognormal(5.2, 0.8)
            login_attempts = 1
            duration = np.random.randint(60, 120)
            channel = np.random.choice(['Online', 'Branch'])
            balance = np.random.lognormal(8, 0.8)
            is_fraud = True
        
        # Common attributes
        age = np.random.randint(25, 65)
        transaction_type = 'Debit'
        customer_occupation = np.random.choice(['Engineer', 'Doctor', 'Student', 'Retired'])
        location = np.random.choice(['San Jose', 'New York', 'Chicago', 'Austin','Miami'])
        
        # Normal business hours for edge cases
        hour = np.random.randint(9, 18)
        
        base_date = datetime(2023, 1, 1)
        transaction_date = base_date.replace(
            month=np.random.randint(1, 13),
            day=np.random.randint(1, 28),
            hour=hour,
            minute=np.random.randint(0, 60)
        )
        previous_date = transaction_date - timedelta(days=np.random.randint(2, 10))
        
        return {
            'TransactionID': f'TX_EDGE_{index+1:03d}',
            'AccountID': f'AC{np.random.randint(10000, 99999)}',
            'TransactionAmount': round(amount, 2),
            'TransactionDate': transaction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'TransactionType': transaction_type,
            'Location': location,
            'DeviceID': f'D{np.random.randint(100000, 999999)}',
            'IP Address': f'{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}',
            'MerchantID': f'M{np.random.randint(100, 999)}',
            'Channel': channel,
            'CustomerAge': age,
            'CustomerOccupation': customer_occupation,
            'TransactionDuration': duration,
            'LoginAttempts': login_attempts,
            'AccountBalance': round(balance, 2),
            'PreviousTransactionDate': previous_date.strftime('%Y-%m-%d %H:%M:%S'),
            'expected_result': 'fraud' if is_fraud else 'normal'
        }
    
    def generate_test_data(self, n_samples: int = 10, fraud_rate: float = 0.3, seed: int = 42, include_edge_cases: bool = True) -> pd.DataFrame:
        """
        Generate synthetic test data with known labels.
        
        Args:
            n_samples (int): Number of test samples to generate
            fraud_rate (float): Proportion of fraud cases in test data
            seed (int): Random seed for reproducibility
            include_edge_cases (bool): Whether to include edge cases
            
        Returns:
            pd.DataFrame: Test data with expected_result column
        """
        np.random.seed(seed)
        
        if include_edge_cases and n_samples >= 20:
            # Reserve 20% for edge cases
            n_edge_cases = max(4, int(n_samples * 0.2))
            n_regular = n_samples - n_edge_cases
        else:
            n_edge_cases = 0
            n_regular = n_samples
        
        n_fraud = int(n_regular * fraud_rate)
        n_normal = n_regular - n_fraud
        
        print(f"Generating test data:")
        print(f"  Total samples: {n_samples}")
        print(f"  Regular samples: {n_regular}")
        print(f"    - Fraud: {n_fraud} ({fraud_rate*100:.0f}%)")
        print(f"    - Normal: {n_normal}")
        if n_edge_cases > 0:
            print(f"  Edge cases: {n_edge_cases} (challenging scenarios)")
        
        test_records = []
        
        # Generate normal transactions
        for i in range(n_normal):
            record = self.generate_transaction(is_fraud=False, index=i)
            test_records.append(record)
        
        # Generate fraud transactions
        for i in range(n_fraud):
            record = self.generate_transaction(is_fraud=True, index=i + n_normal)
            test_records.append(record)
        
        # Generate edge cases
        if n_edge_cases > 0:
            for i in range(n_edge_cases):
                record = self.generate_edge_case_transaction(index=i + n_regular)
                test_records.append(record)
        
        # Shuffle the records
        np.random.shuffle(test_records)
        
        # Create DataFrame
        test_df = pd.DataFrame(test_records)
        
        # Save to CSV
        test_df.to_csv(self.output_path, index=False)
        print(f"\n✓ Test data saved to {self.output_path}")
        
        # Print summary
        print(f"\nTest data summary:")
        print(f"  Shape: {test_df.shape}")
        print(f"  Columns: {list(test_df.columns)}")
        print(f"\nLabel distribution:")
        print(test_df['expected_result'].value_counts())
        
        # Additional statistics
        print(f"\nTransaction characteristics:")
        print(f"  Amount range: ${test_df['TransactionAmount'].min():.2f} - ${test_df['TransactionAmount'].max():.2f}")
        print(f"  Avg amount: ${test_df['TransactionAmount'].mean():.2f}")
        
        fraud_mask = test_df['expected_result'] == 'fraud'
        if fraud_mask.any():
            print(f"\nFraud vs Normal patterns:")
            print(f"  Avg amount (fraud): ${test_df[fraud_mask]['TransactionAmount'].mean():.2f}")
            print(f"  Avg amount (normal): ${test_df[~fraud_mask]['TransactionAmount'].mean():.2f}")
            print(f"  Avg login attempts (fraud): {test_df[fraud_mask]['LoginAttempts'].mean():.1f}")
            print(f"  Avg login attempts (normal): {test_df[~fraud_mask]['LoginAttempts'].mean():.1f}")
            
        print(f"\nNote: This test data includes realistic variations and edge cases")
        print(f"      to properly evaluate model performance on unseen patterns.")
        
        return test_df


def main():
    """
    Main function to generate test data when script is run directly.
    """
    print("="*60)
    print("FRAUD DETECTION TEST DATA GENERATOR")
    print("="*60)
    
    # Default configuration
    n_samples = 10
    fraud_rate = 0.3
    output_path = 'test_data.csv'
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        try:
            n_samples = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of samples: {sys.argv[1]}")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            fraud_rate = float(sys.argv[2])
            if not 0 <= fraud_rate <= 1:
                raise ValueError("Fraud rate must be between 0 and 1")
        except ValueError as e:
            print(f"Invalid fraud rate: {sys.argv[2]} - {e}")
            sys.exit(1)
    
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    
    # Generate test data
    generator = TestDataGenerator(output_path=output_path)
    test_df = generator.generate_test_data(n_samples=n_samples, fraud_rate=fraud_rate)
    
    print(f"\n✓ Test data generation completed successfully!")
    print(f"\nUsage: python test_data_generator.py [n_samples] [fraud_rate] [output_path]")
    print(f"Example: python test_data_generator.py 100 0.2 test_data.csv")


if __name__ == "__main__":
    main()