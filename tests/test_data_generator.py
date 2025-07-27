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
        self.fraud_patterns = {
            'amount': {'mean': 7, 'std': 1.5},  # Higher amounts
            'login_attempts': {'distribution': [3, 4, 5], 'probabilities': [0.4, 0.4, 0.2]},
            'duration': {'distribution': [15, 20, 25], 'probabilities': [0.5, 0.3, 0.2]},
            'channel': {'distribution': ['Online', 'ATM'], 'probabilities': [0.7, 0.3]},
            'age': {'min': 18, 'max': 45},
            'balance': {'mean': 6, 'std': 1}  # Lower balance
        }
        
        self.normal_patterns = {
            'amount': {'mean': 5, 'std': 1},  # Normal amounts
            'login_attempts': {'distribution': [1, 2], 'probabilities': [0.9, 0.1]},
            'duration': {'distribution': [60, 90, 120], 'probabilities': [0.3, 0.4, 0.3]},
            'channel': {'distribution': ['Online', 'ATM', 'Branch'], 'probabilities': [0.33, 0.33, 0.34]},
            'age': {'min': 25, 'max': 65},
            'balance': {'mean': 8, 'std': 1}  # Higher balance
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
        patterns = self.fraud_patterns if is_fraud else self.normal_patterns
        
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
        
        # Generate dates
        base_date = datetime(2023, 1, 1)
        transaction_date = base_date.replace(
            month=np.random.randint(1, 13),
            day=np.random.randint(1, 28),
            hour=np.random.randint(0, 24),
            minute=np.random.randint(0, 60)
        )
        previous_date = transaction_date - timedelta(days=np.random.randint(1, 30))
        
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
    
    def generate_test_data(self, n_samples: int = 10, fraud_rate: float = 0.3, seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic test data with known labels.
        
        Args:
            n_samples (int): Number of test samples to generate
            fraud_rate (float): Proportion of fraud cases in test data
            seed (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Test data with expected_result column
        """
        np.random.seed(seed)
        
        n_fraud = int(n_samples * fraud_rate)
        n_normal = n_samples - n_fraud
        
        print(f"Generating test data:")
        print(f"  Total samples: {n_samples}")
        print(f"  Fraud samples: {n_fraud} ({fraud_rate*100:.0f}%)")
        print(f"  Normal samples: {n_normal} ({(1-fraud_rate)*100:.0f}%)")
        
        test_records = []
        
        # Generate normal transactions
        for i in range(n_normal):
            record = self.generate_transaction(is_fraud=False, index=i)
            test_records.append(record)
        
        # Generate fraud transactions
        for i in range(n_fraud):
            record = self.generate_transaction(is_fraud=True, index=i + n_normal)
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