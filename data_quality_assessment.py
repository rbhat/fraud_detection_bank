"""
Data quality assessment tools for fraud detection analysis.
Simple class to check data issues and clean things up.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class DataQualityAssessment:
    """
    Basic data quality checker for ML projects.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def check_missing_values(self):
        """Check for missing values in the dataset"""
        print("Missing values analysis")
        print("="*30)

        # Check for explicit missing values
        print("\n1. Explicit missing values:")
        print("-" * 25)
        missing_summary = self.df.isnull().sum()
        missing_percentage = (missing_summary / len(self.df)) * 100

        missing_df = pd.DataFrame({
            'Column': missing_summary.index,
            'Missing_Count': missing_summary.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Count', ascending=False)

        print("Missing values by column:")
        has_missing = False
        for _, row in missing_df.iterrows():
            if row['Missing_Count'] > 0:
                print(f"  {row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']:.2f}%)")
                has_missing = True
            
        if not has_missing:
            print("  ✓ No explicit missing values found")

        # Check for implicit missing values
        print("\n2. Implicit missing values:")
        print("-" * 25)

        implicit_issues = {}

        # Check for empty strings in string columns
        string_cols = self.df.select_dtypes(include=['object']).columns
        for col in string_cols:
            empty_strings = (self.df[col] == '').sum()
            whitespace_only = self.df[col].str.isspace().sum() if self.df[col].dtype == 'object' else 0
            if empty_strings > 0 or whitespace_only > 0:
                implicit_issues[col] = {'empty_strings': empty_strings, 'whitespace': whitespace_only}

        # Check for suspicious zeros in amount fields
        if 'TransactionAmount' in self.df.columns:
            zero_amounts = (self.df['TransactionAmount'] == 0).sum()
            if zero_amounts > 0:
                implicit_issues['TransactionAmount'] = {'zero_amounts': zero_amounts}

        # Check for suspicious patterns in IDs
        id_cols = [col for col in self.df.columns if 'ID' in col.upper()]
        for col in id_cols:
            if self.df[col].dtype == 'object':
                pattern_issues = self.df[col].str.contains('null|none|unknown|missing|na|n/a', case=False, na=False).sum()
                if pattern_issues > 0:
                    implicit_issues[col] = {'pattern_issues': pattern_issues}

        print("Patterns found:")
        if implicit_issues:
            for col, issues in implicit_issues.items():
                print(f"  {col}: {issues}")
        else:
            print("  ✓ No suspicious patterns found")

        # Check for impossible/invalid values
        print("\n3. Invalid values:")
        print("-" * 15)

        invalid_values = {}

        # Check for negative values where they shouldn't be
        if 'TransactionAmount' in self.df.columns:
            negative_amounts = (self.df['TransactionAmount'] < 0).sum()
            if negative_amounts > 0:
                invalid_values['TransactionAmount'] = f"{negative_amounts} negative amounts"

        if 'CustomerAge' in self.df.columns:
            invalid_ages = ((self.df['CustomerAge'] < 0) | (self.df['CustomerAge'] > 150)).sum()
            if invalid_ages > 0:
                invalid_values['CustomerAge'] = f"{invalid_ages} invalid ages"

        if 'LoginAttempts' in self.df.columns:
            invalid_logins = (self.df['LoginAttempts'] < 1).sum()
            if invalid_logins > 0:
                invalid_values['LoginAttempts'] = f"{invalid_logins} invalid login attempts"

        print("Issues found:")
        if invalid_values:
            for col, issue in invalid_values.items():
                print(f"  {col}: {issue}")
        else:
            print("  ✓ No invalid values detected")

        # Data completeness summary
        print("\n4. Overall data completeness:")
        print("-" * 30)
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100

        print(f"Dataset dimensions: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Total cells: {total_cells:,}")
        print(f"Missing cells: {missing_cells:,}")
        print(f"Data completeness: {completeness:.2f}%")

        if completeness >= 95:
            print("✓ Dataset has excellent completeness")
        elif completeness >= 90:
            print("⚠ Dataset has good completeness")
        else:
            print("❌ Dataset has poor completeness - investigate further")
        
        return {
            'missing_summary': missing_df,
            'implicit_issues': implicit_issues,
            'invalid_values': invalid_values,
            'completeness_percentage': completeness
        }

    def check_duplicates(self):
        """Find and remove duplicate records"""
        print("Duplicate detection and cleanup")
        print("="*35)

        # Check for exact duplicates
        print("\n1. Exact duplicates:")
        print("-" * 20)
        exact_duplicates = self.df.duplicated().sum()
        print(f"Exact duplicate rows: {exact_duplicates}")

        if exact_duplicates > 0:
            print("Sample of exact duplicates:")
            duplicate_rows = self.df[self.df.duplicated(keep=False)].sort_values(self.df.columns[0])
            print(duplicate_rows.head())
        else:
            print("✓ No exact duplicates found")

        # Check business logic duplicates
        print("\n2. Business logic duplicates:")
        print("-" * 30)

        tx_id_duplicates = 0
        if 'TransactionID' in self.df.columns:
            tx_id_duplicates = self.df['TransactionID'].duplicated().sum()
            print(f"Duplicate TransactionIDs: {tx_id_duplicates}")
            if tx_id_duplicates > 0:
                print("Duplicate TransactionID examples:")
                dup_tx_ids = self.df[self.df['TransactionID'].duplicated(keep=False)]['TransactionID'].unique()[:5]
                for tx_id in dup_tx_ids:
                    print(f"  {tx_id}: {self.df[self.df['TransactionID'] == tx_id].shape[0]} occurrences")

        # Remove duplicates
        print("\n3. Cleanup:")
        print("-" * 10)

        original_count = len(self.df)

        # Remove exact duplicates if any
        if exact_duplicates > 0:
            self.df = self.df.drop_duplicates()
            removed_exact = original_count - len(self.df)
            print(f"Removed {removed_exact} exact duplicate rows")
        else:
            print("No exact duplicates to remove")

        # Handle business logic duplicates
        if 'TransactionID' in self.df.columns and tx_id_duplicates > 0:
            self.df = self.df.drop_duplicates(subset=['TransactionID'], keep='first')
            removed_tx_duplicates = original_count - len(self.df)
            print(f"Removed {removed_tx_duplicates} duplicate TransactionID rows")

        print(f"\nOriginal dataset: {original_count:,} rows")
        print(f"Cleaned dataset: {len(self.df):,} rows")
        print(f"Total removed: {original_count - len(self.df):,} rows ({((original_count - len(self.df))/original_count)*100:.2f}%)")

        if len(self.df) < original_count:
            print(f"\n✓ Dataset cleaned: {original_count - len(self.df)} duplicate rows removed")
        else:
            print("\n✓ No duplicates needed removal - dataset is clean")
            
        return self.df

    def validate_data_types(self):
        """Check and fix data types"""
        print("Data type validation")
        print("="*25)

        # Current data types
        print("\n1. Current data types:")
        print("-" * 20)
        print("Data types by column:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            unique_count = self.df[col].nunique()
            sample_values = self.df[col].head(3).tolist()
            print(f"  {col}: {dtype} (unique: {unique_count}, samples: {sample_values})")

        # Check what needs fixing
        print("\n2. Validation check:")
        print("-" * 18)

        corrections_needed = {}

        # Check if date columns are properly parsed
        date_columns = ['TransactionDate', 'PreviousTransactionDate']
        for col in date_columns:
            if col in self.df.columns:
                if self.df[col].dtype == 'object':
                    print(f"⚠ {col} is object type, should be datetime")
                    corrections_needed[col] = 'datetime'
                else:
                    print(f"✓ {col} has correct datetime type")

        # Check if ID columns should be strings
        id_columns = ['TransactionID', 'AccountID', 'DeviceID', 'MerchantID']
        for col in id_columns:
            if col in self.df.columns:
                if self.df[col].dtype != 'object':
                    print(f"⚠ {col} is {self.df[col].dtype}, should be string/object")
                    corrections_needed[col] = 'string'
                else:
                    print(f"✓ {col} has correct string type")

        # Check if numerical columns are appropriate
        numerical_should_be = {
            'TransactionAmount': 'float64',
            'CustomerAge': 'int64', 
            'TransactionDuration': 'int64',
            'LoginAttempts': 'int64',
            'AccountBalance': 'float64'
        }

        for col, expected_type in numerical_should_be.items():
            if col in self.df.columns:
                current_type = str(self.df[col].dtype)
                if current_type != expected_type:
                    print(f"⚠ {col} is {current_type}, expected {expected_type}")
                    corrections_needed[col] = expected_type
                else:
                    print(f"✓ {col} has correct {expected_type} type")

        # Apply fixes
        print("\n3. Applying fixes:")
        print("-" * 16)

        if corrections_needed:
            for col, target_type in corrections_needed.items():
                try:
                    if target_type == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                        print(f"✓ Converted {col} to datetime")
                    elif target_type == 'string':
                        self.df[col] = self.df[col].astype(str)
                        print(f"✓ Converted {col} to string")
                    elif target_type in ['int64', 'float64']:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(target_type)
                        print(f"✓ Converted {col} to {target_type}")
                except Exception as e:
                    print(f"❌ Failed to convert {col}: {e}")
        else:
            print("✓ No data type corrections needed")

        print("\nData type validation completed")
        return self.df

    def assess_and_clean(self):
        """Run full data quality pipeline"""
        print("Running data quality assessment")
        print("=" * 35)
        
        # Run all checks
        missing_analysis = self.check_missing_values()
        print("\n" + "=" * 35)
        
        self.df = self.check_duplicates()
        print("\n" + "=" * 35)
        
        self.df = self.validate_data_types()
        print("\n" + "=" * 35)
        
        print("Data quality assessment completed")
        print("=" * 35)
        print(f"Original dataset shape: {self.original_shape}")
        print(f"Final dataset shape: {self.df.shape}")
        
        return self.df

    def get_cleaned_data(self):
        """Get the cleaned dataset"""
        return self.df


# Quick function for simple usage
def assess_data_quality(df):
    """Run data quality assessment on a dataframe"""
    dqa = DataQualityAssessment(df)
    return dqa.assess_and_clean()