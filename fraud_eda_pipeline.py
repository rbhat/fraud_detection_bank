# fraud_eda_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class FraudEDACapstone:
    """
    Comprehensive EDA pipeline for fraud detection capstone project.
    Handles data quality assessment, feature analysis, and visualization.
    """
    
    def __init__(self, df, target_column=None):
        """
        Initialize the EDA pipeline
        
        Parameters:
        df (pd.DataFrame): Raw dataset
        target_column (str): Name of target column if it exists
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.target_column = target_column
        self.categorical_cols = []
        self.numerical_cols = []
        self.date_cols = []
        self.outlier_scores = None
        self.feature_importance = None
        
        # Auto-detect column types
        self._detect_column_types()
        
        print(f"FraudEDACapstone initialized with dataset: {self.df.shape}")
        print(f"Detected {len(self.numerical_cols)} numerical, {len(self.categorical_cols)} categorical, {len(self.date_cols)} date columns")
    
    def _detect_column_types(self):
        """Automatically detect column types"""
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                if self.df[col].nunique() < 10 and col.lower() in ['loginattempts', 'dayofweek', 'month', 'hour']:
                    self.categorical_cols.append(col)
                else:
                    self.numerical_cols.append(col)
            elif self.df[col].dtype == 'object':
                # Check if it looks like a date
                if any(word in col.lower() for word in ['date', 'time']):
                    self.date_cols.append(col)
                else:
                    self.categorical_cols.append(col)
    
    def comprehensive_data_quality_assessment(self):
        """
        Comprehensive data quality assessment using existing data_quality_assessment.py
        """
        print("COMPREHENSIVE DATA QUALITY ASSESSMENT")
        print("=" * 50)
        
        # Import and use existing data quality assessment
        try:
            from data_quality_assessment import assess_data_quality
            print("Using existing data_quality_assessment.py module")
            
            # Run the existing assessment
            self.df = assess_data_quality(self.df)
            
            print("\n✓ Data quality assessment completed using existing module")
            
        except ImportError:
            print("⚠ data_quality_assessment.py not found, using built-in assessment")
            self._fallback_data_quality_assessment()
        
        return self.df
    
    def _fallback_data_quality_assessment(self):
        """Fallback data quality assessment if module not available"""
        print("\nRunning fallback data quality assessment...")
        
        # Basic missing value check
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() == 0:
            print("✓ No missing values found")
        else:
            print("Missing values found:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  {col}: {count}")
        
        # Basic duplicate check
        exact_dupes = self.df.duplicated().sum()
        if exact_dupes > 0:
            print(f"Removing {exact_dupes} duplicate rows...")
            self.df = self.df.drop_duplicates()
        else:
            print("✓ No duplicate rows found")
        
        # Convert date columns
        for col in self.date_cols:
            if self.df[col].dtype == 'object':
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    print(f"✓ Converted {col} to datetime")
                except:
                    print(f"⚠ Could not convert {col} to datetime")
        
        print("✓ Fallback data quality assessment completed")
    
    def detailed_categorical_analysis(self):
        """Detailed analysis of categorical features"""
        print("\nDETAILED CATEGORICAL FEATURES ANALYSIS")
        print("=" * 45)
        
        categorical_summary = {}
        
        for col in self.categorical_cols:
            if col in self.df.columns:
                value_counts = self.df[col].value_counts()
                
                print(f"\n{col}:")
                print(f"  Unique values: {self.df[col].nunique()}")
                print(f"  Most common values:")
                
                # Show top 5 values
                for value, count in value_counts.head().items():
                    percentage = (count / len(self.df)) * 100
                    print(f"    {value}: {count} ({percentage:.2f}%)")
                
                if len(value_counts) > 5:
                    print(f"    ... and {len(value_counts) - 5} other values")
                
                # Store summary
                categorical_summary[col] = {
                    'unique_count': self.df[col].nunique(),
                    'top_value': value_counts.index[0],
                    'top_count': value_counts.iloc[0],
                    'top_percentage': (value_counts.iloc[0] / len(self.df)) * 100
                }
        
        self.categorical_summary = categorical_summary
        print("\n✓ Categorical analysis completed")
        return categorical_summary
    
    def outlier_detection_analysis(self):
        """Multi-method outlier detection"""
        print("\nOUTLIER DETECTION ANALYSIS")
        print("=" * 30)
        
        outlier_methods = {}
        
        print("1. IQR Method:")
        print("-" * 15)
        
        outlier_methods['IQR'] = {}
        for col in self.numerical_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_count = outliers_mask.sum()
                outlier_percentage = (outlier_count / len(self.df)) * 100
                
                outlier_methods['IQR'][col] = {
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'mask': outliers_mask
                }
                
                print(f"  {col}: {outlier_count} outliers ({outlier_percentage:.2f}%)")
        
        print("\n2. Z-Score Method:")
        print("-" * 18)
        
        outlier_methods['Z_Score'] = {}
        for col in self.numerical_cols:
            if col in self.df.columns:
                z_scores = np.abs(stats.zscore(self.df[col]))
                outliers_mask = z_scores > 3
                outlier_count = outliers_mask.sum()
                outlier_percentage = (outlier_count / len(self.df)) * 100
                
                outlier_methods['Z_Score'][col] = {
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'mask': outliers_mask
                }
                
                print(f"  {col}: {outlier_count} outliers ({outlier_percentage:.2f}%)")
        
        # Consensus scoring
        print("\n3. Consensus Outlier Scoring:")
        print("-" * 30)
        
        outlier_scores = pd.DataFrame(index=self.df.index)
        
        for col in self.numerical_cols:
            if col in self.df.columns:
                col_score = pd.Series(0, index=self.df.index)
                
                # Add points for each method that detects an outlier
                for method in outlier_methods:
                    if col in outlier_methods[method]:
                        col_score += outlier_methods[method][col]['mask'].astype(int)
                
                outlier_scores[f'{col}_outlier_score'] = col_score
        
        # Total outlier score per row
        outlier_scores['total_outlier_score'] = outlier_scores.sum(axis=1)
        
        # High-confidence outliers
        high_confidence_outliers = outlier_scores['total_outlier_score'] >= 2
        print(f"High-confidence outliers (≥2 methods): {high_confidence_outliers.sum()} ({high_confidence_outliers.sum()/len(self.df)*100:.2f}%)")
        
        # Add outlier flags to dataframe
        self.df['is_outlier'] = high_confidence_outliers
        self.df['outlier_score'] = outlier_scores['total_outlier_score']
        
        self.outlier_methods = outlier_methods
        self.outlier_scores = outlier_scores
        
        print("✓ Outlier detection completed")
        return outlier_methods
    
    def multivariate_analysis(self):
        """Comprehensive multivariate analysis"""
        print("\nMULTIVARIATE ANALYSIS")
        print("=" * 25)
        
        # 1. Correlation Analysis
        print("1. Correlation Matrix Analysis")
        print("-" * 30)
        
        if len(self.numerical_cols) > 1:
            # Get numerical columns that exist in dataframe
            available_numerical = [col for col in self.numerical_cols if col in self.df.columns]
            
            if len(available_numerical) > 1:
                corr_matrix = self.df[available_numerical].corr()
                
                # Find strong correlations
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.3:
                            strong_correlations.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': corr_val
                            })
                
                print(f"Strong correlations (|r| > 0.3):")
                for corr in sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True):
                    print(f"  {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}")
                
                self.correlation_matrix = corr_matrix
                self.strong_correlations = strong_correlations
            else:
                print("Not enough numerical columns for correlation analysis")
        
        # 2. Network Analysis (if applicable)
        self._network_analysis()
        
        print("✓ Multivariate analysis completed")
    
    def _network_analysis(self):
        """Analyze network patterns in device/IP data"""
        print("\n2. Network Pattern Analysis")
        print("-" * 25)
        
        # Look for device/IP sharing patterns
        device_cols = [col for col in self.df.columns if 'device' in col.lower()]
        ip_cols = [col for col in self.df.columns if 'ip' in col.lower()]
        account_cols = [col for col in self.df.columns if 'account' in col.lower()]
        
        if device_cols and account_cols:
            device_col = device_cols[0]
            account_col = account_cols[0]
            
            device_usage = self.df.groupby(device_col).agg({
                account_col: 'nunique',
                self.df.columns[0]: 'count'  # Transaction count
            }).rename(columns={
                account_col: 'unique_accounts',
                self.df.columns[0]: 'transaction_count'
            })
            
            shared_devices = device_usage[device_usage['unique_accounts'] > 1]
            print(f"Devices used by multiple accounts: {len(shared_devices)} out of {len(device_usage)} ({len(shared_devices)/len(device_usage)*100:.2f}%)")
            
            self.device_usage = device_usage
        
        if ip_cols and account_cols:
            ip_col = ip_cols[0]
            account_col = account_cols[0]
            location_cols = [col for col in self.df.columns if 'location' in col.lower()]
            
            if location_cols:
                location_col = location_cols[0]
                ip_usage = self.df.groupby(ip_col).agg({
                    account_col: 'nunique',
                    location_col: 'nunique',
                    self.df.columns[0]: 'count'
                }).rename(columns={
                    account_col: 'unique_accounts',
                    location_col: 'unique_locations',
                    self.df.columns[0]: 'transaction_count'
                })
                
                multi_location_ips = ip_usage[ip_usage['unique_locations'] > 1]
                print(f"IPs used from multiple locations: {len(multi_location_ips)} out of {len(ip_usage)} ({len(multi_location_ips)/len(ip_usage)*100:.2f}%)")
                
                self.ip_usage = ip_usage
    
    def feature_importance_analysis(self, target_col=None):
        """
        Analyze feature importance using multiple methods
        Must be called after target creation
        """
        if target_col is None and self.target_column is None:
            print("⚠ No target column specified. Skipping feature importance analysis.")
            return None
        
        target_col = target_col or self.target_column
        
        if target_col not in self.df.columns:
            print(f"⚠ Target column '{target_col}' not found. Skipping feature importance analysis.")
            return None
        
        print("\nFEATURE IMPORTANCE ANALYSIS")
        print("=" * 30)
        
        # Prepare features for analysis
        all_potential_features = self.numerical_cols + self.categorical_cols
        
        # Remove duplicates and ensure target column is excluded
        feature_cols = []
        seen_cols = set()
        
        for col in all_potential_features:
            if (col in self.df.columns and 
                col != target_col and 
                col not in seen_cols and
                not col.startswith('Unnamed')):  # Skip unnamed columns
                feature_cols.append(col)
                seen_cols.add(col)
        
        print(f"Features for analysis: {len(feature_cols)} columns")
        print(f"Excluded target: {target_col}")
        
        if len(feature_cols) == 0:
            print("No features available for importance analysis")
            return None
        
        # Encode categorical variables for analysis
        X_encoded = self.df[feature_cols].copy()
        
        # Ensure target is a 1D array
        if isinstance(target_col, list):
            target_col = target_col[0]  # Take first column if list
        
        y = self.df[target_col].copy()
        
        # Ensure y is 1D
        if hasattr(y, 'values'):
            y = y.values
        if len(y.shape) > 1:
            y = y.flatten()
        
        print(f"Target variable: {target_col}")
        print(f"Target shape: {y.shape}")
        print(f"Target unique values: {np.unique(y)}")
        
        # Encode categorical columns
        label_encoders = {}
        for col in self.categorical_cols:
            if col in X_encoded.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                label_encoders[col] = le
        
        # 1. Random Forest Feature Importance
        print("1. Random Forest Feature Importance")
        print("-" * 35)
        
        try:
            # Check if we have enough samples
            if len(y) < 10:
                print("Not enough samples for Random Forest analysis")
                raise ValueError("Insufficient samples")
            
            # Determine if classification or regression
            unique_values = np.unique(y)
            if len(unique_values) <= 10 and all(isinstance(val, (int, np.integer)) for val in unique_values):
                # Classification
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
                print(f"Using Random Forest Classifier ({len(unique_values)} classes)")
            else:
                # Regression
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
                print(f"Using Random Forest Regressor")
            
            # Fit the model
            rf.fit(X_encoded, y)
            
            rf_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 10 features by Random Forest importance:")
            for i, (_, row) in enumerate(rf_importance.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:<25} {row['importance']:.4f}")
            
            self.rf_importance = rf_importance
        
        except Exception as e:
            print(f"Random Forest analysis failed: {e}")
        
        # 2. Mutual Information
        print("\n2. Mutual Information Analysis")
        print("-" * 30)
        
        try:
            # Check for sufficient samples and variance
            if len(y) < 10:
                print("Not enough samples for Mutual Information analysis")
                raise ValueError("Insufficient samples")
            
            unique_values = np.unique(y)
            if len(unique_values) <= 20:  # Classification or few unique values
                mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
                print(f"Using Mutual Information Classification ({len(unique_values)} classes)")
            else:  # Regression
                from sklearn.feature_selection import mutual_info_regression
                mi_scores = mutual_info_regression(X_encoded, y, random_state=42)
                print(f"Using Mutual Information Regression")
            
            mi_importance = pd.DataFrame({
                'feature': feature_cols,
                'mutual_info': mi_scores
            }).sort_values('mutual_info', ascending=False)
            
            print("Top 10 features by Mutual Information:")
            for i, (_, row) in enumerate(mi_importance.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:<25} {row['mutual_info']:.4f}")
            
            self.mi_importance = mi_importance
        
        except Exception as e:
            print(f"Mutual Information analysis failed: {e}")
        
        print("✓ Feature importance analysis completed")
        return getattr(self, 'rf_importance', None)
    
    def advanced_visualizations(self):
        """Create advanced visualizations for the analysis"""
        print("\nADVANCED VISUALIZATIONS")
        print("=" * 25)
        
        # 1. Correlation Heatmap
        if hasattr(self, 'correlation_matrix'):
            self._plot_correlation_heatmap()
        
        # 2. Outlier Visualizations
        if hasattr(self, 'outlier_methods'):
            self._plot_outlier_analysis()
        
        # 3. Categorical Distribution Plots
        self._plot_categorical_distributions()
        
        # 4. Feature Importance Plots
        if hasattr(self, 'rf_importance'):
            self._plot_feature_importance()
        
        print("✓ Advanced visualizations completed")
    
    def _plot_correlation_heatmap(self):
        """Plot correlation matrix heatmap"""
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(self.correlation_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": .8})
        
        plt.title('Correlation Matrix of Numerical Features', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
    
    def _plot_outlier_analysis(self):
        """Plot outlier detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Outlier counts by method
        outlier_counts = {}
        for method in self.outlier_methods:
            counts = [self.outlier_methods[method][col]['count'] 
                     for col in self.numerical_cols if col in self.outlier_methods[method]]
            outlier_counts[method] = sum(counts)
        
        methods = list(outlier_counts.keys())
        counts = list(outlier_counts.values())
        
        axes[0, 0].bar(methods, counts, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Outliers Detected by Method')
        axes[0, 0].set_ylabel('Number of Outliers')
        
        # Outlier score distribution
        if 'total_outlier_score' in self.df.columns:
            score_dist = self.df['total_outlier_score'].value_counts().sort_index()
            axes[0, 1].bar(score_dist.index, score_dist.values, color='lightgreen')
            axes[0, 1].set_title('Outlier Score Distribution')
            axes[0, 1].set_xlabel('Outlier Score')
            axes[0, 1].set_ylabel('Number of Transactions')
        
        # Feature-wise outlier comparison (top 4 features)
        top_features = self.numerical_cols[:4]
        for i, feature in enumerate(top_features):
            if i < 2 and feature in self.df.columns:
                row = 1
                col = i
                
                iqr_outliers = self.outlier_methods.get('IQR', {}).get(feature, {}).get('count', 0)
                zscore_outliers = self.outlier_methods.get('Z_Score', {}).get(feature, {}).get('count', 0)
                
                axes[row, col].bar(['IQR', 'Z-Score'], [iqr_outliers, zscore_outliers], 
                                 color=['orange', 'purple'])
                axes[row, col].set_title(f'Outliers in {feature}')
                axes[row, col].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_categorical_distributions(self):
        """Plot categorical feature distributions"""
        n_categorical = len(self.categorical_cols)
        if n_categorical == 0:
            return
        
        # Limit to first 6 categorical features for visualization
        cat_cols_to_plot = [col for col in self.categorical_cols[:6] if col in self.df.columns]
        
        if len(cat_cols_to_plot) == 0:
            return
        
        n_cols = min(3, len(cat_cols_to_plot))
        n_rows = (len(cat_cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cat_cols_to_plot):
            if i < len(axes):
                value_counts = self.df[col].value_counts()
                
                # Show top 10 values max
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                
                axes[i].bar(range(len(value_counts)), value_counts.values, color='skyblue')
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_ylabel('Count')
                
                if len(value_counts) <= 5:
                    axes[i].set_xticks(range(len(value_counts)))
                    axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                else:
                    axes[i].set_xlabel('Categories (ordered by frequency)')
        
        # Hide extra subplots
        for i in range(len(cat_cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_feature_importance(self):
        """Plot feature importance analysis results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Random Forest importance
        if hasattr(self, 'rf_importance'):
            top_rf = self.rf_importance.head(10)
            axes[0].barh(range(len(top_rf)), top_rf['importance'], color='lightgreen')
            axes[0].set_yticks(range(len(top_rf)))
            axes[0].set_yticklabels(top_rf['feature'])
            axes[0].set_title('Random Forest Feature Importance')
            axes[0].set_xlabel('Importance Score')
            axes[0].invert_yaxis()
        
        # Mutual Information importance
        if hasattr(self, 'mi_importance'):
            top_mi = self.mi_importance.head(10)
            axes[1].barh(range(len(top_mi)), top_mi['mutual_info'], color='lightcoral')
            axes[1].set_yticks(range(len(top_mi)))
            axes[1].set_yticklabels(top_mi['feature'])
            axes[1].set_title('Mutual Information Feature Importance')
            axes[1].set_xlabel('Mutual Information Score')
            axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    def generate_eda_report(self):
        """Generate a comprehensive EDA report"""
        print("\nEDA PIPELINE SUMMARY REPORT")
        print("=" * 40)
        
        print(f"Dataset: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        if hasattr(self, 'categorical_summary'):
            print(f"\nCategorical Features: {len(self.categorical_summary)}")
            for col, summary in self.categorical_summary.items():
                print(f"  {col}: {summary['unique_count']} unique values")
        
        if hasattr(self, 'outlier_methods'):
            total_outliers = sum([method_data[col]['count'] 
                                for method_data in self.outlier_methods.values() 
                                for col in method_data])
            print(f"\nOutliers detected: {total_outliers} total across all methods")
        
        if hasattr(self, 'strong_correlations'):
            print(f"\nStrong correlations found: {len(self.strong_correlations)}")
        
        if hasattr(self, 'rf_importance'):
            top_feature = self.rf_importance.iloc[0]
            print(f"\nMost important feature: {top_feature['feature']} (importance: {top_feature['importance']:.3f})")
        
        print(f"\n✓ EDA pipeline completed successfully")
        print(f"✓ Dataset ready for feature engineering and modeling")
        
        return {
            'shape': self.df.shape,
            'categorical_features': len(self.categorical_cols),
            'numerical_features': len(self.numerical_cols),
            'outliers_detected': hasattr(self, 'outlier_methods'),
            'correlations_found': len(getattr(self, 'strong_correlations', [])),
            'feature_importance_completed': hasattr(self, 'rf_importance')
        }
    
    def get_cleaned_dataset(self):
        """Return the cleaned and processed dataset"""
        return self.df.copy()
    
    def update_dataset(self, new_df):
        """Update the internal dataset (useful after feature engineering)"""
        self.df = new_df.copy()
        # Re-detect column types for new features
        self._detect_column_types()
        print(f"Dataset updated: {self.df.shape}")
    
    def run_post_feature_engineering_analysis(self, target_column=None):
        """
        Run analysis after feature engineering (outliers, correlations, feature importance)
        """
        print("\nPOST-FEATURE ENGINEERING ANALYSIS")
        print("=" * 40)
        
        # Update target column if provided
        if target_column:
            self.target_column = target_column
        
        # Re-run analyses that benefit from engineered features
        self.outlier_detection_analysis()
        self.multivariate_analysis()
        
        # Feature importance (if target available)
        if self.target_column and self.target_column in self.df.columns:
            self.feature_importance_analysis()
        
        # Generate visualizations
        self.advanced_visualizations()
        
        print("✓ Post-feature engineering analysis completed")
        return self.df
    
    def run_full_pipeline(self, target_column=None):
        """
        Run the complete EDA pipeline
        
        Parameters:
        target_column (str): Name of target column for feature importance analysis
        """
        print("RUNNING COMPREHENSIVE EDA PIPELINE")
        print("=" * 50)
        
        # Update target column if provided
        if target_column:
            self.target_column = target_column
        
        # Run all analyses
        self.comprehensive_data_quality_assessment()
        self.detailed_categorical_analysis()
        self.outlier_detection_analysis()
        self.multivariate_analysis()
        
        # Feature importance (if target available)
        if self.target_column and self.target_column in self.df.columns:
            self.feature_importance_analysis()
        
        # Generate visualizations
        self.advanced_visualizations()
        
        # Generate final report
        report = self.generate_eda_report()
        
        return self.df, report