# Fraud Detection EDA Plan

## 1. Data Quality & Initial Exploration
- [x] Check dataset shape and structure (10,000 rows, 17 columns)
- [x] Identify fraud target variable (⚠️ NO FRAUD LABEL FOUND - need synthetic labels)
- [x] Check for missing values (None found)
- [x] Identify duplicate transactions (0 duplicates)
- [x] Verify data types and perform necessary conversions
- [x] Extract datetime features from TransactionDate (Hour, DayOfWeek, Month, DayOfMonth)

## 2. Univariate Analysis

### Numerical Features
- [x] **TransactionAmount**
  - Distribution analysis (histogram, box plot) ✓
  - Statistical summary (mean: 248.25, median: 174.51)
  - Identify outliers and extreme values (15.22% outliers)
  - High skewness (1.436) and kurtosis (2.298)

- [x] **CustomerAge**
  - Age distribution analyzed ✓
  - Range: 18-64 years, mean: 40.84
  - Normal distribution (skew: -0.019)

- [x] **TransactionDuration**
  - Distribution of transaction times ✓
  - Mean: 59.97 seconds, highly variable
  - 15.22% outliers detected

- [x] **LoginAttempts**
  - Frequency distribution analyzed ✓
  - Mean: 1.5 attempts
  - Multiple attempts flagged as risk indicator

- [x] **AccountBalance**
  - Balance distribution analyzed ✓
  - Mean: $49,991, well distributed
  - Balance to transaction amount ratio calculated

### Categorical Features
- [x] **TransactionType (Debit/Credit)**
  - Debit: 59.84%, Credit: 40.16%
  - Proportion analysis completed

- [x] **Channel (ATM/Online)**
  - Online: 59.72%, ATM: 40.28%
  - Channel usage patterns identified

- [x] **Location**
  - 20 unique locations identified
  - Top location: Location_10 (5.16%)
  - Geographic distribution fairly uniform

- [x] **CustomerOccupation**
  - 10 occupations identified
  - Top: Engineer (10.38%), Doctor (10.09%)
  - Distribution analyzed

- [x] **MerchantID**
  - 50 unique merchants
  - Top merchant: Merchant_12 (2.32%)
  - Fairly distributed transactions

### Temporal Features
- [x] **TransactionDate Analysis**
  - Extracted hour, day, month, day of week ✓
  - Peak hours: 12-14 (lunch time)
  - All weekdays present in data
  - Monthly patterns show variation

- [x] **Time Since Previous Transaction**
  - Time gaps calculated ✓
  - Median: 360.36 hours
  - Quick transactions (<0.5h) flagged as risk

## 3. Risk Scoring (Since No Fraud Labels)
- [x] Created synthetic risk score based on anomaly patterns
- [x] Risk indicators identified:
  - High transaction amounts (>95th percentile)
  - Multiple login attempts (>2)
  - Unusual hours (0-5 AM)
  - Quick successive transactions (<0.5 hours)
  - High amount-to-balance ratio (>95th percentile)
- [x] Risk score distribution:
  - Score 0: 60.79%
  - Score 1: 28.18%
  - Score 2: 8.83%
  - Score 3: 1.90%
  - Score 4: 0.29%
  - Score 5: 0.01%
- [x] High-risk transactions (score ≥3): 2.20%

## 4. Multivariate Analysis

### Complex Pattern Detection
- [ ] **Transaction Behavior Profiles**
  - Amount + Time + Channel combinations
  - Customer demographic risk profiles
  - Merchant + Location + Time patterns

- [ ] **Anomaly Indicators**
  - Unusual transaction sequences
  - Cross-account linked behaviors
  - Network analysis (shared devices/IPs)

- [ ] **Feature Interactions**
  - Correlation matrix for numerical features
  - Chi-square tests for categorical associations
  - Feature importance preliminary assessment

## 5. Data Preparation for Modeling

### Feature Engineering Ideas
- [ ] Transaction velocity features
- [ ] Account age and history features
- [ ] Time-based aggregations
- [ ] Risk scores for merchants/locations
- [ ] Customer behavior deviation metrics

### Data Quality Improvements
- [ ] Handle missing values strategy
- [ ] Outlier treatment approach
- [ ] Feature scaling considerations
- [ ] Categorical encoding plans

## 6. Visualization Summary

### Key Visualizations to Create
- [ ] Fraud rate overview dashboard
- [ ] Transaction amount distribution plots
- [ ] Time-based pattern heatmaps
- [ ] Geographic fraud distribution
- [ ] Feature correlation heatmap
- [ ] Top risk factors summary chart

## 7. EDA Findings Documentation

### Document Key Insights:
- [ ] Main fraud indicators identified
- [ ] Data quality issues found
- [ ] Feature engineering opportunities
- [ ] Modeling recommendations
- [ ] Business insights for stakeholders