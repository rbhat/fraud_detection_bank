import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self, X, y, test_size=0.2, random_state=42, cv_folds=5):
        """
        Initialize the model comparison framework
        
        Parameters:
        X: Feature matrix
        y: Target vector
        test_size: Test set proportion
        random_state: Random seed for reproducibility
        cv_folds: Number of cross-validation folds
        """
        self.X = X
        self.y = y
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        # Initialize CV strategy
        self.cv = cv_folds
        
        # Initialize scaler
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        # Results storage
        self.model_results = {
            'model_name': [],
            'cv_f1_mean': [],
            'cv_f1_std': [],
            'cv_precision_mean': [],
            'cv_precision_std': [],
            'cv_recall_mean': [],
            'cv_recall_std': [],
            'cv_roc_auc_mean': [],
            'cv_roc_auc_std': []
        }
        
        # Store fitted models
        self.fitted_models = {}
        
        # Define all models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=random_state, class_weight='balanced'),
            'Extra Trees': ExtraTreesClassifier(random_state=random_state, class_weight='balanced'),
            'SVM': SVC(random_state=random_state, class_weight='balanced', probability=True),
            'Neural Network': MLPClassifier(random_state=random_state, max_iter=1000)
        }
        
        # Define parameter grids for tuning
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Decision Tree': {
                'max_depth': [5, 10, 20, None],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
    
    def evaluate_model_cv(self, model, model_name, use_scaled=False):
        """
        Evaluate a single model using cross-validation
        
        Parameters:
        model: Sklearn model object
        model_name: String name for the model
        use_scaled: Whether to use scaled features
        """
        print(f"Evaluating {model_name}...")
        
        # Choose data
        X_data = self.X_scaled if use_scaled else self.X
        
        # Cross-validation scoring
        scoring_metrics = ['f1', 'precision', 'recall', 'roc_auc']
        cv_scores = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X_data, self.y, cv=self.cv, scoring=metric)
            cv_scores[metric] = scores
        
        # Store results
        self.model_results['model_name'].append(model_name)
        self.model_results['cv_f1_mean'].append(cv_scores['f1'].mean())
        self.model_results['cv_f1_std'].append(cv_scores['f1'].std())
        self.model_results['cv_precision_mean'].append(cv_scores['precision'].mean())
        self.model_results['cv_precision_std'].append(cv_scores['precision'].std())
        self.model_results['cv_recall_mean'].append(cv_scores['recall'].mean())
        self.model_results['cv_recall_std'].append(cv_scores['recall'].std())
        self.model_results['cv_roc_auc_mean'].append(cv_scores['roc_auc'].mean())
        self.model_results['cv_roc_auc_std'].append(cv_scores['roc_auc'].std())
        
        # Fit model and store
        X_data = self.X_scaled if use_scaled else self.X
        model.fit(X_data, self.y)
        self.fitted_models[model_name] = model
        
        print(f"{model_name} - F1: {cv_scores['f1'].mean():.3f} (+/- {cv_scores['f1'].std()*2:.3f})")
        
        return cv_scores
    
    def evaluate_all_models(self):
        """
        Evaluate all models using cross-validation
        """
        print("Starting model comparison with 5-fold cross-validation...\n")
        
        # Models that need scaled features
        scaled_models = ['SVM', 'Neural Network']
        
        for name, model in self.models.items():
            use_scaled = name in scaled_models
            self.evaluate_model_cv(model, name, use_scaled)
            print()
        
        print("Model evaluation complete!")
        return pd.DataFrame(self.model_results)
    
    def tune_hyperparameters(self, model_names=None):
        """
        Tune hyperparameters for specified models using GridSearchCV
        
        Parameters:
        model_names: List of model names to tune (if None, tune all available)
        """
        if model_names is None:
            model_names = list(self.param_grids.keys())
        
        tuned_models = {}
        scaled_models = ['SVM', 'Neural Network']
        
        for name in model_names:
            if name not in self.param_grids:
                print(f"No parameter grid defined for {name}")
                continue
                
            print(f"Tuning {name}...")
            
            model = self.models[name]
            param_grid = self.param_grids[name]
            use_scaled = name in scaled_models
            X_data = self.X_scaled if use_scaled else self.X
            
            search = GridSearchCV(model, param_grid, cv=self.cv, scoring='f1', n_jobs=-1)
            search.fit(X_data, self.y)
            tuned_models[name] = search.best_estimator_
            
            print(f"Best {name} score: {search.best_score_:.3f}")
            print(f"Best {name} params: {search.best_params_}\n")
        
        return tuned_models
    
    def plot_model_comparison(self, figsize=(15, 10)):
        """
        Create comprehensive model comparison plots
        """
        if not self.model_results['model_name']:
            print("No model results to plot. Run evaluate_all_models() first.")
            return
        
        df_results = pd.DataFrame(self.model_results)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Comparison - Cross-Validation Results', fontsize=16, fontweight='bold')
        
        metrics = ['cv_f1_mean', 'cv_precision_mean', 'cv_recall_mean', 'cv_roc_auc_mean']
        titles = ['F1 Score', 'Precision', 'Recall', 'ROC-AUC']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            # Create bar plot with error bars
            std_metric = metric.replace('_mean', '_std')
            bars = ax.bar(df_results['model_name'], df_results[metric], 
                         yerr=df_results[std_metric], capsize=5, alpha=0.7)
            
            # Color bars
            colors = plt.cm.Set3(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, (bar, val) in enumerate(zip(bars, df_results[metric])):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + df_results[std_metric].iloc[j],
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_cv_score_distributions(self):
        """
        Plot distribution of CV scores for each model
        """
        if not self.fitted_models:
            print("No fitted models found. Run evaluate_all_models() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cross-Validation Score Distributions', fontsize=16, fontweight='bold')
        
        metrics = ['f1', 'precision', 'recall', 'roc_auc']
        titles = ['F1 Score Distribution', 'Precision Distribution', 'Recall Distribution', 'ROC-AUC Distribution']
        
        # Recalculate CV scores for box plots
        cv_data = {metric: [] for metric in metrics}
        model_names = []
        
        scaled_models = ['SVM', 'Neural Network']
        
        for name, model in self.models.items():
            if name in self.fitted_models:
                use_scaled = name in scaled_models
                X_data = self.X_scaled if use_scaled else self.X
                
                for metric in metrics:
                    scores = cross_val_score(model, X_data, self.y, cv=self.cv, scoring=metric)
                    cv_data[metric].extend(scores)
                
                model_names.extend([name] * self.cv_folds)
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            # Create DataFrame for seaborn
            plot_data = pd.DataFrame({
                'Model': model_names,
                'Score': cv_data[metric]
            })
            
            sns.boxplot(data=plot_data, x='Model', y='Score', ax=ax)
            ax.set_title(title, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def get_results_summary(self):
        """
        Get a summary of all model results
        """
        if not self.model_results['model_name']:
            return "No results available. Run evaluate_all_models() first."
        
        df = pd.DataFrame(self.model_results)
        df = df.round(3)
        
        # Add ranking
        df['f1_rank'] = df['cv_f1_mean'].rank(ascending=False).astype(int)
        df['roc_auc_rank'] = df['cv_roc_auc_mean'].rank(ascending=False).astype(int)
        
        # Sort by F1 score
        df = df.sort_values('cv_f1_mean', ascending=False)
        
        return df
    
    def plot_feature_importance(self, top_n=15):
        """
        Plot feature importance for tree-based models
        """
        tree_models = ['Random Forest', 'Extra Trees', 'Decision Tree']
        available_models = [name for name in tree_models if name in self.fitted_models]
        
        if not available_models:
            print("No tree-based models found for feature importance analysis.")
            return
        
        n_models = len(available_models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        for i, model_name in enumerate(available_models):
            model = self.fitted_models[model_name]
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = [f'feature_{i}' for i in range(len(importance))]
                
                # Sort and get top N
                indices = np.argsort(importance)[::-1][:top_n]
                
                axes[i].barh(range(top_n), importance[indices])
                axes[i].set_yticks(range(top_n))
                axes[i].set_yticklabels([feature_names[j] for j in indices])
                axes[i].set_xlabel('Importance')
                axes[i].set_title(f'{model_name}\nFeature Importance')
                axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return fig