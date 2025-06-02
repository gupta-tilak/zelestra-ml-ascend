import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    def __init__(self, df, target_column='efficiency'):
        self.df = df.copy()
        self.target_column = target_column
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results = {}
        
    def preprocess_data(self):
        """Preprocess the data for analysis"""
        print("Preprocessing data...")
        
        # Separate target variable
        self.y = self.df[self.target_column]
        self.X = self.df.drop([self.target_column], axis=1)
        
        # Handle categorical variables
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            print(f"Encoding categorical variables: {list(categorical_cols)}")
            le = LabelEncoder()
            for col in categorical_cols:
                if self.X[col].dtype == 'category':
                    self.X[col] = self.X[col].astype(str)
                self.X[col] = le.fit_transform(self.X[col])
        
        # Remove ID column if present
        if 'id' in self.X.columns:
            self.X = self.X.drop('id', axis=1)
        
        self.feature_names = list(self.X.columns)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"Data shape: {self.X.shape}")
        print(f"Features: {len(self.feature_names)}")
        
    def correlation_analysis(self):
        """Analyze correlation between features and target"""
        print("\n=== Correlation Analysis ===")
        
        correlations = []
        for feature in self.feature_names:
            corr_coef, p_value = pearsonr(self.X[feature], self.y)
            correlations.append({
                'Feature': feature,
                'Correlation': abs(corr_coef),
                'Correlation_Raw': corr_coef,
                'P_Value': p_value
            })
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
        self.results['correlation'] = corr_df
        
        print("Top 10 features by absolute correlation:")
        print(corr_df.head(10)[['Feature', 'Correlation_Raw', 'P_Value']])
        
        return corr_df
    
    def univariate_selection(self):
        """Perform univariate feature selection"""
        print("\n=== Univariate Feature Selection ===")
        
        # F-test
        f_selector = SelectKBest(score_func=f_regression, k='all')
        f_selector.fit(self.X_train, self.y_train)
        
        f_scores = pd.DataFrame({
            'Feature': self.feature_names,
            'F_Score': f_selector.scores_,
            'F_P_Value': f_selector.pvalues_
        }).sort_values('F_Score', ascending=False)
        
        # Mutual Information
        mi_scores = mutual_info_regression(self.X_train, self.y_train, random_state=42)
        mi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        self.results['f_test'] = f_scores
        self.results['mutual_info'] = mi_df
        
        print("Top 10 features by F-score:")
        print(f_scores.head(10)[['Feature', 'F_Score']])
        
        print("\nTop 10 features by Mutual Information:")
        print(mi_df.head(10))
        
        return f_scores, mi_df
    
    def tree_based_importance(self):
        """Calculate feature importance using tree-based models"""
        print("\n=== Tree-Based Feature Importance ===")
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        
        rf_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'RF_Importance': rf.feature_importances_
        }).sort_values('RF_Importance', ascending=False)
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        
        gb_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'GB_Importance': gb.feature_importances_
        }).sort_values('GB_Importance', ascending=False)
        
        self.results['random_forest'] = rf_importance
        self.results['gradient_boosting'] = gb_importance
        
        print("Top 10 features by Random Forest importance:")
        print(rf_importance.head(10))
        
        print("\nTop 10 features by Gradient Boosting importance:")
        print(gb_importance.head(10))
        
        return rf_importance, gb_importance
    
    def regularization_based_selection(self):
        """Feature selection using regularization methods"""
        print("\n=== Regularization-Based Feature Selection ===")
        
        # Standardize features for regularization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # LASSO with cross-validation
        lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
        lasso.fit(X_train_scaled, self.y_train)
        
        lasso_coef = pd.DataFrame({
            'Feature': self.feature_names,
            'Lasso_Coefficient': abs(lasso.coef_),
            'Lasso_Coefficient_Raw': lasso.coef_
        }).sort_values('Lasso_Coefficient', ascending=False)
        
        # Ridge with cross-validation
        ridge = RidgeCV(cv=5)
        ridge.fit(X_train_scaled, self.y_train)
        
        ridge_coef = pd.DataFrame({
            'Feature': self.feature_names,
            'Ridge_Coefficient': abs(ridge.coef_),
            'Ridge_Coefficient_Raw': ridge.coef_
        }).sort_values('Ridge_Coefficient', ascending=False)
        
        self.results['lasso'] = lasso_coef
        self.results['ridge'] = ridge_coef
        
        print("Top 10 features by LASSO coefficient magnitude:")
        print(lasso_coef.head(10)[['Feature', 'Lasso_Coefficient_Raw']])
        
        # Features selected by LASSO (non-zero coefficients)
        selected_features = lasso_coef[lasso_coef['Lasso_Coefficient'] > 0]['Feature'].tolist()
        print(f"\nLASSO selected {len(selected_features)} features out of {len(self.feature_names)}")
        
        return lasso_coef, ridge_coef
    
    def recursive_feature_elimination(self):
        """Perform Recursive Feature Elimination"""
        print("\n=== Recursive Feature Elimination ===")
        
        # RFE with Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rfe = RFE(estimator=rf, n_features_to_select=10, step=1)
        rfe.fit(self.X_train, self.y_train)
        
        rfe_results = pd.DataFrame({
            'Feature': self.feature_names,
            'Selected': rfe.support_,
            'Ranking': rfe.ranking_
        }).sort_values('Ranking')
        
        self.results['rfe'] = rfe_results
        
        selected_features = rfe_results[rfe_results['Selected']]['Feature'].tolist()
        print(f"RFE selected top 10 features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")
        
        return rfe_results
    
    def variance_threshold_selection(self):
        """Remove features with low variance"""
        print("\n=== Variance Threshold Selection ===")
        
        # Calculate variance for each feature
        variances = pd.DataFrame({
            'Feature': self.feature_names,
            'Variance': self.X.var()
        }).sort_values('Variance', ascending=False)
        
        # Apply variance threshold
        threshold = 0.01  # Remove features with variance < 0.01
        var_selector = VarianceThreshold(threshold=threshold)
        var_selector.fit(self.X)
        
        selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) 
                           if var_selector.get_support()[i]]
        
        print(f"Features with variance > {threshold}: {len(selected_features)}")
        print(f"Removed {len(self.feature_names) - len(selected_features)} low-variance features")
        
        self.results['variance'] = variances
        
        return variances
    
    def create_summary_ranking(self):
        """Create a comprehensive ranking of features"""
        print("\n=== Creating Summary Ranking ===")
        
        # Normalize rankings from different methods
        rankings = pd.DataFrame({'Feature': self.feature_names})
        
        # Add correlation ranking
        corr_ranking = self.results['correlation'].reset_index(drop=True)
        corr_ranking['Corr_Rank'] = range(1, len(corr_ranking) + 1)
        rankings = rankings.merge(corr_ranking[['Feature', 'Corr_Rank']], on='Feature')
        
        # Add F-test ranking
        f_ranking = self.results['f_test'].reset_index(drop=True)
        f_ranking['F_Rank'] = range(1, len(f_ranking) + 1)
        rankings = rankings.merge(f_ranking[['Feature', 'F_Rank']], on='Feature')
        
        # Add MI ranking
        mi_ranking = self.results['mutual_info'].reset_index(drop=True)
        mi_ranking['MI_Rank'] = range(1, len(mi_ranking) + 1)
        rankings = rankings.merge(mi_ranking[['Feature', 'MI_Rank']], on='Feature')
        
        # Add RF ranking
        rf_ranking = self.results['random_forest'].reset_index(drop=True)
        rf_ranking['RF_Rank'] = range(1, len(rf_ranking) + 1)
        rankings = rankings.merge(rf_ranking[['Feature', 'RF_Rank']], on='Feature')
        
        # Add GB ranking
        gb_ranking = self.results['gradient_boosting'].reset_index(drop=True)
        gb_ranking['GB_Rank'] = range(1, len(gb_ranking) + 1)
        rankings = rankings.merge(gb_ranking[['Feature', 'GB_Rank']], on='Feature')
        
        # Calculate average ranking
        ranking_cols = ['Corr_Rank', 'F_Rank', 'MI_Rank', 'RF_Rank', 'GB_Rank']
        rankings['Average_Rank'] = rankings[ranking_cols].mean(axis=1)
        rankings['Std_Rank'] = rankings[ranking_cols].std(axis=1)
        
        # Sort by average ranking
        final_ranking = rankings.sort_values('Average_Rank')
        
        self.results['final_ranking'] = final_ranking
        
        print("Top 15 features by average ranking:")
        print(final_ranking.head(15)[['Feature', 'Average_Rank', 'Std_Rank']])
        
        return final_ranking
    
    def visualize_results(self):
        """Create visualizations for feature importance analysis"""
        print("\n=== Creating Visualizations ===")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Feature Importance Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Correlation with target
        top_corr = self.results['correlation'].head(15)
        axes[0, 0].barh(range(len(top_corr)), top_corr['Correlation_Raw'])
        axes[0, 0].set_yticks(range(len(top_corr)))
        axes[0, 0].set_yticklabels(top_corr['Feature'], fontsize=8)
        axes[0, 0].set_title('Top 15 Features by Correlation')
        axes[0, 0].set_xlabel('Correlation Coefficient')
        
        # 2. Random Forest Importance
        top_rf = self.results['random_forest'].head(15)
        axes[0, 1].barh(range(len(top_rf)), top_rf['RF_Importance'])
        axes[0, 1].set_yticks(range(len(top_rf)))
        axes[0, 1].set_yticklabels(top_rf['Feature'], fontsize=8)
        axes[0, 1].set_title('Top 15 Features by Random Forest')
        axes[0, 1].set_xlabel('Feature Importance')
        
        # 3. Mutual Information
        top_mi = self.results['mutual_info'].head(15)
        axes[0, 2].barh(range(len(top_mi)), top_mi['MI_Score'])
        axes[0, 2].set_yticks(range(len(top_mi)))
        axes[0, 2].set_yticklabels(top_mi['Feature'], fontsize=8)
        axes[0, 2].set_title('Top 15 Features by Mutual Information')
        axes[0, 2].set_xlabel('MI Score')
        
        # 4. LASSO Coefficients
        top_lasso = self.results['lasso'].head(15)
        colors = ['red' if x < 0 else 'blue' for x in top_lasso['Lasso_Coefficient_Raw']]
        axes[1, 0].barh(range(len(top_lasso)), top_lasso['Lasso_Coefficient_Raw'], color=colors)
        axes[1, 0].set_yticks(range(len(top_lasso)))
        axes[1, 0].set_yticklabels(top_lasso['Feature'], fontsize=8)
        axes[1, 0].set_title('Top 15 Features by LASSO Coefficients')
        axes[1, 0].set_xlabel('LASSO Coefficient')
        
        # 5. Final Ranking
        top_final = self.results['final_ranking'].head(15)
        axes[1, 1].barh(range(len(top_final)), 1/top_final['Average_Rank'])  # Inverse for better visualization
        axes[1, 1].set_yticks(range(len(top_final)))
        axes[1, 1].set_yticklabels(top_final['Feature'], fontsize=8)
        axes[1, 1].set_title('Top 15 Features by Average Ranking')
        axes[1, 1].set_xlabel('Ranking Score (Higher = Better)')
        
        # 6. Ranking Comparison Heatmap
        top_features = self.results['final_ranking'].head(10)['Feature'].tolist()
        ranking_matrix = []
        methods = ['Corr_Rank', 'F_Rank', 'MI_Rank', 'RF_Rank', 'GB_Rank']
        
        for feature in top_features:
            row = []
            for method in methods:
                rank = self.results['final_ranking'][
                    self.results['final_ranking']['Feature'] == feature
                ][method].iloc[0]
                row.append(rank)
            ranking_matrix.append(row)
        
        im = axes[1, 2].imshow(ranking_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[1, 2].set_xticks(range(len(methods)))
        axes[1, 2].set_xticklabels(['Corr', 'F-test', 'MI', 'RF', 'GB'], rotation=45)
        axes[1, 2].set_yticks(range(len(top_features)))
        axes[1, 2].set_yticklabels(top_features, fontsize=8)
        axes[1, 2].set_title('Ranking Heatmap (Top 10 Features)')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
        
        plt.tight_layout()
        plt.show()
    
    def model_performance_comparison(self, top_k_features=[5, 10, 15, 20]):
        """Compare model performance with different numbers of top features"""
        print("\n=== Model Performance Comparison ===")
        
        # Get top features from final ranking
        top_features = self.results['final_ranking']['Feature'].tolist()
        
        results = []
        
        for k in top_k_features:
            selected_features = top_features[:k]
            
            # Prepare data with selected features
            X_train_selected = self.X_train[selected_features]
            X_test_selected = self.X_test[selected_features]
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_selected, self.y_train)
            
            # Predictions
            y_pred = rf.predict(X_test_selected)
            
            # Metrics
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            results.append({
                'Features': k,
                'MSE': mse,
                'R2_Score': r2,
                'RMSE': np.sqrt(mse)
            })
            
            print(f"Top {k} features - R² Score: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        # Performance with all features
        rf_all = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_all.fit(self.X_train, self.y_train)
        y_pred_all = rf_all.predict(self.X_test)
        mse_all = mean_squared_error(self.y_test, y_pred_all)
        r2_all = r2_score(self.y_test, y_pred_all)
        
        results.append({
            'Features': len(self.feature_names),
            'MSE': mse_all,
            'R2_Score': r2_all,
            'RMSE': np.sqrt(mse_all)
        })
        
        print(f"All {len(self.feature_names)} features - R² Score: {r2_all:.4f}, RMSE: {np.sqrt(mse_all):.4f}")
        
        return pd.DataFrame(results)
    
    def run_complete_analysis(self):
        """Run the complete feature importance and selection analysis"""
        print("Starting Complete Feature Importance Analysis")
        print("=" * 50)
        
        # Preprocessing
        self.preprocess_data()
        
        # Run all analyses
        self.correlation_analysis()
        self.univariate_selection()
        self.tree_based_importance()
        self.regularization_based_selection()
        self.recursive_feature_elimination()
        self.variance_threshold_selection()
        
        # Create summary
        final_ranking = self.create_summary_ranking()
        
        # Performance comparison
        performance_df = self.model_performance_comparison()
        
        # Visualizations
        self.visualize_results()
        
        # Summary recommendations
        print("\n" + "=" * 50)
        print("FEATURE SELECTION RECOMMENDATIONS")
        print("=" * 50)
        
        top_20_features = final_ranking.head(20)['Feature'].tolist()
        print(f"\nTop 20 most important features for efficiency prediction:")
        for i, feature in enumerate(top_20_features, 1):
            print(f"{i:2d}. {feature}")
        
        # Find optimal number of features
        best_performance = performance_df.loc[performance_df['R2_Score'].idxmax()]
        print(f"\nOptimal performance achieved with {int(best_performance['Features'])} features:")
        print(f"R² Score: {best_performance['R2_Score']:.4f}")
        print(f"RMSE: {best_performance['RMSE']:.4f}")
        
        return {
            'final_ranking': final_ranking,
            'performance_comparison': performance_df,
            'top_features': top_20_features,
            'all_results': self.results
        }

# Usage Example:
# Assuming your dataframe is loaded as 'df'
# analyzer = FeatureAnalyzer(df, target_column='efficiency')
# results = analyzer.run_complete_analysis()

# To get the top features:
# top_features = results['top_features']
# print("Selected features for modeling:", top_features)