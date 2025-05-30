import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class ImputationPipeline:
    
    def __init__(self):
        self.temp_from_module_model = None
        self.module_from_temp_model = None
        self.mice_imputer = None
        self.knn_imputer = None
        self.error_code_fill = None
        self.installation_type_classifier = None  # Changed from fill to classifier
        self.installation_type_features = None    # Store features used for prediction
        self.installation_type_encoder = None     # For encoding categorical features
        self.cloud_coverage_median = None
        self.mice_cols = ['irradiance', 'voltage', 'current', 'panel_age', 'cloud_coverage', 'maintenance_count','soiling_ratio']
        self.knn_cols = ['wind_speed', 'pressure', 'temperature', 'module_temperature', 'humidity']
        
        # Features to use for installation_type prediction (customize based on your dataset)
        self.installation_type_predictor_cols = [
            'irradiance', 'voltage', 'current', 'panel_age', 'wind_speed', 
            'pressure', 'temperature', 'module_temperature', 'humidity',
            'cloud_coverage', 'maintenance_count', 'soiling_ratio'
        ]

    def _apply_training_cleaning_rules(self, df):
        """Apply cleaning rules for training data - drops rows that don't meet criteria"""
        df_cleaned = df.copy()
        drop_indices = set()
        
        # Rule 1: Drop temperature > 70
        if 'temperature' in df_cleaned.columns:
            bad_temp = df_cleaned[df_cleaned['temperature'] > 70]
            print(f"Dropping {len(bad_temp)} rows with temperature > 70")
            drop_indices.update(bad_temp.index)

        # Rule 2: Drop irradiance < 0
        if 'irradiance' in df_cleaned.columns:
            bad_irr = df_cleaned[df_cleaned['irradiance'] < 0]
            print(f"Dropping {len(bad_irr)} rows with irradiance < 0")
            drop_indices.update(bad_irr.index)

        # Rule 3: Impute cloud_coverage > 100 (for training, we still impute)
        if 'cloud_coverage' in df_cleaned.columns:
            over_100 = df_cleaned['cloud_coverage'] > 100
            count_over_100 = over_100.sum()
            if count_over_100 > 0:
                # Calculate median from valid cloud coverage values
                valid_cloud_mask = (df_cleaned['cloud_coverage'] <= 100) & (df_cleaned['cloud_coverage'].notna())
                median_cloud = df_cleaned.loc[valid_cloud_mask, 'cloud_coverage'].median()
                print(f"Imputing {count_over_100} rows where cloud_coverage > 100 with median = {median_cloud}")
                df_cleaned.loc[over_100, 'cloud_coverage'] = median_cloud
                # Store median for prediction pipeline
                self.cloud_coverage_median = median_cloud
        
        # Drop the problematic rows
        if drop_indices:
            print(f"Total rows to drop: {len(drop_indices)}")
            df_cleaned = df_cleaned.drop(index=drop_indices)
            df_cleaned = df_cleaned.reset_index(drop=True)
        
        return df_cleaned

    def _apply_prediction_cleaning_rules(self, df):
        """Apply cleaning rules for prediction data - fixes values instead of dropping"""
        df_cleaned = df.copy()
        
        # Rule 1: Keep temperature as is (no changes for prediction)
        
        # Rule 2: Take absolute value of negative irradiance
        if 'irradiance' in df_cleaned.columns:
            negative_irr = df_cleaned['irradiance'] < 0
            count_negative = negative_irr.sum()
            if count_negative > 0:
                print(f"Converting {count_negative} negative irradiance values to absolute values")
                df_cleaned.loc[negative_irr, 'irradiance'] = df_cleaned.loc[negative_irr, 'irradiance'].abs()

        # Rule 3: Fill cloud_coverage > 100 with training median
        if 'cloud_coverage' in df_cleaned.columns:
            # Initialize cloud_coverage_median if it doesn't exist (backward compatibility)
            if not hasattr(self, 'cloud_coverage_median'):
                self.cloud_coverage_median = None
                
            if self.cloud_coverage_median is not None:
                over_100 = df_cleaned['cloud_coverage'] > 100
                count_over_100 = over_100.sum()
                if count_over_100 > 0:
                    print(f"Filling {count_over_100} rows where cloud_coverage > 100 with training median = {self.cloud_coverage_median}")
                    df_cleaned.loc[over_100, 'cloud_coverage'] = self.cloud_coverage_median
            else:
                # Fallback: if no training median available, use current data median
                over_100 = df_cleaned['cloud_coverage'] > 100
                count_over_100 = over_100.sum()
                if count_over_100 > 0:
                    valid_cloud_mask = (df_cleaned['cloud_coverage'] <= 100) & (df_cleaned['cloud_coverage'].notna())
                    if valid_cloud_mask.any():
                        fallback_median = df_cleaned.loc[valid_cloud_mask, 'cloud_coverage'].median()
                        print(f"Warning: No training median available. Filling {count_over_100} rows where cloud_coverage > 100 with current data median = {fallback_median}")
                        df_cleaned.loc[over_100, 'cloud_coverage'] = fallback_median
                    else:
                        print(f"Warning: Cannot compute median for cloud_coverage. Leaving {count_over_100} values > 100 unchanged.")
        
        return df_cleaned

    def _prepare_installation_type_features(self, df):
        """Prepare features for installation_type prediction"""
        # Get available predictor columns
        available_cols = [col for col in self.installation_type_predictor_cols if col in df.columns]
        
        if not available_cols:
            return None, []
        
        feature_df = df[available_cols].copy()
        
        # Handle any remaining missing values in predictor columns
        # Use simple forward fill or median for numerical columns
        for col in feature_df.columns:
            if feature_df[col].dtype in ['int64', 'float64']:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
            else:
                feature_df[col] = feature_df[col].fillna(feature_df[col].mode()[0] if len(feature_df[col].mode()) > 0 else 'UNKNOWN')
        
        return feature_df, available_cols

    def fit(self, df, is_training=True):
        """
        Fit the imputation pipeline
        
        Parameters:
        df: DataFrame to fit on
        is_training: Boolean flag to indicate if this is training data (applies cleaning rules)
        """
        df = df.copy()
        
        # Apply appropriate cleaning rules
        if is_training:
            df = self._apply_training_cleaning_rules(df)
        else:
            # For prediction data during fit (shouldn't happen normally)
            df = self._apply_prediction_cleaning_rules(df)

        # Regression imputation: temperature from module_temperature
        temp_mask = df['temperature'].notna() & df['module_temperature'].notna()
        if temp_mask.any():
            self.temp_from_module_model = LinearRegression().fit(
                df.loc[temp_mask, ['module_temperature']], df.loc[temp_mask, 'temperature']
            )

        # Regression imputation: module_temperature from temperature
        module_mask = df['temperature'].notna() & df['module_temperature'].notna()
        if module_mask.any():
            self.module_from_temp_model = LinearRegression().fit(
                df.loc[module_mask, ['temperature']], df.loc[module_mask, 'module_temperature']
            )

        # MICE imputer - only fit on columns that exist
        available_mice_cols = [col for col in self.mice_cols if col in df.columns]
        if available_mice_cols:
            self.mice_imputer = IterativeImputer(random_state=42, max_iter=10, sample_posterior=False)
            self.mice_imputer.fit(df[available_mice_cols])

        # KNN imputer - only fit on columns that exist
        available_knn_cols = [col for col in self.knn_cols if col in df.columns]
        if available_knn_cols:
            self.knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
            self.knn_imputer.fit(df[available_knn_cols])

        # Categorical fills
        if 'error_code' in df.columns:
            self.error_code_fill = 'NO_ERROR'

        # Fit installation_type classifier
        if 'installation_type' in df.columns:
            # Get rows with non-null installation_type for training
            non_null_mask = df['installation_type'].notna()
            
            if non_null_mask.sum() > 0:  # Check if we have any non-null values
                # Prepare features for training the classifier
                feature_df, available_cols = self._prepare_installation_type_features(df)
                
                if feature_df is not None and len(available_cols) > 0:
                    # Store the features used
                    self.installation_type_features = available_cols
                    
                    # Get training data
                    X_train = feature_df.loc[non_null_mask]
                    y_train = df.loc[non_null_mask, 'installation_type']
                    
                    # Train Random Forest classifier
                    self.installation_type_classifier = RandomForestClassifier(
                        n_estimators=100, 
                        random_state=42, 
                        max_depth=10,
                        min_samples_split=10,
                        min_samples_leaf=5
                    )
                    self.installation_type_classifier.fit(X_train, y_train)
                    
                    print(f"Trained installation_type classifier using features: {available_cols}")
                    print(f"Training accuracy: {self.installation_type_classifier.score(X_train, y_train):.3f}")
                else:
                    print("Warning: No suitable features found for installation_type prediction. Will use mode fallback.")
                    # Fallback to mode
                    self.installation_type_fill = df['installation_type'].mode()[0] if len(df['installation_type'].mode()) > 0 else 'UNKNOWN'
            else:
                print("Warning: No non-null installation_type values found for training classifier.")
                self.installation_type_fill = 'UNKNOWN'

        # Store cloud coverage median if not already stored and we have valid data
        if self.cloud_coverage_median is None and 'cloud_coverage' in df.columns:
            valid_cloud_mask = (df['cloud_coverage'] <= 100) & (df['cloud_coverage'].notna())
            if valid_cloud_mask.any():
                self.cloud_coverage_median = df.loc[valid_cloud_mask, 'cloud_coverage'].median()

        return self

    def transform(self, df, is_training=False):
        """
        Transform the data using fitted imputers
        
        Parameters:
        df: DataFrame to transform
        is_training: Boolean flag to indicate if this is training data
        """
        df = df.copy()

        # Apply appropriate cleaning rules
        if is_training:
            df = self._apply_training_cleaning_rules(df)
        else:
            df = self._apply_prediction_cleaning_rules(df)

        # Regression imputation: temperature from module_temperature
        if self.temp_from_module_model is not None:
            mask = df['temperature'].isna() & df['module_temperature'].notna()
            if mask.any():
                df.loc[mask, 'temperature'] = self.temp_from_module_model.predict(df.loc[mask, ['module_temperature']])

        # Regression imputation: module_temperature from temperature
        if self.module_from_temp_model is not None:
            mask = df['module_temperature'].isna() & df['temperature'].notna()
            if mask.any():
                df.loc[mask, 'module_temperature'] = self.module_from_temp_model.predict(df.loc[mask, ['temperature']])

        # MICE imputation
        if self.mice_imputer is not None:
            available_mice_cols = [col for col in self.mice_cols if col in df.columns]
            if available_mice_cols:
                mice_data = df[available_mice_cols]
                mice_imputed = self.mice_imputer.transform(mice_data)
                df[available_mice_cols] = pd.DataFrame(mice_imputed, columns=available_mice_cols, index=df.index)

        # KNN imputation
        if self.knn_imputer is not None:
            available_knn_cols = [col for col in self.knn_cols if col in df.columns]
            if available_knn_cols:
                knn_data = df[available_knn_cols]
                knn_imputed = self.knn_imputer.transform(knn_data)
                df[available_knn_cols] = pd.DataFrame(knn_imputed, columns=available_knn_cols, index=df.index)

        # Categorical fills
        if 'error_code' in df.columns and self.error_code_fill is not None:
            df['error_code'] = df['error_code'].fillna(self.error_code_fill)

        # Predictive imputation for installation_type
        if 'installation_type' in df.columns:
            missing_mask = df['installation_type'].isna()
            
            if missing_mask.any() and self.installation_type_classifier is not None:
                # Prepare features for prediction
                feature_df, _ = self._prepare_installation_type_features(df)
                
                if feature_df is not None:
                    # Get features for missing values
                    X_missing = feature_df.loc[missing_mask, self.installation_type_features]
                    
                    # Predict missing values
                    predicted_values = self.installation_type_classifier.predict(X_missing)
                    df.loc[missing_mask, 'installation_type'] = predicted_values
                    
                    print(f"Predicted {missing_mask.sum()} missing installation_type values using classifier")
                else:
                    print("Warning: Could not prepare features for installation_type prediction")
                    # Fallback to mode if available
                    if hasattr(self, 'installation_type_fill') and self.installation_type_fill is not None:
                        df['installation_type'] = df['installation_type'].fillna(self.installation_type_fill)
            elif missing_mask.any() and hasattr(self, 'installation_type_fill') and self.installation_type_fill is not None:
                # Fallback to mode imputation
                df['installation_type'] = df['installation_type'].fillna(self.installation_type_fill)
                print(f"Used mode fallback for {missing_mask.sum()} missing installation_type values")

        return df

    def fit_transform(self, df, is_training=True):
        """
        Fit and transform in one step
        
        Parameters:
        df: DataFrame to fit and transform
        is_training: Boolean flag to indicate if this is training data
        """
        self.fit(df, is_training=is_training)
        return self.transform(df, is_training=is_training)

    # Backward compatibility methods - these maintain the old interface
    def transform_training(self, df):
        """Transform training data - applies training rules"""
        return self.transform(df, is_training=True)
    
    def transform_prediction(self, df):
        """Transform prediction data - applies prediction rules"""
        return self.transform(df, is_training=False)