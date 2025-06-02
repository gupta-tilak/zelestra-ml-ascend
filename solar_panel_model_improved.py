import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PowerTransformer, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.visualization import plot_optimization_history, plot_param_importances

# Neural Network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

# Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from utils.imputation import ImputationPipeline
from utils.data_augmentation import DataAugmentationPipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_optuna_objective(model_name, X_train, y_train, cv=5):
    """
    Create an Optuna objective function for a given model
    
    Parameters:
    -----------
    model_name : str
        Name of the model to optimize
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    objective : callable
        Optuna objective function
    """
    def objective(trial):
        if model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7)
            }
            model = XGBRegressor(**params, random_state=42)
            
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = LGBMRegressor(**params, random_state=42, verbose=-1)
            
        elif model_name == 'Gradient Boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            model = GradientBoostingRegressor(**params, random_state=42)
            
        elif model_name == 'Ridge':
            params = {
                'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True)
            }
            model = Ridge(**params, random_state=42)
            
        elif model_name == 'Lasso':
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 1.0, log=True)
            }
            model = Lasso(**params, random_state=42)
            
        elif model_name == 'ANN':
            params = {
                'neurons': trial.suggest_int('neurons', 32, 256),
                'layers': trial.suggest_int('layers', 2, 5),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'l2_reg': trial.suggest_float('l2_reg', 0.0001, 0.1, log=True),
                'batch_size': trial.suggest_int('batch_size', 16, 128)
            }
            model = ANNRegressor(**params, verbose=0)
            
        else:
            raise ValueError(f"Model {model_name} not supported for Optuna optimization")
        
        # Use cross-validation to evaluate the model
        scores = cross_val_score(model, X_train, y_train, 
                               cv=cv, scoring='neg_root_mean_squared_error',
                               n_jobs=-1)
        
        # Return the mean RMSE (negative because Optuna minimizes)
        return -scores.mean()
    
    return objective

class ANNRegressor:
    """
    Custom ANN Regressor wrapper that mimics scikit-learn interface
    """
    def __init__(self, neurons=128, layers=3, dropout_rate=0.3, 
                 learning_rate=0.001, l1_reg=0.0, l2_reg=0.01,
                 epochs=200, batch_size=32, validation_split=0.2,
                 patience=20, verbose=0):
        self.neurons = neurons
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.verbose = verbose
        self.model_ = None
        self.history_ = None
        
    def _build_model(self, input_dim):
        """Build the neural network model with batch normalization"""
        model = Sequential()
        
        # Input layer with batch normalization
        model.add(Dense(self.neurons, 
                       input_dim=input_dim,
                       activation='relu',
                       kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers with batch normalization
        for i in range(self.layers - 1):
            # Gradually decrease neurons in deeper layers
            layer_neurons = max(self.neurons // (2 ** i), 32)
            model.add(Dense(layer_neurons,
                           activation='relu',
                           kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, X, y, **kwargs):
        """Fit the neural network"""
        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Ensure y is 1D
        if len(y.shape) > 1:
            y = y.flatten()
        
        # Build model
        self.model_ = self._build_model(X.shape[1])
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(patience=self.patience, restore_best_weights=True),
            ReduceLROnPlateau(patience=self.patience//2, factor=0.5, min_lr=1e-6)
        ]
        
        # Train model
        self.history_ = self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
            
        predictions = self.model_.predict(X, verbose=0)
        return predictions.flatten()
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'neurons': self.neurons,
            'layers': self.layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'patience': self.patience,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

class SolarPanelModelSelector:
    def __init__(self, data_path='dataset/train.csv', test_size=0.3, random_state=42, features_to_drop=None):
        """
        Initialize the model selector with data loading and basic setup
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.preprocessor = None
        self.imputer = None
        self.features_to_drop = features_to_drop or []

    def fix_data_types(self, df, dataset_name):
        """Fix data type inconsistencies for specific columns"""
        df_fixed = df.copy()
        
        # Define columns that should be numeric
        numeric_columns_to_fix = ['humidity', 'wind_speed', 'pressure']
        
        print(f"\n=== FIXING DATA TYPES FOR {dataset_name} ===")
        
        for col in numeric_columns_to_fix:
            if col in df_fixed.columns:
                print(f"\nProcessing {col}:")
                print(f"Original dtype: {df_fixed[col].dtype}")
                
                # Check for non-numeric values before conversion
                if df_fixed[col].dtype == 'object':
                    # Display unique non-numeric values
                    try:
                        # Try to convert to numeric and see what fails
                        numeric_conversion = pd.to_numeric(df_fixed[col], errors='coerce')
                        non_numeric_mask = pd.isna(numeric_conversion) & df_fixed[col].notna()
                        
                        if non_numeric_mask.any():
                            print(f"Non-numeric values found in {col}:")
                            non_numeric_values = df_fixed.loc[non_numeric_mask, col].value_counts()
                            print(non_numeric_values.head(10))
                            
                            # Handle common non-numeric patterns
                            df_fixed[col] = df_fixed[col].astype(str)
                            
                            # Remove common problematic characters
                            df_fixed[col] = df_fixed[col].str.replace(r'[^\d.-]', '', regex=True)
                            df_fixed[col] = df_fixed[col].str.strip()
                            
                            # Handle empty strings
                            df_fixed[col] = df_fixed[col].replace('', np.nan)
                            df_fixed[col] = df_fixed[col].replace('nan', np.nan)
                            
                        # Convert to numeric
                        df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                        
                        print(f"Converted dtype: {df_fixed[col].dtype}")
                        print(f"Missing values after conversion: {df_fixed[col].isnull().sum()}")
                        print(f"Valid numeric values: {df_fixed[col].notna().sum()}")
                        
                        # Basic statistics for converted column
                        if df_fixed[col].notna().any():
                            print(f"Min: {df_fixed[col].min():.3f}")
                            print(f"Max: {df_fixed[col].max():.3f}")
                            print(f"Mean: {df_fixed[col].mean():.3f}")
                            
                    except Exception as e:
                        print(f"Error converting {col}: {str(e)}")
                else:
                    print(f"{col} is already numeric type: {df_fixed[col].dtype}")
        
        return df_fixed
    
    def load_and_prepare_data(self):
        """
        Load raw data, fix data types, and apply imputation pipeline
        """
        print("Loading raw data...")
        self.df_raw = pd.read_csv(self.data_path)
        print(f"Raw dataset shape: {self.df_raw.shape}")
        print(f"Missing values in raw data:\n{self.df_raw.isnull().sum()[self.df_raw.isnull().sum() > 0]}")
        
        # Step 1: Fix data types BEFORE imputation
        print("\nStep 1: Fixing data types...")
        self.df_fixed = self.fix_data_types(self.df_raw, "TRAINING DATA")
        
        # Verify the fixes
        print("\n=== DATA TYPE VERIFICATION ===")
        print("Data types after fixing:")
        for col in ['humidity', 'wind_speed', 'pressure']:
            if col in self.df_fixed.columns:
                print(f"{col}: {self.df_fixed[col].dtype}")
        
        # Step 2: Initialize and apply imputation pipeline (WITHOUT feature creation)
        print("\nStep 2: Applying imputation pipeline...")
        self.imputer = ImputationPipeline()
        self.df_imputed = self.imputer.fit_transform(self.df_fixed, is_training=True)
        
        print(f"Dataset shape after imputation: {self.df_imputed.shape}")
        remaining_missing = self.df_imputed.isnull().sum().sum()
        print(f"Remaining missing values after imputation: {remaining_missing}")
        
        # Step 3: NOW create features AFTER imputation is complete
        print("\nStep 3: Creating engineered features...")
        from utils.feature_engineering import SolarFeatureEngineering
        feature_engineer = SolarFeatureEngineering()
        self.df = feature_engineer.create_solar_features(self.df_imputed)

        print(f"Dataset shape after feature engineering: {self.df.shape}")
        
        # Verify no missing values in new features
        new_missing = self.df.isnull().sum().sum()
        if new_missing > 0:
            print(f"Warning: {new_missing} missing values found after feature engineering")
            print("Missing values by column:")
            print(self.df.isnull().sum()[self.df.isnull().sum() > 0])

        # Step 4: Drop selected features if specified
        if self.features_to_drop:
            print(f"\nStep 4: Dropping features: {self.features_to_drop}")
            available_features = [col for col in self.features_to_drop if col in self.df.columns]
            unavailable_features = [col for col in self.features_to_drop if col not in self.df.columns]
            
            if available_features:
                self.df = self.df.drop(columns=available_features)
                print(f"Dropped features: {available_features}")
            
            if unavailable_features:
                print(f"Warning: Features not found in dataset: {unavailable_features}")
            
            print(f"Dataset shape after dropping features: {self.df.shape}")
        
        # Separate features and target
        self.target_col = 'efficiency'
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]
        
        # Identify categorical and numerical columns
        self.categorical_cols = self.df[self.feature_cols].select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = self.df[self.feature_cols].select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        print(f"Categorical columns: {self.categorical_cols}")
        print(f"Numerical columns: {self.numerical_cols}")
        print(f"Total features: {len(self.feature_cols)}")
        
        return self.df
    
    def create_preprocessing_pipeline(self):
        """
        Create preprocessing pipeline for numerical and categorical features
        """
        print("Creating preprocessing pipeline...")
        
        # For neural networks, we need StandardScaler instead of RobustScaler
        # as neural networks work better with standardized inputs
        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler())  # Neural networks prefer StandardScaler
        ])
        
        # Categorical preprocessing pipeline
        categorical_pipeline = Pipeline([
            ('encoder', 'passthrough')  # Will be handled separately
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_cols),
            ('cat', categorical_pipeline, self.categorical_cols)
        ])
        
        return self.preprocessor

    def prepare_train_test_split(self):
        """
        Prepare train-test split with proper preprocessing and stratification
        """
        print("Preparing train-test split...")
        
        X = self.df[self.feature_cols].copy()
        y = self.df[self.target_col].copy()
        
        # Create bins for stratification
        n_bins = 10
        kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        y_binned = kbd.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        # Store original target values for later use
        self.y_original = y.copy()
        
        # Apply power transformation to target if it's skewed
        self.target_transformer = PowerTransformer(method='yeo-johnson')
        y_transformed = self.target_transformer.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y_binned
        )
        
        # Also split original target for evaluation
        _, _, self.y_train_original, self.y_test_original = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y_binned
        )
        
        # Create preprocessing pipeline
        transformers = []
        
        # Numerical preprocessing
        if self.numerical_cols:
            numerical_transformer = StandardScaler()
            transformers.append(('num', numerical_transformer, self.numerical_cols))
        
        # Categorical preprocessing with OneHotEncoder
        if self.categorical_cols:
            categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            transformers.append(('cat', categorical_transformer, self.categorical_cols))
        
        # Create the preprocessor
        self.preprocessor_with_onehot = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # Fit and transform the data
        X_train_processed = self.preprocessor_with_onehot.fit_transform(X_train)
        X_test_processed = self.preprocessor_with_onehot.transform(X_test)
        
        # Get feature names after transformation
        try:
            feature_names = self.preprocessor_with_onehot.get_feature_names_out()
        except AttributeError:
            feature_names = []
            if self.numerical_cols:
                feature_names.extend(self.numerical_cols)
            if self.categorical_cols:
                cat_transformer = self.preprocessor_with_onehot.named_transformers_['cat']
                cat_feature_names = cat_transformer.get_feature_names_out(self.categorical_cols)
                feature_names.extend(cat_feature_names)
        
        # Convert to DataFrames
        self.X_train = pd.DataFrame(X_train_processed, columns=feature_names)
        self.X_test = pd.DataFrame(X_test_processed, columns=feature_names)
        self.y_train, self.y_test = y_train, y_test
        
        # Apply data augmentation to training data
        print("\nApplying data augmentation to training data...")
        augmentation_pipeline = DataAugmentationPipeline(random_state=self.random_state)
        
        # Store original training data before augmentation
        X_train_orig = self.X_train.copy()
        y_train_orig = self.y_train.copy()
        
        # Apply augmentation
        X_train_aug, y_train_aug = augmentation_pipeline.fit_transform(
            self.X_train.values, 
            self.y_train,
            apply_smote=True,
            apply_noise=True,
            apply_dropout=True
        )
        
        # Update training data with augmented samples
        self.X_train = pd.DataFrame(X_train_aug, columns=feature_names)
        self.y_train = y_train_aug
        
        # Also update the original target values for the augmented samples
        # We need to transform the augmented target values back to original scale
        y_train_aug_reshaped = y_train_aug.reshape(-1, 1)
        y_train_orig_aug = self.target_transformer.inverse_transform(y_train_aug_reshaped).flatten()
        self.y_train_original = y_train_orig_aug
        
        # Store feature names and preprocessor
        self.final_feature_names = list(feature_names)
        self.preprocessor = self.preprocessor_with_onehot
        
        print(f"Training set shape after augmentation: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Number of features after one-hot encoding: {len(self.final_feature_names)}")
        if self.categorical_cols:
            print(f"Categorical columns '{', '.join(self.categorical_cols)}' have been one-hot encoded")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def define_models(self):
        """
        Define all models to be tested including ANN
        """
        print("Defining models...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(random_state=self.random_state),
            'Lasso': Lasso(random_state=self.random_state),
            'ElasticNet': ElasticNet(random_state=self.random_state),
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            'Random Forest': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            'Extra Trees': ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=self.random_state),
            'XGBoost': XGBRegressor(random_state=self.random_state, eval_metric='rmse'),
            'LightGBM': LGBMRegressor(random_state=self.random_state, verbose=-1),
            'CatBoost': CatBoostRegressor(random_state=self.random_state, verbose=False),
            'KNN': KNeighborsRegressor(),
            'SVR': SVR(),
            'ANN': ANNRegressor(verbose=0)  # Use custom ANN wrapper
        }
        
        return self.models
    
    def inverse_transform_predictions(self, y_pred):
        """
        Apply inverse transformation to predictions to get them back to original scale
        """
        y_pred_reshaped = y_pred.reshape(-1, 1)
        y_pred_original = self.target_transformer.inverse_transform(y_pred_reshaped).flatten()
        return y_pred_original
    
    def custom_score_function(self, y_true, y_pred):
        """
        Custom scoring function as per problem statement
        Score = 100*(1-sqrt(MSE))
        Note: This should be calculated on original scale, not transformed scale
        """
        mse = mean_squared_error(y_true, y_pred)
        score = 100 * (1 - np.sqrt(mse))
        return score
    
    def evaluate_base_models(self):
        """
        Evaluate all base models using cross-validation
        """
        print("Evaluating base models...")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Cross-validation scores (on transformed target)
                if name == 'ANN':
                    # For ANN, use fewer CV folds due to computational cost
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, 
                                              scoring='neg_mean_squared_error', n_jobs=1)
                else:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, 
                                              scoring='neg_mean_squared_error', n_jobs=-1)
                
                # Fit model for additional metrics
                model.fit(self.X_train, self.y_train)
                
                # Get predictions on transformed scale
                y_pred_train_transformed = model.predict(self.X_train)
                y_pred_test_transformed = model.predict(self.X_test)
                
                # Transform predictions back to original scale
                y_pred_train_original = self.inverse_transform_predictions(y_pred_train_transformed)
                y_pred_test_original = self.inverse_transform_predictions(y_pred_test_transformed)
                
                # Calculate metrics on ORIGINAL scale
                train_rmse = np.sqrt(mean_squared_error(self.y_train_original, y_pred_train_original))
                test_rmse = np.sqrt(mean_squared_error(self.y_test_original, y_pred_test_original))
                train_r2 = r2_score(self.y_train_original, y_pred_train_original)
                test_r2 = r2_score(self.y_test_original, y_pred_test_original)
                
                # Custom score on original scale
                train_custom_score = self.custom_score_function(self.y_train_original, y_pred_train_original)
                test_custom_score = self.custom_score_function(self.y_test_original, y_pred_test_original)
                
                # CV RMSE on transformed scale (for comparison)
                cv_rmse_transformed = np.sqrt(-cv_scores.mean())
                
                self.results[name] = {
                    'CV_RMSE_transformed': cv_rmse_transformed,
                    'CV_RMSE_std': np.sqrt(cv_scores.std()),
                    'Train_RMSE': train_rmse,
                    'Test_RMSE': test_rmse,
                    'Train_R2': train_r2,
                    'Test_R2': test_r2,
                    'Train_Custom_Score': train_custom_score,
                    'Test_Custom_Score': test_custom_score,
                    'Model': model
                }
                
                print(f"  ✓ {name} completed - Test Custom Score: {test_custom_score:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)}")
                continue
        
        return self.results

    def hyperparameter_tuning(self, top_n=5, n_trials=50):
        """
        Perform hyperparameter tuning using Optuna
        
        Parameters:
        -----------
        top_n : int
            Number of top models to tune
        n_trials : int
            Number of trials for each model
        """
        print(f"Performing hyperparameter tuning for top {top_n} models using Optuna...")
        
        # Sort models by test RMSE
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['Test_RMSE'])
        top_models = [name for name, _ in sorted_models[:top_n]]
        
        tuned_results = {}
        
        for model_name in top_models:
            print(f"\nTuning {model_name}...")
            
            try:
                # Create study
                study = optuna.create_study(direction='minimize')
                
                # Create objective function
                objective = create_optuna_objective(model_name, self.X_train, self.y_train)
                
                # Optimize
                study.optimize(objective, n_trials=n_trials)
                
                # Get best parameters
                best_params = study.best_params
                print(f"Best parameters for {model_name}:")
                for param, value in best_params.items():
                    print(f"  {param}: {value}")
                
                # Create and train model with best parameters
                if model_name == 'Random Forest':
                    best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
                elif model_name == 'XGBoost':
                    best_model = XGBRegressor(**best_params, random_state=42)
                elif model_name == 'LightGBM':
                    best_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
                elif model_name == 'Gradient Boosting':
                    best_model = GradientBoostingRegressor(**best_params, random_state=42)
                elif model_name == 'Ridge':
                    best_model = Ridge(**best_params, random_state=42)
                elif model_name == 'Lasso':
                    best_model = Lasso(**best_params, random_state=42)
                elif model_name == 'ANN':
                    best_model = ANNRegressor(**best_params, verbose=0)
                
                # Fit the model
                best_model.fit(self.X_train, self.y_train)
                
                # Get predictions
                y_pred_train_transformed = best_model.predict(self.X_train)
                y_pred_test_transformed = best_model.predict(self.X_test)
                
                # Transform predictions back to original scale
                y_pred_train_original = self.inverse_transform_predictions(y_pred_train_transformed)
                y_pred_test_original = self.inverse_transform_predictions(y_pred_test_transformed)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(self.y_train_original, y_pred_train_original))
                test_rmse = np.sqrt(mean_squared_error(self.y_test_original, y_pred_test_original))
                train_r2 = r2_score(self.y_train_original, y_pred_train_original)
                test_r2 = r2_score(self.y_test_original, y_pred_test_original)
                train_custom_score = self.custom_score_function(self.y_train_original, y_pred_train_original)
                test_custom_score = self.custom_score_function(self.y_test_original, y_pred_test_original)
                
                tuned_results[f'{model_name}_Tuned'] = {
                    'Best_Params': best_params,
                    'CV_RMSE_transformed': study.best_value,
                    'Train_RMSE': train_rmse,
                    'Test_RMSE': test_rmse,
                    'Train_R2': train_r2,
                    'Test_R2': test_r2,
                    'Train_Custom_Score': train_custom_score,
                    'Test_Custom_Score': test_custom_score,
                    'Model': best_model,
                    'Study': study  # Store the study for later analysis
                }
                
                print(f"  ✓ {model_name} tuning completed - Test Custom Score: {test_custom_score:.4f}")
                
                # Plot optimization history and parameter importance
                try:
                    fig1 = plot_optimization_history(study)
                    fig2 = plot_param_importances(study)
                    fig1.write_html(f"optuna_plots/{model_name}_optimization_history.html")
                    fig2.write_html(f"optuna_plots/{model_name}_param_importance.html")
                except Exception as e:
                    print(f"  Warning: Could not create Optuna plots: {str(e)}")
                
            except Exception as e:
                print(f"  ✗ Error tuning {model_name}: {str(e)}")
                continue
        
        self.tuned_results = tuned_results
        return tuned_results

    def select_best_model(self):
        """
        Select the best model based on test performance
        """
        print("Selecting best model...")
        
        # Combine base and tuned results
        all_results = {**self.results}
        if hasattr(self, 'tuned_results'):
            all_results.update(self.tuned_results)
        
        # Find best model based on test custom score
        best_model_name = max(all_results.keys(), 
                            key=lambda x: all_results[x]['Test_Custom_Score'])
        
        self.best_model_name = best_model_name
        self.best_model = all_results[best_model_name]['Model']
        self.best_score = all_results[best_model_name]['Test_Custom_Score']
        
        print(f"Best Model: {best_model_name}")
        print(f"Best Test Custom Score: {self.best_score:.4f}")
        
        return self.best_model_name, self.best_model
    
    def predict(self, X_raw):
        """
        Make predictions on raw data using the complete pipeline
        
        Parameters:
        X_raw: Raw input data (DataFrame) - will be processed through the entire pipeline
        
        Returns:
        y_pred_original: Predictions on original scale
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Run the pipeline first.")
        
        if self.imputer is None:
            raise ValueError("Imputation pipeline not fitted. Run training first.")
        
        # Step 1: Fix data types (same as training)
        X_fixed = self.fix_data_types(X_raw, "PREDICTION DATA")
        
        # Step 2: Apply imputation
        X_imputed = self.imputer.transform_prediction(X_fixed)

        # Step 2.5: Apply feature engineering (same as training)
        from utils.feature_engineering import SolarFeatureEngineering
        feature_engineer = SolarFeatureEngineering()
        X_engineered = feature_engineer.create_solar_features(X_imputed)

        # Step 3: Drop features and select features
        if self.features_to_drop:
            available_features = [col for col in self.features_to_drop if col in X_engineered.columns]
            if available_features:
                X_engineered = X_engineered.drop(columns=available_features)

        # Select features
        X_features = X_engineered[self.feature_cols].copy()
        
        # Step 4: Apply preprocessing (including one-hot encoding for categorical variables)
        X_processed = self.preprocessor.transform(X_features)

        # Convert back to DataFrame for consistency with final feature names
        X_processed = pd.DataFrame(X_processed, columns=self.final_feature_names)
        
        # Step 6: Get predictions on transformed scale
        y_pred_transformed = self.best_model.predict(X_processed)
        
        # Step 7: Transform back to original scale
        y_pred_original = self.inverse_transform_predictions(y_pred_transformed)
        
        return y_pred_original
    
    def print_results_summary(self):
        """
        Print comprehensive results summary
        """
        print("\n" + "="*80)
        print("MODEL SELECTION RESULTS SUMMARY")
        print("="*80)
        
        # Base models results
        print("\nBASE MODELS PERFORMANCE (on original scale):")
        print("-" * 50)
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('Test_Custom_Score', ascending=False)
        
        display_cols = ['Test_Custom_Score', 'Test_RMSE', 'Test_R2', 'CV_RMSE_transformed']
        print(results_df[display_cols].round(4).to_string())
        
        # Tuned models results
        if hasattr(self, 'tuned_results') and self.tuned_results:
            print("\nTUNED MODELS PERFORMANCE (on original scale):")
            print("-" * 50)
            tuned_df = pd.DataFrame(self.tuned_results).T
            tuned_df = tuned_df.sort_values('Test_Custom_Score', ascending=False)
            print(tuned_df[display_cols].round(4).to_string())
        
        print(f"\nBEST MODEL: {self.best_model_name}")
        print(f"BEST SCORE: {self.best_score:.4f}")
        print("\nNote: All metrics except CV_RMSE_transformed are calculated on original scale")
        
    def save_best_model(self, filepath='model/best_solar_model.pkl'):
        """
        Save the best model and all preprocessing components
        """
        import pickle
        
        model_package = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'target_transformer': self.target_transformer,
            'imputer': self.imputer,
            'feature_names': self.feature_cols,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'features_to_drop': self.features_to_drop,
            'final_feature_names': self.final_feature_names  # Add this for consistency
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"Best model with complete pipeline saved to {filepath}")
    
    def load_model(self, filepath='best_solar_model.pkl'):
        """
        Load a saved model with complete pipeline
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        self.best_model = model_package['model']
        self.preprocessor = model_package['preprocessor']
        self.final_feature_names = model_package.get('final_feature_names', [])
        self.target_transformer = model_package['target_transformer']
        self.imputer = model_package['imputer']  # Load the imputation pipeline
        self.feature_cols = model_package['feature_names']
        self.categorical_cols = model_package['categorical_cols']
        self.numerical_cols = model_package['numerical_cols']
        self.best_model_name = model_package.get('best_model_name', 'Unknown')
        self.best_score = model_package.get('best_score', 0)
        self.features_to_drop = model_package.get('features_to_drop', []) 
        
        print(f"Model with complete pipeline loaded successfully: {self.best_model_name}")
        
    def run_complete_pipeline(self):
        """
        Run the complete model selection pipeline including data type fixing and imputation
        """
        print("Starting Solar Panel Performance Model Selection Pipeline...")
        print("="*60)
        
        # Step 1: Load raw data, fix data types, and apply imputation
        self.load_and_prepare_data()
        
        # Step 2: Create preprocessing pipeline
        self.create_preprocessing_pipeline()
        
        # Step 3: Prepare train-test split
        self.prepare_train_test_split()
        
        # Step 4: Define models
        self.define_models()
        
        # Step 5: Evaluate base models
        self.evaluate_base_models()
        
        # Step 6: Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Step 7: Select best model
        self.select_best_model()
        
        # Step 8: Print results
        self.print_results_summary()
        
        # Step 9: Save best model with complete pipeline
        self.save_best_model()
        
        return self.best_model, self.best_model_name, self.best_score

# Usage example
if __name__ == "__main__":

        # Define features to drop

    features_to_drop = [
        'id',                           # Identifier column
        'temperature',                  # Environmental feature
        'irradiance',                   # Raw irradiance (you have normalized version)
        # 'humidity',                     # Environmental feature
        'maintenance_count',            # Maintenance related (you have frequency)
        'voltage',                      # Electrical parameter
        # 'module_temperature',           # Temperature measurement
        # 'cloud_coverage',               # Weather feature
        'wind_speed',                   # Weather feature
        'pressure',                     # Environmental feature
        'string_id',                    # Categorical identifier
        'error_code',                   # Categorical feature
        # 'installation_type',            # Categorical feature
        # 'temp_difference',              # Temperature derived feature
        'temp_coefficient_effect',      # Temperature related
        'soiling_loss',                 # Soiling related (you have ratio)
        # 'age_category',                 # Categorical age (you have numerical age)
        # 'environmental_stress',         # Composite environmental feature
        'wind_cooling_effect',          # Wind related feature
        # 'effective_module_temp'         # Temperature derived feature
    ]

    # Selected features to KEEP (plus target 'efficiency')
    selected_features = [
        'irradiance_normalized',        # Key: Normalized solar irradiance
        'soiling_ratio',               # Key: Panel cleanliness factor
        'panel_age',                   # Key: Age of solar panel
        'expected_irradiance_clean',   # Key: Expected clean irradiance
        'age_degradation_factor',      # Key: Age-related degradation
        'current',                     # Key: Electrical current output
        'maintenance_frequency',       # Key: Maintenance frequency
        'irradiance_cloud_ratio',      # Key: Cloud impact on irradiance
        'power_output',               # Key: Power generation
        'efficiency'                  # Target variable
    ]

    # Initialize the model selector
    # Note: Use raw data path here, not pre-engineered data
    selector = SolarPanelModelSelector(data_path='dataset/train.csv', features_to_drop=features_to_drop)
    
    # Run the complete pipeline
    best_model, best_model_name, best_score = selector.run_complete_pipeline()
    
    print(f"\nPipeline completed successfully!")
    print(f"Best model: {best_model_name} with score: {best_score:.4f}")