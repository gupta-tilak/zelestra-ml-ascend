import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import optuna
import pickle
import warnings
warnings.filterwarnings('ignore')

from utils.imputation import ImputationPipeline
from utils.data_augmentation import DataAugmentationPipeline

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class ClusterSpecificANN:
    """
    ANN model specifically designed for a cluster
    """
    def __init__(self, cluster_id, neurons=128, layers=3, dropout_rate=0.3, 
                 learning_rate=0.001, l1_reg=0.0, l2_reg=0.01,
                 epochs=200, batch_size=32, validation_split=0.2,
                 patience=20, verbose=0):
        self.cluster_id = cluster_id
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
        self.is_fitted = False
        
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
        
        # Check if we have enough samples
        if len(X) < 10:
            print(f"Warning: Cluster {self.cluster_id} has only {len(X)} samples. Using simplified model.")
            # Use a simpler model for small clusters
            self.neurons = min(self.neurons, 32)
            self.layers = min(self.layers, 2)
        
        # Build model
        self.model_ = self._build_model(X.shape[1])
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(patience=self.patience, restore_best_weights=True),
            ReduceLROnPlateau(patience=self.patience//2, factor=0.5, min_lr=1e-6)
        ]
        
        # Adjust validation split for small datasets
        val_split = min(self.validation_split, 0.1) if len(X) < 50 else self.validation_split
        
        # Train model
        self.history_ = self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=min(self.batch_size, len(X)//2) if len(X) > 10 else len(X),
            validation_split=val_split,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted or self.model_ is None:
            raise ValueError(f"Cluster {self.cluster_id} model must be fitted before making predictions")
            
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
            
        predictions = self.model_.predict(X, verbose=0)
        return predictions.flatten()

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'cluster_id': self.cluster_id,
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

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        """Return the negative mean squared error on the given test data and labels."""
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)

class DBSCANClusteringEnsemble:
    """
    DBSCAN Clustering + Individual ANN Models Ensemble
    """
    def __init__(self, eps=None, min_samples=None, random_state=42):
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.dbscan = None
        self.cluster_models = {}
        self.noise_model = None
        self.cluster_labels = None
        self.n_clusters = 0
        self.noise_points = 0
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _optimize_dbscan_parameters(self, X):
        """
        Optimize DBSCAN parameters using k-distance graph and silhouette score
        """
        print("Optimizing DBSCAN parameters...")
        
        # Scale the data for clustering
        X_scaled = self.scaler.fit_transform(X)
        
        # Method 1: K-distance graph for eps estimation
        if self.eps is None:
            k = max(4, min(10, X.shape[1]))  # Adaptive k based on feature count
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors_fit = neighbors.fit(X_scaled)
            distances, indices = neighbors_fit.kneighbors(X_scaled)
            
            # Sort distances and find the elbow point
            distances = np.sort(distances[:, k-1], axis=0)
            
            # Simple elbow detection using second derivative
            if len(distances) > 10:
                second_derivative = np.diff(distances, 2)
                elbow_idx = np.argmax(second_derivative) + 2
                self.eps = distances[elbow_idx]
            else:
                self.eps = np.percentile(distances, 90)
        
        # Method 2: Grid search for min_samples if not provided
        if self.min_samples is None:
            # Rule of thumb: min_samples = 2 * dimensions
            self.min_samples = max(2, min(2 * X.shape[1], len(X) // 10))
        
        # Validate parameters with silhouette score
        best_eps = self.eps
        best_min_samples = self.min_samples
        best_score = -1
        
        eps_range = [self.eps * 0.5, self.eps, self.eps * 1.5]
        min_samples_range = [max(2, self.min_samples - 1), self.min_samples, self.min_samples + 1]
        
        for eps_val in eps_range:
            for min_samples_val in min_samples_range:
                try:
                    dbscan_temp = DBSCAN(eps=eps_val, min_samples=min_samples_val)
                    labels = dbscan_temp.fit_predict(X_scaled)
                    
                    if len(set(labels)) > 1 and len(set(labels)) < len(X) * 0.8:  # Reasonable number of clusters
                        # Calculate silhouette score (excluding noise points)
                        if len(set(labels[labels != -1])) > 1:
                            mask = labels != -1
                            if np.sum(mask) > 1:
                                score = silhouette_score(X_scaled[mask], labels[mask])
                                if score > best_score:
                                    best_score = score
                                    best_eps = eps_val
                                    best_min_samples = min_samples_val
                except:
                    continue
        
        self.eps = best_eps
        self.min_samples = best_min_samples
        
        print(f"Optimized DBSCAN parameters: eps={self.eps:.4f}, min_samples={self.min_samples}")
        return self.eps, self.min_samples
    
    def _perform_clustering(self, X):
        """
        Perform DBSCAN clustering
        """
        print("Performing DBSCAN clustering...")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Initialize DBSCAN
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        
        # Fit and predict clusters
        self.cluster_labels = self.dbscan.fit_predict(X_scaled)
        
        # Analyze clustering results
        unique_labels = set(self.cluster_labels)
        self.n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.noise_points = list(self.cluster_labels).count(-1)
        
        print(f"Clustering Results:")
        print(f"  Number of clusters: {self.n_clusters}")
        print(f"  Number of noise points: {self.noise_points}")
        print(f"  Percentage of noise: {self.noise_points/len(X)*100:.2f}%")
        
        # Print cluster sizes
        for label in sorted(unique_labels):
            if label != -1:
                cluster_size = list(self.cluster_labels).count(label)
                print(f"  Cluster {label}: {cluster_size} points ({cluster_size/len(X)*100:.2f}%)")
        
        return self.cluster_labels
    
    def _train_cluster_models(self, X, y):
        """
        Train individual ANN models for each cluster
        """
        print("Training individual ANN models for each cluster...")
        
        unique_labels = set(self.cluster_labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points for now
            
            # Get data points for this cluster
            cluster_mask = self.cluster_labels == label
            X_cluster = X[cluster_mask]
            y_cluster = y[cluster_mask]
            
            print(f"\nTraining model for Cluster {label} ({len(X_cluster)} samples)...")
            
            # Optimize hyperparameters for this cluster using Optuna
            best_params = self._optimize_cluster_model(X_cluster, y_cluster, label)
            
            # Create and train the model with best parameters
            cluster_model = ClusterSpecificANN(
                cluster_id=label,
                **best_params,
                verbose=0
            )
            
            cluster_model.fit(X_cluster, y_cluster)
            self.cluster_models[label] = cluster_model
            
            print(f"✓ Cluster {label} model trained successfully")
        
        # Train a separate model for noise points if they exist
        if self.noise_points > 0:
            print(f"\nTraining model for noise points ({self.noise_points} samples)...")
            noise_mask = self.cluster_labels == -1
            X_noise = X[noise_mask]
            y_noise = y[noise_mask]
            
            # Use simpler parameters for noise model
            noise_params = {
                'neurons': 64,
                'layers': 2,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'l2_reg': 0.01,
                'epochs': 100,
                'patience': 10
            }
            
            self.noise_model = ClusterSpecificANN(
                cluster_id=-1,
                **noise_params,
                verbose=0
            )
            
            self.noise_model.fit(X_noise, y_noise)
            print("✓ Noise model trained successfully")
    
    def _optimize_cluster_model(self, X_cluster, y_cluster, cluster_id):
        """
        Optimize hyperparameters for a specific cluster using Optuna
        """
        def objective(trial):
            params = {
                'neurons': trial.suggest_int('neurons', 32, 128),
                'layers': trial.suggest_int('layers', 2, 4),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'l2_reg': trial.suggest_float('l2_reg', 0.0001, 0.1, log=True),
                'epochs': 100,  # Fixed for optimization speed
                'patience': 10,
                'verbose': 0
            }
            
            # Create model
            model = ClusterSpecificANN(cluster_id=cluster_id, **params)
            
            # Use cross-validation if we have enough samples
            if len(X_cluster) >= 20:
                scores = cross_val_score(model, X_cluster, y_cluster, cv=3, 
                                       scoring='neg_mean_squared_error', n_jobs=1)
                return -scores.mean()
            else:
                # For small clusters, use simple train-validation split
                if len(X_cluster) > 5:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_cluster, y_cluster, test_size=0.3, random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    return mean_squared_error(y_val, y_pred)
                else:
                    # Very small cluster - just fit and return training error
                    model.fit(X_cluster, y_cluster)
                    y_pred = model.predict(X_cluster)
                    return mean_squared_error(y_cluster, y_pred)
        
        # Optimize with fewer trials for speed
        n_trials = 20 if len(X_cluster) >= 50 else 10
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def fit(self, X, y):
        """
        Fit the clustering ensemble model
        """
        print("="*60)
        print("DBSCAN CLUSTERING ENSEMBLE TRAINING")
        print("="*60)
        
        # Convert to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Store training data
        self.training_data_original = X.copy()
        
        # Step 1: Optimize DBSCAN parameters
        self._optimize_dbscan_parameters(X)
        
        # Step 2: Perform clustering
        self._perform_clustering(X)
        
        # Step 3: Train individual models for each cluster
        self._train_cluster_models(X, y)
        
        self.is_fitted = True
        print("\n✓ Clustering ensemble training completed successfully!")
        return self
    
    def predict(self, X):
        """
        Make predictions using the clustering ensemble
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy array
        if hasattr(X, 'values'):
            X = X.values
        
        # Scale the data using the same scaler
        X_scaled = self.scaler.transform(X)
        
        # Initialize predictions array
        predictions = np.zeros(len(X))
        
        # Pre-compute scaled training data and cluster centroids
        if hasattr(self, 'training_data_original'):
            training_data_scaled = self.scaler.transform(self.training_data_original)
            cluster_centroids = {}
            cluster_training_points = {}
            
            # Calculate centroids and store training points for each cluster
            for label in self.cluster_models.keys():
                cluster_mask = self.cluster_labels == label
                if np.any(cluster_mask):
                    cluster_points = training_data_scaled[cluster_mask]
                    cluster_centroids[label] = np.mean(cluster_points, axis=0)
                    cluster_training_points[label] = cluster_points
        else:
            # Fallback: use cluster models directly without distance-based weighting
            for i, x_test in enumerate(X_scaled):
                x_original = self.scaler.inverse_transform(x_test.reshape(1, -1))
                cluster_preds = []
                
                for label, model in self.cluster_models.items():
                    try:
                        pred = model.predict(x_original)[0]
                        cluster_preds.append(pred)
                    except Exception as e:
                        print(f"Warning: Error predicting with cluster {label} model: {e}")
                        continue
                
                if cluster_preds:
                    predictions[i] = np.mean(cluster_preds)
                else:
                    predictions[i] = 0  # Fallback value
                    
            return predictions
        
        # For each test point, find the closest cluster
        for i, x_test in enumerate(X_scaled):
            min_distance = float('inf')
            assigned_cluster = -1
            
            # Find closest cluster centroid
            for label, centroid in cluster_centroids.items():
                distance = np.linalg.norm(x_test - centroid)
                if distance < min_distance:
                    min_distance = distance
                    assigned_cluster = label
            
            # Make prediction using the assigned cluster model
            try:
                if assigned_cluster in self.cluster_models:
                    # Transform back to original scale for prediction
                    x_original = self.scaler.inverse_transform(x_test.reshape(1, -1))
                    predictions[i] = self.cluster_models[assigned_cluster].predict(x_original)[0]
                elif self.noise_model is not None:
                    x_original = self.scaler.inverse_transform(x_test.reshape(1, -1))
                    predictions[i] = self.noise_model.predict(x_original)[0]
                else:
                    # Fallback: use weighted average of all cluster models
                    cluster_preds = []
                    cluster_weights = []
                    x_original = self.scaler.inverse_transform(x_test.reshape(1, -1))
                    
                    for label, model in self.cluster_models.items():
                        try:
                            pred = model.predict(x_original)[0]
                            # Weight by inverse distance to cluster centroid
                            distance = np.linalg.norm(x_test - cluster_centroids[label])
                            weight = 1.0 / (distance + 1e-8)  # Add small epsilon to avoid division by zero
                            cluster_preds.append(pred)
                            cluster_weights.append(weight)
                        except Exception as e:
                            print(f"Warning: Error predicting with cluster {label} model: {e}")
                            continue
                    
                    if cluster_preds:
                        # Weighted average
                        cluster_weights = np.array(cluster_weights)
                        cluster_weights = cluster_weights / np.sum(cluster_weights)
                        predictions[i] = np.average(cluster_preds, weights=cluster_weights)
                    else:
                        predictions[i] = 0  # Fallback value
                        
            except Exception as e:
                print(f"Warning: Error making prediction for point {i}: {e}")
                predictions[i] = 0  # Fallback value
        
        return predictions
    
    def get_cluster_info(self):
        """
        Get information about the clusters
        """
        if not self.is_fitted:
            return None
        
        info = {
            'n_clusters': self.n_clusters,
            'noise_points': self.noise_points,
            'eps': self.eps,
            'min_samples': self.min_samples,
            'cluster_sizes': {}
        }
        
        unique_labels = set(self.cluster_labels)
        for label in unique_labels:
            cluster_size = list(self.cluster_labels).count(label)
            info['cluster_sizes'][label] = cluster_size
        
        return info

class ClusteringEnsemblePipeline:
    """
    Complete pipeline integrating clustering ensemble with the existing solar panel pipeline
    """
    def __init__(self, base_selector, eps=None, min_samples=None):
        self.base_selector = base_selector
        self.clustering_ensemble = DBSCANClusteringEnsemble(eps=eps, min_samples=min_samples)
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the clustering ensemble pipeline
        """
        print("Training Clustering Ensemble Pipeline...")
        
        # Fit the clustering ensemble
        self.clustering_ensemble.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the clustering ensemble
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        return self.clustering_ensemble.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the clustering ensemble model
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        custom_score = 100 * (1 - np.sqrt(mean_squared_error(y_test, y_pred)))
        
        results = {
            'RMSE': rmse,
            'R2': r2,
            'Custom_Score': custom_score
        }
        
        return results

class EnhancedSolarPanelModelSelector:
    """
    Enhanced version that includes clustering ensemble approach
    """
    def __init__(self, data_path='dataset/train.csv', test_size=0.3, random_state=42, features_to_drop=None):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.features_to_drop = features_to_drop or []
        self.clustering_ensemble = None
        self.clustering_results = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.preprocessor = None
        self.target_transformer = None
        self.imputer = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_original = None
        self.y_test_original = None
        self.feature_cols = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.final_feature_names = None

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
        
        # Step 2: Initialize and apply imputation pipeline
        print("\nStep 2: Applying imputation pipeline...")
        self.imputer = ImputationPipeline()
        self.df_imputed = self.imputer.fit_transform(self.df_fixed, is_training=True)
        
        print(f"Dataset shape after imputation: {self.df_imputed.shape}")
        remaining_missing = self.df_imputed.isnull().sum().sum()
        print(f"Remaining missing values after imputation: {remaining_missing}")
        
        # Step 3: Create features AFTER imputation is complete
        print("\nStep 3: Creating engineered features...")

        # from utils.feature_engineering import SolarFeatureEngineering
        # feature_engineer = SolarFeatureEngineering()
        # self.df = feature_engineer.create_solar_features(self.df_imputed)
        
        # Use imputed data directly instead of engineered features
        self.df = self.df_imputed.copy()

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
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # Fit and transform the data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Get feature names after transformation
        try:
            feature_names = self.preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = []
            if self.numerical_cols:
                feature_names.extend(self.numerical_cols)
            if self.categorical_cols:
                cat_transformer = self.preprocessor.named_transformers_['cat']
                cat_feature_names = cat_transformer.get_feature_names_out(self.categorical_cols)
                feature_names.extend(cat_feature_names)
        
        # Convert to DataFrames
        self.X_train = pd.DataFrame(X_train_processed, columns=feature_names)
        self.X_test = pd.DataFrame(X_test_processed, columns=feature_names)
        self.y_train, self.y_test = y_train, y_test
        
        # Store feature names and preprocessor
        self.final_feature_names = list(feature_names)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Number of features after one-hot encoding: {len(self.final_feature_names)}")
        if self.categorical_cols:
            print(f"Categorical columns '{', '.join(self.categorical_cols)}' have been one-hot encoded")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_clustering_ensemble(self, eps=None, min_samples=None):
        """
        Train the clustering ensemble model
        """
        print("\n" + "="*60)
        print("TRAINING CLUSTERING ENSEMBLE MODEL")
        print("="*60)
        
        # Initialize clustering ensemble
        self.clustering_ensemble = DBSCANClusteringEnsemble(
            eps=eps, 
            min_samples=min_samples,
            random_state=self.random_state
        )
        
        # Train the ensemble
        self.clustering_ensemble.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        y_pred_test_transformed = self.clustering_ensemble.predict(self.X_test)
        y_pred_test_original = self.inverse_transform_predictions(y_pred_test_transformed)
        
        # Calculate metrics on original scale
        test_rmse = np.sqrt(mean_squared_error(self.y_test_original, y_pred_test_original))
        test_r2 = r2_score(self.y_test_original, y_pred_test_original)
        test_custom_score = self.custom_score_function(self.y_test_original, y_pred_test_original)
        
        # Store results
        self.clustering_results = {
            'Test_RMSE': test_rmse,
            'Test_R2': test_r2,
            'Test_Custom_Score': test_custom_score,
            'Model': self.clustering_ensemble,
            'Cluster_Info': self.clustering_ensemble.get_cluster_info()
        }
        
        # Set as best model
        self.best_model = self.clustering_ensemble
        self.best_model_name = "Clustering_Ensemble"
        self.best_score = test_custom_score
        
        print(f"\nClustering Ensemble Results:")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test Custom Score: {test_custom_score:.4f}")
        
        # Print cluster information
        cluster_info = self.clustering_ensemble.get_cluster_info()
        print(f"\nCluster Information:")
        print(f"Number of clusters: {cluster_info['n_clusters']}")
        print(f"Noise points: {cluster_info['noise_points']}")
        print(f"DBSCAN parameters: eps={cluster_info['eps']:.4f}, min_samples={cluster_info['min_samples']}")
        
        return self.clustering_results

    def custom_score_function(self, y_true, y_pred):
        """
        Custom scoring function as per problem statement
        Score = 100*(1-sqrt(MSE))
        """
        mse = mean_squared_error(y_true, y_pred)
        score = 100 * (1 - np.sqrt(mse))
        return score

    def inverse_transform_predictions(self, y_pred):
        """
        Apply inverse transformation to predictions to get them back to original scale
        """
        if self.target_transformer is not None:
            y_pred_reshaped = y_pred.reshape(-1, 1)
            y_pred_original = self.target_transformer.inverse_transform(y_pred_reshaped).flatten()
            return y_pred_original
        return y_pred

    def predict(self, X_raw):
        """
        Make predictions on raw data using the complete pipeline
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        if self.imputer is None:
            raise ValueError("Imputation pipeline not fitted")
        
        # Step 1: Fix data types
        X_fixed = self.fix_data_types(X_raw, "PREDICTION DATA")
        
        # Step 2: Apply imputation
        X_imputed = self.imputer.transform_prediction(X_fixed)
        
        # Step 3: Apply feature engineering
        # from utils.feature_engineering import SolarFeatureEngineering
        # feature_engineer = SolarFeatureEngineering()
        # X_engineered = feature_engineer.create_solar_features(X_imputed)
        
        # Use imputed data directly instead of engineered features
        X_engineered = X_imputed.copy()

        # Step 4: Drop features if specified
        if self.features_to_drop:
            available_features = [col for col in self.features_to_drop if col in X_engineered.columns]
            if available_features:
                X_engineered = X_engineered.drop(columns=available_features)
        
        # Select features
        X_features = X_engineered[self.feature_cols].copy()
        
        # Step 5: Apply preprocessing
        X_processed = self.preprocessor.transform(X_features)
        X_processed = pd.DataFrame(X_processed, columns=self.final_feature_names)
        
        # Step 6: Make predictions
        y_pred_transformed = self.best_model.predict(X_processed)
        
        # Step 7: Transform back to original scale
        y_pred_original = self.inverse_transform_predictions(y_pred_transformed)
        
        return y_pred_original

    def save_model(self, filepath='model/best_clustering_ensemble.pkl'):
        """
        Save the clustering ensemble model and all necessary components
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        model_package = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'target_transformer': self.target_transformer,
            'imputer': self.imputer,
            'feature_cols': self.feature_cols,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols,
            'final_feature_names': self.final_feature_names,
            'features_to_drop': self.features_to_drop,
            'model_type': 'clustering_ensemble',
            'score': self.best_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"Clustering ensemble model saved to {filepath}")

    def load_model(self, filepath='model/best_clustering_ensemble.pkl'):
        """
        Load a saved clustering ensemble model
        """
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        self.best_model = model_package['model']
        self.preprocessor = model_package['preprocessor']
        self.target_transformer = model_package['target_transformer']
        self.imputer = model_package['imputer']
        self.feature_cols = model_package['feature_cols']
        self.categorical_cols = model_package['categorical_cols']
        self.numerical_cols = model_package['numerical_cols']
        self.final_feature_names = model_package['final_feature_names']
        self.features_to_drop = model_package['features_to_drop']
        self.best_score = model_package.get('score', 0)
        self.best_model_name = "Clustering_Ensemble"
        
        print(f"Clustering ensemble model loaded successfully")

    def run_pipeline(self):
        """
        Run the complete pipeline including data preparation and model training
        """
        print("Starting Clustering Ensemble Pipeline...")
        print("="*70)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Prepare train-test split
        self.prepare_train_test_split()
        
        # Step 3: Train clustering ensemble
        self.train_clustering_ensemble()
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"Best model: {self.best_model_name}")
        print(f"Best score: {self.best_score:.4f}")
        
        # Save the model
        self.save_model()
        
        return self.best_model, self.best_model_name, self.best_score

# Usage Example
if __name__ == "__main__":
    # Define features to drop
    features_to_drop = [
        'id',
        # 'id', 'temperature', 'irradiance', 'maintenance_count', 'voltage',
        # 'wind_speed', 'pressure', 'string_id', 'error_code',
        # 'temp_coefficient_effect', 'soiling_loss', 'wind_cooling_effect'
    ]
    
    # Initialize the enhanced model selector
    enhanced_selector = EnhancedSolarPanelModelSelector(
        data_path='dataset/train.csv',
        features_to_drop=features_to_drop
    )
    
    # Run the complete pipeline
    best_model, best_model_name, best_score = enhanced_selector.run_pipeline()