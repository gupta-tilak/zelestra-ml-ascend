import numpy as np
from sklearn.neighbors import NearestNeighbors

class RegressionSMOTE:
    """
    SMOTE implementation for regression problems
    """
    
    def __init__(self, k_neighbors=5, sampling_strategy='auto', random_state=42):
        """
        Initialize RegressionSMOTE
        
        Parameters:
        -----------
        k_neighbors : int
            Number of nearest neighbors to use for synthetic sample generation
        sampling_strategy : str or dict
            Strategy for sampling. 'auto' will balance low-density regions
        random_state : int
            Random state for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.nn_model = None
        
    def _find_low_density_regions(self, X, y, n_bins=10):
        """
        Identify low-density regions in the target variable
        """
        # Create bins for target variable
        bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(y, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Count samples in each bin
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        
        # Identify low-density bins (below median count)
        median_count = np.median(bin_counts[bin_counts > 0])
        low_density_bins = np.where(bin_counts < median_count)[0]
        
        return bin_indices, low_density_bins, bin_counts
    
    def _generate_synthetic_samples(self, X_minority, y_minority, n_synthetic):
        """
        Generate synthetic samples using k-nearest neighbors
        """
        if len(X_minority) < self.k_neighbors:
            k = len(X_minority) - 1
        else:
            k = self.k_neighbors
            
        if k <= 0:
            return X_minority, y_minority
        
        # Fit nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1)
        nn_model.fit(X_minority)
        
        # Generate synthetic samples
        synthetic_X = []
        synthetic_y = []
        
        np.random.seed(self.random_state)
        
        for _ in range(n_synthetic):
            # Randomly select a sample
            idx = np.random.randint(0, len(X_minority))
            sample = X_minority[idx:idx+1]
            target = y_minority[idx]
            
            # Find k nearest neighbors
            distances, indices = nn_model.kneighbors(sample)
            neighbor_indices = indices[0][1:]  # Exclude the sample itself
            
            if len(neighbor_indices) == 0:
                continue
                
            # Randomly select a neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = X_minority[neighbor_idx]
            neighbor_target = y_minority[neighbor_idx]
            
            # Generate synthetic sample
            alpha = np.random.random()
            synthetic_sample = sample[0] + alpha * (neighbor - sample[0])
            synthetic_target = target + alpha * (neighbor_target - target)
            
            synthetic_X.append(synthetic_sample)
            synthetic_y.append(synthetic_target)
        
        if synthetic_X:
            return np.array(synthetic_X), np.array(synthetic_y)
        else:
            return np.empty((0, X_minority.shape[1])), np.array([])
    
    def fit_resample(self, X, y):
        """
        Apply SMOTE for regression
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target values
            
        Returns:
        --------
        X_resampled, y_resampled : resampled data
        """
        X = np.array(X)
        y = np.array(y)
        
        # Find low-density regions
        bin_indices, low_density_bins, bin_counts = self._find_low_density_regions(X, y)
        
        if len(low_density_bins) == 0:
            print("No low-density regions found. Returning original data.")
            return X, y
        
        # Calculate target count (median of non-zero bins)
        target_count = int(np.median(bin_counts[bin_counts > 0]))
        
        X_resampled = [X]
        y_resampled = [y]
        
        print(f"Applying RegressionSMOTE to {len(low_density_bins)} low-density regions...")
        
        for bin_idx in low_density_bins:
            # Get samples in this bin
            mask = bin_indices == bin_idx
            if not np.any(mask):
                continue
                
            X_bin = X[mask]
            y_bin = y[mask]
            
            current_count = len(X_bin)
            n_synthetic = max(0, target_count - current_count)
            
            if n_synthetic > 0:
                X_synthetic, y_synthetic = self._generate_synthetic_samples(
                    X_bin, y_bin, n_synthetic
                )
                
                if len(X_synthetic) > 0:
                    X_resampled.append(X_synthetic)
                    y_resampled.append(y_synthetic)
                    print(f"  Generated {len(X_synthetic)} synthetic samples for bin {bin_idx}")
        
        # Combine all samples
        X_final = np.vstack(X_resampled)
        y_final = np.concatenate(y_resampled)
        
        print(f"Original samples: {len(X)}, Final samples: {len(X_final)}")
        
        return X_final, y_final

class NoiseInjection:
    """
    Strategic noise injection for data augmentation
    """
    
    def __init__(self, noise_factor=0.05, random_state=42):
        """
        Initialize noise injection
        
        Parameters:
        -----------
        noise_factor : float
            Factor controlling noise magnitude relative to feature std
        random_state : int
            Random state for reproducibility
        """
        self.noise_factor = noise_factor
        self.random_state = random_state
        
    def add_gaussian_noise(self, X, y=None, augmentation_factor=0.5):
        """
        Add Gaussian noise to features
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like, optional
            Target values
        augmentation_factor : float
            Fraction of original data to augment
            
        Returns:
        --------
        X_augmented, y_augmented : augmented data
        """
        X = np.array(X)
        n_samples = int(len(X) * augmentation_factor)
        
        np.random.seed(self.random_state)
        
        # Randomly select samples to augment
        indices = np.random.choice(len(X), size=n_samples, replace=True)
        X_selected = X[indices]
        
        # Calculate noise based on feature standard deviations
        feature_stds = np.std(X, axis=0)
        noise = np.random.normal(0, self.noise_factor * feature_stds, size=X_selected.shape)
        
        # Add noise
        X_noisy = X_selected + noise
        
        if y is not None:
            y = np.array(y)
            y_selected = y[indices]
            return X_noisy, y_selected
        
        return X_noisy
    
    def add_feature_dropout(self, X, y=None, dropout_rate=0.1, augmentation_factor=0.3):
        """
        Apply feature dropout as augmentation technique
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like, optional
            Target values
        dropout_rate : float
            Fraction of features to drop for each sample
        augmentation_factor : float
            Fraction of original data to augment
            
        Returns:
        --------
        X_augmented, y_augmented : augmented data
        """
        X = np.array(X)
        n_samples = int(len(X) * augmentation_factor)
        
        np.random.seed(self.random_state + 1)
        
        # Randomly select samples to augment
        indices = np.random.choice(len(X), size=n_samples, replace=True)
        X_selected = X[indices].copy()
        
        # Apply feature dropout
        n_features_to_drop = int(X.shape[1] * dropout_rate)
        
        for i in range(len(X_selected)):
            features_to_drop = np.random.choice(
                X.shape[1], size=n_features_to_drop, replace=False
            )
            X_selected[i, features_to_drop] = 0  # or use feature mean
        
        if y is not None:
            y = np.array(y)
            y_selected = y[indices]
            return X_selected, y_selected
        
        return X_selected

class DataAugmentationPipeline:
    """
    Comprehensive data augmentation pipeline for regression
    """
    
    def __init__(self, random_state=42):
        """
        Initialize data augmentation pipeline
        """
        self.random_state = random_state
        self.smote = RegressionSMOTE(random_state=random_state)
        self.noise_injector = NoiseInjection(random_state=random_state)
        self.is_fitted = False
        
    def fit_transform(self, X, y, apply_smote=True, apply_noise=True, apply_dropout=True):
        """
        Apply comprehensive data augmentation
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target values
        apply_smote : bool
            Whether to apply SMOTE for regression
        apply_noise : bool
            Whether to apply noise injection
        apply_dropout : bool
            Whether to apply feature dropout
            
        Returns:
        --------
        X_augmented, y_augmented : augmented data
        """
        print("Applying data augmentation pipeline...")
        
        X_augmented = [np.array(X)]
        y_augmented = [np.array(y)]
        
        original_size = len(X)
        
        # Apply SMOTE for low-density regions
        if apply_smote:
            print("Applying RegressionSMOTE...")
            X_smote, y_smote = self.smote.fit_resample(X, y)
            
            # Only add the synthetic samples (not the original ones)
            n_original = len(X)
            if len(X_smote) > n_original:
                X_synthetic = X_smote[n_original:]
                y_synthetic = y_smote[n_original:]
                X_augmented.append(X_synthetic)
                y_augmented.append(y_synthetic)
                print(f"  Added {len(X_synthetic)} SMOTE samples")
        
        # Apply noise injection
        if apply_noise:
            print("Applying noise injection...")
            X_noise, y_noise = self.noise_injector.add_gaussian_noise(
                X, y, augmentation_factor=0.3
            )
            X_augmented.append(X_noise)
            y_augmented.append(y_noise)
            print(f"  Added {len(X_noise)} noise-augmented samples")
        
        # Apply feature dropout
        if apply_dropout:
            print("Applying feature dropout...")
            X_dropout, y_dropout = self.noise_injector.add_feature_dropout(
                X, y, dropout_rate=0.15, augmentation_factor=0.2
            )
            X_augmented.append(X_dropout)
            y_augmented.append(y_dropout)
            print(f"  Added {len(X_dropout)} dropout-augmented samples")
        
        # Combine all augmented data
        X_final = np.vstack(X_augmented)
        y_final = np.concatenate(y_augmented)
        
        # Shuffle the augmented dataset
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(X_final))
        X_final = X_final[indices]
        y_final = y_final[indices]
        
        print(f"Data augmentation completed:")
        print(f"  Original size: {original_size}")
        print(f"  Augmented size: {len(X_final)}")
        print(f"  Augmentation ratio: {len(X_final) / original_size:.2f}x")
        
        self.is_fitted = True
        return X_final, y_final 