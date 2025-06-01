import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

class SolarFeatureEngineering:
    """
    Separate class for creating solar panel domain-specific features
    """
    
    @staticmethod
    def create_solar_features(df):
        """Create domain-specific features for solar panel analysis"""
        df_engineered = df.copy()
        
        # Power calculation (P = V * I)
        if 'voltage' in df.columns and 'current' in df.columns:
            df_engineered['power_output'] = df_engineered['voltage'] * df_engineered['current']
        
        # Temperature difference (module vs ambient)
        if 'module_temperature' in df.columns and 'temperature' in df.columns:
            df_engineered['temp_difference'] = (df_engineered['module_temperature'] - 
                                            df_engineered['temperature'])
        
        # Performance ratio (considering irradiance and temperature effects)
        if 'irradiance' in df.columns and 'temperature' in df.columns:
            # Normalized irradiance (relative to standard test conditions: 1000 W/m²)
            df_engineered['irradiance_normalized'] = df_engineered['irradiance'] / 1000
            
            # Temperature coefficient effect (typical -0.4%/°C)
            df_engineered['temp_coefficient_effect'] = 1 - 0.004 * (df_engineered['temperature'] - 25)
        
        # Soiling impact on expected performance
        if 'soiling_ratio' in df.columns and 'irradiance' in df.columns:
            df_engineered['expected_irradiance_clean'] = (df_engineered['irradiance'] / 
                                                        df_engineered['soiling_ratio'])
            df_engineered['soiling_loss'] = (df_engineered['expected_irradiance_clean'] - 
                                            df_engineered['irradiance'])
        
        # Weather interaction features
        if 'cloud_coverage' in df.columns and 'irradiance' in df.columns:
            df_engineered['irradiance_cloud_ratio'] = (df_engineered['irradiance'] / 
                                                    (100 - df_engineered['cloud_coverage'] + 1))
        
        # Aging effects
        if 'panel_age' in df.columns:
            # Typical degradation rate: 0.5-0.8% per year
            df_engineered['age_degradation_factor'] = 1 - (0.006 * df_engineered['panel_age'])
            df_engineered['age_category'] = pd.cut(df_engineered['panel_age'], 
                                                bins=[0, 2, 5, 10, float('inf')],
                                                labels=['New', 'Young', 'Mature', 'Old'])
        
        # Maintenance effectiveness
        if 'maintenance_count' in df.columns and 'panel_age' in df.columns:
            df_engineered['maintenance_frequency'] = (df_engineered['maintenance_count'] / 
                                                    (df_engineered['panel_age'] + 1))
        
        # Environmental stress factors
        if 'humidity' in df.columns and 'temperature' in df.columns:
            # Heat index approximation
            df_engineered['environmental_stress'] = (df_engineered['humidity'] * 
                                                df_engineered['temperature'] / 100)
        
        # Wind cooling effect
        if 'wind_speed' in df.columns and 'module_temperature' in df.columns:
            df_engineered['wind_cooling_effect'] = df_engineered['wind_speed'] * 2  # Simplified model
            df_engineered['effective_module_temp'] = (df_engineered['module_temperature'] - 
                                                    df_engineered['wind_cooling_effect'])
        
        # # ===== ADDITIONAL ADVANCED FEATURE ENGINEERING =====
        
        # if 'power_output' in df_engineered.columns:
        #     df_engineered['power_output_log'] = np.log1p(df_engineered['power_output'])
        
        # # Robust scaling for outlier-prone features
        # if 'temp_difference' in df_engineered.columns:
        #     scaler = RobustScaler()
        #     df_engineered['temp_difference_robust'] = scaler.fit_transform(
        #         df_engineered[['temp_difference']]).flatten()
        
        # # Winsorization for extreme values
        # if 'irradiance' in df.columns:
        #     df_engineered['irradiance_winsorized'] = np.clip(df_engineered['irradiance'], 
        #                                                    np.percentile(df_engineered['irradiance'], 1),
        #                                                    np.percentile(df_engineered['irradiance'], 99))
        
        # # Theoretical maximum power calculation
        # required_cols = ['irradiance_normalized', 'temp_coefficient_effect', 'age_degradation_factor', 'soiling_ratio']
        # if all(col in df_engineered.columns for col in required_cols):
        #     df_engineered['theoretical_max_power'] = (df_engineered['irradiance_normalized'] * 
        #                                              df_engineered['temp_coefficient_effect'] * 
        #                                              df_engineered['age_degradation_factor'] * 
        #                                              df_engineered['soiling_ratio'])
        
        # # Performance deviation metrics
        # if 'power_output' in df_engineered.columns and 'theoretical_max_power' in df_engineered.columns:
        #     df_engineered['performance_deviation'] = (df_engineered['power_output'] - 
        #                                              df_engineered['theoretical_max_power'])
        #     # Avoid division by zero
        #     df_engineered['efficiency_ratio'] = np.where(
        #         df_engineered['theoretical_max_power'] != 0,
        #         df_engineered['power_output'] / df_engineered['theoretical_max_power'],
        #         0
        #     )
        
        # # String-level aggregations
        # if 'string_id' in df.columns and 'power_output' in df_engineered.columns:
        #     string_stats = df_engineered.groupby('string_id')['power_output'].agg(['mean', 'std', 'min', 'max'])
        #     string_stats.columns = ['power_output_string_mean', 'power_output_string_std', 
        #                            'power_output_string_min', 'power_output_string_max']
        #     df_engineered = df_engineered.merge(string_stats, left_on='string_id', right_index=True, how='left')
        #     df_engineered['power_vs_string_mean'] = (df_engineered['power_output'] - 
        #                                            df_engineered['power_output_string_mean'])
        
        # # Error frequency and patterns
        # if 'error_code' in df.columns:
        #     df_engineered['error_indicator'] = (df_engineered['error_code'] != 0).astype(int)
            
        #     if 'string_id' in df.columns:
        #         df_engineered['consecutive_errors'] = df_engineered.groupby('string_id')['error_indicator'].transform(
        #             lambda x: x.groupby((x != x.shift()).cumsum()).cumsum())
        
        # # Performance anomaly detection
        # anomaly_features = ['power_output', 'irradiance', 'module_temperature']
        # available_anomaly_features = [col for col in anomaly_features if col in df_engineered.columns]
        
        # if len(available_anomaly_features) >= 2:
        #     iso_forest = IsolationForest(contamination=0.1, random_state=42)
        #     df_engineered['anomaly_score'] = iso_forest.fit_predict(
        #         df_engineered[available_anomaly_features])
        
        # # Cluster similar operating conditions
        # clustering_features = ['irradiance_normalized', 'temp_difference', 'environmental_stress']
        # available_clustering_features = [col for col in clustering_features if col in df_engineered.columns]
        
        # if len(available_clustering_features) >= 2:
        #     # Handle missing values for clustering
        #     clustering_data = df_engineered[available_clustering_features].fillna(df_engineered[available_clustering_features].mean())
            
        #     kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        #     df_engineered['operating_regime'] = kmeans.fit_predict(clustering_data)
            
        #     # Regime-specific performance metrics
        #     if 'power_output' in df_engineered.columns:
        #         regime_performance = df_engineered.groupby('operating_regime')['power_output'].mean()
        #         df_engineered['regime_expected_power'] = df_engineered['operating_regime'].map(regime_performance)
        #         df_engineered['regime_performance_deviation'] = (df_engineered['power_output'] - 
        #                                                        df_engineered['regime_expected_power'])
        
        return df_engineered