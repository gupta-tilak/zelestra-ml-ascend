import pandas as pd
import numpy as np

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
        
        # Installation type encoding (this creates installation_type_tracking)
        if 'installation_type' in df.columns:
            df_engineered['installation_type_tracking'] = (df_engineered['installation_type'] == 'tracking').astype(int)
        
        return df_engineered