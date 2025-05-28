import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

class ImputationPipeline:
    
    def __init__(self):
        self.temp_from_module_model = None
        self.module_from_temp_model = None
        self.mice_imputer = None
        self.knn_imputer = None
        self.error_code_fill = None
        self.installation_type_fill = None
        self.mice_cols = ['irradiance', 'voltage', 'current', 'panel_age', 'cloud_coverage', 'maintenance_count','soiling_ratio']
        self.knn_cols = ['wind_speed', 'pressure', 'temperature', 'module_temperature', 'humidity']

    def fit(self, df):
        df = df.copy()

        # Regression imputation: temperature from module_temperature
        temp_mask = df['temperature'].notna() & df['module_temperature'].notna()
        self.temp_from_module_model = LinearRegression().fit(
            df.loc[temp_mask, ['module_temperature']], df.loc[temp_mask, 'temperature']
        )

        # Regression imputation: module_temperature from temperature
        module_mask = df['temperature'].notna() & df['module_temperature'].notna()
        self.module_from_temp_model = LinearRegression().fit(
            df.loc[module_mask, ['temperature']], df.loc[module_mask, 'module_temperature']
        )

        # MICE imputer
        self.mice_imputer = IterativeImputer(random_state=42, max_iter=10, sample_posterior=False)
        self.mice_imputer.fit(df[self.mice_cols])

        # KNN imputer
        self.knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
        self.knn_imputer.fit(df[self.knn_cols])

        # Categorical fills
        self.error_code_fill = df['error_code'].fillna('NO_ERROR')
        self.installation_type_fill = df['installation_type'].mode()[0] if 'installation_type' in df else 'UNKNOWN'

        return self

    def transform(self, df):
        df = df.copy()

        # Regression imputation: temperature from module_temperature
        mask = df['temperature'].isna() & df['module_temperature'].notna()
        if mask.any():
            df.loc[mask, 'temperature'] = self.temp_from_module_model.predict(df.loc[mask, ['module_temperature']])

        # Regression imputation: module_temperature from temperature
        mask = df['module_temperature'].isna() & df['temperature'].notna()
        if mask.any():
            df.loc[mask, 'module_temperature'] = self.module_from_temp_model.predict(df.loc[mask, ['temperature']])

        # MICE imputation
        mice_data = df[self.mice_cols]
        mice_imputed = self.mice_imputer.transform(mice_data)
        df[self.mice_cols] = pd.DataFrame(mice_imputed, columns=self.mice_cols, index=df.index)

        # KNN imputation
        knn_data = df[self.knn_cols]
        knn_imputed = self.knn_imputer.transform(knn_data)
        df[self.knn_cols] = pd.DataFrame(knn_imputed, columns=self.knn_cols, index=df.index)

        # Categorical fills
        if 'error_code' in df:
            df['error_code'] = df['error_code'].fillna('NO_ERROR')

        if 'installation_type' in df:
            df['installation_type'] = df['installation_type'].fillna(self.installation_type_fill)

        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)