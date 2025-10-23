"""
Advanced Feature Engineering Module for Real Estate Price Prediction
=====================================================================
This module implements sophisticated feature engineering techniques including:
- Polynomial features
- Interaction terms
- Location encoding
- Derived features
- Target encoding
- Feature selection
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PolynomialFeatures, LabelEncoder, OneHotEncoder
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from category_encoders import TargetEncoder, BinaryEncoder, OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AdvancedFeatureEngineer:
    """Advanced feature engineering for real estate data"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
        self.location_encoder = None
        self.target_encoder = None
        
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic derived features from existing columns"""
        logger.info("Creating basic derived features...")
        df = df.copy()
        
        # REMOVED: price_per_sqft feature - it was causing data leakage by using target column
        # This feature allowed the model to reverse-engineer the price during training
            
        # Total rooms
        if 'bath' in df.columns and 'bhk' in df.columns:
            df['total_rooms'] = df['bath'] + df['bhk']
            
        # Bath to BHK ratio
        if 'bath' in df.columns and 'bhk' in df.columns:
            df['bath_bhk_ratio'] = df['bath'] / (df['bhk'] + 1)
            
        # Balcony to BHK ratio
        if 'balcony' in df.columns and 'bhk' in df.columns:
            df['balcony_bhk_ratio'] = df['balcony'] / (df['bhk'] + 1)
            
        # Luxury indicator (more bathrooms than bedrooms)
        if 'bath' in df.columns and 'bhk' in df.columns:
            df['is_luxury'] = (df['bath'] > df['bhk']).astype(int)
            
        # Size category
        if 'total_sqft' in df.columns:
            df['size_category'] = pd.cut(df['total_sqft'], 
                                        bins=[0, 500, 1000, 1500, 2000, 3000, np.inf],
                                        labels=['tiny', 'small', 'medium', 'large', 'xlarge', 'huge'])
            
        # BHK category
        if 'bhk' in df.columns:
            df['bhk_category'] = pd.cut(df['bhk'],
                                       bins=[0, 1, 2, 3, 4, np.inf],
                                       labels=['studio', '1bhk', '2bhk', '3bhk', '4bhk+'])
        
        # Area type encoding (if exists)
        if 'area_type' in df.columns:
            area_priority = {
                'Super built-up  Area': 4,
                'Built-up  Area': 3,
                'Plot  Area': 2,
                'Carpet  Area': 1
            }
            df['area_priority'] = df['area_type'].map(area_priority).fillna(0)
        
        # Availability encoding
        if 'availability' in df.columns:
            df['is_ready'] = df['availability'].apply(lambda x: 1 if x == 'Ready To Move' else 0)
        
        logger.info(f"Created {len(df.columns) - len(df.columns)} basic features")
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                  features: List[str],
                                  degree: int = 2,
                                  include_bias: bool = False) -> pd.DataFrame:
        """Create polynomial and interaction features"""
        logger.info(f"Creating polynomial features of degree {degree}...")
        
        if not features:
            return df
            
        # Select only the specified features that exist
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            return df
            
        X = df[available_features].fillna(0)
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(available_features)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
        
        # Remove duplicate columns (original features)
        poly_df = poly_df.drop(columns=available_features, errors='ignore')
        
        # Combine with original DataFrame
        result = pd.concat([df, poly_df], axis=1)
        
        logger.info(f"Created {len(poly_df.columns)} polynomial features")
        return result
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        logger.info("Creating interaction features...")
        df = df.copy()
        
        # Define important interactions
        interactions = [
            ('total_sqft', 'bhk', 'sqft_per_bhk'),
            ('total_sqft', 'bath', 'sqft_per_bath'),
            ('bhk', 'bath', 'bhk_bath_product'),
            ('bhk', 'balcony', 'bhk_balcony_product'),
        ]
        
        for feat1, feat2, new_name in interactions:
            if feat1 in df.columns and feat2 in df.columns:
                df[new_name] = df[feat1] * df[feat2]
        
        # Square footage ratios
        if 'total_sqft' in df.columns:
            if 'bhk' in df.columns:
                df['sqft_per_room'] = df['total_sqft'] / (df['bhk'] + df.get('bath', 0) + 1)
            
            # Log transformation of square footage
            df['log_sqft'] = np.log1p(df['total_sqft'])
            
            # Square root transformation
            df['sqrt_sqft'] = np.sqrt(df['total_sqft'])
        
        return df
    
    def encode_location_features(self, df: pd.DataFrame, 
                                target_col: Optional[str] = None,
                                is_training: bool = True,
                                location_stats: Optional[Dict] = None) -> pd.DataFrame:
        """Advanced location encoding WITHOUT data leakage"""
        logger.info("Encoding location features...")
        
        if 'location' not in df.columns:
            return df
            
        df = df.copy()
        
        # 1. Frequency encoding (safe - doesn't use target)
        location_counts = df['location'].value_counts()
        df['location_frequency'] = df['location'].map(location_counts)
        
        # 2. REMOVED target encoding - causes severe data leakage
        # Target encoding must be done separately with proper cross-validation
        # to avoid leaking information from the target variable
        
        # 3. Location clustering based on non-target features
        # Cluster by frequency and property characteristics only
        if len(location_counts) > 5:
            # Use only non-target features for clustering
            numeric_cols = [col for col in ['total_sqft', 'bhk', 'bath', 'balcony'] 
                           if col in df.columns]
            if numeric_cols:
                location_features = df.groupby('location')[numeric_cols].agg(['mean', 'std']).reset_index()
                location_features.columns = ['location'] + [f'{col}_{stat}' for col, stat in location_features.columns[1:]]
                location_features['location_frequency'] = location_features['location'].map(location_counts)
                
                feature_cols = [col for col in location_features.columns if col != 'location']
                location_features_filled = location_features[feature_cols].fillna(0)
                
                kmeans = KMeans(n_clusters=min(5, len(location_features)), random_state=42)
                location_features['location_cluster'] = kmeans.fit_predict(location_features_filled)
                
                cluster_map = dict(zip(location_features['location'], location_features['location_cluster']))
                df['location_cluster'] = df['location'].map(cluster_map).fillna(0).astype(int)
        
        # 4. Is popular location (top 20%) - safe, doesn't use target
        top_locations = location_counts.head(int(len(location_counts) * 0.2)).index
        df['is_popular_location'] = df['location'].isin(top_locations).astype(int)
        
        return df
    
    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned versions of continuous features"""
        logger.info("Creating binned features...")
        df = df.copy()
        
        # Bin continuous features
        binning_configs = [
            ('total_sqft', [0, 600, 1000, 1500, 2000, 3000, np.inf], 
             ['very_small', 'small', 'medium', 'large', 'very_large', 'huge']),
            ('bath', [0, 1, 2, 3, 4, np.inf], 
             ['1bath', '2bath', '3bath', '4bath', '5+bath']),
            ('bhk', [0, 1, 2, 3, 4, np.inf],
             ['studio', '1bhk', '2bhk', '3bhk', '4+bhk'])
        ]
        
        for col, bins, labels in binning_configs:
            if col in df.columns:
                df[f'{col}_bin'] = pd.cut(df[col], bins=bins, labels=labels)
                # Convert to numeric codes
                df[f'{col}_bin_code'] = pd.Categorical(df[f'{col}_bin']).codes
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame, 
                                   group_cols: List[str] = ['location']) -> pd.DataFrame:
        """Create statistical features based on groupings (NO TARGET LEAKAGE)"""
        logger.info("Creating statistical features...")
        df = df.copy()
        
        # ONLY use non-target numeric columns
        numeric_cols = ['total_sqft', 'bath', 'bhk', 'balcony']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        for group_col in group_cols:
            if group_col not in df.columns:
                continue
                
            for num_col in numeric_cols:
                # Group statistics (safe - no target column used)
                group_stats = df.groupby(group_col)[num_col].agg(['mean', 'std', 'min', 'max'])
                
                # Map to original dataframe
                df[f'{num_col}_{group_col}_mean'] = df[group_col].map(group_stats['mean'])
                df[f'{num_col}_{group_col}_std'] = df[group_col].map(group_stats['std']).fillna(0)
                
                # Deviation from group mean
                df[f'{num_col}_{group_col}_deviation'] = df[num_col] - df[f'{num_col}_{group_col}_mean']
                
                # Normalized deviation
                df[f'{num_col}_{group_col}_norm_deviation'] = (
                    df[f'{num_col}_{group_col}_deviation'] / 
                    (df[f'{num_col}_{group_col}_std'] + 1)
                )
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'mutual_info',
                       n_features: int = 30) -> Tuple[pd.DataFrame, List[str]]:
        """Select most important features"""
        logger.info(f"Selecting top {n_features} features using {method}...")
        
        # Remove non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Handle missing values
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, len(X_numeric.columns)))
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(n_features, len(X_numeric.columns)))
        elif method == 'rfe':
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(n_features, len(X_numeric.columns)))
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        X_selected = selector.fit_transform(X_numeric, y)
        
        # Get selected feature names
        if hasattr(selector, 'get_support'):
            selected_features = X_numeric.columns[selector.get_support()].tolist()
        else:
            selected_features = X_numeric.columns.tolist()[:n_features]
        
        logger.info(f"Selected {len(selected_features)} features")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def apply_pca(self, X: pd.DataFrame, n_components: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """Apply PCA for dimensionality reduction"""
        logger.info(f"Applying PCA with {n_components} variance explained...")
        
        # Select numeric columns only
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Handle missing values
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame with PCA components
        pca_cols = [f'pca_{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
        
        logger.info(f"PCA reduced {X_numeric.shape[1]} features to {X_pca.shape[1]} components")
        logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca_df, pca
    
    def transform(self, df: pd.DataFrame, 
                 use_polynomial: bool = True,
                 use_interactions: bool = True,
                 use_binning: bool = True,
                 use_statistical: bool = True,
                 polynomial_degree: int = 2) -> pd.DataFrame:
        """Apply all feature engineering transformations"""
        logger.info("Applying comprehensive feature engineering...")
        
        # Create basic features
        df = self.create_basic_features(df)
        
        # Create interaction features
        if use_interactions:
            df = self.create_interaction_features(df)
        
        # Encode location features
        df = self.encode_location_features(df)
        
        # Create binned features
        if use_binning:
            df = self.create_binned_features(df)
        
        # Create statistical features
        if use_statistical:
            df = self.create_statistical_features(df)
        
        # Create polynomial features for key numeric columns
        if use_polynomial:
            numeric_features = ['total_sqft', 'bath', 'bhk', 'balcony']
            numeric_features = [f for f in numeric_features if f in df.columns]
            if numeric_features:
                df = self.create_polynomial_features(df, numeric_features[:3], degree=polynomial_degree)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df


def create_feature_pipeline(df: pd.DataFrame, 
                          target_col: str = 'price',
                          feature_selection: bool = True,
                          n_features: int = 50) -> Tuple[pd.DataFrame, List[str]]:
    """Complete feature engineering pipeline"""
    
    engineer = AdvancedFeatureEngineer()
    
    # Apply all transformations
    df_transformed = engineer.transform(df)
    
    # Feature selection if requested
    if feature_selection and target_col in df_transformed.columns:
        X = df_transformed.drop(columns=[target_col])
        y = df_transformed[target_col]
        
        X_selected, selected_features = engineer.select_features(X, y, n_features=n_features)
        
        # Add back the target column
        df_final = pd.concat([X_selected, y], axis=1)
        
        return df_final, selected_features
    
    return df_transformed, df_transformed.columns.tolist()


if __name__ == "__main__":
    logger.info("Advanced Feature Engineering Module loaded successfully")
