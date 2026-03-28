"""
Data Preprocessing Module for TG-GAT DDoS Detection System.

This module handles comprehensive data preprocessing including:
- Data loading and integration from multiple datasets
- Feature engineering and selection
- Data cleaning and normalization
- Train/validation/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for network traffic datasets.
    
    Handles multiple datasets (CIC-DDoS2019, CSE-CIC-IDS2018, InSDN) with
    unified preprocessing and feature engineering.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'Label'
        self.processed_data = None
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'CIC-DDoS2019': {
                'label_mapping': {'Benign': 'Benign', 'DDoS': 'DDoS'},
                'required_columns': self._get_cic_ddos2019_columns()
            },
            'CSE-CIC-IDS2018': {
                'label_mapping': {'Benign': 'Benign'},
                'attack_mapping': {
                    'Bot': 'DDoS', 'DDoS': 'DDoS', 'DoS': 'DDoS',
                    'Web': 'DDoS', 'Bruteforce': 'DDoS', 'Infiltration': 'DDoS'
                },
                'required_columns': self._get_cic_ids2018_columns()
            },
            'InSDN': {
                'label_mapping': {'Normal': 'Benign', 'Attack': 'DDoS'},
                'required_columns': self._get_insdn_columns()
            }
        }
    
    def load_dataset(self, dataset_path: str, dataset_type: str) -> pd.DataFrame:
        """
        Load dataset from file path based on dataset type.
        
        Args:
            dataset_path: Path to the dataset file
            dataset_type: Type of dataset ('CIC-DDoS2019', 'CSE-CIC-IDS2018', 'InSDN')
            
        Returns:
            Loaded DataFrame
        """
        try:
            if dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            elif dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset_path}")
            
            logger.info(f"Loaded {dataset_type} dataset: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {dataset_path}: {str(e)}")
            raise
    
    def preprocess_dataset(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Preprocess a single dataset.
        
        Args:
            df: Input DataFrame
            dataset_type: Type of dataset
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Preprocessing {dataset_type} dataset")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # 1. Handle missing values
        df = self._handle_missing_values(df)
        
        # 2. Standardize column names
        df = self._standardize_column_names(df, dataset_type)
        
        # 3. Map labels to unified format
        df = self._map_labels(df, dataset_type)
        
        # 4. Feature engineering
        df = self._feature_engineering(df, dataset_type)
        
        # 5. Remove invalid samples
        df = self._remove_invalid_samples(df)
        
        logger.info(f"Preprocessed {dataset_type} dataset: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values")
            
            # Fill numerical columns with median
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            
            # Fill categorical columns with mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != self.target_column:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def _standardize_column_names(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Standardize column names across datasets."""
        # Common column mappings
        column_mappings = {
            'Src IP': 'src_ip',
            'Dst IP': 'dst_ip', 
            'Src Port': 'src_port',
            'Dst Port': 'dst_port',
            'Protocol': 'protocol',
            'Timestamp': 'timestamp',
            'Flow Duration': 'flow_duration',
            'Label': 'Label'
        }
        
        # Apply mappings
        df = df.rename(columns=column_mappings)
        
        # Ensure required columns exist
        required_cols = self.dataset_configs[dataset_type]['required_columns']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns in {dataset_type}: {missing_cols}")
            # Create missing columns with default values
            for col in missing_cols:
                df[col] = 0
        
        return df
    
    def _map_labels(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Map labels to unified format (Benign/DDoS)."""
        config = self.dataset_configs[dataset_type]
        
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found")
            return df
        
        # Map labels
        label_mapping = config['label_mapping']
        df['Label'] = df[self.target_column].map(label_mapping)
        
        # Handle attack mapping if exists
        if 'attack_mapping' in config:
            attack_mapping = config['attack_mapping']
            df['Label'] = df['Label'].fillna(df[self.target_column].map(attack_mapping))
        
        # Fill unmapped labels
        df['Label'] = df['Label'].fillna('Unknown')
        
        # Remove unknown labels
        df = df[df['Label'] != 'Unknown']
        
        return df
    
    def _feature_engineering(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Perform feature engineering on the dataset."""
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Create ratio features
        if all(col in df.columns for col in ['Tot Fwd Pkts', 'Tot Bwd Pkts']):
            df['fwd_bwd_pkts_ratio'] = df['Tot Fwd Pkts'] / (df['Tot Bwd Pkts'] + 1e-8)
        
        if all(col in df.columns for col in ['TotLen Fwd Pkts', 'TotLen Bwd Pkts']):
            df['fwd_bwd_bytes_ratio'] = df['TotLen Fwd Pkts'] / (df['TotLen Bwd Pkts'] + 1e-8)
        
        # Create packet size features
        if all(col in df.columns for col in ['TotLen Fwd Pkts', 'Tot Fwd Pkts']):
            df['avg_fwd_packet_size'] = df['TotLen Fwd Pkts'] / (df['Tot Fwd Pkts'] + 1e-8)
        
        if all(col in df.columns for col in ['TotLen Bwd Pkts', 'Tot Bwd Pkts']):
            df['avg_bwd_packet_size'] = df['TotLen Bwd Pkts'] / (df['Tot Bwd Pkts'] + 1e-8)
        
        # Create flow rate features
        if all(col in df.columns for col in ['Tot Fwd Pkts', 'flow_duration']):
            df['fwd_flow_rate'] = df['Tot Fwd Pkts'] / (df['flow_duration'] + 1e-8)
        
        if all(col in df.columns for col in ['Tot Bwd Pkts', 'flow_duration']):
            df['bwd_flow_rate'] = df['Tot Bwd Pkts'] / (df['flow_duration'] + 1e-8)
        
        # Encode categorical features
        categorical_columns = ['protocol', 'src_ip', 'dst_ip'] if all(col in df.columns for col in ['protocol', 'src_ip', 'dst_ip']) else []
        
        for col in categorical_columns:
            if col in df.columns and col != self.target_column:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def _remove_invalid_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid samples from the dataset."""
        initial_size = len(df)
        
        # Remove samples with invalid flow duration
        if 'flow_duration' in df.columns:
            df = df[df['flow_duration'] > 0]
        
        # Remove samples with invalid packet counts
        packet_cols = ['Tot Fwd Pkts', 'Tot Bwd Pkts']
        for col in packet_cols:
            if col in df.columns:
                df = df[df[col] >= 0]
        
        # Remove samples with infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        final_size = len(df)
        logger.info(f"Removed {initial_size - final_size} invalid samples")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            df: Input DataFrame
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Normalized DataFrame
        """
        # Identify numerical columns (excluding target and categorical)
        exclude_columns = ['Label', 'src_ip', 'dst_ip', 'timestamp']
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        numerical_columns = [col for col in numerical_columns if col not in exclude_columns]
        
        if numerical_columns:
            if fit_scaler:
                df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
            else:
                df[numerical_columns] = self.scaler.transform(df[numerical_columns])
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=1-train_split-val_split,
            stratify=df['Label'],
            random_state=42
        )
        
        # Second split: train vs val
        val_size = val_split / (train_split + val_split)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            stratify=train_val_df['Label'],
            random_state=42
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get feature columns for model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature column names
        """
        exclude_columns = ['Label', 'timestamp', 'src_ip', 'dst_ip']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        self.feature_columns = feature_columns
        logger.info(f"Selected {len(feature_columns)} feature columns")
        
        return feature_columns
    
    def process_multiple_datasets(self, dataset_paths: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Process multiple datasets and combine them.
        
        Args:
            dataset_paths: Dictionary mapping dataset types to list of file paths
            
        Returns:
            Combined and preprocessed DataFrame
        """
        all_dataframes = []
        
        for dataset_type, paths in dataset_paths.items():
            logger.info(f"Processing {dataset_type} datasets")
            
            dataset_dfs = []
            for path in paths:
                try:
                    df = self.load_dataset(path, dataset_type)
                    df = self.preprocess_dataset(df, dataset_type)
                    dataset_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
                    continue
            
            if dataset_dfs:
                combined_df = pd.concat(dataset_dfs, ignore_index=True)
                all_dataframes.append(combined_df)
        
        if not all_dataframes:
            raise ValueError("No valid datasets found")
        
        # Combine all datasets
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Normalize features
        final_df = self.normalize_features(final_df, fit_scaler=True)
        
        # Get feature columns
        self.get_feature_columns(final_df)
        
        self.processed_data = final_df
        logger.info(f"Combined dataset shape: {final_df.shape}")
        
        return final_df
    
    def _get_cic_ddos2019_columns(self) -> List[str]:
        """Get required columns for CIC-DDoS2019 dataset."""
        return [
            'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
            'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Label'
        ]
    
    def _get_cic_ids2018_columns(self) -> List[str]:
        """Get required columns for CSE-CIC-IDS2018 dataset."""
        return [
            'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
            'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Label'
        ]
    
    def _get_insdn_columns(self) -> List[str]:
        """Get required columns for InSDN dataset."""
        return [
            'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
            'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Label'
        ]
