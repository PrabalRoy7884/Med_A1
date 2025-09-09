"""
Data preprocessing module for disease prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import os

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.symptom_columns = None
        self.target_column = 'prognosis'

    def load_data(self, train_path, test_path=None):
        """Load training and testing data"""
        self.train_data = pd.read_csv(train_path)
        if test_path:
            self.test_data = pd.read_csv(test_path)
        else:
            self.test_data = None

        print(f"Training data shape: {self.train_data.shape}")
        if self.test_data is not None:
            print(f"Testing data shape: {self.test_data.shape}")

        # Get symptom columns (all columns except prognosis)
        self.symptom_columns = [col for col in self.train_data.columns if col != self.target_column]
        print(f"Number of symptom features: {len(self.symptom_columns)}")
        print(f"Number of unique diseases: {self.train_data[self.target_column].nunique()}")

        return self.train_data, self.test_data

    def encode_labels(self, data=None):
        """Encode disease labels to numerical values"""
        if data is None:
            data = self.train_data

        # Fit the encoder on training data
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(self.train_data[self.target_column])

        # Transform the labels
        encoded_labels = self.label_encoder.transform(data[self.target_column])
        return encoded_labels

    def prepare_features_and_target(self, data=None, encode_target=True):
        """Prepare features (X) and target (y) for model training"""
        if data is None:
            data = self.train_data

        X = data[self.symptom_columns].values

        if encode_target:
            y = self.encode_labels(data)
        else:
            y = data[self.target_column].values

        return X, y

    def balance_data(self, X, y, random_state=42):
        """Balance the dataset using RandomOverSampler"""
        ros = RandomOverSampler(random_state=random_state)
        X_balanced, y_balanced = ros.fit_resample(X, y)

        print(f"Original data shape: {X.shape}")
        print(f"Balanced data shape: {X_balanced.shape}")

        return X_balanced, y_balanced

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and validation sets"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")

        return X_train, X_val, y_train, y_val

    def get_disease_names(self):
        """Get the list of disease names"""
        return self.label_encoder.classes_.tolist()

    def get_symptom_names(self):
        """Get the list of symptom names"""
        return self.symptom_columns

    def decode_predictions(self, predictions):
        """Convert numerical predictions back to disease names"""
        return self.label_encoder.inverse_transform(predictions)

    def save_processed_data(self, X_train, X_val, y_train, y_val, output_dir='data/processed'):
        """Save processed data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        # Create DataFrames
        train_processed = pd.DataFrame(X_train, columns=self.symptom_columns)
        train_processed[self.target_column] = y_train

        val_processed = pd.DataFrame(X_val, columns=self.symptom_columns)
        val_processed[self.target_column] = y_val

        # Save to CSV
        train_processed.to_csv(f'{output_dir}/train_processed.csv', index=False)
        val_processed.to_csv(f'{output_dir}/val_processed.csv', index=False)

        print(f"Processed data saved to {output_dir}/")

    def preprocess_user_input(self, selected_symptoms):
        """Preprocess user input for prediction"""
        # Create a feature vector with all symptoms set to 0
        user_data = np.zeros(len(self.symptom_columns))

        # Set selected symptoms to 1
        for symptom in selected_symptoms:
            if symptom in self.symptom_columns:
                symptom_idx = self.symptom_columns.index(symptom)
                user_data[symptom_idx] = 1

        return user_data.reshape(1, -1)
