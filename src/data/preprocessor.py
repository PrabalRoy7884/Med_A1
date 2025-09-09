"""
Data preprocessing module for disease prediction
"""

import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2

class FixedDataPreprocessor:
    def __init__(self, label_encoder=None, symptom_columns=None, selected_features=None, 
                 selected_feature_names=None, feature_selector=None):
        self.label_encoder = label_encoder
        self.symptom_columns = symptom_columns
        self.selected_features = selected_features
        self.selected_feature_names = selected_feature_names
        self.feature_selector = feature_selector
        
    def get_symptom_names(self):
        return self.symptom_columns
        
    def get_disease_names(self):
        return list(self.label_encoder.classes_) if self.label_encoder else []
        
    def decode_predictions(self, predictions):
        return self.label_encoder.inverse_transform(predictions)
    
    def preprocess_user_input(self, selected_symptoms):
        # Create a feature vector with all symptoms set to 0
        user_data = np.zeros(len(self.symptom_columns))
        
        # Set selected symptoms to 1
        for symptom in selected_symptoms:
            if symptom in self.symptom_columns:
                symptom_idx = self.symptom_columns.index(symptom)
                user_data[symptom_idx] = 1
        
        # Apply feature selection
        user_data_selected = self.feature_selector.transform(user_data.reshape(1, -1))
        
        return user_data_selected
