"""
Prediction module for disease prediction
"""

import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Tuple
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import FixedDataPreprocessor

class DiseasePredictionService:
    def __init__(self, model_path='models/best_model.pkl'):
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.preprocessor = None
        self.is_loaded = False

    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model_data = joblib.load(self.model_path)
            self.model = self.model_data['model']
            self.preprocessor = self.model_data.get('preprocessor', None)

            print(f"Model loaded successfully: {self.model_data['model_name']}")
            print(f"CV Score: {self.model_data['cv_score']:.4f}")

            self.is_loaded = True
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict_disease(self, symptoms: List[str]) -> Dict:
        """Predict disease based on list of symptoms"""
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "Model not loaded"}

        if not self.preprocessor:
            return {"error": "Preprocessor not available"}

        try:
            # Preprocess user input
            user_features = self.preprocessor.preprocess_user_input(symptoms)

            # Make prediction
            prediction = self.model.predict(user_features)[0]

            # Get prediction probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(user_features)[0]
                # Get top 5 predictions with probabilities
                top_indices = np.argsort(proba)[::-1][:5]
                probabilities = []
                for idx in top_indices:
                    disease_name = self.preprocessor.decode_predictions([idx])[0]
                    confidence = proba[idx]
                    probabilities.append({
                        'disease': disease_name,
                        'confidence': float(confidence)
                    })

            # Decode prediction
            predicted_disease = self.preprocessor.decode_predictions([prediction])[0]

            return {
                'predicted_disease': predicted_disease,
                'input_symptoms': symptoms,
                'probabilities': probabilities,
                'model_name': self.model_data['model_name'],
                'model_accuracy': self.model_data['cv_score']
            }

        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

    def predict_with_confidence(self, symptoms: List[str], threshold: float = 0.5) -> Dict:
        """Predict disease with confidence threshold"""
        result = self.predict_disease(symptoms)

        if "error" in result:
            return result

        # Check if top prediction meets confidence threshold
        if result['probabilities'] and len(result['probabilities']) > 0:
            top_confidence = result['probabilities'][0]['confidence']

            if top_confidence < threshold:
                return {
                    **result,
                    'warning': f'Low confidence prediction ({top_confidence:.2f}). Consider consulting a healthcare professional.',
                    'reliable_prediction': False
                }
            else:
                return {
                    **result,
                    'reliable_prediction': True
                }
        else:
            return result

    def get_similar_diseases(self, symptoms: List[str], top_n: int = 5) -> List[Dict]:
        """Get top N most similar diseases based on symptoms"""
        result = self.predict_disease(symptoms)

        if "error" in result or not result.get('probabilities'):
            return []

        return result['probabilities'][:top_n]

    def explain_prediction(self, symptoms: List[str]) -> Dict:
        """Provide explanation for the prediction"""
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "Model not loaded"}

        try:
            # Get basic prediction
            prediction_result = self.predict_disease(symptoms)

            if "error" in prediction_result:
                return prediction_result

            # Get feature importance if available
            explanation = {
                'prediction': prediction_result['predicted_disease'],
                'input_symptoms': symptoms,
                'contributing_factors': [],
                'model_confidence': prediction_result['probabilities'][0]['confidence'] if prediction_result['probabilities'] else None
            }

            # Add symptom analysis
            symptom_analysis = []
            for symptom in symptoms:
                symptom_analysis.append({
                    'symptom': symptom,
                    'relevance': 'high',  # Simplified for this example
                    'description': f'Symptom "{symptom}" is associated with the predicted condition'
                })

            explanation['symptom_analysis'] = symptom_analysis

            return explanation

        except Exception as e:
            return {"error": f"Explanation failed: {e}"}

    def batch_predict(self, symptom_lists: List[List[str]]) -> List[Dict]:
        """Predict diseases for multiple symptom lists"""
        results = []
        for symptoms in symptom_lists:
            result = self.predict_disease(symptoms)
            results.append(result)
        return results

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}

        return {
            'model_name': self.model_data['model_name'],
            'cv_score': self.model_data['cv_score'],
            'model_type': type(self.model).__name__,
            'available_diseases': len(self.preprocessor.get_disease_names()) if self.preprocessor else 0,
            'available_symptoms': len(self.preprocessor.get_symptom_names()) if self.preprocessor else 0
        }

    def validate_symptoms(self, symptoms: List[str]) -> Tuple[List[str], List[str]]:
        """Validate input symptoms and return valid/invalid lists"""
        if not self.preprocessor:
            return symptoms, []

        available_symptoms = self.preprocessor.get_symptom_names()
        valid_symptoms = [s for s in symptoms if s in available_symptoms]
        invalid_symptoms = [s for s in symptoms if s not in available_symptoms]

        return valid_symptoms, invalid_symptoms
