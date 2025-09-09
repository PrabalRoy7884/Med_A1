"""
Model training module for disease prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
import joblib
import os

class DiseasePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0

    def initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
        }

        print(f"Initialized {len(self.models)} models for training")

    def train_and_evaluate_models(self, X_train, y_train, X_val, y_val, cv_folds=5):
        """Train and evaluate all models"""
        self.results = {}

        print("Training and evaluating models...")
        print("="*50)

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            try:
                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_val)

                # Calculate accuracy
                accuracy = accuracy_score(y_val, y_pred)

                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                # Store results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred
                }

                print(f"{name} - Validation Accuracy: {accuracy:.4f}")
                print(f"{name} - CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")

                # Track best model
                if cv_mean > self.best_score:
                    self.best_score = cv_mean
                    self.best_model = model
                    self.best_model_name = name

            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        print(f"\nBest model: {self.best_model_name} with CV score: {self.best_score:.4f}")
        return self.results

    def optimize_hyperparameters(self, X_train, y_train, model_name='Random Forest', n_trials=50):
        """Optimize hyperparameters using Optuna"""

        def objective(trial):
            if model_name == 'Random Forest':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_name == 'XGBoost':
                model = XGBClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    random_state=42,
                    eval_metric='mlogloss'
                )
            elif model_name == 'CatBoost':
                model = CatBoostClassifier(
                    iterations=trial.suggest_int('iterations', 100, 500),
                    depth=trial.suggest_int('depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 10),
                    random_state=42,
                    verbose=False
                )
            else:
                raise ValueError(f"Hyperparameter optimization not implemented for {model_name}")

            # Cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()

        print(f"\nOptimizing hyperparameters for {model_name}...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials)

        print(f"Best trial: {study.best_trial.value:.4f}")
        print(f"Best params: {study.best_trial.params}")

        # Train best model
        if model_name == 'Random Forest':
            best_model = RandomForestClassifier(**study.best_trial.params, random_state=42, n_jobs=-1)
        elif model_name == 'XGBoost':
            best_model = XGBClassifier(**study.best_trial.params, random_state=42, eval_metric='mlogloss')
        elif model_name == 'CatBoost':
            best_model = CatBoostClassifier(**study.best_trial.params, random_state=42, verbose=False)

        best_model.fit(X_train, y_train)

        # Update best model if better
        if study.best_trial.value > self.best_score:
            self.best_score = study.best_trial.value
            self.best_model = best_model
            self.best_model_name = f"{model_name} (Optimized)"

        return best_model, study.best_trial.params

    def create_ensemble_model(self, X_train, y_train):
        """Create ensemble model using voting classifier"""
        from sklearn.ensemble import VotingClassifier

        # Select top 3 models based on CV scores
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
        top_3_models = [(name, result['model']) for name, result in sorted_models[:3]]

        print(f"Creating ensemble with top 3 models: {[name for name, _ in top_3_models]}")

        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=top_3_models,
            voting='soft'  # Use probability-based voting
        )

        ensemble.fit(X_train, y_train)

        # Evaluate ensemble
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
        ensemble_score = cv_scores.mean()

        print(f"Ensemble CV Score: {ensemble_score:.4f}")

        # Update best model if ensemble is better
        if ensemble_score > self.best_score:
            self.best_score = ensemble_score
            self.best_model = ensemble
            self.best_model_name = "Ensemble (Top 3)"

        return ensemble

    def save_best_model(self, filepath='models/best_model.pkl', preprocessor=None):
        """Save the best trained model"""
        if self.best_model is None:
            print("No trained model to save!")
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model and metadata
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'cv_score': self.best_score,
            'preprocessor': preprocessor
        }

        joblib.dump(model_data, filepath)
        print(f"Best model ({self.best_model_name}) saved to {filepath}")
        print(f"CV Score: {self.best_score:.4f}")

    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model"""
        if self.best_model is None:
            print("No trained model available!")
            return None

        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                importances = np.abs(self.best_model.coef_[0])
            else:
                print("Feature importance not available for this model type")
                return None

            # Create DataFrame with feature importance
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            return feature_importance_df

        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return None
