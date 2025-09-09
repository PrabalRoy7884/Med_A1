"""
Helper utilities for the disease prediction project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

def load_data(file_path):
    """Load CSV data"""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(f"Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def save_model(model, filepath, model_name="Model"):
    """Save trained model to disk"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"{model_name} saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving {model_name}: {e}")

def load_model(filepath):
    """Load trained model from disk"""
    try:
        model = joblib.load(filepath)
        print(f"Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        return None

def plot_class_distribution(data, target_column, title="Class Distribution"):
    """Plot distribution of target classes"""
    plt.figure(figsize=(15, 8))

    # Count plot
    plt.subplot(1, 2, 1)
    sns.countplot(data=data, y=target_column, order=data[target_column].value_counts().index)
    plt.title(f"{title} - Count Plot")
    plt.xlabel("Count")

    # Pie chart for top 10 classes
    plt.subplot(1, 2, 2)
    top_10_classes = data[target_column].value_counts().head(10)
    plt.pie(top_10_classes.values, labels=top_10_classes.index, autopct='%1.1f%%')
    plt.title(f"{title} - Top 10 Classes")

    plt.tight_layout()
    plt.show()

def plot_symptom_correlation(data, symptom_columns, n_symptoms=20):
    """Plot correlation matrix of top symptoms"""
    # Calculate symptom frequency
    symptom_freq = data[symptom_columns].sum().sort_values(ascending=False)
    top_symptoms = symptom_freq.head(n_symptoms).index.tolist()

    # Create correlation matrix
    corr_matrix = data[top_symptoms].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Correlation Matrix of Top {n_symptoms} Symptoms')
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results_dict, title="Model Comparison"):
    """Plot comparison of different models"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models = list(results_dict.keys())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, metric in enumerate(metrics):
        metric_values = [results_dict[model].get(metric.lower().replace('-', '_'), 0) for model in models]
        axes[i].bar(models, metric_values)
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for j, v in enumerate(metric_values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def create_interactive_symptom_plot(data, symptom_columns):
    """Create interactive plot showing symptom frequencies"""
    symptom_freq = data[symptom_columns].sum().sort_values(ascending=False)

    fig = px.bar(
        x=symptom_freq.values[:30], 
        y=symptom_freq.index[:30],
        orientation='h',
        title="Top 30 Most Common Symptoms",
        labels={'x': 'Frequency', 'y': 'Symptoms'}
    )
    fig.update_layout(height=800)
    return fig

def create_disease_symptom_heatmap(data, symptom_columns, target_column, top_diseases=10, top_symptoms=20):
    """Create heatmap showing relationship between diseases and symptoms"""
    # Get top diseases and symptoms
    top_disease_names = data[target_column].value_counts().head(top_diseases).index
    symptom_freq = data[symptom_columns].sum().sort_values(ascending=False)
    top_symptom_names = symptom_freq.head(top_symptoms).index

    # Create disease-symptom matrix
    disease_symptom_matrix = []
    for disease in top_disease_names:
        disease_data = data[data[target_column] == disease]
        symptom_prevalence = disease_data[top_symptom_names].mean()
        disease_symptom_matrix.append(symptom_prevalence)

    disease_symptom_df = pd.DataFrame(disease_symptom_matrix, 
                                     index=top_disease_names, 
                                     columns=top_symptom_names)

    fig = px.imshow(
        disease_symptom_df,
        title=f"Disease-Symptom Relationship Heatmap (Top {top_diseases} Diseases, Top {top_symptoms} Symptoms)",
        color_continuous_scale="RdYlBu_r",
        aspect="auto"
    )
    fig.update_layout(height=600)
    return fig

def calculate_metrics(y_true, y_pred, average='weighted'):
    """Calculate classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

    return metrics

def print_classification_report(y_true, y_pred, target_names=None):
    """Print detailed classification report"""
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print("\nDetailed Classification Report:")
    print("="*50)
    print(report)
