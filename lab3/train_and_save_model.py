import pandas as pd
import joblib
import io
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
from google.cloud import storage


def download_data():
    """Load Iris dataset"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y


def preprocess_data(X, y):
    """Split data into train and test sets"""
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    """Train Logistic Regression model (same as Lab 2)"""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    return accuracy, f1


def save_model_to_gcs(model, bucket_name, blob_name):
    """Saves model directly to GCS using an in-memory buffer"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        print(f"Model uploaded to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"Failed to upload model: {e}")


def save_metrics_to_gcs(accuracy, f1, bucket_name, blob_name):
    """Save metrics to GCS"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        metrics_content = f"Accuracy: {accuracy}\nF1 Score: {f1}\n"
        blob.upload_from_string(metrics_content, content_type='text/plain')
        print(f"Metrics saved to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"Failed to save metrics: {e}")


def main():
    bucket_name = os.getenv("GCS_BUCKET_NAME", "your-bucket-name")
    
    print("Loading Iris data...")
    X, y = download_data()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    print("Training Logistic Regression model...")
    model = train_model(X_train, y_train)
    
    print("Evaluating model...")
    accuracy, f1 = evaluate_model(model, X_test, y_test)
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    print("Uploading model to GCS...")
    save_model_to_gcs(model, bucket_name, f"models/iris_lr_{timestamp}.joblib")
    save_model_to_gcs(model, bucket_name, "models/latest_model.joblib")
    
    print("Uploading metrics to GCS...")
    save_metrics_to_gcs(accuracy, f1, bucket_name, f"metrics/evaluation_{timestamp}.txt")
    save_metrics_to_gcs(accuracy, f1, bucket_name, "metrics/latest_evaluation.txt")
    
    print("Done!")


if __name__ == "__main__":
    main()
