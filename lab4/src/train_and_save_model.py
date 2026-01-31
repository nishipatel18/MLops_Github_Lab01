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
    """Load Iris dataset (same as Lab 2 & 3)"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y


def preprocess_data(X, y):
    """Split data into train and test sets (80/20 split)"""
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    """Train Logistic Regression model (same as Lab 2 & 3)"""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return accuracy and F1 score"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    return accuracy, f1


def get_model_version(bucket_name, version_file_name):
    """Get current model version from GCS"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(version_file_name)
        
        if blob.exists():
            version = int(blob.download_as_text().strip())
            return version
        return 0
    except Exception as e:
        print(f"❌ Error getting version: {e}")
        return 0


def update_model_version(bucket_name, version_file_name, new_version):
    """Update model version in GCS"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(version_file_name)
        blob.upload_from_string(str(new_version), content_type='text/plain')
        print(f"Version updated to {new_version}")
    except Exception as e:
        print(f"Error updating version: {e}")


def save_model_to_gcs(model, bucket_name, blob_name):
    """Save model directly to GCS using in-memory buffer"""
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


def save_model_locally(model, path):
    """Save model to local filesystem for Docker"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved locally at {path}")


def main():
    # Get environment variables
    bucket_name = os.getenv("GCS_BUCKET_NAME", "your-bucket-name")
    version_file_name = os.getenv("VERSION_FILE_NAME", "model_version.txt")
    
    print("=" * 50)
    print("Starting ML Pipeline - Lab 4")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n📥 Step 1: Loading Iris data...")
    X, y = download_data()
    print(f"   Data shape: {X.shape}")
    
    # Step 2: Preprocess
    print("\nStep 2: Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print(f"   Train size: {len(X_train)}")
    print(f"   Test size: {len(X_test)}")
    
    # Step 3: Train
    print("\n🏋️ Step 3: Training Logistic Regression model...")
    model = train_model(X_train, y_train)
    print("   Model trained successfully!")
    
    # Step 4: Evaluate
    print("\nStep 4: Evaluating model...")
    accuracy, f1 = evaluate_model(model, X_test, y_test)
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    
    # Step 5: Get version
    print("\nStep 5: Getting model version...")
    current_version = get_model_version(bucket_name, version_file_name)
    new_version = current_version + 1
    print(f"   Current version: {current_version}")
    print(f"   New version: {new_version}")
    
    # Step 6: Save locally
    print("\nStep 6: Saving model locally...")
    local_model_path = "trained_models/model.joblib"
    save_model_locally(model, local_model_path)
    
    # Step 7: Save to GCS
    print("\nStep 7: Uploading model to GCS...")
    save_model_to_gcs(model, bucket_name, f"trained_models/model_v{new_version}.joblib")
    save_model_to_gcs(model, bucket_name, "trained_models/latest_model.joblib")
    
    # Step 8: Update version
    print("\n🔢 Step 8: Updating model version...")
    update_model_version(bucket_name, version_file_name, new_version)
    
    print("\n" + "=" * 50)
    print(f"Pipeline complete! Model version: {new_version}")
    print("=" * 50)
    
    # Output version for GitHub Actions (important!)
    print(f"\nMODEL_VERSION={new_version}")


if __name__ == "__main__":
    main()
