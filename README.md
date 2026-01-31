# MLOps GitHub Labs

## Overview
This repository contains 4 MLOps labs demonstrating CI/CD pipelines, model training, testing, and cloud deployment using GitHub Actions and Google Cloud Platform.

## Dataset & Model
- **Dataset**: Iris (built into scikit-learn)
- **Model**: Logistic Regression

## Project Structure
```
MLops_GIthub_Lab01/
├── .github/workflows/
│   ├── lab1_pytest_action.yml
│   ├── lab1_unittest_action.yml
│   ├── model_retraining_on_push.yml
│   ├── model_calibration_on_push.yml
│   ├── lab3_train_and_upload.yml
│   └── lab4_cicd_pipeline.yml
├── lab1/ (Python Testing)
├── lab2/ (Model Training)
├── lab3/ (GCP Storage)
└── lab4/ (CI/CD + Docker)
```

## GCP Screenshots
I have attached images of GCP for Lab 3 and Lab 4:
- **Lab 3**: Google Cloud Storage bucket with trained models![WhatsApp Image 2026-01-31 at 12 34 54 AM (1)](https://github.com/user-attachments/assets/d74c5fd3-7878-43aa-835e-730ef4da65d0)

- **Lab 4**: Artifact Regist![WhatsApp Image 2026-01-31 at 1 47 30 AM](https://github.com/user-attachments/assets/ba469eef-6b81-4ff4-9712-f9fb0094e176)
ry with Docker images

## Technologies Used
- Python
- GitHub Actions
- Google Cloud Platform (GCS, Artifact Registry)
- Docker
- Pytest & Unittest
- scikit-learn
