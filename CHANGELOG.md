# Changelog

## [v0.1.0] - 2025-10-15

### 🚀 Features
- Implemented ML microservice for short-term diabetes risk prediction using Scikit-learn.
- Developed Flask & Gunicorn-based API service for model inference.
- Integrated MLflow for experiment tracking and model management.
- Added Docker-based infrastructure for reproducible builds and deployments.
- Created MLOps pipeline with GitHub Actions for training, testing, packaging, and deployment.

### 🛠️ Improvements
- Introduced lazy loading for model in `serve/` to optimize performance.
- Cleaned up repository structure and removed redundant files.
- Added `.flake8` configuration and resolved linting issues.
- Enhanced local development workflow with `docker-compose.yml`.

### 🧪 Testing
- Added unit tests using `pytest` for training and serving modules.
- Configured GitHub Actions to run flake8 and pytest checks automatically.

### 📦 Dependencies
- Defined Python requirements in `requirements.txt` for consistent environment setup.
- Specified environment variables in `.env` for local builds.

### 📁 Repository Layout
- `.github/workflows/` – CI/CD pipeline files
- `docker/` – Docker configuration
- `serve/` – API service code
- `train/` – Model training notebooks and scripts

---


### 📊 Model Details

| Attribute        | Value                                  |
|------------------|----------------------------------------|
| Model Type       | Linear Regression                      |
| Preprocessing    | StandardScaler                         |
| Input Features   | 10 (age, sex, bmi, bp, s1–s6)          |
| Output           | Continuous progression risk score      |
| Test Partion Size| 25 %                                   |
| RandomSeed       | 8                                      |
| Evaluation Metric| RMSE = 3108.05 (on held-out test split)|
| Model Size       | ~2 KB (serialized `.pkl` file)         |
| Training Dataset | Scikit-learn Diabetes Dataset          |
| Version          | v0.1.0                                 |
| Docker Image Size| ~750 MB                                |

---

Initial release of the Virtual Diabetes Clinic Triage microservice. 