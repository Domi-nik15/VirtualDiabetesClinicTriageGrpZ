# 🩺 Virtual Diabetes Clinic Triage — Group Z

Ein ML-Mikroservice zur Vorhersage des kurzfristigen Diabetes-Risikos, entwickelt für virtuelle Klinik-Triage-Dashboards. Das Projekt nutzt das offene Scikit-learn Diabetes-Dataset und bietet eine vollständig reproduzierbare MLOps-Pipeline mit Docker und GitHub Actions.

## 👥 Team
Dominic Behling, Filippo Besana, Dominik Eder, Chang Liu

---

## 🚀 Features

- Vorhersage des Diabetes-Risikos mit scikit-learn
- MLOps-Pipeline für Training, Testing, Packaging & Deployment
- Docker-basierte Infrastruktur mit MLflow & Python
- API-Service mit Flask & Gunicorn

---

## Architecture

```
 Model API (container :9696) ─ loads model from container file system
```

---

## Prerequisites

- **Docker Desktop** (Windows/macOS/Linux).
- **Python Environment** Version 3.11 (optional only for local tests)

---

## Repository layout

```
VirtualDiabetesClinicTriageGrpZ/
├─ .github/workflows/ # pipeline files
├─ docker/            # docker files
├─ images/            # images for the documentation
├─ serve/             # Flask/Gunicorn app (API)
├─ train/             # notebooks & training code
├─ .env               # definition files for environment variables
├─ .flake8            # configuration file for flake8 tests
├─ requirements.txt   # python requriments for the tests
└─ docker-compose.yml
```

---

## Quick start

### Build and run locally

```bash
# Train, build and run application
docker compose up -d train_and_run --build

```

### Health test

Test local container:
```bash
curl -X GET http://localhost:9696/health
```

Expected request:
```json
{"model":"short-term_disease_progression","model_version":"v0.1","status":"ok"}
```

### Prediction test

```bash
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d "{\"sex\": 0.2, \"age\": 0.1}"
```

Expected request:
```json
{"prediction":97.21090730549051}
```

---

## Services & ports

* Model API: http://localhost:9696

> If a port is busy, change the **left** side of the mapping in `docker-compose.yml` (e.g., `9697:9696`) and open the new host port.

---

## Clean up

stop containers
```bash
docker compose down
```

---

## Run prebuild container

```bash
docker run -d -p 9696:9696  ghcr.io/domi-nik15/mlflow-diabetes:latest
```

---

## Local tests

Start your python environment and install [requirements](requirements.txt)

### flake8 (lint tests)

```bash
flake8 train serve
```

### pytest (unit tests)

```bash
pytest -v --maxfail=1 --disable-warnings
```

---