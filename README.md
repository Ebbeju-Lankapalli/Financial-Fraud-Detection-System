# 🛡️ Financial Fraud Detection System — Production ML + AI Investigation Agent

> Detect high-risk financial transactions with production-grade machine learning, persistent audit history, model evaluation, and grounded AI-assisted fraud investigation.

![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-Build%20Tool-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Production%20ML-FFCC00?style=for-the-badge)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Persistence-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-GPT--OSS%20Agent-F55036?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-Backend%20Hosting-46E3B7?style=for-the-badge&logo=render&logoColor=black)
![Vercel](https://img.shields.io/badge/Vercel-Frontend-000000?style=for-the-badge&logo=vercel&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-21%20Tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

---

## 🌐 Live Deployment

### Frontend

**Financial Fraud Detection System**

https://financial-fraud-detection-system-alpha.vercel.app

### Backend API

https://fraud-detection-backend-wc2p.onrender.com

Health:

https://fraud-detection-backend-wc2p.onrender.com/health

### Production CatBoost Model Service

https://fraud-catboost-model-service.onrender.com

Health:

https://fraud-catboost-model-service.onrender.com/health

Model information:

https://fraud-catboost-model-service.onrender.com/catboost/model/info

> The production deployment uses Vercel for the React frontend, Render for the FastAPI backend and CatBoost inference service, Render PostgreSQL for persistent audit history, and Groq for GPT-OSS fraud-investigation explanations.

> Free Render services can experience cold starts after periods of inactivity.

---

## 📌 Overview

**Financial Fraud Detection System** is an end-to-end AI/ML application designed to screen financial transactions for fraud risk and assist human investigators with grounded explanations.

The system combines:

- Production tabular machine learning
- CatBoost fraud classification
- IEEE-CIS Fraud Detection data
- Validation-based decision threshold optimization
- HIGH / LOW fraud-risk classification
- Fraud probability scoring
- Persistent transaction audit history
- Dashboard analytics
- Model evaluation and comparison
- Grounded AI fraud investigation
- GPT-OSS through Groq
- Qwen2.5 + QLoRA research experiments
- Dockerized inference services
- PostgreSQL cloud persistence
- React + FastAPI full-stack architecture
- Vercel + Render deployment

The production fraud decision is made entirely by the **CatBoost classifier**.

The LLM is used only to explain the result and provide investigation guidance.

---

## 🎯 Problem Statement

Financial institutions process large volumes of transactions, making manual inspection of every transaction impractical.

A fraud-screening system should help identify transactions that deserve additional investigation.

This project implements the following workflow:

```text
Financial Transaction
        ↓
Feature Validation
        ↓
Production CatBoost Model
        ↓
Fraud Probability
        ↓
Validation-Selected Threshold
        ↓
HIGH / LOW Risk
        ↓
Audit Persistence
        ↓
Human Review
        ↓
Optional AI Investigation Agent
```

The system provides a **screening signal**, not legal proof that fraud occurred.

---

## ✨ Key Features

### 🔍 Production Fraud Detection

Users can submit a transaction containing the 20 features required by the production IEEE-CIS CatBoost model.

The system returns:

```text
Risk Classification
Fraud Probability
Decision Threshold
Model Name
Feature Count
Decision Source
Output Validation Status
```

Example verified production result:

```text
Risk                HIGH
Fraud Probability   99.72%
Threshold           83%
Model               catboost_reduced_fraud_detector
Features            20
Decision Source     catboost_ieee_cis
Valid Output        True
```

---

### ⚖️ Validation-Based Threshold Selection

The production decision threshold is:

```text
0.83
```

The threshold was selected using the **validation dataset**, optimizing F1 score.

```text
Validation Set
      ↓
Candidate Thresholds
      ↓
Precision / Recall / F1
      ↓
Best Validation F1
      ↓
Threshold = 0.83
```

The held-out test dataset was **not used for threshold selection**.

Production decision rule:

```text
Fraud Probability >= 0.83
        ↓
      HIGH

Fraud Probability < 0.83
        ↓
       LOW
```

---

## 📊 Production Model Performance

The active production model is:

```text
catboost_reduced_fraud_detector
```

Dataset:

```text
IEEE-CIS Fraud Detection
```

Production feature count:

```text
20
```

Evaluation split:

```text
Chronological Held-Out Test Set
```

Test rows:

```text
88,581
```

Fraud rows:

```text
3,083
```

### Test Metrics

| Metric | Score |
|---|---:|
| Accuracy | **96.91%** |
| Precision | **58.02%** |
| Recall | **40.61%** |
| F1 Score | **47.78%** |
| ROC-AUC | **88.51%** |
| PR-AUC | **46.36%** |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | **84,592** | **906** |
| Actual Fraud | **1,831** | **1,252** |

```text
True Negatives   84,592
False Positives     906
False Negatives   1,831
True Positives    1,252
```

Because fraud detection is highly imbalanced, the project reports **precision, recall, F1, ROC-AUC, and PR-AUC** rather than relying only on accuracy.

---

## 🧠 Why CatBoost?

Fraud detection in this project is primarily a structured tabular-data problem.

CatBoost was selected for the production model because gradient-boosted decision trees perform strongly on tabular datasets and can learn complex nonlinear relationships between transaction features.

The project includes both:

```text
61-feature CatBoost reference model
            ↓
Feature importance analysis
            ↓
Reduced 20-feature model
            ↓
Validation threshold optimization
            ↓
Production deployment
```

The reduced model provides a smaller production inference interface while preserving strong fraud-ranking performance.

---

## 🤖 AI Fraud Investigation Agent

The project includes a fraud-investigation agent powered by:

```text
Groq
  ↓
openai/gpt-oss-20b
```

The agent receives:

```text
Transaction Data
      +
Authoritative CatBoost Prediction
      +
Optional Investigator Question
```

and generates:

```text
Human-Readable Explanation
Investigation Context
Recommended Analyst Actions
```

---

## 🔒 CatBoost Is the Decision Authority

The LLM **does not classify the transaction**.

The architecture deliberately separates classification from explanation:

```text
            CatBoost
               ↓
       HIGH / LOW Decision
               ↓
       Fraud Probability
               ↓
               │
               ▼
        Fraud Investigation
             Agent
               ↓
            GPT-OSS
               ↓
     Explanation + Guidance
```

GPT-OSS is not allowed to change:

```text
HIGH → LOW
```

or:

```text
LOW → HIGH
```

The CatBoost result remains authoritative throughout the workflow.

---

## 🛡️ LLM Grounding & Hallucination Protection

IEEE-CIS includes anonymized or encoded features such as:

```text
card1
card2
card5
addr1
C*
D*
M*_enc
```

Their exact real-world business meanings are not available from the dataset.

The agent is explicitly instructed not to invent meanings such as:

```text
card1 = actual credit-card number
addr1 = billing address
card5 = bank issuer
```

A grounding regression test verifies this behavior.

Example adversarial question:

```text
Is card1 the customer's actual card number
and is addr1 their billing address?
```

Verified agent behavior:

```text
No.

card1 and addr1 are anonymized or encoded IEEE-CIS features.
Their real-world meanings cannot be inferred from the dataset.
```

This protects the investigation layer from presenting unsupported interpretations as facts.

---

## 📋 Transaction Audit History

Every analyzed transaction receives:

```text
analysis_id
created_at
transaction inputs
risk
fraud probability
threshold
model
feature count
decision source
valid output status
```

Records can later be retrieved through the History page or API.

This provides basic model-decision traceability and auditability.

---

## 📊 Dashboard

The production dashboard provides:

```text
Total Analyses
HIGH Risk Count
LOW Risk Count
HIGH Risk Percentage
Valid Output Count
Invalid Output Count
Risk Distribution
Recent Fraud Probabilities
Recent Analyses
```

The deployed dashboard reads persisted transaction records from PostgreSQL.

---

## 🔎 Transaction Detail

Each stored transaction can be opened individually.

The transaction-detail page displays:

- Analysis ID
- Timestamp
- HIGH / LOW classification
- Fraud probability
- Decision threshold
- Model name
- Decision source
- Valid output status
- All 20 submitted production features
- AI investigation option

This makes model predictions inspectable after inference.

---

## 📈 Model Evaluation

The application exposes model-evaluation endpoints and a frontend evaluation dashboard.

Production evaluation includes:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
PR-AUC
Confusion Matrix
Validation Threshold
Full 61-Feature Reference
Reduced 20-Feature Production Model
```

Production evaluation is intentionally separated from the Qwen/QLoRA research experiment.

---

## 🧪 Qwen2.5 + QLoRA Research

The repository preserves an earlier LLM fine-tuning experiment using:

```text
Qwen/Qwen2.5-1.5B-Instruct
        ↓
4-bit Quantization
        ↓
QLoRA
        ↓
PEFT
        ↓
SFTTrainer
        ↓
Fine-Tuned Fraud Classification Experiment
```

The fine-tuned model experiment is preserved for research and comparison.

It is **not the production fraud decision source**.

Production:

```text
IEEE-CIS
   ↓
CatBoost
   ↓
HIGH / LOW
```

Research:

```text
Transaction Prompt
      ↓
Qwen2.5
      ↓
QLoRA Fine-Tuning
      ↓
Experimental HIGH / LOW Classification
```

The repository keeps these two concerns explicitly separated.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    React + Vite Frontend                    │
│                                                             │
│ Home │ Dashboard │ Analyze │ Agent │ History │ Evaluation  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ HTTPS / REST
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
│                                                             │
│ Transactions │ Dashboard │ Agent │ Evaluation │ Persistence│
└───────────────┬─────────────────────┬───────────────────────┘
                │                     │
                │                     │
                ▼                     ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│ CatBoost Model Service   │   │ Fraud Investigation Agent   │
│                          │   │                              │
│ IEEE-CIS                 │   │ Groq                        │
│ 20 Features              │   │ GPT-OSS-20B                 │
│ Threshold 0.83           │   │ Grounded Explanation        │
└─────────────┬────────────┘   └──────────────────────────────┘
              │
              ▼
      HIGH / LOW + Score
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PostgreSQL Database                      │
│                                                             │
│ Transaction Inputs │ Predictions │ Audit History           │
└─────────────────────────────────────────────────────────────┘
```

---

## ☁️ Production Deployment Architecture

```text
User Browser
     ↓
Vercel
React + Vite Frontend
     ↓
Render
FastAPI Backend
     ├──────────────→ Render PostgreSQL
     │
     ├──────────────→ Render CatBoost Model Service
     │                       ↓
     │                CatBoost .cbm Artifact
     │
     └──────────────→ Groq API
                             ↓
                       GPT-OSS-20B
```

Production services are independently deployable.

---

## 🐳 Docker Architecture

The production system contains separate Docker images for:

```text
backend/
└── Dockerfile

model-service/
└── Dockerfile
```

The production CatBoost container intentionally excludes heavyweight Qwen research dependencies.

Production model-service dependencies include:

```text
FastAPI
Uvicorn
Pydantic
CatBoost
Pandas
NumPy
```

Research dependencies are preserved separately in:

```text
model-service/requirements-research.txt
```

This keeps the production image significantly smaller and simpler.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, Vite, React Router, Tailwind CSS, Recharts, Lucide React |
| Backend | Python, FastAPI, Pydantic |
| Production ML | CatBoost |
| Dataset | IEEE-CIS Fraud Detection |
| AI Agent | Groq, GPT-OSS-20B |
| Research LLM | Qwen2.5-1.5B-Instruct |
| Fine-Tuning | QLoRA, LoRA, PEFT, SFT |
| Persistence | PostgreSQL, SQLite |
| Model Service | FastAPI, CatBoost |
| Testing | Pytest |
| Linting | Ruff |
| Containers | Docker |
| Frontend Deployment | Vercel |
| Backend Deployment | Render |
| Model Deployment | Render |
| Database Deployment | Render PostgreSQL |
| Version Control | Git, GitHub |

---

## 📂 Project Structure

```text
Financial-Fraud-Detection-System/
│
├── artifacts/
│   ├── ieee_cis/
│   │   ├── catboost_fraud_detector.cbm
│   │   ├── catboost_reduced_fraud_detector.cbm
│   │   ├── final_test_metrics.json
│   │   ├── reduced_final_test_metrics.json
│   │   ├── feature_importance.json
│   │   └── reduced_threshold_optimization.json
│   └── tabular/
│
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   ├── api/
│   │   ├── core/
│   │   ├── repositories/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── tools/
│   │   └── utils/
│   ├── tests/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── agent/
│   │   │   ├── common/
│   │   │   ├── dashboard/
│   │   │   ├── evaluation/
│   │   │   ├── layout/
│   │   │   └── transaction/
│   │   ├── pages/
│   │   ├── services/
│   │   └── utils/
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── vercel.json
│
├── model-service/
│   ├── catboost_model/
│   ├── model/
│   ├── schemas/
│   ├── utils/
│   ├── production_app.py
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements-research.txt
│
├── ml/
│   ├── configs/
│   ├── data/
│   ├── notebooks/
│   └── src/
│       ├── data/
│       ├── evaluation/
│       ├── ieee_cis/
│       ├── tabular/
│       ├── training/
│       └── utils/
│
├── evaluation/
│   ├── figures/
│   └── results/
│
├── docs/
│   ├── architecture/
│   ├── screenshots/
│   ├── MODEL_CARD.md
│   └── PROJECT_REPORT.md
│
├── scripts/
├── .github/
│   └── workflows/
│
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🌐 REST API

### Backend

#### Health

```http
GET /health
```

#### Transactions

```http
POST /api/transactions/analyze
GET  /api/transactions/history
GET  /api/transactions/{analysis_id}
```

#### Dashboard

```http
GET /api/dashboard
```

#### Fraud Agent

```http
GET  /api/agent
POST /api/agent/analyze
```

#### Production Evaluation

```http
GET /api/evaluation
GET /api/evaluation/production
GET /api/evaluation/production/threshold
GET /api/evaluation/production/full-reference
```

#### Research Evaluation

```http
GET /api/evaluation/research
GET /api/evaluation/base
GET /api/evaluation/finetuned
GET /api/evaluation/comparison
```

---

## 🤖 Model-Service API

### Health

```http
GET /health
```

### Model Information

```http
GET /catboost/model/info
```

### Prediction

```http
POST /catboost/predict
```

Example response:

```json
{
  "risk": "HIGH",
  "fraud_probability": 0.9971981836996825,
  "threshold": 0.83,
  "model": "catboost_reduced_fraud_detector",
  "feature_count": 20,
  "decision_source": "catboost_ieee_cis",
  "valid_output": true
}
```

---

## 🧪 Automated Testing

The verified backend test suite currently contains:

```text
Tests:       21 passed
Failures:    0
```

Covered areas include:

- Transaction analysis
- Transaction persistence
- Transaction lookup
- Transaction history
- Input validation
- Dashboard metrics
- Empty dashboard handling
- Agent deterministic fallback
- Groq agent integration
- Agent API
- Agent grounding protection
- Production evaluation
- Research evaluation
- Threshold endpoint
- Missing evaluation artifacts
- Health API
- Database backend detection

Run:

```bash
PYTHONPATH=backend pytest backend/tests -v
```

---

## 🧹 Code Quality

Python code is validated with Ruff:

```bash
ruff check \
backend/app \
backend/tests \
model-service \
ml/src
```

Git whitespace verification:

```bash
git diff --check
```

Frontend production build:

```bash
cd frontend
npm run build
```

---

## ✅ Verified Production Flow

The complete deployed architecture has been manually tested.

```text
Public Vercel Website
        ↓
Render FastAPI Backend
        ↓
Render CatBoost Model Service
        ↓
20-Feature IEEE-CIS Input
        ↓
Fraud Probability
        ↓
0.83 Decision Threshold
        ↓
HIGH / LOW Classification
        ↓
PostgreSQL Persistence
        ↓
Dashboard + History
        ↓
Transaction Detail
        ↓
Optional Fraud Investigation Agent
        ↓
Groq GPT-OSS
        ↓
Grounded Explanation
```

Verified examples include:

```text
HIGH
99.72% fraud probability
Threshold 83%
```

and:

```text
LOW
58.46% fraud probability
Threshold 83%
```

The deployed application has also been verified for:

```text
Dashboard analytics
Transaction persistence
Transaction history
Individual transaction lookup
Model evaluation
Research evaluation
Fraud investigation
LLM grounding
PostgreSQL persistence
CatBoost lazy loading
Public frontend accessibility
Frontend → Backend CORS
Backend → Model-Service communication
```

---

## 📸 Screenshots

### Dashboard

![Dashboard](docs/screenshots/dashboard.png)

### Transaction Analysis

![Transaction Analysis](docs/screenshots/transaction_analysis.png)

### Fraud Investigation Agent

![Fraud Investigation Agent](docs/screenshots/agent.png)

### Model Evaluation

![Model Evaluation](docs/screenshots/evaluation.png)

---

## 📐 Architecture Diagrams

### System Architecture

![System Architecture](docs/architecture/system_architecture.png)

### Training Pipeline

![Training Pipeline](docs/architecture/training_pipeline.png)

### Agent Workflow

![Agent Workflow](docs/architecture/agent_workflow.png)

---

## 🚀 Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ebbeju-Lankapalli/Financial-Fraud-Detection-System.git

cd "Financial-Fraud-Detection-System"
```

### 2. Create / Activate Python Environment

Example with Conda:

```bash
conda create -n mlproject python=3.12
conda activate mlproject
```

### 3. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Install Production Model-Service Dependencies

```bash
pip install -r model-service/requirements.txt
```

For Qwen/QLoRA research:

```bash
pip install -r model-service/requirements-research.txt
```

### 5. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## 🔐 Environment Configuration

Create the backend environment file:

```bash
cp backend/.env.example backend/.env
```

Example local configuration:

```env
APP_NAME=Financial Fraud Detection System
APP_ENV=development

MODEL_SERVICE_URL=http://127.0.0.1:8001
MODEL_SERVICE_TIMEOUT_SECONDS=120

DATABASE_URL=sqlite:///./app.db

GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TIMEOUT_SECONDS=60

CORS_ORIGINS=http://localhost:5173
```

Frontend:

```bash
cp frontend/.env.example frontend/.env
```

```env
VITE_API_BASE_URL=http://localhost:8000
```

> Never commit real API keys, PostgreSQL credentials, `.env` files, or other secrets.

---

## ▶️ Run Locally

### Terminal 1 — Production CatBoost Model Service

```bash
cd model-service

uvicorn production_app:app \
  --host 127.0.0.1 \
  --port 8001 \
  --reload
```

### Terminal 2 — FastAPI Backend

From the project root:

```bash
PYTHONPATH=backend uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

### Terminal 3 — React Frontend

```bash
cd frontend

npm run dev
```

Open:

```text
http://localhost:5173
```

Backend:

```text
http://127.0.0.1:8000
```

Model service:

```text
http://127.0.0.1:8001
```

---

## 🐳 Docker

### Build Production CatBoost Service

From the repository root:

```bash
docker build \
  -f model-service/Dockerfile \
  -t fraud-catboost-model-service \
  .
```

Run:

```bash
docker run --rm \
  -p 8001:8001 \
  -e PORT=8001 \
  fraud-catboost-model-service
```

### Build Backend

```bash
docker build \
  -f backend/Dockerfile \
  -t fraud-detection-backend \
  .
```

---

## 🗄️ Persistence

The repository supports two database backends.

### Local Development

```env
DATABASE_URL=sqlite:///./app.db
```

### Cloud Deployment

```env
DATABASE_URL=postgresql://...
```

Backend selection is automatic based on the configured database URL.

PostgreSQL is used by the deployed application to preserve fraud-analysis records across service restarts and redeployments.

---

## ☁️ Deployment

The current deployment architecture is:

```text
Frontend
Vercel
        ↓
FastAPI Backend
Render
        ↓
        ├── Render PostgreSQL
        ├── Render CatBoost Service
        └── Groq GPT-OSS
```

Frontend production variable:

```env
VITE_API_BASE_URL=https://YOUR_BACKEND_DOMAIN
```

Backend production variables:

```env
APP_NAME=Financial Fraud Detection System

MODEL_SERVICE_URL=https://YOUR_MODEL_SERVICE_DOMAIN
MODEL_SERVICE_TIMEOUT_SECONDS=120

DATABASE_URL=postgresql://...

GROQ_API_KEY=your_server_side_key
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TIMEOUT_SECONDS=60

CORS_ORIGINS=https://YOUR_FRONTEND_DOMAIN
```

Secrets remain server-side.

Never expose `GROQ_API_KEY` or `DATABASE_URL` through a `VITE_*` environment variable.

---

## 🔐 Security & Reliability Practices

The project includes:

- Server-side API keys
- `.env` exclusion through `.gitignore`
- CORS allow-list configuration
- Pydantic request validation
- Strict response schemas
- Model-output validation
- Persistent audit IDs
- Explicit model decision source
- LLM grounding instructions
- Regression testing for anonymized feature hallucinations
- Production / research model separation
- Docker service isolation
- PostgreSQL cloud persistence
- Automated backend tests
- Ruff linting
- GitHub Actions workflow

Important rules:

```text
Never commit GROQ_API_KEY
Never commit PostgreSQL passwords
Never commit .env files
Never expose backend secrets through VITE_* variables
Never allow the LLM to override the CatBoost classification
```

---

## 🏆 Project Highlights

- End-to-end deployed fraud detection application
- Real production CatBoost classifier
- IEEE-CIS Fraud Detection dataset
- Chronological held-out test evaluation
- Validation-only threshold optimization
- Reduced 20-feature production model
- 61-feature reference model preserved
- HIGH / LOW fraud classification
- Fraud probability scoring
- Persistent PostgreSQL audit history
- Interactive fraud analytics dashboard
- Individual transaction inspection
- GPT-OSS fraud investigation agent
- Groq production LLM inference
- Explicit ML → LLM separation
- LLM hallucination / grounding protection
- Qwen2.5 + QLoRA research experiment
- Dockerized backend
- Dockerized model service
- React + Vite frontend
- FastAPI REST APIs
- Vercel deployment
- Render deployment
- 21 automated tests passing
- Ruff validation
- GitHub Actions workflow

---

## 🎯 Skills Demonstrated

### Machine Learning Engineering

```text
Fraud Detection
Tabular Machine Learning
CatBoost
Feature Selection
Feature Importance
Class Imbalance
Threshold Optimization
Precision / Recall Analysis
F1 Optimization
ROC-AUC
PR-AUC
Confusion Matrix
Model Evaluation
Train / Validation / Test Separation
Production Inference
```

### Generative AI & Agentic AI

```text
LLMs
GPT-OSS
Groq API
Prompt Engineering
Grounded Generation
Agentic Workflows
LLM Safety
Hallucination Reduction
Human-in-the-Loop Investigation
QLoRA
LoRA
PEFT
Supervised Fine-Tuning
Qwen2.5
```

### Full-Stack Engineering

```text
React
Vite
FastAPI
REST APIs
Pydantic
PostgreSQL
SQLite
Recharts
Tailwind CSS
API Integration
Responsive UI
```

### Production / MLOps

```text
Docker
Model Serving
Service Separation
Environment Configuration
Secret Management
Cloud Databases
Render
Vercel
Health Checks
GitHub Actions
Git
GitHub
Testing
Linting
Production Deployment
```

---

## 📚 What I Learned

Building the Financial Fraud Detection System provided practical experience across the complete AI/ML lifecycle:

```text
Dataset Preparation
        ↓
Feature Engineering
        ↓
Model Comparison
        ↓
CatBoost Training
        ↓
Feature Reduction
        ↓
Threshold Optimization
        ↓
Held-Out Evaluation
        ↓
Production Model Serving
        ↓
FastAPI Integration
        ↓
Persistent Audit Storage
        ↓
React Visualization
        ↓
AI Investigation Agent
        ↓
LLM Grounding
        ↓
Docker
        ↓
Cloud Deployment
        ↓
Production Verification
```

The project also demonstrates the importance of separating **predictive ML decisions** from **generative AI explanations** in systems where traceability and reliability matter.

---

## 🚧 Future Improvements

Potential future improvements include:

- SHAP-based local feature explanations
- Cost-sensitive fraud classification
- Improved recall optimization
- Precision-recall threshold profiles
- Probability calibration
- Real-time streaming transaction ingestion
- Kafka / event-driven fraud pipelines
- Device and behavioral features
- Velocity-rule features
- Account-level fraud graphs
- Graph neural networks
- Analyst feedback loops
- Active learning
- Model monitoring
- Data-drift detection
- Prediction-drift detection
- Prometheus / Grafana monitoring
- Role-based analyst authentication
- Investigation case management
- Evidence attachment workflows
- Model registry
- CI/CD deployment pipeline
- Automated model retraining
- Dedicated managed PostgreSQL production tier
- Cloud object storage for artifacts

---

## 🛣️ Development Status

```text
IEEE-CIS Data Pipeline          ✅
CatBoost Training               ✅
Feature Importance              ✅
20-Feature Production Model     ✅
Threshold Optimization          ✅
Held-Out Test Evaluation        ✅
Production Model Service        ✅
FastAPI Backend                 ✅
Transaction Analysis API        ✅
PostgreSQL Persistence          ✅
SQLite Local Support            ✅
Dashboard                       ✅
Transaction History             ✅
Transaction Detail              ✅
Fraud Investigation Agent       ✅
Groq GPT-OSS Integration        ✅
LLM Grounding Protection        ✅
Qwen2.5 Research                ✅
QLoRA Fine-Tuning Research      ✅
Model Evaluation UI             ✅
React Frontend                  ✅
Docker Model Service            ✅
Docker Backend                  ✅
Automated Tests                 ✅
Vercel Frontend Deployment      ✅
Render Backend Deployment       ✅
Render Model Deployment         ✅
Render PostgreSQL               ✅
End-to-End Production Test      ✅
```

---

## ⚠️ Disclaimer

This project is an educational and research-oriented fraud-risk screening system.

A HIGH-risk classification represents a model-generated screening signal and does **not** prove that fraud occurred.

Predictions should not be used as the sole basis for financial, legal, account-blocking, or other consequential decisions.

Human review and additional operational context are required.

The IEEE-CIS dataset contains anonymized features whose exact real-world meanings are not necessarily available. The application deliberately avoids assigning unsupported semantic meanings to those fields.

---

## 👨‍💻 Author

**Ebbeju Lankapalli**

B.Tech Computer Science and Engineering
**Artificial Intelligence & Machine Learning**

Aspiring **AI / ML Engineer**

GitHub:

https://github.com/Ebbeju-Lankapalli

Areas of interest:

```text
Machine Learning
Deep Learning
Generative AI
Agentic AI
LLMs
Fraud Detection
MLOps
AI Systems
Model Deployment
Production AI Engineering
```

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐.

---

# 🛡️ Financial Fraud Detection System

> Production machine learning for fraud screening, persistent auditability, and grounded AI-assisted investigation.
