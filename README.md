# Spectre Gold AI Platform

An AI-powered gold trading platform built on the **Google AI Ecosystem Stack** — Gemini as the intelligence layer, Firebase Genkit for agentic orchestration, Vertex AI Agent Builder (ADK) for tool-calling agents, and Google Cloud Run for serverless deployment. It provides live gold price data, LSTM-based price prediction, sentiment-adjusted forecasts, simulated trading, a multi-tier AI chat assistant, and a structured market intelligence report grounded in real-time Google Search results.

---

## Google AI Stack Alignment

| Mandate Component | Implementation |
|---|---|
| **Gemini (Brain)** | `gemini-2.5-flash` for all LLM calls — market reports, chat, Genkit flows |
| **Firebase Genkit (Orchestrator)** | `marketIntelligenceFlow` + `goldChatFlow` TypeScript flows with structured schemas |
| **Vertex AI Agent Builder (ADK)** | `agent_service/` — ADK-compatible agent with 5 registered tools, `_SpectreAgentShim` fallback |
| **Cloud Run (Deployment)** | 3 independent services: FastAPI backend, Genkit flow server, ADK agent |
| **Grounded Context / RAG** | Dedicated `context_provider.py` adapter — Google Search grounding by default, Vertex AI Search plug-in when configured |

---

## Features

- **Live Gold Price** — real-time GC=F data via `yfinance`
- **AI Price Prediction** — LSTM model predicts next-day close price
- **Sentiment Adjustment** — news sentiment shifts the technical forecast up/down
- **Interactive Chart** — Plotly candlestick/line chart with MA7 and range selectors
- **Simulated Trading** — buy/sell gold with a virtual wallet; positions persist in `portfolio.json`
- **Smart Trade Automation** — configurable take-profit, stop-loss, and add-on triggers
- **AI Market Intelligence Report** — structured report (sentiment, key drivers, risk factors, price outlook) powered by Genkit `marketIntelligenceFlow` with Google Search-grounded source links
- **Spectre Gold AI Chatbot** — multi-tier chat assistant with visible source labels:
  - Tier 0: Genkit `goldChatFlow` (explanation / advice questions)
  - Tier 1: ADK Agent Service (if configured)
  - Tier 2: Local WorkflowRunner (4-agent pipeline)
  - Tier 3: Direct Gemini single-shot fallback
- **Risk Guard** — evaluates signal consistency, news credibility, and market volatility
- **Smart Alerts** — detects large price movements, sentiment reversals, high-risk conditions, and conflicting signals
- **Suggested Action** — explainable Buy / Hold / Sell / Wait decision card

---

## Architecture

```
Streamlit (app.py)
    │
    ├── FastAPI Backend (Backend/main.py, port 8010)
    │       ├── POST /genkit/market-intelligence  ← Genkit flow + merged grounded sources
    │       ├── POST /genkit/chat                 ← Genkit goldChatFlow
    │       ├── POST /agentic-chat                ← ADK Agent → WorkflowRunner fallback
    │       ├── POST /risk-analysis
    │       ├── POST /suggested-action
    │       ├── POST /smart-alerts
    │       └── GET  /market-sentiment
    │
    ├── Genkit Flow Server (genkit_flows/, port 4001)  ← REST API; Dev UI on 4000
    │       ├── POST /marketIntelligenceFlow      ← structured market report (Gemini)
    │       └── POST /goldChatFlow                ← gold chat Q&A (Gemini)
    │
    ├── ADK Agent Service (agent_service/, port 8020) [optional Cloud Run]
    │       └── POST /agent/run                   ← 5-tool ADK agent
    │
    └── Context / Retrieval Layer (News/context_provider.py)
            ├── GoogleSearchGroundingProvider      ← default (Gemini built-in tool)
            └── VertexAISearchProvider             ← optional (set VERTEX_SEARCH_DATASTORE_ID)
```

### Component Responsibilities

| Layer | Files | Role |
|---|---|---|
| **Gemini Brain** | via `google-genai` SDK | LLM inference for all AI responses |
| **Genkit Orchestrator** | `genkit_flows/src/flows/` | Typed agentic flows with structured output schemas |
| **ADK Agent** | `agent_service/root_agent.py` | Tool-calling agent (market snapshot, risk, alerts, decisions) |
| **Deterministic Finance** | `News/agents.py`, `risk_guard.py`, `decision_engine.py`, `smart_alerts.py` | Rule-based Python logic — no hallucination risk |
| **Context / RAG Layer** | `News/context_provider.py` | Abstraction over retrieval backends (Google Search / Vertex AI Search) |
| **FastAPI** | `Backend/main.py` | Orchestration API bridging all layers |
| **Streamlit** | `app.py` | 3-page UI: Dashboard / Trading / AI Intelligence |

---

## Project Structure

```
app.py                            # Streamlit frontend (3 pages)
Backend/
    main.py                       # FastAPI backend
Models/
    gold_price_model.keras        # Trained LSTM model
    gold_scaler.bin               # Feature scaler
News/
    news.py                       # Gemini + Google Search grounding pipeline
    context_provider.py           # Retrieval abstraction (Google Search / Vertex AI Search)
    chatbot.py                    # Multi-tier chat dispatcher (agentic_dispatch)
    agents.py                     # Local WorkflowRunner (4-agent pipeline)
    risk_guard.py                 # Risk evaluation module
    decision_engine.py            # Suggested action engine
    smart_alerts.py               # Alert detection module
genkit_flows/                     # Firebase Genkit TypeScript flows
    src/
        genkit.ts                 # Genkit ai instance + dotenv init
        index.ts                  # Flow registration + HTTP server (port 4001)
        flows/
            marketIntelligence.ts # marketIntelligenceFlow + goldChatFlow
    package.json
    Dockerfile                    # Cloud Run image (port 4001)
agent_service/                    # Vertex AI Agent Builder (ADK) service
    server.py                     # FastAPI wrapper (port 8020 local / 8080 Cloud Run)
    root_agent.py                 # ADK Agent + _SpectreAgentShim fallback
    tools.py                      # 5 FunctionTool wrappers
    llm_client.py                 # Gemini LLM client
    Dockerfile                    # Cloud Run image
deploy/
    deploy_cloud_run.sh           # Deploy all 3 services to Cloud Run
    .env.example                  # Environment variable reference
requirements.txt                  # Python dependencies
portfolio.json                    # Simulated portfolio state (auto-created)
.env                              # API keys (not committed)
```

---

## Setup

### 1. Python environment

```powershell
conda activate resume-ai
pip install -r requirements.txt
```

### 2. Node environment (Genkit flows)

```powershell
cd genkit_flows
npm install
```

### 3. Configure `.env` in the project root

```env
GOOGLE_API_KEY=your_gemini_api_key_here

# Service URLs (defaults work for local dev — change for Cloud Run)
API_BASE_URL=http://localhost:8010
GENKIT_FLOW_URL=http://localhost:4001
ADK_AGENT_SERVICE_URL=          # leave blank to use local WorkflowRunner fallback

# Optional: Vertex AI Search (leave blank to use Google Search grounding)
VERTEX_SEARCH_DATASTORE_ID=     # projects/PROJECT/locations/global/.../dataStores/STORE_ID
VERTEX_SEARCH_LOCATION=global
```

---

## Run (Local Development)

Open **three terminals** in order:

**Terminal 1 — Genkit flow server**
```powershell
cd genkit_flows
npm run genkit:ui
# Dev UI: http://localhost:4000
# REST API: http://localhost:4001  ← FastAPI talks to this
```

**Terminal 2 — FastAPI backend**
```powershell
cd Backend
uvicorn main:app --reload --host 127.0.0.1 --port 8010
```

**Terminal 3 — Streamlit frontend**
```powershell
streamlit run app.py
```

App available at **http://localhost:8501**

---

## Cloud Run Deployment

```bash
# From project root — requires gcloud CLI authenticated
export GOOGLE_API_KEY=your_key
export GCP_PROJECT_ID=your-project-id
./deploy/deploy_cloud_run.sh
```

The script deploys three Cloud Run services:
1. `spectre-gold-agent` — ADK Agent Service
2. `spectre-gold-genkit` — Genkit Flow Server
3. `spectre-gold-backend` — FastAPI Backend

It prints all service URLs and the `.env` values to set in Streamlit.

---

## Context / Retrieval Layer

The system uses a **dedicated retrieval abstraction** (`News/context_provider.py`) that can be swapped without changing any other code:

| Provider | When active | What it does |
|---|---|---|
| `GoogleSearchGroundingProvider` | Default (no extra config) | Gemini's built-in Google Search grounding tool |
| `VertexAISearchProvider` | When `VERTEX_SEARCH_DATASTORE_ID` is set | Queries a Vertex AI Search datastore for RAG |

The active provider label is surfaced in the UI as **Context Source: Google Search Grounding** or **Context Source: Vertex AI Search**.

---

## Optional: ADK Agent Service (local)

```powershell
cd agent_service
uvicorn server:app --port 8020
```

Then set `ADK_AGENT_SERVICE_URL=http://localhost:8020` in `.env`.

---

## Environment Notes

```powershell
# Suppress TensorFlow oneDNN warnings
$env:TF_ENABLE_ONEDNN_OPTS = "0"

# Override Genkit port
$env:FLOW_SERVER_PORT = "4002"
$env:GENKIT_FLOW_URL = "http://localhost:4002"
```

---

## Features

- **Live Gold Price** — real-time GC=F data via `yfinance`
- **AI Price Prediction** — LSTM model predicts next-day close price
- **Sentiment Adjustment** — news sentiment shifts the technical forecast up/down
- **Interactive Chart** — Plotly candlestick/line chart with MA7 and range selectors
- **Simulated Trading** — buy/sell gold with a virtual wallet; positions persist in `portfolio.json`
- **Smart Trade Automation** — configurable take-profit, stop-loss, and add-on triggers
- **AI Market Intelligence Report** — structured report (sentiment, key drivers, risk factors, price outlook) powered by Genkit `marketIntelligenceFlow` with Google Search-grounded source links
- **Spectre Gold AI Chatbot** — multi-tier chat assistant with visible source labels:
  - Tier 0: Genkit `goldChatFlow` (explanation / advice questions)
  - Tier 1: ADK Agent Service (if configured)
  - Tier 2: Local WorkflowRunner (4-agent pipeline)
  - Tier 3: Direct Gemini single-shot fallback
- **Risk Guard** — evaluates signal consistency, news credibility, and market volatility
- **Smart Alerts** — detects large price movements, sentiment reversals, high-risk conditions, and conflicting signals
- **Suggested Action** — explainable Buy / Hold / Sell / Wait decision card

---

## Architecture

```
Streamlit (app.py)
    │
    ├── FastAPI (Backend/main.py, port 8010)
    │       ├── POST /genkit/market-intelligence  ← Genkit marketIntelligenceFlow + Python grounded sources
    │       ├── POST /genkit/chat                 ← Genkit goldChatFlow
    │       ├── POST /agentic-chat                ← ADK Agent proxy → WorkflowRunner fallback
    │       ├── POST /risk-analysis
    │       ├── POST /suggested-action
    │       ├── POST /smart-alerts
    │       └── GET  /market-sentiment
    │
    ├── Genkit Flow Server (genkit_flows/, port 4001)
    │       ├── POST /marketIntelligenceFlow      ← structured market report
    │       └── POST /goldChatFlow                ← gold chat Q&A
    │
    └── ADK Agent Service (agent_service/, port 8020) [optional]
            └── POST /agent/run
```

---

## Project Structure

```
app.py                            # Streamlit frontend (3 pages: Dashboard / Trading / AI Intelligence)
Backend/
    main.py                       # FastAPI backend
Models/
    gold_price_model.keras        # Trained LSTM model
    gold_scaler.bin               # Feature scaler
News/
    news.py                       # Google Search-grounded market report (Python)
    chatbot.py                    # Multi-tier chat dispatcher (agentic_dispatch)
    agents.py                     # Local WorkflowRunner (4-agent pipeline)
    risk_guard.py                 # Risk evaluation module
    decision_engine.py            # Suggested action engine
    smart_alerts.py               # Alert detection module
genkit_flows/                     # Google Genkit TypeScript flows
    src/
        genkit.ts                 # Genkit ai instance + dotenv init
        index.ts                  # Flow registration + HTTP server (port 4001)
        flows/
            marketIntelligence.ts # marketIntelligenceFlow + goldChatFlow
    package.json
agent_service/                    # ADK Agent Service (optional, port 8020)
    server.py
    root_agent.py
    tools.py
    llm_client.py
    Dockerfile
deploy/
    deploy_cloud_run.sh           # Cloud Run deploy script
    .env.example                  # Environment variable reference
Data_preparation/
    1_data_exploration.ipynb
    GC_F_historical.csv
portfolio.json                    # Simulated portfolio state (auto-created)
requirements.txt                  # Python dependencies (pip install -r requirements.txt)
.env                              # API keys (not committed)
```

---

## Setup

### 1. Python environment

```powershell
conda activate resume-ai
pip install -r requirements.txt
```

### 2. Node environment (Genkit flows)

```powershell
cd genkit_flows
npm install
```

### 3. Configure `.env` in the project root

```env
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional — only needed when services run on non-default ports
API_BASE_URL=http://localhost:8010
GENKIT_FLOW_URL=http://localhost:4001
ADK_AGENT_SERVICE_URL=          # leave blank unless running agent_service/
```

---

## Run

Open **three terminals** in order:

**Terminal 1 — Genkit flow server**
```powershell
cd genkit_flows
npm run genkit:ui
# Starts: Dev UI on http://localhost:4000
#         REST API on http://localhost:4001
```

**Terminal 2 — FastAPI backend**
```powershell
cd Backend
uvicorn main:app --reload --host 127.0.0.1 --port 8010
```

**Terminal 3 — Streamlit frontend**
```powershell
streamlit run app.py
```

App available at **http://localhost:8501**

---

## Optional: ADK Agent Service

```powershell
cd agent_service
uvicorn server:app --port 8020
```

Then set `ADK_AGENT_SERVICE_URL=http://localhost:8020` in `.env`.

---

## Environment Notes

Suppress TensorFlow oneDNN warnings:
```powershell
$env:TF_ENABLE_ONEDNN_OPTS = "0"
```

Genkit flow port can be overridden:
```powershell
$env:FLOW_SERVER_PORT = "4002"   # genkit_flows side
$env:GENKIT_FLOW_URL = "http://localhost:4002"  # FastAPI side
```

