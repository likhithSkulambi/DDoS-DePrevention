# DDoS Detection & Prevention via Federated Learning

A robust, privacy-preserving **Federated Learning-based DDoS Detection and Real-Time Prevention System** evaluated on the **CICIDS2017** benchmark dataset. The system features a multi-client Federated Averaging (FedAvg) engine, an automated live packet capture and flow extraction pipeline, and an interactive FastAPI web dashboard.

---

## 📌 Features

- **Federated Training Engine**: Simulates distributed, collaborative model training across multiple clients without centralizing raw network data using FedAvg.
- **22 Canonical Flow Features**: Real-time extraction of bidirectional statistical features matching the CICIDS2017 benchmark (durations, packet length statistics, inter-arrival times, segment sizes).
- **Interactive Web Dashboard**: Modern FastAPI-powered UI with live traffic event streaming, real-time metrics, packet capture management, and batch CSV predictions.
- **Live Traffic Sniffer & Classifier**: Captures layer-3/4 packets, reconstructs bidirectional conversation flows, normalizes via a pre-fitted scaler, and classifies traffic with high confidence.
- **DDoS Traffic Simulator**: Generates controlled, non-destructive burst traffic to validate live interception and model response.

---

## 📁 Project Structure

```
├── data/
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv  # CICIDS2017 dataset
│   └── README.md
├── docs/
│   ├── global_model.h5                                   # Trained global model
│   ├── live_capture_pipeline.md                          # Technical capture pipeline docs
│   ├── avg_convergence_plot.png                          # Training performance plots
│   └── roc_curve.png
├── models/
│   └── README.md
├── notebooks/                                            # Interactive analysis notebooks
│   ├── explore_fraud_data.ipynb
│   ├── federated_training_demo.ipynb
│   └── model_inference_demo.ipynb
├── src/                                                  # Core Python modules
│   ├── capture_live.py                                   # Live packet sniffer
│   ├── ddos_detector.py                                  # Model architecture & evaluation
│   ├── ddos_main.py                                      # Federated training CLI
│   ├── feature_extractor.py                              # Flow reconstruction & feature extractor
│   ├── generate_scaler.py                                # Scaler generator utility
│   ├── live_predict.py                                   # Offline flow inference
│   ├── simulate_ddos.py                                  # Live DDoS traffic generator
│   └── utils.py                                          # Shared utilities & loggers
├── webapp/                                               # Web Dashboard
│   ├── app.py                                            # FastAPI backend
│   └── static/index.html                                 # Single-page web UI
├── .gitignore
├── requirements.txt                                      # Core ML & sniffer dependencies
├── requirements-web.txt                                  # Web dashboard dependencies
└── README.md
```

---

## ⚙️ Prerequisites

1. **Python 3.10 or 3.11** (Python 3.11 recommended for TensorFlow compatibility on Windows).
2. **Npcap (Windows users only)**: Required for live packet capture (`capture_live.py` and Web UI sniffer).
   - Download and install from [npcap.com](https://npcap.com/).
   - During installation, check **"Install Npcap in WinPcap API-compatible Mode"**.

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/likhithSkulambi/DDoS-DePrevention.git
   cd DDoS-DePrevention
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-web.txt
   ```

4. **Generate the Pre-Fitted Feature Scaler (Run Once):**
   Fits `StandardScaler` on the 22 canonical features from the dataset and generates `scaler.pkl`:
   ```bash
   python src/generate_scaler.py
   ```

---

## 🖥️ How to Run

### 1. Web Application Dashboard (Recommended)

Launch the interactive dashboard to monitor traffic, test CSV files, and control live packet captures:

```bash
python webapp/app.py
```
*(Alternatively: `uvicorn webapp.app:app --host 127.0.0.1 --port 8000`)*

- Open your browser at: **`http://127.0.0.1:8000`**
- **Batch CSV Analysis**: Upload flow CSV files for instant predictions (`Normal` vs `DDoS`).
- **Live Capture**: Start/Stop sniffing on physical network interfaces and view live classified flow events.

---

### 2. Federated Learning Training Simulation

Train the global model across distributed client nodes using Federated Averaging:

```bash
# Default training (3 clients, 10 FedAvg rounds)
python src/ddos_main.py

# Custom clients, rounds, and test split
python src/ddos_main.py --clients 5 --rounds 15 --test-size 0.2
```

---

### 3. Live Traffic Capture & Attack Simulation (CLI)

Demonstrate live network packet capture and real-time inference:

- **Scenario A: Normal Ambient Traffic**
  ```bash
  python src/capture_live.py
  # Enter capture duration (e.g., 0.5 for 30 seconds)
  ```

- **Scenario B: Live Attack Detection (Two Terminals)**
  - **Terminal 1** (Sniffer):
    ```bash
    python src/capture_live.py
    ```
  - **Terminal 2** (Attack Generator):
    ```bash
    python src/simulate_ddos.py --continuous --interval 2
    # Press Ctrl+C after 3-5 bursts
    ```
    *Terminal 1 will capture the packets, extract 22 statistical features, and identify the attack flows in real time.*

- **Scenario C: Offline Prediction on Existing Flow CSV**
  ```bash
  python src/live_predict.py
  ```

---

### 4. Interactive Jupyter Notebooks

For data exploration and model evaluations:

```bash
pip install notebook
jupyter notebook
```
Navigate to `notebooks/` and open `explore_fraud_data.ipynb` or `federated_training_demo.ipynb`.

---

## 🛡️ License

This project is licensed under the Apache-2.0 License.
