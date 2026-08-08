# AI-Driven Predictive Maintenance & Diagnostic System for Kirloskar 1.5 kW Motors

An end-to-end industrial predictive maintenance and automated diagnostic system built specifically for the **Kirloskar 1.5 kW 3-Phase Induction Motor**. This repository combines physics-grounded telemetry generation, hierarchical machine learning models, automated PDF reporting, and a Generative AI maintenance advisor powered by Google Gemini.

---

## 📌 Project Overview

This system monitors three primary telemetry parameters—**Temperature**, **Vibration**, and **Current**—to detect anomaly severity and pinpoint specific root-cause motor faults before catastrophic failures occur. Grounded in ISO 10816 vibration standards and original Kirloskar motor manufacturer specifications, it provides automated risk categorization, diagnostic visual analytics, and natural-language corrective maintenance guidance.

---

## 🚀 Key Features

* **Physics-Grounded Data Generation**: Synthesizes multi-channel motor telemetry using real physical baseline constants (e.g., rated current, thermal limits, vibration severity zones).


* **Hierarchical Machine Learning Architecture**:
* **Level 1 Classifier**: Evaluates motor operational health state (*Healthy*, *Warning*, *Fault*).


* **Level 2 Classifier**: Identifies specific underlying fault mechanisms when Level 1 flags an operational anomaly (*Bearing Wear*, *Imbalance*, *Misalignment*, *Rotor Bar Fault*, *Overload*, *Overheating*).




* **Automated Feature Engineering**: Generates 15+ domain-specific derived features, including interaction terms, threshold flags, and normalized electrical/thermal ratios.


* **Automated Report Generation**: Produces multi-page PDF diagnostic reports containing health state distributions, feature importance breakdowns, confusion matrices, and maintenance roadmaps.


* **GenAI Maintenance Advisor**: Leverages the `google-genai` SDK (`gemini-2.0-flash`) to process live telemetry JSON data and optional thermal/physical motor images to generate immediate corrective actions and urgency ratings.



---

## 📁 Repository Structure

| File Name | Category | Description |
| --- | --- | --- |
| **`dataset_generator_hierarchical.py`** | Data Generation | Synthesizes physical sensor telemetry based on Kirloskar specs and ISO 10816 standards.

 |
| **`Datasheet.pdf`** | Technical Reference | Ground-truth engineering datasheet for the Kirloskar 1.5 kW 3-Phase motor.

 |
| **`hierarchical_motor_fault_dataset.csv`** | Dataset | Synthesized historical sensor telemetry dataset with two-level labels.

 |
| **`ML_model_hierarchical.py`** | Machine Learning | Training pipeline for feature extraction, model fitting, validation, and PDF report generation.

 |
| **`motor_fault_model_hierarchical.pkl`** | Model Artifact | Serialized model package storing fitted Random Forest classifiers, encoders, and threshold metadata.

 |
| **`Model_Training_Report_final.pdf`** | Diagnostic Report | Multi-page executive and engineering report produced after model evaluation.

 |
| **`ml_model.json`** | Data Payload | Standardized JSON payload containing inference results for downstream AI advisor consumption.

 |
| **`motor_advisor.py`** | GenAI Advisor | Generative AI advisor script using Gemini API for root-cause diagnosis and maintenance advice.

 |

---

## ⚙️ Hardware & Operating Specifications

The telemetry generation and ML model feature limits are directly derived from the official Kirloskar datasheet:

* **Motor Model**: Kirloskar 1.5 kW (2.0 HP) 3-Phase Induction Motor


* **Rated Voltage**: 415 V (± 10%)


* **Full Load Current (FLA)**: 3.13 A


* **Synchronous / Full Load Speed**: 3000 RPM / 2840 RPM


* **Enclosure / Bearing Types**: IP55 / 6205ZZ (Drive End), 6204ZZ (Non-Drive End)


* **Maximum Thermal Operating Limit**: 120°C


* **Vibration Standard**: ISO 10816-3 Group 2 Industrial Machinery



---

## 🛠️ Installation & Setup

### 1. Prerequisites

Ensure you have Python 3.9+ installed along with the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn reportlab google-genai pillow

```

### 2. Set Up API Key (for GenAI Advisor)

Export your Gemini API key as an environment variable:

```bash
# On Linux/macOS
export GEMINI_API_KEY="your_actual_api_key_here"

# On Windows (PowerShell)
$env:GEMINI_API_KEY="your_actual_api_key_here"

```

---

## 🔄 Execution Workflow

Follow these steps to run the complete predictive maintenance pipeline:

### Step 1: Generate Telemetry Dataset

Creates the physical sensor dataset (`hierarchical_motor_fault_dataset.csv`) using rule-based motor parameters.

```bash
python dataset_generator_hierarchical.py

```

### Step 2: Train Machine Learning Models & Generate PDF Report

Processes telemetry, engineers derived features, trains the Level 1 and Level 2 Random Forest classifiers, saves the binary artifact (`motor_fault_model_hierarchical.pkl`), simulates real-time ESP32 edge predictions, updates `ml_model.json`, and exports the diagnostic PDF report.

```bash
python ML_model_hierarchical.py

```

### Step 3: Run GenAI Maintenance Advisor

Infers root-cause diagnosis from `ml_model.json` (and optional physical inspection images like `image.jpg`), outputting actionable maintenance recommendations.

```bash
python motor_advisor.py

```

---

## 🧠 Model Architecture & Methodology

```
                   ┌──────────────────────────────────┐
                   │     Raw Telemetry Ingestion      │
                   │ (Temp: °C, Vib: mm/s, Curr: A)   │
                   └────────────────┘─────────────────┘
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │       Feature Engineering        │
                   │ (15+ derived ratios & flags)     │
                   └────────────────┘─────────────────┘
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │    Level 1 RF Classifier         │
                   │ (Healthy / Warning / Fault)      │
                   └────────────────┬─────────────────┘
                                    │
                         [If State == "Fault"]
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │    Level 2 RF Classifier         │
                   │    (Specific Fault Type)         │
                   └────────────────┬─────────────────┘
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │   JSON Payload & GenAI Advisor   │
                   │ (Gemini 2.0 Maintenance Insights)│
                   └──────────────────────────────────┘

```

* **Level 1 Classification**: Filters out normal operational noise and identifies early-stage degradation (`Warning`) or critical operating state (`Fault`).


* **Level 2 Classification**: Triggers upon a `Fault` classification to differentiate between electrical overload, thermal breakdown, mechanical misalignment, rotor bar cracks, or bearing failure.


* **Generative AI Diagnostics**: Ingests JSON predictions and output metrics to generate prioritized corrective action plans, required tooling checklists, and urgency ratings.
