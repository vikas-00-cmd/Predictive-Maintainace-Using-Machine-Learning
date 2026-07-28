"""
Hierarchical Motor Fault ML Model (Upgraded)
==============================================
Implements the research report's full fault taxonomy with:

  1. HIERARCHICAL PREDICTION:
     - Level 1: Motor Condition (Healthy / Warning / Fault)
     - Level 2: Fault Type (Bearing_Wear / Imbalance / Misalignment /
                             Rotor_Bar_Fault / Overload / Overheating)

  2. CONFIDENCE SCORES via predict_proba()

  3. DATASHEET-GROUNDED THRESHOLDS built into the feature pipeline:
     - Kirloskar 1.5 kW: Rated Current 3.10 A, Ambient 50C + Rise 70C
     - ISO 10816-3: Vibration limits for Group 2 small motors
     - Normalized features: current/rated, temp/limit, vib/iso_threshold

  4. ENGINEERED FEATURES from research report Section 4:
     - Raw sensor values + datasheet-normalized ratios
     - Cross-sensor interaction features (temp*vib, curr*vib, etc.)
     - Threshold exceedance flags

  5. COMPREHENSIVE PDF REPORT with training results, confusion matrices,
     and per-class performance for both models.

Input:  3 sensor values from ESP32 (Temperature, Vibration, Current)
Output: {"motor_condition": "Fault", "fault_type": "Bearing_Wear", "confidence": 0.96}

Usage:
    python ML_model_hierarchical.py                     # GUI file dialog
    python ML_model_hierarchical.py dataset.csv         # CLI argument
    python ML_model_hierarchical.py --generate           # Generate dataset first, then train
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KIRLOSKAR 1.5 kW DATASHEET CONSTANTS (from research report)
# ============================================================
DATASHEET = {
    'rated_current_a': 3.10,
    'rated_voltage_v': 415.0,
    'rated_speed_rpm': 2840.0,
    'ambient_temp_c': 50.0,
    'temp_rise_c': 70.0,
    'max_thermal_limit_c': 120.0,
    # ISO 10816-3 Group 2 vibration thresholds
    'vib_healthy_max': 2.8,     # mm/s RMS
    'vib_warning_max': 4.5,     # mm/s RMS
    # Current thresholds (derived)
    'curr_healthy_max': 3.10 * 1.10,  # 3.41 A (110%)
    'curr_warning_max': 3.10 * 1.30,  # 4.03 A (130%)
    # Temperature thresholds (motor surface)
    'temp_healthy_max': 65.0,
    'temp_warning_max': 85.0,
}

# Fault taxonomy from research report
FAULT_TYPES = [
    'None',              # Healthy or Warning (no specific fault)
    'Bearing_Wear',      # Mechanical: high vibration + moderate temp
    'Imbalance',         # Mechanical: very high vibration, normal current
    'Misalignment',      # Mechanical: moderate vibration + elevated current
    'Rotor_Bar_Fault',   # Electrical: high fluctuating current, mild vibration
    'Overload',          # Electrical: very high current + high temp, low vibration
    'Overheating',       # Thermal: very high temp, normal current & vibration
]

CONDITION_CLASSES = ['Healthy', 'Warning', 'Fault']


def clean_text(text):
    """Sanitize strings for FPDF standard Latin-1 encoding."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '°': ' deg ',
        '\u2013': '-',
        '\u2014': '-',
        '\u2026': '...',
        '\u2022': '*',
        '\u201c': '"',
        '\u201d': '"',
        '\u2018': "'",
        '\u2019': "'"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')


class MotorHealthPDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 14)
        self.set_text_color(24, 43, 73)
        self.cell(0, 8, clean_text("HIERARCHICAL MOTOR FAULT DIAGNOSTIC REPORT"), ln=True, align='C')
        self.set_font("Arial", 'I', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 4, clean_text("AI-Driven Multi-Fault Predictive Maintenance (Kirloskar 1.5 kW Reference)"), ln=True, align='C')
        self.ln(3)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, clean_text(f"Page {self.page_no()} | Hierarchical Motor Fault Prediction System"), align='C')


# ============================================================
# DATASHEET-GROUNDED FEATURE ENGINEERING
# ============================================================
def engineer_features(df, datasheet=DATASHEET):
    """
    Transforms raw sensor readings into an enriched feature set grounded
    in the Kirloskar datasheet and ISO standards.

    Input columns: Motor_Temp_C, Vibration_mm_s, Current_A
    Output: DataFrame with 15+ engineered features

    Feature categories (from research report Section 4):
      1. Raw sensor values (3)
      2. Datasheet-normalized ratios (3)
      3. Threshold exceedance flags (6)
      4. Cross-sensor interaction features (3)
    """
    features = pd.DataFrame()

    # 1. Raw sensor values
    features['temp'] = df['Motor_Temp_C']
    features['vibration'] = df['Vibration_mm_s']
    features['current'] = df['Current_A']

    # 2. Datasheet-normalized ratios
    #    These let the model reason about "how far from normal" each reading is
    features['temp_ratio'] = df['Motor_Temp_C'] / datasheet['max_thermal_limit_c']
    features['vib_ratio'] = df['Vibration_mm_s'] / datasheet['vib_healthy_max']
    features['curr_ratio'] = df['Current_A'] / datasheet['rated_current_a']

    # 3. Threshold exceedance flags (binary indicators)
    #    Temperature zone flags
    features['temp_warning_flag'] = (df['Motor_Temp_C'] > datasheet['temp_healthy_max']).astype(int)
    features['temp_fault_flag'] = (df['Motor_Temp_C'] > datasheet['temp_warning_max']).astype(int)
    #    Vibration zone flags
    features['vib_warning_flag'] = (df['Vibration_mm_s'] > datasheet['vib_healthy_max']).astype(int)
    features['vib_fault_flag'] = (df['Vibration_mm_s'] > datasheet['vib_warning_max']).astype(int)
    #    Current zone flags
    features['curr_warning_flag'] = (df['Current_A'] > datasheet['curr_healthy_max']).astype(int)
    features['curr_fault_flag'] = (df['Current_A'] > datasheet['curr_warning_max']).astype(int)

    # 4. Cross-sensor interaction features
    #    These capture the COMPLEX multi-sensor relationships that distinguish
    #    similar faults (e.g., Overload vs Overheating vs Bearing Wear)
    features['temp_x_vib'] = features['temp_ratio'] * features['vib_ratio']
    features['curr_x_vib'] = features['curr_ratio'] * features['vib_ratio']
    features['temp_x_curr'] = features['temp_ratio'] * features['curr_ratio']

    return features


# ============================================================
# HIERARCHICAL PREDICTION FUNCTION
# ============================================================
def predict_hierarchical(condition_model, fault_model, condition_encoder, fault_encoder,
                         feature_df, datasheet=DATASHEET):
    """
    Performs hierarchical prediction:
      Step 1: Predict Motor_Condition (Healthy / Warning / Fault)
      Step 2: If Fault → predict Fault_Type with confidence
      Step 3: Return structured output with probabilities

    Returns list of dicts:
    [
        {
            "motor_condition": "Fault",
            "condition_confidence": 0.97,
            "fault_type": "Bearing_Wear",
            "fault_confidence": 0.92,
            "all_condition_probs": {"Healthy": 0.01, "Warning": 0.02, "Fault": 0.97},
            "all_fault_probs": {"Bearing_Wear": 0.92, "Imbalance": 0.04, ...}
        },
        ...
    ]
    """
    results = []

    # Step 1: Predict condition
    condition_pred = condition_model.predict(feature_df)
    condition_proba = condition_model.predict_proba(feature_df)
    condition_labels = condition_encoder.classes_

    # Step 2: Predict fault type for ALL samples (we'll filter later)
    fault_pred = fault_model.predict(feature_df)
    fault_proba = fault_model.predict_proba(feature_df)
    fault_labels = fault_encoder.classes_

    for i in range(len(feature_df)):
        cond = condition_labels[condition_pred[i]]
        cond_conf = float(np.max(condition_proba[i]))

        # Build condition probability dict
        cond_probs = {label: round(float(condition_proba[i][j]), 4)
                      for j, label in enumerate(condition_labels)}

        if cond == 'Fault':
            fault = fault_labels[fault_pred[i]]
            fault_conf = float(np.max(fault_proba[i]))
            fault_probs = {label: round(float(fault_proba[i][j]), 4)
                           for j, label in enumerate(fault_labels)}
        else:
            fault = None
            fault_conf = None
            fault_probs = None

        results.append({
            'motor_condition': cond,
            'condition_confidence': round(cond_conf, 4),
            'fault_type': fault,
            'fault_confidence': round(fault_conf, 4) if fault_conf else None,
            'all_condition_probs': cond_probs,
            'all_fault_probs': fault_probs,
        })

    return results


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == '__main__':

    # =====================
    # 1. DATASET SELECTION
    # =====================
    DATASET_PATH = None
    GENERATE_MODE = '--generate' in sys.argv

    if GENERATE_MODE:
        print("=" * 55)
        print("MODE: Generate dataset + Train model")
        print("=" * 55)
        try:
            from dataset_generator_hierarchical import generate_hierarchical_dataset
            DATASET_PATH = 'hierarchical_motor_fault_dataset.csv'
            generate_hierarchical_dataset(
                samples_per_class=2500,
                output_csv=DATASET_PATH
            )
            print()
        except ImportError as e:
            print(f"ERROR: Could not import dataset generator: {e}")
            sys.exit(1)
    elif len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        DATASET_PATH = sys.argv[1]
        print(f"Using dataset from CLI argument: {DATASET_PATH}")
    else:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            print("Opening file dialog to select training dataset CSV...")
            DATASET_PATH = filedialog.askopenfilename(
                title="Select Hierarchical Dataset CSV for Training",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            root.destroy()
        except Exception as e:
            print(f"GUI dialog unavailable: {e}")

    if not DATASET_PATH or not os.path.exists(DATASET_PATH):
        default_csv = 'hierarchical_motor_fault_dataset.csv'
        if os.path.exists(default_csv):
            print(f"Using default dataset: {default_csv}")
            DATASET_PATH = default_csv
        else:
            print("ERROR: No dataset found. Run with --generate flag first.")
            print("Usage: python ML_model_hierarchical.py --generate")
            sys.exit(1)

    # =====================
    # 2. LOAD & VALIDATE
    # =====================
    print(f"\nLoading dataset: {os.path.basename(DATASET_PATH)}...")
    df = pd.read_csv(DATASET_PATH)

    required_cols = ['Motor_Temp_C', 'Vibration_mm_s', 'Current_A', 'Motor_Condition', 'Fault_Type']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        print("Dataset must have: Motor_Temp_C, Vibration_mm_s, Current_A, Motor_Condition, Fault_Type")
        sys.exit(1)

    total_records = len(df)
    print(f"Dataset loaded: {total_records:,} records")
    print(f"\nMotor Condition distribution:")
    print(df['Motor_Condition'].value_counts().to_string())
    print(f"\nFault Type distribution:")
    print(df['Fault_Type'].value_counts().to_string())

    # =====================
    # 3. FEATURE ENGINEERING
    # =====================
    print("\n" + "=" * 55)
    print("FEATURE ENGINEERING (Datasheet-Grounded)")
    print("=" * 55)

    X = engineer_features(df)
    feature_names = list(X.columns)
    print(f"Engineered {len(feature_names)} features from 3 raw sensor inputs:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name}")

    # =====================
    # 4. TRAIN LEVEL 1: Motor Condition Classifier
    # =====================
    print("\n" + "=" * 55)
    print("TRAINING LEVEL 1: Motor Condition Classifier")
    print("  Classes: Healthy / Warning / Fault")
    print("=" * 55)

    y_condition = df['Motor_Condition']
    le_condition = LabelEncoder()
    y_cond_encoded = le_condition.fit_transform(y_condition)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cond_encoded, test_size=0.2, random_state=42, stratify=y_cond_encoded
    )

    condition_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    condition_model.fit(X_train_c, y_train_c)

    y_pred_c = condition_model.predict(X_test_c)
    y_proba_c = condition_model.predict_proba(X_test_c)

    cond_accuracy = accuracy_score(y_test_c, y_pred_c)
    cond_f1 = f1_score(y_test_c, y_pred_c, average='weighted')
    cond_report = classification_report(
        y_test_c, y_pred_c,
        target_names=le_condition.classes_,
        output_dict=True
    )
    cond_cm = confusion_matrix(y_test_c, y_pred_c)

    print(f"\n  Accuracy:     {cond_accuracy*100:.2f}%")
    print(f"  Weighted F1:  {cond_f1*100:.2f}%")
    print(f"\n  Per-class performance:")
    for cls in le_condition.classes_:
        r = cond_report[cls]
        print(f"    {cls:10s}  Precision: {r['precision']:.3f}  Recall: {r['recall']:.3f}  F1: {r['f1-score']:.3f}")

    # =====================
    # 5. TRAIN LEVEL 2: Fault Type Classifier
    # =====================
    print("\n" + "=" * 55)
    print("TRAINING LEVEL 2: Fault Type Classifier")
    print("  Classes: Bearing_Wear / Imbalance / Misalignment /")
    print("           Rotor_Bar_Fault / Overload / Overheating")
    print("=" * 55)

    # Train on FAULT samples only (Motor_Condition == 'Fault')
    fault_mask = df['Motor_Condition'] == 'Fault'
    df_faults = df[fault_mask].copy()
    X_faults = engineer_features(df_faults)

    y_fault = df_faults['Fault_Type']
    le_fault = LabelEncoder()
    y_fault_encoded = le_fault.fit_transform(y_fault)

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_faults, y_fault_encoded, test_size=0.2, random_state=42, stratify=y_fault_encoded
    )

    fault_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    fault_model.fit(X_train_f, y_train_f)

    y_pred_f = fault_model.predict(X_test_f)
    y_proba_f = fault_model.predict_proba(X_test_f)

    fault_accuracy = accuracy_score(y_test_f, y_pred_f)
    fault_f1 = f1_score(y_test_f, y_pred_f, average='weighted')
    fault_report = classification_report(
        y_test_f, y_pred_f,
        target_names=le_fault.classes_,
        output_dict=True
    )
    fault_cm = confusion_matrix(y_test_f, y_pred_f)

    print(f"\n  Accuracy:     {fault_accuracy*100:.2f}%")
    print(f"  Weighted F1:  {fault_f1*100:.2f}%")
    print(f"\n  Per-class performance:")
    for cls in le_fault.classes_:
        r = fault_report[cls]
        print(f"    {cls:18s}  Precision: {r['precision']:.3f}  Recall: {r['recall']:.3f}  F1: {r['f1-score']:.3f}")

    # =====================
    # 6. LIVE PREDICTION DEMO
    # =====================
    print("\n" + "=" * 55)
    print("LIVE PREDICTION DEMO (ESP32 Sensor Input Simulation)")
    print("=" * 55)

    demo_inputs = [
        {'Motor_Temp_C': 52.0, 'Vibration_mm_s': 1.5, 'Current_A': 2.9,
         'desc': 'Normal operation'},
        {'Motor_Temp_C': 72.0, 'Vibration_mm_s': 6.4, 'Current_A': 3.9,
         'desc': 'Research report example'},
        {'Motor_Temp_C': 74.5, 'Vibration_mm_s': 6.9, 'Current_A': 3.85,
         'desc': 'High vibration + current'},
        {'Motor_Temp_C': 95.0, 'Vibration_mm_s': 1.2, 'Current_A': 4.5,
         'desc': 'High current + temp, low vib (Overload)'},
        {'Motor_Temp_C': 92.0, 'Vibration_mm_s': 1.0, 'Current_A': 2.8,
         'desc': 'High temp only (Overheating)'},
        {'Motor_Temp_C': 58.0, 'Vibration_mm_s': 7.5, 'Current_A': 3.0,
         'desc': 'High vibration only (Imbalance)'},
        {'Motor_Temp_C': 68.0, 'Vibration_mm_s': 5.0, 'Current_A': 3.6,
         'desc': 'Moderate vib + current (Misalignment)'},
        {'Motor_Temp_C': 78.0, 'Vibration_mm_s': 3.5, 'Current_A': 4.0,
         'desc': 'High current + mild vib (Rotor Bar)'},
    ]

    demo_df = pd.DataFrame([{k: v for k, v in d.items() if k != 'desc'} for d in demo_inputs])
    demo_features = engineer_features(demo_df)

    predictions = predict_hierarchical(
        condition_model, fault_model,
        le_condition, le_fault,
        demo_features
    )

    for i, (inp, pred) in enumerate(zip(demo_inputs, predictions)):
        print(f"\n  Input {i+1}: {inp['desc']}")
        print(f"    Temp={inp['Motor_Temp_C']}C  Vib={inp['Vibration_mm_s']}mm/s  Curr={inp['Current_A']}A")
        print(f"    --> Condition: {pred['motor_condition']} ({pred['condition_confidence']*100:.1f}%)")
        if pred['fault_type']:
            print(f"    --> Fault Type: {pred['fault_type']} ({pred['fault_confidence']*100:.1f}%)")
        print(f"    JSON: {json.dumps({k: pred[k] for k in ['motor_condition', 'fault_type', 'condition_confidence', 'fault_confidence']})}")

    # =====================
    # 7. SAVE MODELS
    # =====================
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_PKL_PATH = 'motor_fault_model_hierarchical.pkl'

    # Load previous training history if exists
    historical_traits = []
    if os.path.exists(MODEL_PKL_PATH):
        try:
            with open(MODEL_PKL_PATH, 'rb') as f:
                existing = pickle.load(f)
                historical_traits = existing.get('training_history', [])
        except Exception:
            pass

    traits = {
        'run_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_used': os.path.basename(DATASET_PATH),
        'dataset_size': total_records,
        'condition_accuracy': round(cond_accuracy, 4),
        'condition_f1': round(cond_f1, 4),
        'fault_accuracy': round(fault_accuracy, 4),
        'fault_f1': round(fault_f1, 4),
        'features_used': feature_names,
        'datasheet_constants': DATASHEET,
    }
    historical_traits.append(traits)

    with open(MODEL_PKL_PATH, 'wb') as f:
        pickle.dump({
            'condition_model': condition_model,
            'fault_model': fault_model,
            'condition_encoder': le_condition,
            'fault_encoder': le_fault,
            'feature_names': feature_names,
            'datasheet': DATASHEET,
            'training_history': historical_traits,
        }, f)

    print(f"\n  Models saved to: {MODEL_PKL_PATH}")

    # =====================
    # 8. GENERATE VISUALIZATIONS
    # =====================
    print("\nGenerating visualizations...")
    sns.set_theme(style="whitegrid")

    # --- Chart 1: Condition Distribution ---
    plt.figure(figsize=(7, 3.5))
    colors_cond = {'Healthy': '#2ecc71', 'Warning': '#f39c12', 'Fault': '#e74c3c'}
    ax = sns.countplot(data=df, x='Motor_Condition', hue='Motor_Condition',
                       palette=colors_cond, order=CONDITION_CLASSES, legend=False)
    plt.title('Motor Condition Distribution', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Condition', fontweight='bold')
    plt.ylabel('Sample Count', fontweight='bold')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig('condition_distribution.png', dpi=200)
    plt.close()

    # --- Chart 2: Fault Type Distribution ---
    fault_df = df[df['Motor_Condition'] == 'Fault']
    plt.figure(figsize=(8, 3.5))
    fault_colors = {
        'Bearing_Wear': '#e74c3c', 'Imbalance': '#3498db',
        'Misalignment': '#9b59b6', 'Rotor_Bar_Fault': '#e67e22',
        'Overload': '#1abc9c', 'Overheating': '#f1c40f'
    }
    fault_order = [ft for ft in fault_colors if ft in fault_df['Fault_Type'].unique()]
    ax2 = sns.countplot(data=fault_df, x='Fault_Type', hue='Fault_Type',
                        palette=fault_colors, order=fault_order, legend=False)
    plt.title('Fault Type Distribution (Fault Samples Only)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Fault Type', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    for p in ax2.patches:
        ax2.annotate(f'{int(p.get_height()):,}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig('fault_type_distribution.png', dpi=200)
    plt.close()

    # --- Chart 3: Level 1 Confusion Matrix ---
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cond_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=le_condition.classes_, yticklabels=le_condition.classes_)
    plt.title('Level 1: Motor Condition Confusion Matrix', fontsize=11, fontweight='bold', pad=10)
    plt.ylabel('Actual Condition', fontweight='bold')
    plt.xlabel('Predicted Condition', fontweight='bold')
    plt.tight_layout()
    plt.savefig('condition_confusion_matrix.png', dpi=200)
    plt.close()

    # --- Chart 4: Level 2 Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(fault_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_fault.classes_, yticklabels=le_fault.classes_)
    plt.title('Level 2: Fault Type Confusion Matrix', fontsize=11, fontweight='bold', pad=10)
    plt.ylabel('Actual Fault Type', fontweight='bold')
    plt.xlabel('Predicted Fault Type', fontweight='bold')
    plt.xticks(rotation=35, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('fault_confusion_matrix.png', dpi=200)
    plt.close()

    # --- Chart 5: Feature Importance (Condition Model) ---
    cond_importances = condition_model.feature_importances_
    cond_indices = np.argsort(cond_importances)
    plt.figure(figsize=(7, 4))
    plt.barh([feature_names[i] for i in cond_indices],
             cond_importances[cond_indices], color='#27ae60')
    plt.title('Level 1: Condition Model Feature Importance', fontsize=11, fontweight='bold', pad=10)
    plt.xlabel('Importance Weight', fontweight='bold')
    plt.tight_layout()
    plt.savefig('condition_feature_importance.png', dpi=200)
    plt.close()

    # --- Chart 6: Feature Importance (Fault Model) ---
    fault_importances = fault_model.feature_importances_
    fault_indices = np.argsort(fault_importances)
    plt.figure(figsize=(7, 4))
    plt.barh([feature_names[i] for i in fault_indices],
             fault_importances[fault_indices], color='#2980b9')
    plt.title('Level 2: Fault Type Model Feature Importance', fontsize=11, fontweight='bold', pad=10)
    plt.xlabel('Importance Weight', fontweight='bold')
    plt.tight_layout()
    plt.savefig('fault_feature_importance.png', dpi=200)
    plt.close()

    # --- Chart 7: Sensor Scatter Matrix (3D-like pairplot) ---
    plt.figure(figsize=(8, 4))
    scatter_sample = df.sample(min(3000, len(df)), random_state=42)
    cond_color_map = {'Healthy': '#2ecc71', 'Warning': '#f39c12', 'Fault': '#e74c3c'}
    for cond, color in cond_color_map.items():
        mask = scatter_sample['Motor_Condition'] == cond
        plt.scatter(scatter_sample.loc[mask, 'Vibration_mm_s'],
                   scatter_sample.loc[mask, 'Current_A'],
                   c=color, label=cond, alpha=0.5, s=15)
    plt.xlabel('Vibration (mm/s)', fontweight='bold')
    plt.ylabel('Current (A)', fontweight='bold')
    plt.title('Sensor Space: Vibration vs Current by Condition', fontsize=11, fontweight='bold', pad=10)
    plt.legend(title='Condition', fontsize=9)
    plt.tight_layout()
    plt.savefig('sensor_scatter.png', dpi=200)
    plt.close()

    # =====================
    # 9. BUILD PDF REPORT
    # =====================
    print("Generating comprehensive PDF report...")
    PDF_REPORT_PATH = f'Hierarchical_Model_Report_{timestamp_str}.pdf'
    pdf = MotorHealthPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- PAGE 1: Executive Summary ---
    pdf.add_page()
    pdf.set_fill_color(245, 247, 250)
    pdf.rect(10, pdf.get_y(), 190, 28, 'F')
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(95, 6, clean_text(f" Dataset: {os.path.basename(DATASET_PATH)}"), ln=False)
    pdf.cell(95, 6, clean_text(f" Total Records: {total_records:,}"), ln=True)
    pdf.cell(95, 6, clean_text(f" Timestamp: {traits['run_timestamp']}"), ln=False)
    pdf.cell(95, 6, clean_text(f" Condition Model Acc: {cond_accuracy*100:.2f}%"), ln=True)
    pdf.cell(95, 6, clean_text(f" Fault Model Acc: {fault_accuracy*100:.2f}%"), ln=False)
    pdf.cell(95, 6, clean_text(f" Engineered Features: {len(feature_names)}"), ln=True)
    pdf.cell(95, 6, clean_text(f" Reference: Kirloskar 1.5 kW (3.10A / 2840RPM)"), ln=True)
    pdf.ln(4)

    # Section: Architecture
    pdf.set_font("Arial", 'B', 13)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 7, clean_text("1. Hierarchical Model Architecture"), ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Arial", size=10)
    pdf.set_text_color(40, 40, 40)
    arch_text = (
        "This system implements a TWO-LEVEL hierarchical classification approach:\n\n"
        "LEVEL 1 - Motor Condition Classifier:\n"
        "  Input: 15 engineered features (from 3 ESP32 sensors)\n"
        f"  Output: Healthy / Warning / Fault (Accuracy: {cond_accuracy*100:.2f}%, F1: {cond_f1*100:.2f}%)\n"
        "  Model: Random Forest (200 trees, balanced class weights)\n\n"
        "LEVEL 2 - Fault Type Classifier (activated when Level 1 = 'Fault'):\n"
        "  Input: Same 15 features, trained on fault samples only\n"
        f"  Output: 6 fault types (Accuracy: {fault_accuracy*100:.2f}%, F1: {fault_f1*100:.2f}%)\n"
        "  Model: Random Forest (200 trees, balanced class weights)\n\n"
        "WHY HIERARCHICAL? A single flat classifier must distinguish 8 classes. "
        "The hierarchical approach first separates the easy decision (Is anything wrong?) "
        "from the hard decision (What specific fault?), improving both accuracy and interpretability."
    )
    pdf.multi_cell(0, 5, clean_text(arch_text))
    pdf.ln(4)

    # Datasheet Thresholds Table
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 6, clean_text("Kirloskar Datasheet Thresholds (Built Into Model):"), ln=True)
    pdf.ln(2)

    pdf.set_fill_color(230, 235, 245)
    pdf.set_font("Arial", 'B', 9)
    for header in ["Parameter", "Datasheet", "Healthy", "Warning", "Fault"]:
        pdf.cell(38, 6, clean_text(header), 1, 0, 'C', True)
    pdf.ln()

    pdf.set_font("Arial", size=9)
    threshold_rows = [
        ("Current (A)", "3.10 A rated", "<= 3.41 A", "3.41-4.03 A", "> 4.03 A"),
        ("Temperature (C)", "50C amb + 70C rise", "<= 65 C", "65-85 C", "> 85 C"),
        ("Vibration (mm/s)", "ISO 10816-3", "<= 2.8 mm/s", "2.8-4.5 mm/s", "> 4.5 mm/s"),
    ]
    for row in threshold_rows:
        for val in row:
            pdf.cell(38, 6, clean_text(val), 1, 0, 'C')
        pdf.ln()

    # --- PAGE 2: Sensor Distributions & Condition Distribution ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 13)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 7, clean_text("2. Dataset Analysis & Distributions"), ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    if os.path.exists('condition_distribution.png'):
        pdf.image('condition_distribution.png', x=25, w=160)
        pdf.ln(3)
    if os.path.exists('fault_type_distribution.png'):
        pdf.image('fault_type_distribution.png', x=20, w=170)
        pdf.ln(3)

    pdf.set_font("Arial", size=9.5)
    pdf.set_text_color(40, 40, 40)
    cond_counts = df['Motor_Condition'].value_counts()
    dist_text = (
        f"The dataset contains {total_records:,} telemetry records with balanced class distribution:\n"
        f"- Healthy: {cond_counts.get('Healthy', 0):,} samples ({cond_counts.get('Healthy', 0)/total_records*100:.1f}%)\n"
        f"- Warning: {cond_counts.get('Warning', 0):,} samples ({cond_counts.get('Warning', 0)/total_records*100:.1f}%)\n"
        f"- Fault: {cond_counts.get('Fault', 0):,} samples ({cond_counts.get('Fault', 0)/total_records*100:.1f}%)\n"
        f"Fault samples are subdivided into 6 types based on the research report's multi-sensor signature definitions."
    )
    pdf.multi_cell(0, 4.8, clean_text(dist_text))

    # --- PAGE 3: Level 1 Results ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 13)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 7, clean_text("3. Level 1 Results: Motor Condition Classifier"), ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    if os.path.exists('condition_confusion_matrix.png'):
        pdf.image('condition_confusion_matrix.png', x=40, w=130)
        pdf.ln(3)

    # Per-class table
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, clean_text(f"Overall Accuracy: {cond_accuracy*100:.2f}%  |  Weighted F1: {cond_f1*100:.2f}%"), ln=True)
    pdf.ln(2)

    pdf.set_fill_color(230, 235, 245)
    pdf.set_font("Arial", 'B', 9)
    for h in ["Class", "Precision", "Recall", "F1-Score", "Support"]:
        pdf.cell(38, 6, clean_text(h), 1, 0, 'C', True)
    pdf.ln()
    pdf.set_font("Arial", size=9)
    for cls in le_condition.classes_:
        r = cond_report[cls]
        pdf.cell(38, 6, clean_text(cls), 1, 0, 'C')
        pdf.cell(38, 6, clean_text(f"{r['precision']:.4f}"), 1, 0, 'C')
        pdf.cell(38, 6, clean_text(f"{r['recall']:.4f}"), 1, 0, 'C')
        pdf.cell(38, 6, clean_text(f"{r['f1-score']:.4f}"), 1, 0, 'C')
        pdf.cell(38, 6, clean_text(f"{int(r['support'])}"), 1, 0, 'C')
        pdf.ln()
    pdf.ln(3)

    if os.path.exists('condition_feature_importance.png'):
        pdf.image('condition_feature_importance.png', x=20, w=170)

    # --- PAGE 4: Level 2 Results ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 13)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 7, clean_text("4. Level 2 Results: Fault Type Classifier"), ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    if os.path.exists('fault_confusion_matrix.png'):
        pdf.image('fault_confusion_matrix.png', x=20, w=170)
        pdf.ln(3)

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, clean_text(f"Overall Accuracy: {fault_accuracy*100:.2f}%  |  Weighted F1: {fault_f1*100:.2f}%"), ln=True)
    pdf.ln(2)

    pdf.set_fill_color(230, 235, 245)
    pdf.set_font("Arial", 'B', 8.5)
    for h in ["Fault Type", "Precision", "Recall", "F1-Score", "Support"]:
        pdf.cell(38, 6, clean_text(h), 1, 0, 'C', True)
    pdf.ln()
    pdf.set_font("Arial", size=8.5)
    for cls in le_fault.classes_:
        r = fault_report[cls]
        pdf.cell(38, 6, clean_text(cls.replace('_', ' ')), 1, 0, 'C')
        pdf.cell(38, 6, clean_text(f"{r['precision']:.4f}"), 1, 0, 'C')
        pdf.cell(38, 6, clean_text(f"{r['recall']:.4f}"), 1, 0, 'C')
        pdf.cell(38, 6, clean_text(f"{r['f1-score']:.4f}"), 1, 0, 'C')
        pdf.cell(38, 6, clean_text(f"{int(r['support'])}"), 1, 0, 'C')
        pdf.ln()
    pdf.ln(3)

    if os.path.exists('fault_feature_importance.png'):
        pdf.image('fault_feature_importance.png', x=20, w=170)

    # --- PAGE 5: Sensor Space & Fault Signatures ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 13)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 7, clean_text("5. Sensor Space Visualization & Fault Signature Patterns"), ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    if os.path.exists('sensor_scatter.png'):
        pdf.image('sensor_scatter.png', x=20, w=170)
        pdf.ln(3)

    pdf.set_font("Arial", 'B', 10.5)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 6, clean_text("Why ML Instead of Fixed Thresholds:"), ln=True)
    pdf.set_font("Arial", size=9.5)
    pdf.set_text_color(40, 40, 40)
    ml_why = (
        "A rule-based system (IF temp > 70 AND vibration > 6 THEN Fault) cannot capture:\n"
        "* High current with normal vibration -> Overload (electrical, not mechanical)\n"
        "* High vibration with normal current -> Bearing Wear or Imbalance\n"
        "* Rising temperature with normal current and vibration -> Overheating (cooling failure)\n"
        "* High current with mild vibration -> Rotor Bar Fault (electrical with torque ripple)\n"
        "* Moderate vibration + elevated current -> Misalignment (mechanical resistance)\n\n"
        "The ML model learns these CROSS-SENSOR INTERACTION patterns from the engineered features "
        "(temp_x_vib, curr_x_vib, temp_x_curr) and datasheet-normalized ratios, enabling earlier "
        "and more accurate fault detection than any fixed threshold system."
    )
    pdf.multi_cell(0, 4.8, clean_text(ml_why))
    pdf.ln(4)

    # Fault Signature Reference Table
    pdf.set_font("Arial", 'B', 10.5)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 6, clean_text("Fault Signature Reference (from Research Report):"), ln=True)
    pdf.ln(2)

    pdf.set_fill_color(230, 235, 245)
    pdf.set_font("Arial", 'B', 8)
    for h in ["Fault Type", "Temperature", "Vibration", "Current", "Key Differentiator"]:
        w = 38
        pdf.cell(w, 6, clean_text(h), 1, 0, 'C', True)
    pdf.ln()

    pdf.set_font("Arial", size=7.5)
    sig_rows = [
        ("Bearing Wear", "Moderate +", "HIGH", "Normal", "Vibration primary, temp from friction"),
        ("Imbalance", "Slight +", "VERY HIGH", "Normal", "Extreme vib, low temp rise"),
        ("Misalignment", "Moderate", "Moderate-High", "Slightly +", "Vib + current both elevated"),
        ("Rotor Bar Fault", "Elevated", "Mild +", "HIGH + noisy", "Current dominant + fluctuating"),
        ("Overload", "HIGH", "Low-Normal", "VERY HIGH", "Current primary, vib stays low"),
        ("Overheating", "VERY HIGH", "Normal", "Normal", "Temp only elevated parameter"),
    ]
    for row in sig_rows:
        for val in row:
            pdf.cell(38, 5, clean_text(val), 1, 0, 'C')
        pdf.ln()

    # --- PAGE 6: Live Demo Results ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 13)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 7, clean_text("6. Live Prediction Demonstration"), ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Arial", size=9.5)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, clean_text(
        "Below are sample predictions showing the hierarchical model processing real-world-like ESP32 sensor inputs. "
        "Each prediction includes the motor condition, fault type (if applicable), and confidence scores from predict_proba()."
    ))
    pdf.ln(3)

    for i, (inp, pred) in enumerate(zip(demo_inputs, predictions)):
        pdf.set_font("Arial", 'B', 9)
        pdf.set_text_color(24, 43, 73)
        pdf.cell(0, 5, clean_text(f"Scenario {i+1}: {inp['desc']}"), ln=True)
        pdf.set_font("Arial", size=8.5)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 4.5, clean_text(
            f"  Input: Temp={inp['Motor_Temp_C']}C  Vib={inp['Vibration_mm_s']}mm/s  Curr={inp['Current_A']}A"
        ), ln=True)

        cond_color = {'Healthy': (40, 140, 60), 'Warning': (210, 120, 20), 'Fault': (180, 40, 40)}
        rgb = cond_color.get(pred['motor_condition'], (40, 40, 40))
        pdf.set_text_color(*rgb)
        result_str = f"  Result: {pred['motor_condition']} ({pred['condition_confidence']*100:.1f}%)"
        if pred['fault_type']:
            result_str += f" -> {pred['fault_type']} ({pred['fault_confidence']*100:.1f}%)"
        pdf.cell(0, 4.5, clean_text(result_str), ln=True)
        pdf.set_text_color(40, 40, 40)
        pdf.ln(2)

    # Output PDF
    pdf.output(PDF_REPORT_PATH)

    # Cleanup temp images
    for img_path in [
        'condition_distribution.png', 'fault_type_distribution.png',
        'condition_confusion_matrix.png', 'fault_confusion_matrix.png',
        'condition_feature_importance.png', 'fault_feature_importance.png',
        'sensor_scatter.png'
    ]:
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass

    # =====================
    # 10. FINAL SUMMARY
    # =====================
    print("\n" + "=" * 55)
    print("SUCCESS: Hierarchical Motor Fault Model Complete!")
    print("=" * 55)
    print(f"  Level 1 (Condition):  {cond_accuracy*100:.2f}% accuracy, {cond_f1*100:.2f}% F1")
    print(f"  Level 2 (Fault Type): {fault_accuracy*100:.2f}% accuracy, {fault_f1*100:.2f}% F1")
    print(f"  Features engineered:  {len(feature_names)} (from 3 raw sensors)")
    print(f"  Model saved:          {MODEL_PKL_PATH}")
    print(f"  PDF report:           {PDF_REPORT_PATH}")
    print(f"  Dataset used:         {os.path.basename(DATASET_PATH)}")
    print("=" * 55)
    print("\nTo use the trained model for real-time ESP32 predictions:")
    print("  1. Load the pickle: pickle.load(open('motor_fault_model_hierarchical.pkl', 'rb'))")
    print("  2. Create a DataFrame: pd.DataFrame([{'Motor_Temp_C': 72, 'Vibration_mm_s': 6.4, 'Current_A': 3.9}])")
    print("  3. Engineer features: engineer_features(df)")
    print("  4. Call: predict_hierarchical(condition_model, fault_model, ...)")
    print("  5. Output: {'motor_condition': 'Fault', 'fault_type': 'Bearing_Wear', 'confidence': 0.96}")
