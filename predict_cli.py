"""
Motor Fault Prediction - CLI Interface for 3 Sensors
Usage: python predict_cli.py <vibration> <current> <temperature>

Sensors:
- Vibration: MPU6050 (mm/s)
- Current: ACS712 (A)  
- Temperature: MLX90640 (C)
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json

OUTPUT_FILE = 'ml_model.json'

DATASHEET = {
    'rated_current_a': 3.10,
    'max_thermal_limit_c': 120.0,
    'vib_healthy_max': 2.8,
    'vib_warning_max': 4.5,
    'curr_healthy_max': 3.10 * 1.10,
    'curr_warning_max': 3.10 * 1.30,
    'temp_healthy_max': 65.0,
    'temp_warning_max': 85.0,
}

def engineer_features(df, datasheet=DATASHEET):
    features = pd.DataFrame()
    features['temp'] = df['Motor_Temp_C']
    features['vibration'] = df['Vibration_mm_s']
    features['current'] = df['Current_A']
    features['temp_ratio'] = df['Motor_Temp_C'] / datasheet['max_thermal_limit_c']
    features['vib_ratio'] = df['Vibration_mm_s'] / datasheet['vib_healthy_max']
    features['curr_ratio'] = df['Current_A'] / datasheet['rated_current_a']
    features['temp_warning_flag'] = (df['Motor_Temp_C'] > datasheet['temp_healthy_max']).astype(int)
    features['temp_fault_flag'] = (df['Motor_Temp_C'] > datasheet['temp_warning_max']).astype(int)
    features['vib_warning_flag'] = (df['Vibration_mm_s'] > datasheet['vib_healthy_max']).astype(int)
    features['vib_fault_flag'] = (df['Vibration_mm_s'] > datasheet['vib_warning_max']).astype(int)
    features['curr_warning_flag'] = (df['Current_A'] > datasheet['curr_healthy_max']).astype(int)
    features['curr_fault_flag'] = (df['Current_A'] > datasheet['curr_warning_max']).astype(int)
    features['temp_x_vib'] = features['temp_ratio'] * features['vib_ratio']
    features['curr_x_vib'] = features['curr_ratio'] * features['vib_ratio']
    features['temp_x_curr'] = features['temp_ratio'] * features['curr_ratio']
    return features

def predict_hierarchical(condition_model, fault_model, condition_encoder, fault_encoder, feature_df):
    results = []
    condition_pred = condition_model.predict(feature_df)
    condition_proba = condition_model.predict_proba(feature_df)
    condition_labels = condition_encoder.classes_
    fault_pred = fault_model.predict(feature_df)
    fault_proba = fault_model.predict_proba(feature_df)
    fault_labels = fault_encoder.classes_
    
    for i in range(len(feature_df)):
        cond = condition_labels[condition_pred[i]]
        cond_conf = float(np.max(condition_proba[i]))
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

def main():
    model_path = 'motor_fault_model_hierarchical.pkl'
    MAX_SAMPLES = 10
    
    if not os.path.exists(model_path):
        print("Model not found. Train first with: python ML_model_hierarchical.py --generate")
        return
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    condition_model = model_data['condition_model']
    fault_model = model_data['fault_model']
    condition_encoder = model_data['condition_encoder']
    fault_encoder = model_data['fault_encoder']
    
    samples = []
    
    if len(sys.argv) >= 4 and not sys.argv[1].startswith('--'):
        values = [float(x) for x in sys.argv[1:]]
        num_complete = len(values) // 3
        if num_complete == 0:
            print("Error: Need at least 3 values (vibration current temperature)")
            return
        start_idx = len(values) - (num_complete * 3)
        if start_idx:
            print(f"[INFO] Dropped first {start_idx} value(s) to keep complete groups")
        for i in range(start_idx, len(values), 3):
            samples.append({
                'Motor_Temp_C': values[i+2],
                'Vibration_mm_s': values[i],
                'Current_A': values[i+1]
            })
        if len(samples) > MAX_SAMPLES:
            samples = samples[-MAX_SAMPLES:]
    elif len(sys.argv) == 3 and sys.argv[1] == '--file':
        try:
            df = pd.read_csv(sys.argv[2])
            cols_needed = ['Motor_Temp_C', 'Vibration_mm_s', 'Current_A']
            if all(c in df.columns for c in cols_needed):
                samples = df[cols_needed].to_dict('records')
                if len(samples) > MAX_SAMPLES:
                    samples = samples[-MAX_SAMPLES:]
            else:
                print(f"CSV must have columns: {', '.join(cols_needed)}")
                return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        # Interactive mode - user types values after running the file
        try:
            vals = input("Enter sensor values (vib curr temp): ").strip().split()
            nums = [float(v) for v in vals]
            num_complete = len(nums) // 3
            # Take complete groups from the END to preserve latest readings
            start_idx = len(nums) - (num_complete * 3)
            leftover = start_idx
            for i in range(start_idx, len(nums), 3):
                samples.append({
                    'Motor_Temp_C': nums[i+2],
                    'Vibration_mm_s': nums[i],
                    'Current_A': nums[i+1]
                })
            if leftover:
                print(f"[INFO] Dropped first {int(leftover)} value(s) to keep complete groups")
        except:
            pass
        if not samples:
            print("Using demo values: 6.4 3.9 72")
            samples.append({'Motor_Temp_C': 72, 'Vibration_mm_s': 6.4, 'Current_A': 3.9})
        if len(samples) > MAX_SAMPLES:
            samples = samples[-MAX_SAMPLES:]
    
    # Average multiple samples
    if len(samples) > 1:
        print(f"[INFO] Averaging {len(samples)} samples...")
        avg_sample = {
            'Motor_Temp_C': round(np.mean([s['Motor_Temp_C'] for s in samples]), 2),
            'Vibration_mm_s': round(np.mean([s['Vibration_mm_s'] for s in samples]), 2),
            'Current_A': round(np.mean([s['Current_A'] for s in samples]), 2)
        }
        input_df = pd.DataFrame([avg_sample])
        samples = [avg_sample]
    else:
        input_df = pd.DataFrame(samples)
    
    features = engineer_features(input_df)
    results = predict_hierarchical(
        condition_model, fault_model,
        condition_encoder, fault_encoder,
        features
    )
    
    pred = results[0]
    
    output = {
        'vibration_mm_s': samples[0]['Vibration_mm_s'],
        'current_A': samples[0]['Current_A'],
        'temperature_C': samples[0]['Motor_Temp_C'],
        'motor_condition': pred['motor_condition'],
        'condition_confidence': pred['condition_confidence'],
        'fault_type': pred['fault_type'],
        'fault_confidence': pred['fault_confidence']
    }
    
    print("----------------------------------------")
    print(f"  Input: Vib={samples[0]['Vibration_mm_s']:.2f}mm/s  Curr={samples[0]['Current_A']:.2f}A  Temp={samples[0]['Motor_Temp_C']:.2f}C")
    print(f"  Condition: {pred['motor_condition']} ({pred['condition_confidence']*100:.1f}%)")
    if pred['fault_type']:
        print(f"  Fault Type: {pred['fault_type']} ({pred['fault_confidence']*100:.1f}%)")
    print(f"  JSON: {json.dumps(output)}")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()