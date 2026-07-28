"""
Hierarchical Motor Fault Dataset Generator
============================================
Generates synthetic telemetry grounded in Kirloskar 1.5 kW datasheet specifications
and the deep-research-report fault signature definitions.

Produces a dataset with:
  - 3 sensor inputs: Temperature (Motor_Temp_C), Vibration (Vibration_mm_s), Current (Current_A)
  - Hierarchical labels:
      Motor_Condition: Healthy / Warning / Fault
      Fault_Type: None / Bearing_Wear / Imbalance / Misalignment / Rotor_Bar_Fault / Overload / Overheating

Fault signatures are based on ISO 10816, MCSA literature, and the Kirloskar datasheet:
  - Rated Current: 3.10 A  (Healthy <= 3.4 A, Warning 3.4-4.0 A, Fault > 4.0 A)
  - Temperature: Ambient 50C + Rise 70C = 120C limit
    (Healthy <= 65C surface, Warning 65-85C, Fault > 85C)
  - Vibration: ISO 10816-3 Group 2
    (Healthy <= 2.8 mm/s, Warning 2.8-4.5 mm/s, Fault > 4.5 mm/s)

Each fault type has a UNIQUE multi-sensor signature so the ML model learns
complex cross-sensor patterns (not just simple thresholds).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# Kirloskar 1.5 kW Datasheet Constants (from research report)
# ============================================================
RATED_CURRENT_A = 3.10
RATED_VOLTAGE_V = 415.0
RATED_SPEED_RPM = 2840.0
AMBIENT_TEMP_C = 50.0
TEMP_RISE_C = 70.0
MAX_THERMAL_LIMIT_C = 120.0

# ISO 10816-3 Vibration Thresholds (Group 2 small motors)
VIB_HEALTHY_MAX = 2.8     # mm/s RMS
VIB_WARNING_MAX = 4.5     # mm/s RMS

# Current Thresholds (derived from datasheet)
CURR_HEALTHY_MAX = RATED_CURRENT_A * 1.10   # 3.41 A
CURR_WARNING_MAX = RATED_CURRENT_A * 1.30   # 4.03 A

# Temperature Thresholds (motor surface via MLX90640)
TEMP_HEALTHY_MAX = 65.0   # C
TEMP_WARNING_MAX = 85.0   # C


def generate_healthy_samples(n):
    """
    HEALTHY: All sensors within optimal datasheet ranges.
    Temperature: 40-62C, Vibration: 0.3-2.5 mm/s, Current: 2.2-3.3 A
    """
    temp = np.random.uniform(40.0, 62.0, n) + np.random.normal(0, 2.0, n)
    vib = np.random.uniform(0.3, 2.5, n) + np.random.normal(0, 0.15, n)
    curr = np.random.uniform(2.2, 3.3, n) + np.random.normal(0, 0.08, n)

    # Clamp to realistic bounds
    temp = np.clip(temp, 35.0, 64.9)
    vib = np.clip(vib, 0.2, 2.79)
    curr = np.clip(curr, 1.8, 3.39)

    return temp, vib, curr, ['Healthy'] * n, ['None'] * n


def generate_warning_generic_samples(n):
    """
    WARNING (early degradation): One or more parameters in caution zone,
    but no clear fault signature yet. Simulates gradual aging.
    """
    temp = np.random.uniform(58.0, 78.0, n) + np.random.normal(0, 3.0, n)
    vib = np.random.uniform(2.0, 3.8, n) + np.random.normal(0, 0.2, n)
    curr = np.random.uniform(2.9, 3.7, n) + np.random.normal(0, 0.1, n)

    temp = np.clip(temp, 55.0, 84.9)
    vib = np.clip(vib, 1.5, 4.49)
    curr = np.clip(curr, 2.5, 3.99)

    return temp, vib, curr, ['Warning'] * n, ['None'] * n


def generate_bearing_wear_samples(n):
    """
    BEARING WEAR fault signature (from research report Section 2):
    - HIGH vibration (mechanical deterioration, high-frequency components)
    - MODERATELY elevated temperature (frictional heating in bearing)
    - NORMAL to slightly elevated current (friction increases torque demand slightly)
    
    Key pattern: Vibration is the PRIMARY indicator. Temperature follows due
    to friction. Current stays relatively normal — this distinguishes it from Overload.
    """
    vib = np.random.uniform(4.5, 10.0, n) + np.random.normal(0, 0.5, n)
    temp = np.random.uniform(62.0, 88.0, n) + np.random.normal(0, 3.0, n)
    curr = np.random.uniform(2.8, 3.6, n) + np.random.normal(0, 0.1, n)

    # Correlation: worse vibration → higher temperature
    temp += (vib - 4.5) * 2.0

    vib = np.clip(vib, 4.0, 12.0)
    temp = np.clip(temp, 60.0, 100.0)
    curr = np.clip(curr, 2.5, 3.8)

    return temp, vib, curr, ['Fault'] * n, ['Bearing_Wear'] * n


def generate_imbalance_samples(n):
    """
    IMBALANCE fault signature (from research report Section 2):
    - HIGH vibration at 1x RPM (mass imbalance causes strong 1-per-rev)
    - SLIGHTLY elevated temperature (bearing load from imbalance)
    - NORMAL current (no electrical issue)

    Key pattern: Very high vibration with NORMAL current and only mild temp rise.
    Distinguished from Bearing Wear by generally higher vibration amplitude
    but lower temperature (no direct friction heating).
    """
    vib = np.random.uniform(5.0, 11.0, n) + np.random.normal(0, 0.6, n)
    temp = np.random.uniform(52.0, 72.0, n) + np.random.normal(0, 2.5, n)
    curr = np.random.uniform(2.6, 3.4, n) + np.random.normal(0, 0.08, n)

    vib = np.clip(vib, 4.5, 13.0)
    temp = np.clip(temp, 48.0, 78.0)
    curr = np.clip(curr, 2.3, 3.5)

    return temp, vib, curr, ['Fault'] * n, ['Imbalance'] * n


def generate_misalignment_samples(n):
    """
    MISALIGNMENT fault signature (from research report Section 2):
    - MODERATE-HIGH vibration (strong at 2x RPM harmonics, axial vibration)
    - MODERATE temperature rise (bearing loads from angular/parallel offset)
    - SLIGHTLY elevated current (increased mechanical resistance)

    Key pattern: Moderate vibration (less extreme than Imbalance) combined
    with a noticeable current increase. Temperature is moderate.
    """
    vib = np.random.uniform(3.5, 7.5, n) + np.random.normal(0, 0.4, n)
    temp = np.random.uniform(58.0, 80.0, n) + np.random.normal(0, 3.0, n)
    curr = np.random.uniform(3.2, 3.9, n) + np.random.normal(0, 0.1, n)

    # Correlation: worse misalignment → higher current draw
    curr += (vib - 3.5) * 0.05

    vib = np.clip(vib, 3.0, 9.0)
    temp = np.clip(temp, 55.0, 85.0)
    curr = np.clip(curr, 3.0, 4.2)

    return temp, vib, curr, ['Fault'] * n, ['Misalignment'] * n


def generate_rotor_bar_fault_samples(n):
    """
    ROTOR BAR FAULT signature (from research report Section 2):
    - SLIGHTLY elevated vibration (subtle 2x RPM component from torque ripple)
    - MODERATE temperature rise (reduced efficiency, I²R heating)
    - ELEVATED current with FLUCTUATIONS (slip-frequency sidebands)

    Key pattern: Current is abnormally high AND noisy/fluctuating, while
    vibration is only mildly elevated. This is the OPPOSITE of Bearing Wear.
    """
    curr_base = np.random.uniform(3.4, 4.3, n)
    curr_noise = np.random.normal(0, 0.2, n)  # Extra current fluctuation
    curr = curr_base + curr_noise

    vib = np.random.uniform(2.5, 4.5, n) + np.random.normal(0, 0.3, n)
    temp = np.random.uniform(65.0, 88.0, n) + np.random.normal(0, 3.0, n)

    # Correlation: higher current → higher temperature
    temp += (curr - 3.4) * 5.0

    curr = np.clip(curr, 3.2, 4.8)
    vib = np.clip(vib, 2.0, 5.5)
    temp = np.clip(temp, 62.0, 100.0)

    return temp, vib, curr, ['Fault'] * n, ['Rotor_Bar_Fault'] * n


def generate_overload_samples(n):
    """
    OVERLOAD fault signature (from research report Section 2):
    - VERY HIGH current (sustained > 110-130% of rated)
    - HIGH temperature (I²R heating, thermal runaway)
    - LOW-NORMAL vibration (motor is mechanically sound, just overloaded)

    Key pattern: Current is the PRIMARY indicator (very high), temperature
    follows proportionally, but vibration stays LOW. This is the critical
    differentiator — high current + low vibration = electrical overload,
    NOT a mechanical fault.
    """
    curr = np.random.uniform(3.8, 5.2, n) + np.random.normal(0, 0.15, n)
    temp = np.random.uniform(75.0, 105.0, n) + np.random.normal(0, 3.5, n)
    vib = np.random.uniform(0.8, 3.0, n) + np.random.normal(0, 0.2, n)

    # Strong correlation: higher current → much higher temperature
    temp += (curr - 3.8) * 8.0

    curr = np.clip(curr, 3.6, 5.5)
    temp = np.clip(temp, 72.0, 120.0)
    vib = np.clip(vib, 0.5, 3.5)

    return temp, vib, curr, ['Fault'] * n, ['Overload'] * n


def generate_overheating_samples(n):
    """
    OVERHEATING fault signature (from research report Section 2):
    - VERY HIGH temperature (cooling failure, bad ventilation, or environment)
    - NORMAL current (motor is not electrically overloaded)
    - NORMAL-LOW vibration (motor is mechanically sound)

    Key pattern: Temperature is the SOLE elevated parameter. Current and
    vibration remain near-normal. This distinguishes Overheating from Overload
    (where current is also high) and from Bearing Wear (where vibration is high).
    """
    temp = np.random.uniform(82.0, 115.0, n) + np.random.normal(0, 3.0, n)
    curr = np.random.uniform(2.5, 3.4, n) + np.random.normal(0, 0.08, n)
    vib = np.random.uniform(0.5, 2.5, n) + np.random.normal(0, 0.15, n)

    temp = np.clip(temp, 78.0, 120.0)
    curr = np.clip(curr, 2.2, 3.5)
    vib = np.clip(vib, 0.3, 3.0)

    return temp, vib, curr, ['Fault'] * n, ['Overheating'] * n


def generate_hierarchical_dataset(
    samples_per_class=2500,
    output_csv='hierarchical_motor_fault_dataset.csv',
    seed=42
):
    """
    Generates a complete hierarchical fault dataset with all fault types
    from the research report taxonomy.
    
    Class distribution (default 2500 per class = 20,000 total):
        Healthy:          2500  (12.5%)
        Warning:          2500  (12.5%)
        Bearing_Wear:     2500  (12.5%)
        Imbalance:        2500  (12.5%)
        Misalignment:     2500  (12.5%)
        Rotor_Bar_Fault:  2500  (12.5%)
        Overload:         2500  (12.5%)
        Overheating:      2500  (12.5%)
    """
    np.random.seed(seed)

    generators = [
        generate_healthy_samples,
        generate_warning_generic_samples,
        generate_bearing_wear_samples,
        generate_imbalance_samples,
        generate_misalignment_samples,
        generate_rotor_bar_fault_samples,
        generate_overload_samples,
        generate_overheating_samples,
    ]

    all_temp, all_vib, all_curr = [], [], []
    all_condition, all_fault_type = [], []

    print("=" * 55)
    print("HIERARCHICAL MOTOR FAULT DATASET GENERATOR")
    print("Grounded in Kirloskar 1.5 kW Datasheet + ISO 10816")
    print("=" * 55)

    for gen_func in generators:
        temp, vib, curr, condition, fault_type = gen_func(samples_per_class)
        all_temp.extend(temp)
        all_vib.extend(vib)
        all_curr.extend(curr)
        all_condition.extend(condition)
        all_fault_type.extend(fault_type)
        label = fault_type[0] if fault_type[0] != 'None' else condition[0]
        print(f"  Generated {samples_per_class:,} samples: {label}")

    # Create timestamps (5-minute intervals over simulated period)
    total = len(all_temp)
    start_time = datetime.now() - timedelta(days=60)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(total)]

    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Motor_Temp_C': np.round(all_temp, 1),
        'Vibration_mm_s': np.round(all_vib, 2),
        'Current_A': np.round(all_curr, 2),
        'Motor_Condition': all_condition,
        'Fault_Type': all_fault_type
    })

    # Shuffle to remove ordering bias
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save
    df.to_csv(output_csv, index=False)

    print(f"\n{'=' * 55}")
    print(f"SUCCESS: Generated {total:,} telemetry records")
    print(f"Output CSV: {output_csv}")
    print(f"\nMotor Condition Distribution:")
    print(df['Motor_Condition'].value_counts().to_string())
    print(f"\nFault Type Distribution:")
    print(df['Fault_Type'].value_counts().to_string())
    print(f"\nSensor Summary Statistics:")
    print(df[['Motor_Temp_C', 'Vibration_mm_s', 'Current_A']].describe().round(2).to_string())
    print(f"{'=' * 55}")

    return df


if __name__ == '__main__':
    output_path = 'hierarchical_motor_fault_dataset.csv'
    samples = 2500

    if len(sys.argv) > 1:
        try:
            samples = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    generate_hierarchical_dataset(
        samples_per_class=samples,
        output_csv=output_path
    )
