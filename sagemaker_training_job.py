import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def run_sagemaker_factory():
    print("⚡ Synthesizing genuine IGHF parameter boundaries...")
    np.random.seed(42)
    records = 1200
    
    # Building rows capturing normal baseline parameters
    data = {
        "sep_pressure_psig": np.random.uniform(750, 940, records),
        "sep_temp_f": np.random.uniform(60, 100, records),
        "teg_pressure_psig": np.random.uniform(750, 895, records),
        "teg_temp_f": np.random.uniform(90, 100, records),
        "glycol_lvl_percent": np.random.uniform(25, 50, records),
        "condensate_lvl_percent": np.random.uniform(25, 68, records)
    }
    
    df = pd.DataFrame(data)
    df["has_failure"] = 0
    
    # Pattern 1: Thermal Carryover Event (Inlet Temp leaks over target boundaries)
    thermal_leak = (df["sep_temp_f"] > 105) & (df["teg_temp_f"] > 110)
    df.loc[thermal_leak, "has_failure"] = 1
    
    # Pattern 2: Pressure Divergence / Process Restriction (Cylinder Valve blockage)
    pressure_divergence = (df["sep_pressure_psig"] > 915) & (df["teg_pressure_psig"] < 765)
    df.loc[pressure_divergence, "has_failure"] = 1
    
    features = ["sep_pressure_psig", "sep_temp_f", "teg_pressure_psig", "teg_temp_f", "glycol_lvl_percent", "condensate_lvl_percent"]
    X = df[features]
    y = df["has_failure"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    print("🌲 Fitting ensemble decision trees to plant signatures...")
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    print("\n📊 --- AWS CLOUD PERFORMANCE METRICS ---")
    print(classification_report(y_test, model.predict(X_test)))
    
    # Serialize the asset
    joblib.dump(model, "models/compressor_risk_model.pkl")
    print("🎯 Success: 'compressor_risk_model.pkl' written cleanly to storage asset array.")

if __name__ == "__main__":
    run_sagemaker_factory()