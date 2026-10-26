# Compressor Health Monitoring

My Master's thesis project: a predictive maintenance dashboard for industrial air compressors. You set 14 operating readings (pressures, temperatures, vibration, oil pressure, power, efficiency, filter and valve condition), and a Random Forest model estimates the risk of failure. The dashboard shows the risk as a gauge, flags the readings that push it up, and suggests maintenance steps for the likely failure mode.

**Live app:** https://compreappr-health-monitoring-dzmttiyfuvtmdh5ycig2nb.streamlit.app/

## How it works

```
CSV data (operations and failure history)
        │
        ▼
Model training (scikit-learn Random Forest, 150 trees, 14 features)
        │
        ▼
models/compressor_risk_model.pkl
        │
        ▼
Streamlit dashboard (app.py): sliders → failure probability → risk level and recommended actions
```

Risk levels: **normal** below 40%, **elevated** from 40% to 70%, **critical** above 70%.

`sagemaker_training_job.py` is a training script written to run on Amazon SageMaker.

## What it monitors

**Mechanical**

| Reading | Unit | Normal | Warning | Critical |
| --- | --- | --- | --- | --- |
| Operating hours | hours a year | 6,000 to 8,500 | 8,500 to 9,000 | over 9,000 |
| Suction pressure | bar | 1.0 to 1.4 | 0.9 to 1.0 | under 0.9 |
| Discharge pressure | bar | 7.5 to 9.0 | 9.0 to 10.5 | over 10.5 |
| Vibration | mm/s | 1.8 to 3.0 | 3.0 to 4.0 | over 4.0 |
| Lubricant temperature | °C | 60 to 75 | 75 to 80 | over 80 |

**Thermodynamic**

| Reading | Unit | Baseline | Alarm |
| --- | --- | --- | --- |
| Compression ratio | ratio | 6.5 | over 11.0 |
| Discharge temperature | °C | 85 | over 100 |
| Bearing temperature | °C | 68 | over 88 |
| Oil pressure | bar | 14.5 | under 12.0 |
| Power consumption | kW | 35.0 | over 50.0 |
| Efficiency | % | 82 | under 70 |
| Pressure drop | bar | 0.4 | over 1.2 |

**Condition**

| Reading | Unit | Excellent | Good | Fair | Poor |
| --- | --- | --- | --- | --- | --- |
| Filter differential pressure | bar | under 0.4 | 0.4 to 0.8 | 0.8 to 1.5 | over 1.5 |
| Valve condition score | 1 to 10 | 9 to 10 | 7 to 8 | 5 to 6 | under 5 |

## Failure modes it looks for

| Failure | Warning signs |
| --- | --- |
| Bearing seizure | Vibration over 4.5 mm/s, bearing temperature over 88 °C, oil pressure under 12 bar |
| Valve failure | Valve score under 3, compression ratio over 11, discharge temperature over 100 °C, efficiency down more than 20% |
| Intake filter blockage | Filter differential over 2.2 bar, suction pressure under 0.9 bar, rising power use and vibration |

## The data

| File | What's in it |
| --- | --- |
| `compressor_operational_realistic.csv` | Yearly operating figures for 4 compressors, 2022 to 2024 (12 rows) |
| `compressor_failures_realistic.csv` | 16 failure events: type, downtime, cost, MTBF/MTTR and how many hours of warning signs came first |
| `compressor_raw_logs.csv` | Raw log export (saved in Excel format despite the `.csv` name) |
| `merge_data.py` | Merges the files into one training table |

The dataset is small: 12 yearly records and 16 failure events. It's enough to build and demonstrate the pipeline, but not enough for reliable accuracy figures, so I don't quote any here. A larger, sensor-level dataset is the main next step.

## Run it locally

```bash
git clone https://github.com/cyberN3m3/compressor-health-monitoring.git
cd compressor-health-monitoring
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py        # opens http://localhost:8501
```

Move the sliders in the sidebar; the gauge, status and recommendations update as you go.

## Files

```
app.py                       # Streamlit dashboard
merge_data.py                # combines the data files
sagemaker_training_job.py    # training script for Amazon SageMaker
models/compressor_risk_model.pkl
requirements.txt
style.css
*.csv                        # data (see above)
```

## Next steps

- Collect sensor-level data and report accuracy, precision and recall on a proper held-out test set
- Compare the Random Forest with XGBoost and a time-series model
- Stream live readings instead of manual sliders
- Monitor several compressors at once

---

Anyasi Chineme · Master's thesis · [GitHub](https://github.com/cyberN3m3)
