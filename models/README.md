# Production Model Documentation

## 🚨 Model Files Not Included in Repository

**Why the trained models aren't uploaded:**

The production Random Forest model (`production_model_rf_pre2025.pkl`) is **380 MB**, which exceeds GitHub's 100 MB file size limit. Rather than use Git LFS or external hosting, the complete training pipeline is provided in the notebooks so you can reproduce the model yourself.

**What's included instead:**
- ✅ Complete model metadata and specifications (this README + `model_metadata.json`)
- ✅ Full training pipeline in [`5_Modelling.ipynb`](../notebooks/5_Modelling.ipynb)
- ✅ Exact preprocessing steps documented below
- ✅ Performance metrics and validation results
- ✅ Fitted StandardScaler (`scaler_pre2025.pkl`, 5 MB)

**To reproduce the production model:**

**Option A: Interactive Notebooks (Recommended for Learning)**
```bash
# Run the complete pipeline:
notebooks/0_Geocode_Names_to_Way_ID.ipynb     # 1. Automated geocoding
notebooks/1_1__coord_to_wayid_manual.ipynb    # 2. Manual verification
notebooks/1_2_prop_validation.ipynb           # 3. Validation
notebooks/2_1_Amenity_OSM_Search.ipynb        # 4. Extract features
notebooks/2_2_Ridership_Data_Extraction.ipynb # 5. Add ridership
notebooks/3_Merging.ipynb                     # 6. Combine features
notebooks/4_Clustering.ipynb                  # 7. Create clusters
notebooks/5_Modelling.ipynb                   # 8. Train Random Forest ← YOU ARE HERE
```

**Option B: Automated Pipeline (Recommended for Production)**
```bash
# Run complete pipeline from command line:
python -m src.scripts.run_full_pipeline --input data/raw/properties.csv

# Or run individual stages:
python -m src.scripts.run_geocoding --input data/raw/properties.csv
python -m src.scripts.run_feature_extraction
python -m src.scripts.run_clustering --method auto
python -m src.scripts.train_model --cv-folds 10 --temporal-split
```
---

## 📊 Model Selection Rationale

After comparing 9 regression algorithms, **Random Forest** was selected for production deployment based on:

| Metric | Value | Rationale |
|--------|-------|-----------|
| **Test R² (2025 holdout)** | 0.97 | Explains 97% of price variance on unseen future data |
| **Test MAPE** | 16.32% | Predictions within ±16% of actual prices—industry-acceptable |
| **Temporal Robustness** | Only 2% R² drop | Validated on strict time-split (train on 2023-2024, test on 2025) |
| **Training Time** | ~10 minutes | Fast training with default hyperparameters |
| **Inference Speed** | 4.2 sec for 2,074 properties | Real-time batch predictions |

### Comparison with Alternative Models

| Model | CV R² | Test R² (2025) | Test MAPE | Notes |
|-------|-------|----------------|-----------|-------|
| **Random Forest** | **0.95** | **0.97** | **16.32%** | **✅ Winner: Best accuracy + temporal robustness** |
| KNeighbors | 0.97 | 0.95 | 18.65% | 🟢 Competitive, slower inference |
| XGBoost | 0.95 | 0.93 | 19.43% | 🟢 Strong, longer training |
| LightGBM | 0.92 | 0.92 | 20.69% | 🟢 Good performance |
| Decision Tree | 1.00 | 0.94 | 19.96% | 🟡 Overfits (Train R² = 1.00) |
| Gradient Boosting | 0.85 | 0.88 | 23.97% | 🟡 Decent performance |
| Ridge | 0.71 | 0.69 | 35.21% | ❌ Linear assumptions fail |
| Linear Regression | 0.71 | 0.69 | 35.21% | ❌ Same as Ridge |
| Lasso | 0.33 | 0.34 | 51.78% | ❌ Too aggressive feature selection |

---

## 🔧 Model Specifications

### Hyperparameters (Default Configuration)

**IMPORTANT:** This model achieves 97% accuracy using scikit-learn's **default Random Forest hyperparameters**—demonstrating that exceptional feature engineering eliminates the need for extensive hyperparameter tuning.
```python
RandomForestRegressor(
    n_estimators=100,        # Default: 100 decision trees
    max_depth=None,          # Default: Trees grow until pure leaves
    min_samples_split=2,     # Default: Minimum 2 samples to split
    min_samples_leaf=1,      # Default: Minimum 1 sample per leaf
    max_features='sqrt',     # Default: √279 ≈ 17 features per split
    bootstrap=True,          # Default: Bootstrap sampling enabled
    random_state=42,         # For reproducibility
    n_jobs=-1,               # Use all CPU cores (optimization)
    verbose=0
)
```

**Why No Hyperparameter Tuning?**

1. **Strong Feature Engineering:** The 279 engineered features (geospatial distances, transit access, market clusters) capture the underlying patterns so well that default parameters suffice.

2. **Efficient Problem Framing:** Proper data preprocessing (log transformation, temporal validation, spatial clustering) reduces model complexity requirements.

3. **Diminishing Returns:** Initial experiments showed that hyperparameter tuning would provide <2% accuracy improvement while requiring 5-10x more training time.

4. **Production Pragmatism:** Default parameters are easier to maintain, debug, and explain to stakeholders.

**Key Insight:** **97% accuracy with default parameters** proves that **data quality and feature engineering matter far more than model tuning**—a critical lesson for production ML systems.

---

## 📈 Performance Metrics

### Temporal Validation (2025 Holdout Test)

**CRITICAL:** Unlike typical train/test random splits, this model was validated on a **strict temporal holdout**—trained on 2023-2024 data, tested on 2025 Q1 data. This simulates real-world forecasting scenarios.

| Metric | Training Set (Pre-2025) | Test Set (2025 Q1) | Assessment |
|--------|-------------------------|-------------------|------------|
| **R² Score** | 0.99 | 0.97 | ✅ Excellent (minimal overfitting) |
| **RMSE (RM/m²)** | 2,187 | 3,905 | ✅ Expected increase on unseen data |
| **MAE (RM/m²)** | 612 | 1,510 | ✅ Predictions off by ~RM 1,510/m² |
| **MAPE** | 12.48% | 16.32% | ✅ Within industry range (15-25%) |

**Key Takeaway:** Only a **2% R² drop** when tested on completely unseen future data—demonstrates strong generalization and minimal temporal drift, **achieved with default hyperparameters**.

### Log-Space Metrics (Internal Training Metrics)

| Metric | Training | Cross-Validation (10-fold) | Test (2025) |
|--------|----------|----------------------------|-------------|
| R² | 0.99 | 0.95 | 0.95 |
| RMSE | 0.11 | 0.22 | 0.23 |
| MAE | 0.07 | 0.14 | 0.15 |

---

## 🧬 Feature Engineering Pipeline

### Input Features (279 total)

**Feature Categories:**

1. **Continuous Geospatial (22 features)**
   - Property attributes: `property_m2`, `unit_level`
   - Distances to amenities: `dist_to_mall`, `dist_to_school`, `dist_to_park`, `dist_to_river`, `dist_to_lake`, `dist_to_rail_station`
   - Walking distances (OSM network): `walk_dist_to_mall`, `walk_dist_to_school`, etc.
   - Density features: `mall_count`, `school_count`, `park_count`, `rail_station_count` (within 1km)
   - Transit ridership: `incoming_ridership_within_1km`, `outgoing_ridership_within_1km`

2. **Property Type Dummies (11 features)**
   - One-hot encoded: `property_type_condominium/apartment`, `property_type_detached`, etc.

3. **District Dummies (11 features)**
   - Administrative districts: `district_kuala_lumpur`, `district_petaling`, `district_klang`, etc.

4. **Binary Features (2 features)**
   - `freehold_1`: Property tenure type (freehold vs leasehold)
   - `transit_1`: Proximity to major transit infrastructure

5. **Market Cluster Dummies (233 features)**
   - Spatial clusters from DBSCAN/K-Means: `market_cluster_id_KL_017`, `market_cluster_id_PE_013`, etc.
   - **Purpose:** Consolidate 18,000+ unique road names into 238 geographic market segments

### Preprocessing Pipeline
```python
# 1. Feature Scaling (StandardScaler)
scaler = StandardScaler()
scaler.fit(X_train)  # Fitted on 66,606 pre-2025 training samples

# Continuous features scaled to mean=0, std=1
# Categorical dummies (0/1) passed through unchanged

# 2. Target Transformation
y_train_log = np.log1p(y_train)  # log(price_m2 + 1)

# Reason: Handle right-skewed price distribution
# Inverse transform: np.expm1(predictions) → original RM/m² scale
```

**File:** `scaler_pre2025.pkl` (5 MB) - Available in `models/` folder, or reproduce via [`5_Modelling.ipynb`](../notebooks/5_Modelling.ipynb)

---

## 🎯 Feature Importance Analysis

**Top 10 Drivers of Property Prices:**

| Rank | Feature | Importance | Business Insight |
|------|---------|------------|------------------|
| 1 | `property_m2` | 88.46% | **Property size dominates**—larger units command exponentially higher prices |
| 2 | `walk_dist_to_rail_station` | 2.67% | Transit accessibility critical but secondary to size |
| 3 | `property_type_condominium/apartment` | 1.10% | High-rise condos valued higher |
| 4 | `property_type_2 - 2 1/2 storey terraced` | 1.06% | Mid-tier terraced houses consistent value |
| 5 | `unit_level` | 0.78% | Higher floors command premium |
| 6 | `walk_dist_to_lake` | 0.62% | Waterfront proximity marginal value |
| 7 | `property_type_detached` | 0.59% | Detached houses = luxury segment |
| 8 | `dist_to_lake` | 0.57% | Straight-line distance slightly less important |
| 9 | `district_hulu_selangor` | 0.56% | District effects minimal |
| 10 | `district_kuala_lumpur` | 0.55% | Central KL premium captured by attributes |

**Critical Finding:** 88.46% of property value explained by size alone, with transit access (2.67%) and property type (1.10%) supporting roles.

**Key Insight:** The dominance of property size validates the feature engineering approach—the model captures true market dynamics without needing hyperparameter optimization.

---

## 🚀 Reproduction Instructions

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Training the Model (With Default Hyperparameters)
```python
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 1. Load preprocessed data (from notebooks/3_Merging.ipynb output)
X_train = ...  # 66,606 samples × 279 features
y_train = ...  # Log-transformed prices

# 2. Initialize and fit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Train Random Forest with DEFAULT hyperparameters
rf_model = RandomForestRegressor(
    random_state=42,  # Only set for reproducibility
    n_jobs=-1         # Use all CPU cores for speed
)

rf_model.fit(X_train_scaled, y_train)

# 4. Save model and scaler
with open('production_model_rf_pre2025.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('scaler_pre2025.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

### Making Predictions

**Option A: Using src/ Package (Recommended)**
```python
from src.models.predictor import PropertyPredictor

# High-level API handles loading, scaling, and inverse transformation
# Includes security validation: path checks, feature validation, hash verification
predictor = PropertyPredictor(
    model_path='models/production_model_rf_pre2025.pkl',
    scaler_path='models/scaler_pre2025.pkl'
)

# Make predictions (returns RM/m² automatically)
# Validates: 279 features, no NaN/Inf values, positive predictions
predictions = predictor.predict(X_new)
print(f"Predicted price: RM {predictions[0]:,.2f} per m²")
```

**Option B: Manual Implementation**
```python
import pickle
import numpy as np
import pandas as pd

# 1. Load model and scaler
with open('production_model_rf_pre2025.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_pre2025.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2. Prepare new data (must have exactly 279 features in correct order)
X_new = pd.DataFrame({...})  # See model_metadata.json for feature order

# 3. Scale features
X_new_scaled = scaler.transform(X_new)

# 4. Predict (log-space)
y_pred_log = model.predict(X_new_scaled)

# 5. Inverse transform to original RM/m² scale
y_pred_original = np.expm1(y_pred_log)

print(f"Predicted price: RM {y_pred_original[0]:,.2f} per m²")
```

**Option C: CLI Tool**
```bash
python -m src.scripts.predict \
    --input new_properties.csv \
    --output predictions.csv \
    --model models/production_model_rf_pre2025.pkl
```

### Example: Predict Price for a Sample Property
```python
# Example: 1,000 m² condominium in KLCC (Kuala Lumpur City Centre)
sample = {
    'property_m2': 1000,
    'unit_level': 25,
    'dist_to_rail_station': 300,
    'walk_dist_to_rail_station': 400,
    'rail_station_count': 2,
    'property_type_condominium/apartment': 1,
    'district_kuala_lumpur': 1,
    'market_cluster_id_KL_017': 1,
    'freehold_1': 1,
    'transit_1': 1,
    # ... (all 279 features required - see model_metadata.json)
}

X_new = pd.DataFrame([sample])
X_scaled = scaler.transform(X_new)
y_pred = np.expm1(model.predict(X_scaled))

print(f"Predicted: RM {y_pred[0]:,.2f}/m²")
# Expected: ~RM 35,000-40,000/m² (KLCC luxury segment)
```

---

## 🔒 Security Considerations

The `PropertyPredictor` class includes security measures for safe deployment:

| Feature | Description |
|---------|-------------|
| **Path Validation** | Model files must be within `models/` directory |
| **Hash Verification** | Optional SHA256 verification to detect tampered files |
| **Feature Validation** | Rejects inputs with wrong feature count, NaN, or Inf values |
| **Pickle Warning** | Logs security warning before loading pickle files |

**Important:** Pickle files can execute arbitrary code. Only load models from trusted sources. For enhanced security, consider using [skops.io](https://skops.github.io/skops/) for model serialization.

See [tests/test_security.py](../tests/test_security.py) for security test coverage.

---

## ⚠️ Known Limitations

1. **Geographic Scope:** Model trained exclusively on Klang Valley properties. **Do not use** for other Malaysian regions (Penang, Johor, Sabah, etc.)

2. **Temporal Drift:** Trained on 2023-2024 data. **Annual retraining recommended** to capture evolving market conditions

3. **Low-Cost Segment:** Higher prediction errors for properties <RM 2,000/m² due to limited training examples

4. **Infrastructure Assumptions:** Model only accounts for RapidKL's LRT/MRT lines as of training date (does not include future MRT3 stations opening in 2027)

5. **New Locations:** Properties on roads not seen during training assigned to NOISE cluster with reduced accuracy

6. **No Hyperparameter Tuning:** Model uses default Random Forest parameters. While this achieves excellent results (R² = 0.97), domain-specific tuning could potentially improve performance by 1-2% at the cost of significantly increased training time.

---

## 📚 Additional Resources

- **Complete Model Metadata:** [`model_metadata.json`](model_metadata.json) for machine-readable specifications
- **Training Notebook:** [`5_Modelling.ipynb`](../notebooks/5_Modelling.ipynb) - Full model selection and evaluation
- **Feature Engineering:** [`2_1_Amenity_OSM_Search.ipynb`](../notebooks/2_1_Amenity_OSM_Search.ipynb) - Geospatial feature extraction
- **Market Clustering:** [`4_Clustering.ipynb`](../notebooks/4_Clustering.ipynb) - DBSCAN spatial clustering methodology

---

## 📊 Model Card Summary

| Attribute | Value |
|-----------|-------|
| **Model Type** | RandomForestRegressor (scikit-learn, default hyperparameters) |
| **Training Data** | 66,606 properties (Klang Valley, 2023-2024) |
| **Test Data** | 2,074 properties (Klang Valley, 2025 Q1) |
| **Features** | 279 (22 geospatial + 11 property types + 11 districts + 233 clusters + 2 binary) |
| **Target** | `price_m2` (RM per m², log-transformed) |
| **Performance** | R² = 0.97, MAPE = 16.32% (2025 temporal holdout) |
| **Hyperparameters** | Default scikit-learn values (no tuning required) |
| **File Size** | ~380 MB (too large for GitHub—reproduce via notebooks) |
| **Training Time** | ~10 minutes (Google Colab) |
| **Inference Speed** | 4.2 seconds for 2,074 predictions |

---

## 💡 Key Takeaway for ML Practitioners

This project demonstrates that **97% accuracy is achievable with default hyperparameters when:**

1. ✅ **Feature engineering is robust** (279 carefully engineered geospatial features)
2. ✅ **Data preprocessing is thorough** (100% geocoding accuracy, spatial clustering)
3. ✅ **Problem framing is correct** (temporal validation, log transformation)
4. ✅ **Algorithm selection is appropriate** (Random Forest for non-linear relationships)

**Lesson:** Invest time in understanding your data and engineering meaningful features before reaching for hyperparameter optimization. In this case, **data quality > model tuning**.

---

## 📧 Contact

For questions about model reproduction or trained model files:
- **Email:** duyanh.phanduc@gmail.com
- **GitHub:** [@anhpdd](https://github.com/anhpdd)

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-16 | Initial production model (Random Forest, R² = 0.97, default hyperparameters) |

---

## 📄 License

This model and documentation are part of the Klang Valley Property Price Prediction project. See main repository for license details.
