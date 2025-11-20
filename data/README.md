# 📊 Data Documentation

## Dataset Overview

**68,680 residential property transactions** in Malaysia's Klang Valley region (2023-2025), enriched with 19 geospatial features from OpenStreetMap.

| Metric | Value |
|--------|-------|
| **Rows** | 68,680 |
| **Features** | 31 (19 predictors + 12 metadata fields) |
| **Target Variable** | `price_m2` (transaction price per square meter) |
| **Coverage** | Klang Valley districts: Gombak, Petaling, Hulu Langat, Kuala Lumpur, etc. |
| **Time Period** | January 2023 - March 2025 |

---

## Data Sources

### 1. Property Transactions (Base Dataset)
- **Source:** National Property Information Centre (NAPIC), Malaysia
- **URL:** [napic.jpph.gov.my/en/open-sales-data](https://napic.jpph.gov.my/en/open-sales-data)
- **Fields:** Property type, district, road name, floor area, transaction price, transaction date
- **License:** Public data (Malaysia Open Data Initiative)

### 2. Geospatial Features
- **Source:** OpenStreetMap via OSMnx Python library
- **Coverage:** 19 location-based features (distances to transit, schools, malls, parks; amenity density counts)
- **Extraction:** See notebooks `0-3` for automated query pipeline

### 3. Transit Ridership
- **Source:** Prasarana Malaysia (public transit operator)
- **URL:** [data.gov.my/dashboard/rapid-explorer](https://data.gov.my/dashboard/rapid-explorer)
- **Usage:** LRT/MRT station ridership aggregated within 1km of each property

---

## Sample Data

A **1,000-row anonymized sample** is included in this folder:

📄 **`sample_data.csv`** (1,000 properties × 31 features, ~165 KB)

---

## Reproducing the Full Dataset

### Option A: Interactive Notebooks (Recommended for Learning)

Run the notebooks in sequence (total runtime: ~3 hours):
```bash
0_Geocode_Names_to_Way_ID.ipynb        # Extract OSM IDs for roads (~1 hour)
1_1__coord_to_wayid_manual.ipynb       # Manual verification (~30 min)
1_2_prop_validation.ipynb              # Validate road geometries (~30 min)
2_1_Amenity_OSM_search.ipynb           # Query nearby amenities (~45 min)
2_2_Ridership_data_extraction.ipynb    # Add transit ridership (~15 min)
3_Merging.ipynb                        # Combine features (~10 min)
4_Clustering.ipynb                     # Spatial clustering (DBSCAN/K-Means) (~20 min)
5_Modelling.ipynb                      # Feature engineering & training (~30 min)
```

### Option B: Automated Pipeline (Recommended for Production)

Run the complete pipeline from command line:
```bash
# Full pipeline (geocoding → features → clustering → training)
python -m src.scripts.run_full_pipeline --input data/raw/properties.csv

# Or run individual stages:
python -m src.scripts.run_geocoding --input data/raw/properties.csv
python -m src.scripts.run_feature_extraction --input data/interim/geocoded.csv
python -m src.scripts.run_clustering --method auto
```

### Using the src/ Package in Python

```python
from src.data.loaders import load_raw_data, save_interim_data
from src.data.geocoding import geocode_properties
from src.data.validation import validate_property_data
from src.features.geospatial import extract_amenity_features
from src.features.clustering import cluster_market_segments

# Load and process data
df = load_raw_data('data/raw/properties.csv')
df_geocoded = geocode_properties(df)
df_features = extract_amenity_features(df_geocoded)
df_clustered = cluster_market_segments(df_features)

# Save interim results
save_interim_data(df_clustered, 'data/interim/clustered.csv')
```

**Requirements:**
- Python 3.9+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `osmnx`, `geopandas`, `folium`
- OpenStreetMap API access (free, no key required)

---

## Dataset Schema (Key Columns)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `price_m2` | float | **Target:** Transaction price per m² (RM) | 5520.83 |
| `property_type` | str | Property category | "Semi-detached" |
| `property_m2` | int | Floor area (m²) | 96 |
| `district` | str | Administrative district | "Gombak" |
| `market_cluster_id` | str | Spatial cluster (DBSCAN/K-Means) | "GO_004" |
| `rail_station_count` | int | Rapid KL train station within 1km radius | 1 |
| `walk_dist_to_rail_station` | float | Walking distance to nearest Rapid KL train station (km) | 3.8 |
| `dist_to_rail_station` | float | Driving distance to nearest Rapid KL train station (km) | 6,2 |
| `outgoing_ridership_within_1km` | int | Transit departures at nearby stations | 236,839 |
| `incoming_ridership_within_1km` | int | Transit arrivals at nearby stations | 120,923 |

**Full schema:** 45 columns total (see `sample_data.csv` for complete list)

---

## Privacy & Ethics

- ✅ No owner names, contact details, or exact property addresses
- ✅ Complies with Malaysia's Personal Data Protection Act (PDPA 2010)
- ⚠️ **For academic/portfolio use only** (not for commercial valuation)

---

## Questions?

**Anh Phan (Robin)**  
📧 duyanh.phanduc@gmail.com  

💼 [linkedin.com/in/phan-đức-duy-anh](https://www.linkedin.com/in/phan-%C4%91%E1%BB%A3c-duy-anh/)





