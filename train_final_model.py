import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


print("=" * 60)
print("FINAL COFFEE YIELD MODEL TRAINING")
print("=" * 60)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("\nLoading validated dataset...")

df = pd.read_csv(
    "coffee_ml_validated_dataset_2009_2024.csv"
)

print("Rows:", len(df))


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

features = [
    "Quarter",
    "State",
    "District",
    "Area_ha",
    "Quarterly_Rainfall_mm",
    "Quarterly_Avg_Tmax_C",
    "Quarterly_Avg_Tmin_C",
    "Coffee_Type",
    "Prediction_Season"
]

target = "Yield_MT_per_ha"


X = df[features]
y = df[target]


# --------------------------------------------------
# FEATURE TYPES
# --------------------------------------------------

categorical_features = [
    "State",
    "District",
    "Coffee_Type",
    "Prediction_Season"
]

numerical_features = [
    "Quarter",
    "Area_ha",
    "Quarterly_Rainfall_mm",
    "Quarterly_Avg_Tmax_C",
    "Quarterly_Avg_Tmin_C"
]


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# TRAIN USING ALL AVAILABLE DATA
# --------------------------------------------------

print("\nTraining final model using all data...")
print("Training period: 2009–2024")

pipeline.fit(
    X,
    y
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

filename = "coffee_yield_final_model.pkl"

joblib.dump(
    pipeline,
    filename
)


print("\n" + "=" * 60)
print("FINAL MODEL CREATED SUCCESSFULLY")
print("=" * 60)

print("\nModel file:")
print(filename)

print("\nTraining rows used:")
print(len(df))

print("\nFeatures used:")

for feature in features:
    print("-", feature)