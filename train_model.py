import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 60)
print("COFFEE YIELD PREDICTION MODEL TRAINING")
print("=" * 60)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(
    "coffee_ml_validated_dataset_2009_2024.csv"
)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 2. SELECT FEATURES
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


print("\nFeatures used:")

for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)


# --------------------------------------------------
# 3. DEFINE CATEGORICAL AND NUMERICAL COLUMNS
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
# 4. PREPROCESSING
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
# 5. CREATE MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 6. CREATE PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 7. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# --------------------------------------------------
# 9. MAKE PREDICTIONS
# --------------------------------------------------

predictions = pipeline.predict(
    X_test
)


# --------------------------------------------------
# 10. EVALUATE MODEL
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nMAE:  {mae:.4f} MT/ha")
print(f"RMSE: {rmse:.4f} MT/ha")
print(f"R² Score: {r2:.4f}")


# --------------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------------

model_filename = (
    "coffee_yield_model.pkl"
)

joblib.dump(
    pipeline,
    model_filename
)


print("\nModel saved successfully!")

print(
    "Saved as:",
    model_filename
)


# --------------------------------------------------
# 12. SAMPLE PREDICTIONS
# --------------------------------------------------

comparison = X_test.copy()

comparison["Actual_Yield"] = y_test.values

comparison["Predicted_Yield"] = predictions

comparison["Difference"] = (
    comparison["Predicted_Yield"]
    -
    comparison["Actual_Yield"]
)


print("\nSample Predictions:")

print(
    comparison[
        [
            "District",
            "Coffee_Type",
            "Prediction_Season",
            "Actual_Yield",
            "Predicted_Yield",
            "Difference"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)