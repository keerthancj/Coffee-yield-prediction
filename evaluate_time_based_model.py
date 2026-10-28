import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 60)
print("TIME-BASED COFFEE YIELD MODEL VALIDATION")
print("=" * 60)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(
    "coffee_ml_validated_dataset_2009_2024.csv"
)

print("Dataset loaded successfully.")
print("Total rows:", len(df))


# --------------------------------------------------
# 2. SELECT FEATURES AND TARGET
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


# --------------------------------------------------
# 3. TIME-BASED SPLIT
# --------------------------------------------------

# Train using historical data
train_df = df[
    df["Year"] <= 2021
].copy()

# Test using completely unseen future years
test_df = df[
    df["Year"] >= 2022
].copy()


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


print("\n" + "=" * 60)
print("TIME-BASED DATA SPLIT")
print("=" * 60)

print("\nTraining years:")
print(
    sorted(
        train_df["Year"].unique()
    )
)

print("\nTesting years:")
print(
    sorted(
        test_df["Year"].unique()
    )
)

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))


# --------------------------------------------------
# 4. DEFINE FEATURE TYPES
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
# 5. PREPROCESSING
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
# 6. CREATE MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 7. CREATE PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------

print("\nTraining model using 2009–2021 data...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# --------------------------------------------------
# 9. PREDICT FUTURE YEARS
# --------------------------------------------------

print("\nPredicting 2022–2024 data...")

predictions = pipeline.predict(
    X_test
)


# --------------------------------------------------
# 10. MODEL EVALUATION
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
print("TIME-BASED MODEL PERFORMANCE")
print("=" * 60)

print(f"\nMAE:  {mae:.4f} MT/ha")
print(f"RMSE: {rmse:.4f} MT/ha")
print(f"R² Score: {r2:.4f}")


# --------------------------------------------------
# 11. PERFORMANCE BY YEAR
# --------------------------------------------------

results = test_df[
    [
        "Year",
        "District",
        "Coffee_Type",
        "Prediction_Season",
        "Yield_MT_per_ha"
    ]
].copy()

results["Predicted_Yield"] = predictions


print("\n" + "=" * 60)
print("PERFORMANCE BY YEAR")
print("=" * 60)


for year in sorted(
    results["Year"].unique()
):

    year_data = results[
        results["Year"] == year
    ]

    year_mae = mean_absolute_error(
        year_data["Yield_MT_per_ha"],
        year_data["Predicted_Yield"]
    )

    year_rmse = np.sqrt(
        mean_squared_error(
            year_data["Yield_MT_per_ha"],
            year_data["Predicted_Yield"]
        )
    )

    year_r2 = r2_score(
        year_data["Yield_MT_per_ha"],
        year_data["Predicted_Yield"]
    )

    print(f"\nYear: {year}")
    print(f"MAE:  {year_mae:.4f}")
    print(f"RMSE: {year_rmse:.4f}")
    print(f"R²:   {year_r2:.4f}")


# --------------------------------------------------
# 12. SAMPLE PREDICTIONS
# --------------------------------------------------

results["Difference"] = (
    results["Predicted_Yield"]
    -
    results["Yield_MT_per_ha"]
)


print("\n" + "=" * 60)
print("SAMPLE FUTURE YEAR PREDICTIONS")
print("=" * 60)


print(
    results[
        [
            "Year",
            "District",
            "Coffee_Type",
            "Prediction_Season",
            "Yield_MT_per_ha",
            "Predicted_Yield",
            "Difference"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print("\n" + "=" * 60)
print("TIME-BASED VALIDATION COMPLETED")
print("=" * 60)