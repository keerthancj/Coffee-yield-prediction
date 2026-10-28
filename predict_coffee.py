import pandas as pd
import joblib

from suitability import calculate_suitability


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "coffee_yield_final_model.pkl"

DATA_FILE = "coffee_ml_validated_dataset_2009_2024.csv"


# ============================================================
# DISEASE IMPACT VALUES
# ============================================================

DISEASE_IMPACT = {

    "No Disease": 0.00,

    "Coffee Leaf Rust": 0.80,

    "Coffee Berry Disease": 0.90,

    "White Stem Borer": 0.70,

    "Root Disease": 0.85,

    "Other Disease": 0.50
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)

print("COFFEE YIELD PREDICTION SYSTEM")

print("=" * 60)


print("\nLoading trained model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully!")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def select_from_list(
    title,
    values
):

    print(f"\n{title}")

    for i, value in enumerate(
        values,
        start=1
    ):

        print(
            f"{i}. {value}"
        )


    while True:

        try:

            choice = int(
                input(
                    "\nSelect number: "
                )
            )


            if (
                1 <= choice <= len(values)
            ):

                return values[
                    choice - 1
                ]


            else:

                print(
                    "Invalid selection. Try again."
                )


        except ValueError:

            print(
                "Please enter a valid number."
            )


# ============================================================
# STATE SELECTION
# ============================================================

states = sorted(
    df["State"]
    .unique()
    .tolist()
)


state = select_from_list(

    "Available States:",

    states
)


# ============================================================
# DISTRICT SELECTION
# ============================================================

districts = sorted(

    df[
        df["State"] == state
    ]["District"]

    .unique()

    .tolist()
)


district = select_from_list(

    "Available Districts:",

    districts
)


# ============================================================
# COFFEE TYPE SELECTION
# ============================================================

coffee_types = [

    "Arabica",

    "Robusta"
]


coffee_type = select_from_list(

    "Coffee Types:",

    coffee_types
)


# ============================================================
# QUARTER INPUT
# ============================================================

while True:

    try:

        quarter = int(

            input(
                "\nEnter Quarter (1, 2, 3, or 4): "
            )
        )


        if quarter in [

            1,

            2,

            3,

            4
        ]:

            break


        else:

            print(
                "Quarter must be between 1 and 4."
            )


    except ValueError:

        print(
            "Please enter a valid number."
        )


# ============================================================
# AREA UNIT
# ============================================================

area_units = [

    "Hectare",

    "Acre"
]


area_unit = select_from_list(

    "Area Unit:",

    area_units
)


# ============================================================
# AREA INPUT
# ============================================================

while True:

    try:

        area_input = float(

            input(
                "\nEnter Total Coffee Growing Area: "
            )
        )


        if area_input > 0:

            break


        else:

            print(
                "Area must be greater than zero."
            )


    except ValueError:

        print(
            "Please enter a valid number."
        )


# ============================================================
# CONVERT AREA
# ============================================================

HECTARE_TO_ACRE = 2.47105


if area_unit == "Hectare":

    area_ha = area_input

    area_acres = (

        area_input *

        HECTARE_TO_ACRE
    )


else:

    area_acres = area_input

    area_ha = (

        area_input /

        HECTARE_TO_ACRE
    )


# ============================================================
# RAINFALL INPUT
# ============================================================

while True:

    try:

        rainfall = float(

            input(
                "\nEnter Quarterly Rainfall (mm): "
            )
        )


        if rainfall >= 0:

            break


        else:

            print(
                "Rainfall cannot be negative."
            )


    except ValueError:

        print(
            "Please enter a valid number."
        )


# ============================================================
# MAXIMUM TEMPERATURE
# ============================================================

while True:

    try:

        tmax = float(

            input(
                "\nEnter Average Maximum Temperature (°C): "
            )
        )


        break


    except ValueError:

        print(
            "Please enter a valid number."
        )


# ============================================================
# MINIMUM TEMPERATURE
# ============================================================

while True:

    try:

        tmin = float(

            input(
                "\nEnter Average Minimum Temperature (°C): "
            )
        )


        if tmin <= tmax:

            break


        else:

            print(
                "Minimum temperature cannot be greater "
                "than maximum temperature."
            )


    except ValueError:

        print(
            "Please enter a valid number."
        )


# ============================================================
# PREDICTION SEASON
# ============================================================

prediction_seasons = [

    "Post Blossom",

    "Post Monsoon"
]


prediction_season = select_from_list(

    "Prediction Season:",

    prediction_seasons
)


# ============================================================
# DISEASE SELECTION
# ============================================================

diseases = list(
    DISEASE_IMPACT.keys()
)


disease = select_from_list(

    "Disease Type:",

    diseases
)


# ============================================================
# DISEASE PERCENTAGE
# ============================================================

if disease == "No Disease":

    disease_percentage = 0.0


else:

    while True:

        try:

            disease_percentage = float(

                input(

                    "\nEnter percentage of crop affected "
                    "by disease (0-100): "
                )
            )


            if (

                0 <= disease_percentage <= 100

            ):

                break


            else:

                print(

                    "Enter a value between 0 and 100."
                )


        except ValueError:

            print(

                "Please enter a valid number."
            )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

input_data = pd.DataFrame(

    [

        {

            "Quarter": quarter,

            "State": state,

            "District": district,

            "Area_ha": area_ha,

            "Quarterly_Rainfall_mm": rainfall,

            "Quarterly_Avg_Tmax_C": tmax,

            "Quarterly_Avg_Tmin_C": tmin,

            "Coffee_Type": coffee_type,

            "Prediction_Season": prediction_season

        }

    ]
)


# ============================================================
# YIELD PREDICTION
# ============================================================

print(
    "\nPredicting coffee yield..."
)


predicted_yield = float(

    model.predict(
        input_data
    )[0]
)


# Prevent negative prediction

predicted_yield = max(

    0,

    predicted_yield
)


# ============================================================
# TOTAL PRODUCTION
# ============================================================

predicted_production_mt = (

    predicted_yield *

    area_ha
)


predicted_production_kg = (

    predicted_production_mt *

    1000
)


# ============================================================
# DISEASE IMPACT
# ============================================================

disease_factor = (

    DISEASE_IMPACT[
        disease
    ]
)


production_loss_percentage = (

    disease_percentage *

    disease_factor
)


production_loss_percentage = min(

    production_loss_percentage,

    100
)


production_loss_mt = (

    predicted_production_mt *

    production_loss_percentage / 100
)


production_loss_kg = (

    production_loss_mt *

    1000
)


adjusted_production_mt = (

    predicted_production_mt -

    production_loss_mt
)


adjusted_production_kg = (

    adjusted_production_mt *

    1000
)


# ============================================================
# SUITABILITY ANALYSIS
# ============================================================

print(
    "Analyzing coffee suitability..."
)


suitability_result = calculate_suitability(

    state=state,

    district=district,

    coffee_type=coffee_type,

    quarter=quarter,

    rainfall=rainfall,

    tmax=tmax,

    tmin=tmin
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")

print("=" * 60)

print(
    "COFFEE PREDICTION RESULT"
)

print("=" * 60)


# ------------------------------------------------------------
# USER INPUT SUMMARY
# ------------------------------------------------------------

print(
    "\n--- FARM DETAILS ---"
)


print(
    f"State: {state}"
)


print(
    f"District: {district}"
)


print(
    f"Coffee Type: {coffee_type}"
)


print(
    f"Quarter: {quarter}"
)


print(
    f"Prediction Season: "
    f"{prediction_season}"
)


print(
    f"Disease: {disease}"
)


print(
    f"Area Entered: "
    f"{area_input:.2f} "
    f"{area_unit}"
)


print(
    f"Area in Hectares: "
    f"{area_ha:.4f} ha"
)


print(
    f"Area in Acres: "
    f"{area_acres:.4f} acres"
)


# ------------------------------------------------------------
# YIELD PREDICTION
# ------------------------------------------------------------

print(
    "\n--- YIELD PREDICTION ---"
)


print(
    f"Predicted Yield: "
    f"{predicted_yield:.4f} "
    f"MT/hectare"
)


print(
    f"Predicted Total Production: "
    f"{predicted_production_mt:.2f} "
    f"Metric Tonnes"
)


print(
    f"Predicted Total Production: "
    f"{predicted_production_kg:.2f} "
    f"kg"
)


# ------------------------------------------------------------
# DISEASE IMPACT
# ------------------------------------------------------------

if disease != "No Disease":

    print(
        "\n--- DISEASE IMPACT ---"
    )


    print(
        f"Crop Affected: "
        f"{disease_percentage:.2f}%"
    )


    print(
        f"Estimated Production Loss: "
        f"{production_loss_percentage:.2f}%"
    )


    print(
        f"Estimated Production Loss: "
        f"{production_loss_mt:.2f} "
        f"Metric Tonnes"
    )


    print(
        f"Estimated Production Loss: "
        f"{production_loss_kg:.2f} kg"
    )


print(
    "\n--- ADJUSTED PRODUCTION ---"
)


print(
    f"Adjusted Production: "
    f"{adjusted_production_mt:.2f} "
    f"Metric Tonnes"
)


print(
    f"Adjusted Production: "
    f"{adjusted_production_kg:.2f} kg"
)


# ============================================================
# SUITABILITY RESULTS
# ============================================================

print(
    "\n"
)


print("=" * 60)

print(
    "COFFEE GROWTH SUITABILITY ANALYSIS"
)

print("=" * 60)


print(
    f"Overall Suitability Score: "
    f"{suitability_result['total_score']} / 100"
)

print(
    f"Suitability Result: "
    f"{suitability_result['suitability']}"
)

print(
    f"Raw Score Classification: "
    f"{suitability_result.get('raw_suitability', suitability_result['suitability'])}"
)

print(
    f"Climate Risk Level: "
    f"{suitability_result.get('climate_risk_level', 'Not Available')}"
)

# ------------------------------------------------------------
# FACTOR SCORES
# ------------------------------------------------------------

print(
    "\n--- FACTOR-WISE ANALYSIS ---"
)


print(

    f"Rainfall: "

    f"{suitability_result['rainfall_score']} / 30 "

    f"({suitability_result['rainfall_status']})"
)


print(

    f"Maximum Temperature: "

    f"{suitability_result['tmax_score']} / 25 "

    f"({suitability_result['tmax_status']})"
)


print(

    f"Minimum Temperature: "

    f"{suitability_result['tmin_score']} / 25 "

    f"({suitability_result['tmin_status']})"
)


print(

    f"Coffee Type Compatibility: "

    f"{suitability_result['coffee_type_score']} / 20 "

    f"({suitability_result['coffee_type_status']})"
)


print(

    f"\nAverage Temperature: "

    f"{suitability_result['average_temperature']} °C"
)


# ------------------------------------------------------------
# MAJOR LIMITING FACTOR
# ------------------------------------------------------------

print(

    f"\nMajor Limiting Factor: "

    f"{suitability_result['major_limiting_factor']}"
)


# ------------------------------------------------------------
# LIMITING FACTORS
# ------------------------------------------------------------

if suitability_result[

    "limiting_factors"

]:

    print(

        "\n--- LIMITING FACTORS ---"
    )


    for factor in suitability_result[

        "limiting_factors"

    ]:

        print(

            f"- {factor}"
        )


# ------------------------------------------------------------
# RECOMMENDATIONS
# ------------------------------------------------------------

print(

    "\n--- RECOMMENDATIONS ---"
)


for recommendation in suitability_result[

    "recommendations"

]:

    print(

        f"- {recommendation}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n"
)


print("=" * 60)

print(
    "PREDICTION AND SUITABILITY ANALYSIS COMPLETED"
)

print("=" * 60)