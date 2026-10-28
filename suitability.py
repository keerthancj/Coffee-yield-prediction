import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "coffee_ml_validated_dataset_2009_2024.csv"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    return pd.read_csv(DATA_FILE)


# ============================================================
# GET HISTORICAL LOCAL CLIMATE RANGES
# ============================================================

def get_optimal_ranges(
    state,
    district,
    coffee_type,
    quarter
):

    df = load_dataset()


    # Exact local match

    filtered = df[
        (df["State"] == state) &
        (df["District"] == district) &
        (df["Coffee_Type"] == coffee_type) &
        (df["Quarter"] == quarter)
    ]


    # Fallback 1

    if len(filtered) < 4:

        filtered = df[
            (df["State"] == state) &
            (df["District"] == district) &
            (df["Coffee_Type"] == coffee_type)
        ]


    # Fallback 2

    if len(filtered) < 4:

        filtered = df[
            (df["State"] == state) &
            (df["Coffee_Type"] == coffee_type) &
            (df["Quarter"] == quarter)
        ]


    # Fallback 3

    if len(filtered) < 4:

        filtered = df[
            df["Coffee_Type"] == coffee_type
        ]


    return {

        "rainfall_min": float(
            filtered["Quarterly_Rainfall_mm"].quantile(0.25)
        ),

        "rainfall_max": float(
            filtered["Quarterly_Rainfall_mm"].quantile(0.75)
        ),

        "tmax_min": float(
            filtered["Quarterly_Avg_Tmax_C"].quantile(0.25)
        ),

        "tmax_max": float(
            filtered["Quarterly_Avg_Tmax_C"].quantile(0.75)
        ),

        "tmin_min": float(
            filtered["Quarterly_Avg_Tmin_C"].quantile(0.25)
        ),

        "tmin_max": float(
            filtered["Quarterly_Avg_Tmin_C"].quantile(0.75)
        )

    }


# ============================================================
# BIOLOGICAL TEMPERATURE TOLERANCE
# ============================================================

def get_temperature_tolerance(
    coffee_type
):

    if coffee_type == "Arabica":

        return {

            "tmax_preferred_min": 18,
            "tmax_preferred_max": 32,

            "tmax_tolerance_min": 12,
            "tmax_tolerance_max": 35,

            "tmin_preferred_min": 15,
            "tmin_preferred_max": 24,

            "tmin_tolerance_min": 10,
            "tmin_tolerance_max": 28,

            "avg_preferred_min": 18,
            "avg_preferred_max": 24,

            "avg_tolerance_min": 12,
            "avg_tolerance_max": 28
        }


    elif coffee_type == "Robusta":

        return {

            "tmax_preferred_min": 20,
            "tmax_preferred_max": 33,

            "tmax_tolerance_min": 15,
            "tmax_tolerance_max": 36,

            "tmin_preferred_min": 18,
            "tmin_preferred_max": 26,

            "tmin_tolerance_min": 12,
            "tmin_tolerance_max": 30,

            "avg_preferred_min": 22,
            "avg_preferred_max": 28,

            "avg_tolerance_min": 15,
            "avg_tolerance_max": 31
        }


    # Default safe values

    return {

        "tmax_preferred_min": 18,
        "tmax_preferred_max": 32,

        "tmax_tolerance_min": 12,
        "tmax_tolerance_max": 35,

        "tmin_preferred_min": 15,
        "tmin_preferred_max": 26,

        "tmin_tolerance_min": 10,
        "tmin_tolerance_max": 30,

        "avg_preferred_min": 18,
        "avg_preferred_max": 26,

        "avg_tolerance_min": 12,
        "avg_tolerance_max": 30
    }


# ============================================================
# FLEXIBLE LOCAL RAINFALL TOLERANCE
# ============================================================

def calculate_rainfall_score(
    rainfall,
    local_min,
    local_max,
    max_score=30
):

    # Protection

    if local_max <= local_min:

        return max_score / 2


    local_range = local_max - local_min


    # Biological / flexible tolerance zone

    tolerance_min = max(
        0,
        local_min - local_range
    )

    tolerance_max = local_max + (
        local_range * 1.5
    )


    # Perfect score inside local range

    if local_min <= rainfall <= local_max:

        return float(max_score)


    # Within lower tolerance

    if tolerance_min <= rainfall < local_min:

        distance = local_min - rainfall

        penalty_ratio = distance / (
            local_min - tolerance_min
        )

        score = max_score * (
            1 - 0.6 * penalty_ratio
        )

        return round(
            max(0, score),
            1
        )


    # Within upper tolerance

    if local_max < rainfall <= tolerance_max:

        distance = rainfall - local_max

        penalty_ratio = distance / (
            tolerance_max - local_max
        )

        score = max_score * (
            1 - 0.6 * penalty_ratio
        )

        return round(
            max(0, score),
            1
        )


    # Outside tolerance zone

    return 0.0


# ============================================================
# BIOLOGICAL TEMPERATURE SCORE
# ============================================================

def calculate_temperature_score(
    value,
    preferred_min,
    preferred_max,
    tolerance_min,
    tolerance_max,
    max_score=25
):

    # Preferred zone

    if preferred_min <= value <= preferred_max:

        return float(max_score)


    # Below preferred zone but within tolerance

    if tolerance_min <= value < preferred_min:

        distance = preferred_min - value

        zone = preferred_min - tolerance_min

        ratio = distance / zone

        score = max_score * (
            1 - 0.5 * ratio
        )

        return round(
            max(0, score),
            1
        )


    # Above preferred zone but within tolerance

    if preferred_max < value <= tolerance_max:

        distance = value - preferred_max

        zone = tolerance_max - preferred_max

        ratio = distance / zone

        score = max_score * (
            1 - 0.5 * ratio
        )

        return round(
            max(0, score),
            1
        )


    # Outside tolerance

    return 0.0


# ============================================================
# COFFEE TYPE COMPATIBILITY SCORE
# ============================================================

def calculate_coffee_type_score(
    coffee_type,
    avg_temperature
):

    tolerance = get_temperature_tolerance(
        coffee_type
    )


    preferred_min = tolerance[
        "avg_preferred_min"
    ]

    preferred_max = tolerance[
        "avg_preferred_max"
    ]

    tolerance_min = tolerance[
        "avg_tolerance_min"
    ]

    tolerance_max = tolerance[
        "avg_tolerance_max"
    ]


    return calculate_temperature_score(

        avg_temperature,

        preferred_min,

        preferred_max,

        tolerance_min,

        tolerance_max,

        max_score=20

    )


# ============================================================
# FACTOR STATUS
# ============================================================

def get_factor_status(
    score,
    max_score
):

    percentage = (
        score / max_score
    ) * 100


    if percentage >= 85:

        return "Highly Suitable"


    elif percentage >= 65:

        return "Suitable"


    elif percentage >= 40:

        return "Moderately Suitable"


    elif percentage > 0:

        return "Poor"


    return "Unsuitable"


# ============================================================
# NORMAL SUITABILITY LEVEL
# ============================================================

def get_normal_suitability(
    total_score
):

    if total_score >= 85:

        return "Highly Suitable"

    elif total_score >= 65:

        return "Suitable"

    elif total_score >= 40:

        return "Moderately Suitable"

    elif total_score >= 20:

        return "Low Suitability"

    return "Unsuitable"


# ============================================================
# CLIMATE RISK LEVEL
# ============================================================

def get_climate_risk_level(
    rainfall_score,
    tmax_score,
    tmin_score
):

    scores = [
        rainfall_score / 30,
        tmax_score / 25,
        tmin_score / 25
    ]


    zero_count = sum(
        score == 0
        for score in scores
    )


    poor_count = sum(
        score < 0.4
        for score in scores
    )


    if zero_count >= 2:

        return "Critical"


    if zero_count == 1:

        return "High"


    if poor_count >= 2:

        return "Moderate"


    if poor_count == 1:

        return "Low"


    return "Minimal"


# ============================================================
# FINAL SUITABILITY WITH CRITICAL FACTOR RULE
# ============================================================

def get_final_suitability(
    raw_suitability,
    rainfall_score,
    tmax_score,
    tmin_score
):

    critical_count = sum([

        rainfall_score == 0,

        tmax_score == 0,

        tmin_score == 0

    ])


    # Two or more critical climate factors

    if critical_count >= 2:

        return "Unsuitable"


    # One critical factor

    if critical_count == 1:

        # Cannot be highly suitable

        if raw_suitability == "Highly Suitable":

            return "Suitable"

        return raw_suitability


    return raw_suitability


# ============================================================
# GENERATE ANALYSIS
# ============================================================

def generate_analysis(
    rainfall,
    tmax,
    tmin,
    coffee_type,
    ranges,
    rainfall_score,
    tmax_score,
    tmin_score,
    coffee_type_score
):

    recommendations = []

    limiting_factors = []


    # --------------------------------------------------------
    # RAINFALL
    # --------------------------------------------------------

    local_min = ranges["rainfall_min"]

    local_max = ranges["rainfall_max"]

    local_range = local_max - local_min


    tolerance_max = local_max + (
        local_range * 1.5
    )


    tolerance_min = max(
        0,
        local_min - local_range
    )


    if rainfall > tolerance_max:

        limiting_factors.append(
            "Very high rainfall"
        )

        recommendations.append(
            "Rainfall is above the biological tolerance zone. "
            "Strong drainage and waterlogging prevention are essential."
        )


    elif rainfall < tolerance_min:

        limiting_factors.append(
            "Very low rainfall"
        )

        recommendations.append(
            "Rainfall is substantially below the local climatic range. "
            "Consider irrigation, mulching and soil moisture conservation."
        )


    elif rainfall > local_max:

        limiting_factors.append(
            "High rainfall"
        )

        recommendations.append(
            "Rainfall is above the typical local range. "
            "Ensure adequate drainage and monitor soil moisture."
        )


    elif rainfall < local_min:

        limiting_factors.append(
            "Low rainfall"
        )

        recommendations.append(
            "Rainfall is below the typical local range. "
            "Consider irrigation and soil moisture conservation."
        )


    else:

        recommendations.append(
            "Rainfall is within the typical range for this district and quarter."
        )


    # --------------------------------------------------------
    # TEMPERATURE TOLERANCE
    # --------------------------------------------------------

    tolerance = get_temperature_tolerance(
        coffee_type
    )


    # Tmax

    if tmax < tolerance["tmax_tolerance_min"]:

        limiting_factors.append(
            "Very low maximum temperature"
        )

        recommendations.append(
            "Maximum temperature is below the biological tolerance range. "
            "Monitor plants for severe cold stress."
        )


    elif tmax > tolerance["tmax_tolerance_max"]:

        limiting_factors.append(
            "Very high maximum temperature"
        )

        recommendations.append(
            "Maximum temperature is above the biological tolerance range. "
            "Strong shade management and moisture conservation are required."
        )


    elif tmax < tolerance["tmax_preferred_min"]:

        limiting_factors.append(
            "Low maximum temperature"
        )

        recommendations.append(
            "Maximum temperature is below the preferred range. "
            "Monitor plant growth and cold stress."
        )


    elif tmax > tolerance["tmax_preferred_max"]:

        limiting_factors.append(
            "High maximum temperature"
        )

        recommendations.append(
            "Maximum temperature is above the preferred range. "
            "Use shade management and maintain adequate soil moisture."
        )


    else:

        recommendations.append(
            "Maximum temperature is favourable for this coffee type."
        )


    # Tmin

    if tmin < tolerance["tmin_tolerance_min"]:

        limiting_factors.append(
            "Very low minimum temperature"
        )

        recommendations.append(
            "Minimum temperature is below the biological tolerance range. "
            "Cold protection measures may be required."
        )


    elif tmin > tolerance["tmin_tolerance_max"]:

        limiting_factors.append(
            "Very high minimum temperature"
        )

        recommendations.append(
            "Minimum temperature is above the biological tolerance range "
            "and may increase heat and moisture stress."
        )


    elif tmin < tolerance["tmin_preferred_min"]:

        limiting_factors.append(
            "Low minimum temperature"
        )

        recommendations.append(
            "Minimum temperature is slightly below the preferred range. "
            "Monitor plants for cold stress."
        )


    elif tmin > tolerance["tmin_preferred_max"]:

        limiting_factors.append(
            "High minimum temperature"
        )

        recommendations.append(
            "Minimum temperature is above the preferred range. "
            "Monitor plants for heat stress and moisture loss."
        )


    else:

        recommendations.append(
            "Minimum temperature is favourable for this coffee type."
        )


    # --------------------------------------------------------
    # AVERAGE TEMPERATURE
    # --------------------------------------------------------

    avg_temperature = (
        tmax + tmin
    ) / 2


    if avg_temperature < tolerance["avg_tolerance_min"]:

        limiting_factors.append(
            "Very low average temperature"
        )

        recommendations.append(
            f"Average temperature is too low for {coffee_type}. "
            "Monitor plants for severe cold stress."
        )


    elif avg_temperature > tolerance["avg_tolerance_max"]:

        limiting_factors.append(
            "Very high average temperature"
        )

        recommendations.append(
            f"Average temperature is too high for {coffee_type}. "
            "Use shade and moisture conservation practices."
        )


    elif avg_temperature < tolerance["avg_preferred_min"]:

        limiting_factors.append(
            "Low average temperature"
        )

        recommendations.append(
            f"Average temperature is slightly below the preferred range "
            f"for {coffee_type}."
        )


    elif avg_temperature > tolerance["avg_preferred_max"]:

        limiting_factors.append(
            "High average temperature"
        )

        recommendations.append(
            f"Average temperature is slightly above the preferred range "
            f"for {coffee_type}. Consider shade and moisture management."
        )


    else:

        recommendations.append(
            f"Average temperature is favourable for {coffee_type}."
        )


    # --------------------------------------------------------
    # MAJOR LIMITING FACTOR
    # --------------------------------------------------------

    factor_scores = {

        "Rainfall": rainfall_score / 30,

        "Maximum Temperature": tmax_score / 25,

        "Minimum Temperature": tmin_score / 25,

        "Coffee Type Temperature Compatibility":
        coffee_type_score / 20

    }


    major_limiting_factor = min(
        factor_scores,
        key=factor_scores.get
    )


    return {

        "recommendations": recommendations,

        "limiting_factors": list(
            dict.fromkeys(
                limiting_factors
            )
        ),

        "major_limiting_factor":
        major_limiting_factor

    }


# ============================================================
# MAIN SUITABILITY FUNCTION
# ============================================================

def calculate_suitability(

    state,

    district,

    coffee_type,

    quarter,

    rainfall,

    tmax,

    tmin

):


    # --------------------------------------------------------
    # Historical ranges
    # --------------------------------------------------------

    ranges = get_optimal_ranges(

        state,

        district,

        coffee_type,

        quarter

    )


    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    rainfall_score = calculate_rainfall_score(

        rainfall,

        ranges["rainfall_min"],

        ranges["rainfall_max"],

        30

    )


    tolerance = get_temperature_tolerance(
        coffee_type
    )


    tmax_score = calculate_temperature_score(

        tmax,

        tolerance["tmax_preferred_min"],

        tolerance["tmax_preferred_max"],

        tolerance["tmax_tolerance_min"],

        tolerance["tmax_tolerance_max"],

        25

    )


    tmin_score = calculate_temperature_score(

        tmin,

        tolerance["tmin_preferred_min"],

        tolerance["tmin_preferred_max"],

        tolerance["tmin_tolerance_min"],

        tolerance["tmin_tolerance_max"],

        25

    )


    avg_temperature = (
        tmax + tmin
    ) / 2


    coffee_type_score = calculate_coffee_type_score(

        coffee_type,

        avg_temperature

    )


    # --------------------------------------------------------
    # Total score
    # --------------------------------------------------------

    total_score = round(

        rainfall_score +

        tmax_score +

        tmin_score +

        coffee_type_score,

        1

    )


    # --------------------------------------------------------
    # Raw suitability
    # --------------------------------------------------------

    raw_suitability = get_normal_suitability(
        total_score
    )


    # --------------------------------------------------------
    # Final suitability
    # --------------------------------------------------------

    suitability = get_final_suitability(

        raw_suitability,

        rainfall_score,

        tmax_score,

        tmin_score

    )


    # --------------------------------------------------------
    # Climate risk
    # --------------------------------------------------------

    climate_risk = get_climate_risk_level(

        rainfall_score,

        tmax_score,

        tmin_score

    )


    # --------------------------------------------------------
    # Recommendations and analysis
    # --------------------------------------------------------

    analysis = generate_analysis(

        rainfall,

        tmax,

        tmin,

        coffee_type,

        ranges,

        rainfall_score,

        tmax_score,

        tmin_score,

        coffee_type_score

    )


    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return {

        # Scores

        "total_score": total_score,


        # Final result

        "suitability": suitability,


        # Extra intelligent analysis

        "raw_suitability": raw_suitability,

        "climate_risk_level": climate_risk,


        # Factor scores

        "rainfall_score": rainfall_score,

        "tmax_score": tmax_score,

        "tmin_score": tmin_score,

        "coffee_type_score": coffee_type_score,


        # Factor status

        "rainfall_status": get_factor_status(
            rainfall_score,
            30
        ),

        "tmax_status": get_factor_status(
            tmax_score,
            25
        ),

        "tmin_status": get_factor_status(
            tmin_score,
            25
        ),

        "coffee_type_status": get_factor_status(
            coffee_type_score,
            20
        ),


        # Temperature

        "average_temperature": round(
            avg_temperature,
            2
        ),


        # Historical ranges

        "optimal_ranges": ranges,


        # Analysis

        "recommendations":
        analysis["recommendations"],

        "limiting_factors":
        analysis["limiting_factors"],

        "major_limiting_factor":
        analysis["major_limiting_factor"]

    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    result = calculate_suitability(

        state="Karnataka",

        district="Chikkamagaluru",

        coffee_type="Arabica",

        quarter=2,

        rainfall=500,

        tmax=30,

        tmin=20

    )


    print("\nSUITABILITY RESULT\n")

    print(result)
    