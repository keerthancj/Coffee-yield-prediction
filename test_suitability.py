from suitability import calculate_suitability


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    test_number,
    test_name,
    result
):

    print("\n")
    print("=" * 70)

    print(
        f"TEST {test_number}: {test_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print()

    print(
        f"Overall Score: "
        f"{result['total_score']} / 100"
    )

    print(
        f"Final Suitability: "
        f"{result['suitability']}"
    )

    print(
        f"Raw Score Classification: "
        f"{result['raw_suitability']}"
    )

    print(
        f"Climate Risk Level: "
        f"{result['climate_risk_level']}"
    )


    # --------------------------------------------------------
    # Factor scores
    # --------------------------------------------------------

    print()

    print(
        "---------------- FACTOR SCORES ----------------"
    )

    print()

    print(
        f"Rainfall: "
        f"{result['rainfall_score']} / 30 "
        f"({result['rainfall_status']})"
    )

    print(
        f"Maximum Temperature: "
        f"{result['tmax_score']} / 25 "
        f"({result['tmax_status']})"
    )

    print(
        f"Minimum Temperature: "
        f"{result['tmin_score']} / 25 "
        f"({result['tmin_status']})"
    )

    print(
        f"Coffee Type Temperature Compatibility: "
        f"{result['coffee_type_score']} / 20 "
        f"({result['coffee_type_status']})"
    )


    # --------------------------------------------------------
    # Average temperature
    # --------------------------------------------------------

    print()

    print(
        f"Average Temperature: "
        f"{result['average_temperature']} °C"
    )


    # --------------------------------------------------------
    # Major limiting factor
    # --------------------------------------------------------

    print()

    print(
        f"Major Limiting Factor: "
        f"{result['major_limiting_factor']}"
    )


    # --------------------------------------------------------
    # Historical ranges
    # --------------------------------------------------------

    ranges = result[
        "optimal_ranges"
    ]


    print()

    print(
        "---------------- OPTIMAL HISTORICAL RANGES ----------------"
    )

    print()

    print(
        f"Rainfall Range: "
        f"{ranges['rainfall_min']:.2f} - "
        f"{ranges['rainfall_max']:.2f} mm"
    )

    print(
        f"Maximum Temperature Range: "
        f"{ranges['tmax_min']:.2f} - "
        f"{ranges['tmax_max']:.2f} °C"
    )

    print(
        f"Minimum Temperature Range: "
        f"{ranges['tmin_min']:.2f} - "
        f"{ranges['tmin_max']:.2f} °C"
    )


    # --------------------------------------------------------
    # Limiting factors
    # --------------------------------------------------------

    print()

    print(
        "---------------- LIMITING FACTORS ----------------"
    )

    print()

    if result[
        "limiting_factors"
    ]:

        for factor in result[
            "limiting_factors"
        ]:

            print(
                f"- {factor}"
            )

    else:

        print(
            "No major limiting factors found."
        )


    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    print()

    print(
        "---------------- RECOMMENDATIONS ----------------"
    )

    print()

    for recommendation in result[
        "recommendations"
    ]:

        print(
            f"- {recommendation}"
        )


# ============================================================
# MAIN TESTS
# ============================================================

print()

print("=" * 70)

print(
    "COFFEE SUITABILITY SYSTEM - COMPLETE TEST"
)

print("=" * 70)


# ============================================================
# TEST 1
# ARABICA MODERATE CONDITIONS
# ============================================================

result1 = calculate_suitability(

    state="Karnataka",

    district="Chikkamagaluru",

    coffee_type="Arabica",

    quarter=2,

    rainfall=500,

    tmax=30,

    tmin=20

)


display_result(

    1,

    "ARABICA - MODERATE CONDITIONS",

    result1

)


# ============================================================
# TEST 2
# ARABICA HOT AND DRY
# ============================================================

result2 = calculate_suitability(

    state="Karnataka",

    district="Chikkamagaluru",

    coffee_type="Arabica",

    quarter=2,

    rainfall=100,

    tmax=38,

    tmin=28

)


display_result(

    2,

    "ARABICA - HOT AND DRY CONDITIONS",

    result2

)


# ============================================================
# TEST 3
# ROBUSTA HIGH RAINFALL
# ============================================================

result3 = calculate_suitability(

    state="Karnataka",

    district="Kodagu",

    coffee_type="Robusta",

    quarter=3,

    rainfall=2500,

    tmax=30,

    tmin=20

)


display_result(

    3,

    "ROBUSTA - HIGH RAINFALL CONDITIONS",

    result3

)


# ============================================================
# TEST 4
# ARABICA LOW TEMPERATURE
# ============================================================

result4 = calculate_suitability(

    state="Karnataka",

    district="Chikkamagaluru",

    coffee_type="Arabica",

    quarter=4,

    rainfall=150,

    tmax=18,

    tmin=14

)


display_result(

    4,

    "ARABICA - LOW TEMPERATURE CONDITIONS",

    result4

)


# ============================================================
# TEST 5
# ROBUSTA FAVOURABLE
# ============================================================

result5 = calculate_suitability(

    state="Kerala",

    district="Wayanad",

    coffee_type="Robusta",

    quarter=2,

    rainfall=800,

    tmax=30,

    tmin=24

)


display_result(

    5,

    "ROBUSTA - FAVOURABLE CONDITIONS",

    result5

)


# ============================================================
# SUMMARY
# ============================================================

print("\n")

print("=" * 70)

print(
    "TEST SUMMARY"
)

print("=" * 70)


tests = [

    (
        "Test 1 - Arabica Moderate",
        result1
    ),

    (
        "Test 2 - Arabica Hot/Dry",
        result2
    ),

    (
        "Test 3 - Robusta High Rainfall",
        result3
    ),

    (
        "Test 4 - Arabica Cold",
        result4
    ),

    (
        "Test 5 - Robusta Favourable",
        result5
    )

]


for test_name, result in tests:

    print()

    print(test_name)

    print(
        f"Score: "
        f"{result['total_score']} / 100"
    )

    print(
        f"Final Suitability: "
        f"{result['suitability']}"
    )

    print(
        f"Climate Risk: "
        f"{result['climate_risk_level']}"
    )


print("\n")

print("=" * 70)

print(
    "ALL SUITABILITY TESTS COMPLETED SUCCESSFULLY"
)

print("=" * 70)