from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

import pandas as pd
import joblib

from suitability import calculate_suitability


# ============================================================
# FLASK APP & AUTHENTICATION CONFIG
# ============================================================

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "facet_coffee_yield_glass_key_2026")

DB_FILE = "users.db"

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Pre-seed demo user if not exists
    c.execute("SELECT id FROM users WHERE username = 'admin'")
    if not c.fetchone():
        c.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("admin", "admin@facetcoffee.ai", generate_password_hash("password123"))
        )
    conn.commit()
    conn.close()

init_db()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "coffee_yield_final_model.pkl"

DATA_FILE = "coffee_ml_validated_dataset_2009_2024.csv"

HECTARE_TO_ACRE = 2.47105


# ============================================================
# DISEASE IMPACT
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
# LOAD MODEL AND DATA
# ============================================================

print("Loading Coffee Yield Prediction Model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully!")


print("Loading dataset...")

df = pd.read_csv(
    DATA_FILE
)

print("Dataset loaded successfully!")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    states = sorted(
        df["State"]
        .dropna()
        .unique()
        .tolist()
    )

    logged_in = "user_id" in session
    username = session.get("username", "")

    return render_template(
        "index.html",
        states=states,
        diseases=list(DISEASE_IMPACT.keys()),
        logged_in=logged_in,
        username=username
    )


# ============================================================
# AUTHENTICATION ROUTES (REGISTER, LOGIN, LOGOUT, STATUS)
# ============================================================

@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.form if request.form else request.get_json(silent=True) or {}
        username = str(data.get("username", "")).strip()
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", "")).strip()

        if not username or not email or not password:
            return jsonify({"success": False, "error": "Username, email, and password are required."}), 400

        if len(username) < 3:
            return jsonify({"success": False, "error": "Username must be at least 3 characters."}), 400

        if len(password) < 6:
            return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400

        conn = get_db()
        c = conn.cursor()

        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        if c.fetchone():
            conn.close()
            return jsonify({"success": False, "error": "Username is already registered."}), 400

        c.execute("SELECT id FROM users WHERE email = ?", (email,))
        if c.fetchone():
            conn.close()
            return jsonify({"success": False, "error": "Email is already registered."}), 400

        pw_hash = generate_password_hash(password)
        c.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, pw_hash)
        )
        user_id = c.lastrowid
        conn.commit()
        conn.close()

        session["user_id"] = user_id
        session["username"] = username

        return jsonify({
            "success": True,
            "message": "Account created successfully!",
            "username": username
        })
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.form if request.form else request.get_json(silent=True) or {}
        identifier = str(data.get("identifier", "")).strip()
        password = str(data.get("password", "")).strip()

        if not identifier or not password:
            return jsonify({"success": False, "error": "Please enter your username/email and password."}), 400

        conn = get_db()
        c = conn.cursor()
        c.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ? OR email = ?",
            (identifier, identifier.lower())
        )
        user = c.fetchone()
        conn.close()

        if not user or not check_password_hash(user["password_hash"], password):
            return jsonify({"success": False, "error": "Invalid username or password."}), 401

        session["user_id"] = user["id"]
        session["username"] = user["username"]

        return jsonify({
            "success": True,
            "message": "Signed in successfully!",
            "username": user["username"]
        })
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"success": True, "message": "Logged out successfully."})
    return redirect(url_for("home"))


@app.route("/user/status")
def user_status():
    return jsonify({
        "logged_in": "user_id" in session,
        "username": session.get("username", "")
    })


# ============================================================
# GET DISTRICTS
# ============================================================

@app.route("/districts")
def get_districts():

    state = request.args.get(
        "state"
    )


    districts = sorted(

        df[
            df["State"] == state
        ]["District"]

        .dropna()

        .unique()

        .tolist()
    )


    return jsonify(
        districts
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # CHECK AUTHENTICATION
        # ----------------------------------------------------
        if "user_id" not in session:
            return jsonify({
                "success": False,
                "error": "Authentication required. Please sign in or register to run predictions."
            }), 401

        # ----------------------------------------------------
        # GET FORM DATA
        # ----------------------------------------------------

        state = request.form[
            "state"
        ]


        district = request.form[
            "district"
        ]


        coffee_type = request.form[
            "coffee_type"
        ]


        quarter = int(

            request.form[
                "quarter"
            ]
        )


        area_input = float(

            request.form[
                "area"
            ]
        )


        area_unit = request.form[
            "area_unit"
        ]


        rainfall = float(

            request.form[
                "rainfall"
            ]
        )


        tmax = float(

            request.form[
                "tmax"
            ]
        )


        tmin = float(

            request.form[
                "tmin"
            ]
        )


        prediction_season = request.form[
            "prediction_season"
        ]


        disease = request.form[
            "disease"
        ]


        disease_percentage = float(

            request.form.get(

                "disease_percentage",

                0
            )
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if area_input <= 0:

            return jsonify({

                "success": False,

                "error":
                "Area must be greater than zero."

            })


        if rainfall < 0:

            return jsonify({

                "success": False,

                "error":
                "Rainfall cannot be negative."

            })


        if tmin > tmax:

            return jsonify({

                "success": False,

                "error":
                "Minimum temperature cannot be greater than maximum temperature."

            })


        if not (
            0 <= disease_percentage <= 100
        ):

            return jsonify({

                "success": False,

                "error":
                "Disease percentage must be between 0 and 100."

            })


        # ----------------------------------------------------
        # AREA CONVERSION
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # PREPARE MODEL INPUT
        # ----------------------------------------------------

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

                    "Prediction_Season":
                    prediction_season

                }

            ]
        )


        # ----------------------------------------------------
        # PREDICT YIELD
        # ----------------------------------------------------

        predicted_yield = float(

            model.predict(
                input_data
            )[0]
        )


        predicted_yield = max(

            0,

            predicted_yield
        )


        # ----------------------------------------------------
        # TOTAL PRODUCTION
        # ----------------------------------------------------

        predicted_production_mt = (

            predicted_yield *

            area_ha
        )


        predicted_production_kg = (

            predicted_production_mt *

            1000
        )


        # ----------------------------------------------------
        # DISEASE IMPACT
        # ----------------------------------------------------

        if disease == "No Disease":

            disease_percentage = 0


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

            production_loss_percentage /

            100
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


        # ----------------------------------------------------
        # SUITABILITY ANALYSIS
        # ----------------------------------------------------

        suitability_result = calculate_suitability(

            state=state,

            district=district,

            coffee_type=coffee_type,

            quarter=quarter,

            rainfall=rainfall,

            tmax=tmax,

            tmin=tmin
        )


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return jsonify({

            "success": True,


            # FARM DETAILS

            "farm": {

                "state": state,

                "district": district,

                "coffee_type": coffee_type,

                "quarter": quarter,

                "prediction_season":
                prediction_season,

                "disease": disease,

                "area_input":
                round(area_input, 2),

                "area_unit":
                area_unit,

                "area_ha":
                round(area_ha, 4),

                "area_acres":
                round(area_acres, 4)
            },


            # YIELD

            "yield": {

                "predicted_yield":
                round(predicted_yield, 4),

                "predicted_production_mt":
                round(predicted_production_mt, 2),

                "predicted_production_kg":
                round(predicted_production_kg, 2)
            },


            # DISEASE

            "disease": {

                "crop_affected":
                round(disease_percentage, 2),

                "loss_percentage":
                round(
                    production_loss_percentage,
                    2
                ),

                "loss_mt":
                round(
                    production_loss_mt,
                    2
                ),

                "loss_kg":
                round(
                    production_loss_kg,
                    2
                )
            },


            # ADJUSTED PRODUCTION

            "adjusted_production": {

                "mt":
                round(
                    adjusted_production_mt,
                    2
                ),

                "kg":
                round(
                    adjusted_production_kg,
                    2
                )
            },


            # SUITABILITY

            "suitability":
            suitability_result

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(
                error
            )

        })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5001))
    print(f"\n☕ Starting CoffeeAI Studio on http://127.0.0.1:{port}\n")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=port
    )