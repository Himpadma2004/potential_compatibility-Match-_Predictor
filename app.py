from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# import your compatibility function
from compatibility_model import compute_pair_compatibility, fit_tfidf_on_all_essays, prepare_essay_text

app = Flask(__name__)

# Load your dataset and preprocessing
df = pd.read_csv("okcupid_profiles.csv")

# Keep only selected columns
df_selected = df[[
    "age", "sex", "orientation", "diet", "drinks", "drugs",
    "education", "job", "pets", "religion", "smokes", "essay0"
]].copy()

# Prepare text features (essays)
df_selected = prepare_essay_text(df_selected)

# Fit TF-IDF for essays once
tfidf_vec = fit_tfidf_on_all_essays(df_selected)


@app.route("/", methods=["GET", "POST"])
def index():
    compatibility = None

    if request.method == "POST":
        # Get form data
        userA = {
            "age": int(request.form.get("ageA")),
            "sex": request.form.get("sexA"),
            "orientation": request.form.get("orientationA"),
            "diet": request.form.get("dietA"),
            "drinks": request.form.get("drinksA"),
            "drugs": request.form.get("drugsA"),
            "education": request.form.get("educationA"),
            "job": request.form.get("jobA"),
            "pets": request.form.get("petsA"),
            "religion": request.form.get("religionA"),
            "smokes": request.form.get("smokesA"),
            "essay0": request.form.get("essayA")
        }

        userB = {
            "age": int(request.form.get("ageB")),
            "sex": request.form.get("sexB"),
            "orientation": request.form.get("orientationB"),
            "diet": request.form.get("dietB"),
            "drinks": request.form.get("drinksB"),
            "drugs": request.form.get("drugsB"),
            "education": request.form.get("educationB"),
            "job": request.form.get("jobB"),
            "pets": request.form.get("petsB"),
            "religion": request.form.get("religionB"),
            "smokes": request.form.get("smokesB"),
            "essay0": request.form.get("essayB")
        }

        # Convert dicts to pandas Series for compatibility function
        profileA = pd.Series(userA)
        profileB = pd.Series(userB)

        score, _ = compute_pair_compatibility(
            pd.DataFrame([profileA, profileB]),
            idxA=0, idxB=1,
            essay_cols=["essay0"],
            reuse_tfidf=tfidf_vec
        )
        compatibility = f"{score:.2f}%"

    return render_template("index.html", compatibility=compatibility)


if __name__ == "__main__":
    app.run(debug=True)
