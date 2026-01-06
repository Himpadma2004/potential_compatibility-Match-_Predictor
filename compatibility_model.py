import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Clean essay text
def prepare_essay_text(df):
    df["essay0"] = df["essay0"].fillna("").astype(str)
    return df

# Fit TF-IDF on all essays
def fit_tfidf_on_all_essays(df):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    vectorizer.fit(df["essay0"])
    return vectorizer

# Compute compatibility between two profiles
def compute_pair_compatibility(df, idxA, idxB, essay_cols, reuse_tfidf=None):
    profileA = df.iloc[idxA]
    profileB = df.iloc[idxB]

    # Example scoring: essay similarity
    if reuse_tfidf:
        tfidf_matrix = reuse_tfidf.transform([profileA["essay0"], profileB["essay0"]])
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    else:
        similarity = 0.0

    # Simple rule: essays similarity * 100
    score = similarity * 100

    return score, {"essay_similarity": similarity}
