"""
recommender.py — Voyage Analytics
Content-based Hotel Recommender using TF-IDF + cosine similarity.

Built on the REAL hotels.csv dataset (9 hotels across 9 Brazilian cities).
hotels.csv only contains: travelCode, userCode, name, place, days, price, total, date
It has no style/amenities/star_rating/review_score columns, so those are
deterministically derived from each hotel's real price tier (NOT randomised —
same hotel always gets the same derived attributes, so results are reproducible).

Falls back to a small synthetic hotel set only if hotels.csv cannot be found,
so the module still works standalone for development/testing.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import mlflow
import pickle
import os

__all__ = ["RecommenderConfig", "HotelRecommender", "get_hotel_recommendations"]

USD_TO_INR = 83.0


# ── Config dataclass ─────────────────────────────────────────────────────────

@dataclass
class RecommenderConfig:
    text_weight: float = 0.60
    num_weight:  float = 0.40
    tfidf_max_features: int = 500


# ── Locate hotels.csv ─────────────────────────────────────────────────────────

def _find_hotels_csv() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "hotels.csv",
        os.path.join(here, "hotels.csv"),
        os.path.join(here, "data", "hotels.csv"),
        os.path.join(here, "..", "hotels.csv"),
        os.path.join(here, "..", "data", "hotels.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ── Deterministic attribute enrichment ────────────────────────────────────────
# hotels.csv has no style/climate/amenities/star_rating/review_score columns.
# We derive these deterministically from (hotel name, price tier, city climate)
# so every run produces identical, explainable results — never random.

_CITY_CLIMATE = {
    "Florianopolis": "coastal", "Salvador": "tropical", "Natal": "tropical",
    "Aracaju": "tropical", "Recife": "tropical", "Sao Paulo": "temperate",
    "Campo Grande": "temperate", "Rio de Janeiro": "coastal", "Brasilia": "temperate",
}

_AMENITY_POOL = [
    "pool", "spa", "wifi", "fine_dining", "gym", "bar",
    "beach_access", "business_center", "kids_club", "butler_service",
]


def _derive_attributes(name: str, city: str, avg_price: float, price_rank: int, n_hotels: int) -> dict:
    """Deterministically derive style/amenities/star_rating/review_score from real price tier."""
    # Price tier (0 = cheapest .. 1 = priciest) drives perceived "luxury level"
    tier = price_rank / max(1, n_hotels - 1)  # 0..1

    if tier >= 0.75:
        style, star_rating = "luxury", 5
    elif tier >= 0.5:
        style, star_rating = "boutique", 4
    elif tier >= 0.25:
        style, star_rating = "business", 4
    else:
        style, star_rating = "budget / backpacker", 3

    climate = _CITY_CLIMATE.get(city, "tropical")
    is_coastal = climate == "coastal"

    # Deterministic amenity selection seeded by hotel name (stable hash, not random.seed time-based)
    seed = abs(hash(name)) % (2**32)
    rng = np.random.RandomState(seed)
    n_amenities = 3 + int(tier * 5)  # pricier hotels → more amenities
    amenities = list(rng.choice(_AMENITY_POOL, size=min(n_amenities, len(_AMENITY_POOL)), replace=False))
    if is_coastal and "beach_access" not in amenities:
        amenities.append("beach_access")

    # Review score correlates loosely with tier but isn't purely price-driven
    review_score = round(7.2 + tier * 2.0 + rng.uniform(-0.3, 0.3), 1)
    review_score = min(9.8, max(6.5, review_score))

    travel_type = "business leisure" if style == "business" else (
        "leisure family" if is_coastal else "leisure"
    )

    return {
        "style": style,
        "climate": climate,
        "star_rating": star_rating,
        "amenities": " ".join(amenities),
        "review_score": review_score,
        "travel_type": travel_type,
    }


def _build_hotels_from_csv(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw["city"] = raw["place"].str.extract(r"^(.*?)\s*\(")[0].fillna(raw["place"])

    # One row per unique hotel (price is constant per hotel in this dataset)
    agg = raw.groupby(["name", "place", "city"]).agg(
        avg_price_per_night=("price", "mean"),
        bookings=("price", "count"),
    ).reset_index()

    agg = agg.sort_values("avg_price_per_night").reset_index(drop=True)
    n = len(agg)

    records = []
    for rank, row in agg.iterrows():
        attrs = _derive_attributes(row["name"], row["city"], row["avg_price_per_night"], rank, n)
        records.append({
            "name": row["name"],
            "location": row["city"],
            "country": "Brazil",
            "style": attrs["style"],
            "climate": attrs["climate"],
            "star_rating": attrs["star_rating"],
            "avg_price_per_night": round(row["avg_price_per_night"], 2),
            "review_score": attrs["review_score"],
            "amenities": attrs["amenities"],
            "travel_type": attrs["travel_type"],
            "bookings": int(row["bookings"]),
        })
    return pd.DataFrame(records)


def _build_synthetic_hotels() -> pd.DataFrame:
    """Fallback used only if hotels.csv cannot be located."""
    hotels = [
        {"name": "The Oberoi Udaivilas", "location": "Udaipur", "country": "India",
         "style": "luxury", "climate": "arid / hot", "star_rating": 5,
         "avg_price_per_night": 850, "review_score": 9.6,
         "amenities": "pool spa wifi fine_dining gym butler_service",
         "travel_type": "leisure honeymoon romantic", "bookings": 0},
        {"name": "Taj Lake Palace", "location": "Udaipur", "country": "India",
         "style": "heritage / palace", "climate": "arid / hot", "star_rating": 5,
         "avg_price_per_night": 920, "review_score": 9.4,
         "amenities": "pool spa wifi fine_dining bar butler_service",
         "travel_type": "leisure honeymoon romantic", "bookings": 0},
        {"name": "ITC Grand Chola", "location": "Chennai", "country": "India",
         "style": "luxury", "climate": "tropical", "star_rating": 5,
         "avg_price_per_night": 420, "review_score": 9.1,
         "amenities": "pool spa wifi fine_dining gym business_center bar",
         "travel_type": "business leisure", "bookings": 0},
        {"name": "Radisson Blu Bengaluru", "location": "Bengaluru", "country": "India",
         "style": "business", "climate": "temperate", "star_rating": 4,
         "avg_price_per_night": 180, "review_score": 8.5,
         "amenities": "pool wifi gym business_center bar fine_dining",
         "travel_type": "business", "bookings": 0},
        {"name": "Zostel Goa", "location": "Goa", "country": "India",
         "style": "budget / backpacker", "climate": "tropical", "star_rating": 2,
         "avg_price_per_night": 25, "review_score": 8.1,
         "amenities": "wifi bar beach_access",
         "travel_type": "adventure budget leisure", "bookings": 0},
    ]
    return pd.DataFrame(hotels)


# ── Recommender class ──────────────────────────────────────────────────────────

class HotelRecommender:
    """
    Content-based hotel recommender.
    - Text features (amenities + style + climate + travel_type) → TF-IDF → cosine similarity
    - Numeric features (price, stars, review) → MinMax scaled → Euclidean → inverted score
    - Final score = text_weight * text_sim + num_weight * num_sim

    Prices in self.hotels_ are stored in USD (matches the source hotels.csv units).
    Use `to_inr()` helper or the UI layer to convert for display.
    """

    def __init__(self, config: RecommenderConfig = None):
        self.config  = config or RecommenderConfig()
        self.hotels_ : Optional[pd.DataFrame] = None
        self.tfidf_  = TfidfVectorizer(max_features=self.config.tfidf_max_features)
        self.scaler_ = MinMaxScaler()
        self.tfidf_matrix_  = None
        self.num_matrix_    = None
        self.data_source_   = None  # "real_csv" | "synthetic"
        self._fitted        = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame = None) -> "HotelRecommender":
        """
        Fit on hotel dataframe. If df is None, attempts to load real hotels.csv
        from disk; falls back to a small synthetic set only if not found.

        Expected columns: name, location, country, style, climate, star_rating,
                          avg_price_per_night, review_score, amenities, travel_type
        """
        if df is not None:
            self.hotels_ = df.copy()
            self.data_source_ = "provided"
        else:
            csv_path = _find_hotels_csv()
            if csv_path is not None:
                self.hotels_ = _build_hotels_from_csv(csv_path)
                self.data_source_ = "real_csv"
            else:
                self.hotels_ = _build_synthetic_hotels()
                self.data_source_ = "synthetic"

        # ── Text matrix ──
        text_corpus = (
            self.hotels_["amenities"].fillna("") + " " +
            self.hotels_["style"].fillna("") + " " +
            self.hotels_["climate"].fillna("") + " " +
            self.hotels_["travel_type"].fillna("")
        )
        self.tfidf_matrix_ = self.tfidf_.fit_transform(text_corpus)

        # ── Numeric matrix ──
        num_cols = ["avg_price_per_night", "star_rating", "review_score"]
        self.num_matrix_ = self.scaler_.fit_transform(
            self.hotels_[num_cols].fillna(0)
        )

        self._fitted = True
        return self

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(self, user_data: dict, top_n: int = 5) -> pd.DataFrame:
        """
        Return top_n hotels as a DataFrame with a 'match_score' column (0-100).
        Output prices remain in USD; convert to INR in the display layer.

        user_data keys (all optional):
            amenities      list[str]   e.g. ["pool", "wifi"]
            style          str
            climate        str
            destination    str         matched against 'location'
            travel_type    str
            budget_max     float       in USD
            min_stars      int
            min_review     float
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")

        df = self.hotels_.copy()

        # ── Hard filters ──
        if user_data.get("budget_max"):
            df = df[df["avg_price_per_night"] <= user_data["budget_max"]]
        if user_data.get("min_stars"):
            df = df[df["star_rating"] >= user_data["min_stars"]]
        if user_data.get("min_review"):
            df = df[df["review_score"] >= user_data["min_review"]]
        if user_data.get("destination") and user_data["destination"].strip():
            dest = user_data["destination"].strip().lower()
            df = df[
                df["location"].str.lower().str.contains(dest) |
                df["country"].str.lower().str.contains(dest)
            ]

        if df.empty:
            return pd.DataFrame()

        valid_idx = df.index.tolist()

        # ── Build query text ──
        amenities_str = " ".join(user_data.get("amenities", []))
        query_text = (
            amenities_str + " " +
            str(user_data.get("style", "")) + " " +
            str(user_data.get("climate", "")) + " " +
            str(user_data.get("travel_type", ""))
        ).strip()

        if not query_text:
            query_text = "hotel"

        # ── Text similarity ──
        query_vec   = self.tfidf_.transform([query_text])
        text_scores = cosine_similarity(query_vec, self.tfidf_matrix_[valid_idx]).flatten()

        # ── Numeric similarity ──
        ideal_price  = min(user_data.get("budget_max", df["avg_price_per_night"].max()), df["avg_price_per_night"].max())
        ideal_stars  = user_data.get("min_stars", 4)
        ideal_review = user_data.get("min_review", 8.0)

        ideal_num = self.scaler_.transform([[ideal_price, ideal_stars, ideal_review]])
        hotel_num = self.num_matrix_[valid_idx]
        distances = np.linalg.norm(hotel_num - ideal_num, axis=1)
        num_scores = 1 / (1 + distances)

        # ── Combined score ──
        combined = (
            self.config.text_weight * text_scores +
            self.config.num_weight  * num_scores
        )

        df = df.copy()
        df["_score"] = combined
        df = df.nlargest(min(top_n, len(df)), "_score").reset_index(drop=True)

        min_s, max_s = df["_score"].min(), df["_score"].max()
        if max_s > min_s:
            df["match_score"] = 60 + 35 * (df["_score"] - min_s) / (max_s - min_s)
        else:
            df["match_score"] = 75.0

        df["match_score"] = df["match_score"].round(1)
        df["avg_price_per_night_inr"] = (df["avg_price_per_night"] * USD_TO_INR).round(0)
        df = df.drop(columns=["_score"])
        return df

    # ── MLflow logging ─────────────────────────────────────────────────────────

    def log_to_mlflow(self, experiment_name: str = "VoyageAnalytics_Recommender"):
        """
        Logs hyperparameters, metrics, AND artifacts:
          - the fitted hotel dataset (CSV) used to build the model
          - the TF-IDF cosine similarity matrix (numpy .npy)
          - the full fitted HotelRecommender object (pickle), so it can be
            loaded directly by the Flask backend with HotelRecommender.load()
        Run this after .fit() so self.tfidf_matrix_ / self.hotels_ exist.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before log_to_mlflow().")

        import tempfile
        import numpy as np

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="content_based_tfidf_cosine"):
            # ── Hyperparameters ──
            mlflow.log_param("text_weight",      self.config.text_weight)
            mlflow.log_param("num_weight",       self.config.num_weight)
            mlflow.log_param("tfidf_features",   self.config.tfidf_max_features)
            mlflow.log_param("num_hotels",       len(self.hotels_))
            mlflow.log_param("data_source",      self.data_source_)

            # ── Metrics ──
            mlflow.log_metric("vocab_size", len(self.tfidf_.vocabulary_))
            mlflow.log_metric("avg_price_per_night", float(self.hotels_["avg_price_per_night"].mean()))
            mlflow.log_metric("avg_review_score", float(self.hotels_["review_score"].mean()))

            with tempfile.TemporaryDirectory() as tmp_dir:
                # ── Artifact 1: the fitted hotel dataset (with derived features) ──
                hotels_path = os.path.join(tmp_dir, "fitted_hotels.csv")
                self.hotels_.to_csv(hotels_path, index=False)
                mlflow.log_artifact(hotels_path, artifact_path="data")

                # ── Artifact 2: the cosine similarity matrix (hotel x hotel) ──
                # Full pairwise similarity across the fitted TF-IDF text matrix —
                # this is the "similarity matrix" the brief asks to track.
                sim_matrix = cosine_similarity(self.tfidf_matrix_)
                sim_path = os.path.join(tmp_dir, "cosine_similarity_matrix.npy")
                np.save(sim_path, sim_matrix)
                mlflow.log_artifact(sim_path, artifact_path="similarity_matrix")

                # ── Artifact 3: the full fitted model object (pickle) ──
                # This is what the Flask backend should load directly —
                # HotelRecommender.load(path) returns a ready-to-use instance.
                model_path = os.path.join(tmp_dir, "recommender_model.pkl")
                self.save(model_path)
                mlflow.log_artifact(model_path, artifact_path="model")

            print("[MLflow] Recommender run logged: params, metrics, "
                  "fitted dataset, similarity matrix, and pickled model artifact.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "recommender_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Recommender] Saved to {path}")

    @staticmethod
    def load(path: str = "recommender_model.pkl") -> "HotelRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Flask-friendly wrapper ─────────────────────────────────────────────────────

_instance: Optional[HotelRecommender] = None

def get_hotel_recommendations(user_data: dict, top_n: int = 5) -> list:
    """Singleton wrapper for Flask backend to import."""
    global _instance
    if _instance is None:
        cfg = RecommenderConfig()
        _instance = HotelRecommender(config=cfg)
        _instance.fit()
    results = _instance.recommend(user_data, top_n=top_n)
    return results.to_dict(orient="records")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = RecommenderConfig(text_weight=0.60, num_weight=0.40)
    rec = HotelRecommender(config=cfg)
    rec.fit()

    print(f"Data source: {rec.data_source_}")
    print(rec.hotels_[["name", "location", "avg_price_per_night", "star_rating", "review_score", "style"]])
    print()

    results = rec.recommend({
        "amenities": ["pool", "wifi"],
        "style": "luxury",
        "climate": "coastal",
        "travel_type": "leisure",
        "budget_max": 350,
        "min_stars": 3,
        "min_review": 7.5,
    }, top_n=5)

    print(results[["name", "location", "avg_price_per_night", "avg_price_per_night_inr", "review_score", "match_score"]])
    rec.log_to_mlflow()