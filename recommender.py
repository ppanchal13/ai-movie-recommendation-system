import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Load TMDB Movie Dataset
# --------------------------------------------------

movies = pd.read_csv("tmdb_5000_movies.csv")

# Keep only the columns required by the recommender
movies = movies[
    [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "release_date"
    ]
].copy()

# Clean dataset
movies = movies.dropna(subset=["title"])
movies = movies.drop_duplicates(subset=["title"])
movies = movies.reset_index(drop=True)

movies["overview"] = movies["overview"].fillna("")
movies["genres"] = movies["genres"].fillna("[]")
movies["keywords"] = movies["keywords"].fillna("[]")


# --------------------------------------------------
# Extract Metadata
# --------------------------------------------------

def extract_names(value):
    """
    Extract names from TMDB JSON-style columns.
    """

    try:
        items = ast.literal_eval(value)

        return [
            item["name"].replace(" ", "")
            for item in items
            if "name" in item
        ]

    except (ValueError, SyntaxError, TypeError):
        return []


movies["genres_list"] = movies["genres"].apply(
    extract_names
)

movies["keywords_list"] = movies["keywords"].apply(
    extract_names
)


# --------------------------------------------------
# Build Combined Movie Features
# --------------------------------------------------

def create_tags(row):

    # Genre is given extra importance
    genres = row["genres_list"] * 3

    # Keywords are also strong recommendation signals
    keywords = row["keywords_list"] * 2

    # Plot description
    overview = str(row["overview"]).lower().split()

    tags = genres + keywords + overview

    return " ".join(tags)


movies["tags"] = movies.apply(
    create_tags,
    axis=1
)


# --------------------------------------------------
# TF-IDF Vectorization
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    lowercase=True
)

feature_matrix = vectorizer.fit_transform(
    movies["tags"]
)


# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------

def recommend_movies(
    movie_title,
    number_of_recommendations=5
):

    matches = movies[
        movies["title"].str.lower()
        == movie_title.lower()
    ]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    # Compare selected movie with all other movies
    similarity_scores = cosine_similarity(
        feature_matrix[movie_index],
        feature_matrix
    ).flatten()

    # Highest similarity first
    ranked_indices = similarity_scores.argsort()[::-1]

    recommendations = []

    for index in ranked_indices:

        # Skip the selected movie itself
        if index == movie_index:
            continue

        movie = movies.iloc[index]

        release_year = ""

        if pd.notna(movie["release_date"]):
            release_year = str(
                movie["release_date"]
            )[:4]

        recommendations.append(
            {
                "title": movie["title"],
                "release_year": release_year,
                "similarity_score": round(
                    float(similarity_scores[index]) * 100,
                    2
                )
            }
        )

        if len(recommendations) == number_of_recommendations:
            break

    return recommendations


# --------------------------------------------------
# Terminal Test
# --------------------------------------------------

if __name__ == "__main__":

    print("Movie Recommendation System")
    print("---------------------------")

    movie_title = input(
        "Enter a movie title: "
    ).strip()

    results = recommend_movies(movie_title)

    if results:

        print(
            f"\nMovies similar to {movie_title}:\n"
        )

        for i, recommendation in enumerate(
            results,
            start=1
        ):

            print(
                f"{i}. "
                f"{recommendation['title']} "
                f"({recommendation['release_year']}) "
                f"- "
                f"{recommendation['similarity_score']}% similarity"
            )

    else:

        print("\nMovie not found in dataset.")
