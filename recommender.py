import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# Merge movie metadata with cast and crew
movies = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id",
    suffixes=("", "_credits")
)


# Keep useful columns
movies = movies[
    [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew",
        "release_date"
    ]
].copy()


# Remove duplicates and missing titles
movies = movies.drop_duplicates(subset="title")
movies = movies.dropna(subset=["title"])
movies["overview"] = movies["overview"].fillna("")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def extract_names(value):
    """Extract names from JSON-like TMDB columns."""

    try:
        items = ast.literal_eval(value)

        return [
            item["name"].replace(" ", "")
            for item in items
        ]

    except (ValueError, SyntaxError, TypeError):
        return []


def extract_top_cast(value, limit=3):
    """Extract top cast members."""

    try:
        items = ast.literal_eval(value)

        return [
            item["name"].replace(" ", "")
            for item in items[:limit]
        ]

    except (ValueError, SyntaxError, TypeError):
        return []


def extract_director(value):
    """Extract director from crew."""

    try:
        crew = ast.literal_eval(value)

        for person in crew:

            if person.get("job") == "Director":

                return [
                    person["name"].replace(" ", "")
                ]

    except (ValueError, SyntaxError, TypeError):
        pass

    return []


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

movies["genres_list"] = movies["genres"].apply(
    extract_names
)

movies["keywords_list"] = movies["keywords"].apply(
    extract_names
)

movies["cast_list"] = movies["cast"].apply(
    extract_top_cast
)

movies["director_list"] = movies["crew"].apply(
    extract_director
)


# Convert overview into word tokens
movies["overview_list"] = movies["overview"].apply(
    lambda text: str(text).lower().split()
)


# Build combined movie representation
def create_tags(row):

    # Repeating important metadata gives these
    # features slightly more influence.
    genres = row["genres_list"] * 3
    keywords = row["keywords_list"] * 2
    cast = row["cast_list"] * 2
    director = row["director_list"] * 3

    overview = row["overview_list"]

    tags = (
        genres
        + keywords
        + cast
        + director
        + overview
    )

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

    # Calculate similarity only against selected movie
    scores = cosine_similarity(
        feature_matrix[movie_index],
        feature_matrix
    ).flatten()

    ranked_indices = scores.argsort()[::-1]

    recommendations = []

    for index in ranked_indices:

        if index == movie_index:
            continue

        movie = movies.iloc[index]

        year = ""

        if pd.notna(movie["release_date"]):

            year = str(
                movie["release_date"]
            )[:4]

        recommendations.append(
            {
                "title": movie["title"],
                "release_year": year,
                "similarity_score": round(
                    float(scores[index]) * 100,
                    2
                )
            }
        )

        if (
            len(recommendations)
            == number_of_recommendations
        ):
            break

    return recommendations


# --------------------------------------------------
# Terminal Test
# --------------------------------------------------

if __name__ == "__main__":

    print("Movie Recommendation System")
    print("---------------------------")

    movie = input(
        "Enter a movie title: "
    ).strip()

    results = recommend_movies(movie)

    if results:

        print(
            f"\nMovies similar to {movie}:\n"
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
                f"{recommendation['similarity_score']}% "
                f"similarity"
            )

    else:

        print(
            "\nMovie not found in dataset."
        )
