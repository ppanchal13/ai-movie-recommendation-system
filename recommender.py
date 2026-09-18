import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Load MovieLens movie dataset
# --------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.item"

columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]

movies = pd.read_csv(
    DATA_URL,
    sep="|",
    names=columns,
    encoding="latin-1"
)


genre_columns = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

# Extract release year
movies["release_year"] = pd.to_datetime(
    movies["release_date"],
    errors="coerce"
).dt.year


# Fill missing years using median year
median_year = movies["release_year"].median()
movies["release_year"] = movies["release_year"].fillna(median_year)


# Normalize year to roughly 0-1 range
min_year = movies["release_year"].min()
max_year = movies["release_year"].max()

movies["year_feature"] = (
    (movies["release_year"] - min_year)
    / (max_year - min_year)
)


# Genre features
genre_features = movies[genre_columns].astype(float).copy()


# Give genre information stronger importance
genre_weight = 3.0
genre_features = genre_features * genre_weight


# Give release period a smaller weight
year_weight = 1.2
year_features = movies[["year_feature"]] * year_weight


# Combine genre + release-era features
feature_matrix = pd.concat(
    [
        genre_features.reset_index(drop=True),
        year_features.reset_index(drop=True)
    ],
    axis=1
)


# Normalize vectors before similarity calculation
feature_matrix = normalize(feature_matrix)


# Calculate pairwise cosine similarity
similarity_matrix = cosine_similarity(feature_matrix)


# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------

def recommend_movies(movie_title, number_of_recommendations=5):

    matches = movies[
        movies["title"].str.lower() == movie_title.lower()
    ]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    # Sort highest similarity first
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores:

        # Do not recommend the selected movie itself
        if index == movie_index:
            continue

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "release_year": int(
                movies.iloc[index]["release_year"]
            ),
            "similarity_score": round(
                float(score) * 100,
                2
            )
        })

        if len(recommendations) == number_of_recommendations:
            break

    return recommendations


# --------------------------------------------------
# Terminal Test
# --------------------------------------------------

if __name__ == "__main__":

    print("Movie Recommendation System")
    print("---------------------------")

    movie = input("Enter a movie title: ")

    results = recommend_movies(movie)

    if results:

        print("\nRecommended Movies:")

        for i, recommendation in enumerate(
            results,
            start=1
        ):

            print(
                f"{i}. "
                f"{recommendation['title']} "
                f"- {recommendation['similarity_score']}% similarity"
            )

    else:

        print("\nMovie not found in dataset.")
