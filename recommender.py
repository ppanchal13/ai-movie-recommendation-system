import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load movie dataset
DATA_URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.item"

columns = [
    "movie_id", "title", "release_date", "video_release_date",
    "imdb_url", "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

movies = pd.read_csv(
    DATA_URL,
    sep="|",
    names=columns,
    encoding="latin-1"
)

genre_columns = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]


def create_genre_text(row):
    genres = []

    for genre in genre_columns:
        if row[genre] == 1:
            genres.append(genre)

    return " ".join(genres)


# Convert movie genres into text features
movies["features"] = movies.apply(create_genre_text, axis=1)


# Convert features into numerical vectors
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(movies["features"])


# Calculate similarity between all movies
similarity_matrix = cosine_similarity(feature_matrix)


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

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:number_of_recommendations + 1]:
        recommendations.append({
            "title": movies.iloc[index]["title"],
            "similarity_score": round(float(score) * 100, 2)
        })

    return recommendations


if __name__ == "__main__":
    print("Movie Recommendation System")
    print("---------------------------")

    movie = input("Enter a movie title: ")

    results = recommend_movies(movie)

    if results:
        print("\nRecommended Movies:")

        for i, recommendation in enumerate(results, start=1):
            print(
                f"{i}. {recommendation['title']} "
                f"({recommendation['similarity_score']}% similarity)"
            )
    else:
        print("\nMovie not found in dataset.")
