from flask import Flask, render_template, request
from recommender import movies, recommend_movies

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    selected_movie = None

    # Get all movie titles for dropdown
    movie_titles = sorted(movies["title"].tolist())

    if request.method == "POST":
        selected_movie = request.form.get("movie")

        if selected_movie:
            recommendations = recommend_movies(
                selected_movie,
                number_of_recommendations=5
            )

    return render_template(
        "index.html",
        movie_titles=movie_titles,
        selected_movie=selected_movie,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
