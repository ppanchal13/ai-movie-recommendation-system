# 🎬 AI Movie Recommendation System

A content-based movie recommendation system that suggests similar movies using machine learning and movie metadata.

The system analyzes **genres, keywords, and plot descriptions** from the TMDB movie dataset, converts them into numerical features using **TF-IDF**, and calculates similarity using **cosine similarity**.

A Flask-based web interface allows users to select a movie and instantly receive the top 5 most similar movies.

---

## 🚀 Features

- 🎥 Select a movie from the available dataset
- 🤖 Content-based recommendation engine
- 🧠 TF-IDF feature extraction
- 📐 Cosine similarity for movie comparison
- 🏷️ Uses genres, keywords, and plot descriptions
- 📊 Displays similarity scores for recommendations
- 🌐 Interactive Flask web application
- 📱 Responsive dark-themed interface

---

## 🧠 How It Works

The recommendation pipeline:

```text
Selected Movie
      ↓
Movie Metadata
      ↓
Genres + Keywords + Plot Overview
      ↓
Feature Engineering
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Top 5 Similar Movies
