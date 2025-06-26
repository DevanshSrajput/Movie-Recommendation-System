# 🎬 Netflix-Style Movie Recommender (But Cooler)

Welcome to the **Movie Recommendation System**—because who *doesn't* want a robot to tell them what to watch next? This project is a collaborative filtering-based movie recommender, inspired by Netflix, but with 100% more sarcasm and 0% licensing fees.

## Features

- **User-Based & Item-Based Collaborative Filtering:** Because why settle for one flavor of algorithmic magic?
- **Synthetic Data Generation:** No MovieLens? No problem. We make up our own data (statistically, of course).
- **Precision@K Evaluation:** For when you want to pretend you care about metrics.
- **Tkinter GUI:** A beautiful (okay, functional) interface for those who fear the command line.
- **Pythonic & Pandas-Powered:** Because spreadsheets are just databases with commitment issues.

## How It Works

1. **Synthetic Data:** We generate a fake but statistically plausible set of users, movies, and ratings. Hollywood, beware.
2. **Collaborative Filtering:** We use cosine similarity to find users/movies that are *just like you* (or at least, like your taste in movies).
3. **Recommendations:** Get top picks for any user, with scores that look suspiciously like real predictions.
4. **GUI:** Click buttons, get recommendations, feel like a tech mogul.

## Getting Started

### Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tkinter` (usually comes with Python)

### Installation

Clone this repository (because copy-pasting code from StackOverflow is so 2020):

```sh
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt  # If you actually have a requirements.txt
```

### Running the Command-Line Demo

Want to see recommendations in your terminal? Run:

```sh
python movie_recommender.py
```

### Running the GUI

For those who prefer clicking to typing:

```sh
python recommender_gui.py
```

## File Structure

- [`movie_recommender.py`](movie_recommender.py): The brains of the operation. Handles data, algorithms, and all the heavy lifting.
- [`recommender_gui.py`](recommender_gui.py): The face of the operation. Tkinter-powered GUI for the visually inclined.
- `__pycache__/`: Python’s way of saying “I was here.”

## Example Output

```
🎬 Movie Recommendation System
==================================================
MOVIE RECOMMENDATIONS FOR USER 1
Method: User Based
...
Top 5 Recommendations:
1. Comedy Movie 7
   Predicted Score: 4.23
...
```

## Why Use This?

- You want to learn about collaborative filtering without reading a 50-page research paper.
- You need a project for your portfolio that sounds impressive at parties.
- You like movies. Or robots. Or both.

## Contributing

Pull requests welcome! Just don’t break the sarcasm detector.

## License

MIT. Because sharing is caring (and lawyers are expensive).

---

*“May your recommendations always be binge-worthy.”*