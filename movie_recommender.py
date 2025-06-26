"""
Movie Recommendation System using Collaborative Filtering
Author: AI Internship Portfolio Project
Date: 2025-06-14

A Netflix-style movie recommender that uses collaborative filtering
to suggest movies based on user ratings and preferences.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class MovieRecommender:
    """
    A collaborative filtering movie recommendation system.
    
    Supports both user-based and item-based collaborative filtering
    using cosine similarity for finding similar users/items.
    """
    
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.movies = None
        self.users = None
        
    def create_synthetic_dataset(self, n_users=50, n_movies=20):
        """
        Generate a synthetic MovieLens-style dataset for demonstration.
        
        Args:
            n_users (int): Number of users to generate
            n_movies (int): Number of movies to generate
            
        Returns:
            pd.DataFrame: Ratings data with columns [user_id, movie_id, rating]
        """
        np.random.seed(42)
        
        # Create movie names
        movie_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        movie_names = [f"{np.random.choice(movie_genres)} Movie {i+1}" for i in range(n_movies)]
        
        # Generate ratings (some users rate some movies)
        ratings_data = []
        for user_id in range(1, n_users + 1):
            # Each user rates between 5-15 movies
            n_ratings = np.random.randint(5, 16)
            rated_movies = np.random.choice(range(1, n_movies + 1), n_ratings, replace=False)
            
            for movie_id in rated_movies:
                # Generate realistic ratings (bias towards higher ratings)
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.25, 0.35, 0.25])
                ratings_data.append([user_id, movie_id, rating, movie_names[movie_id-1]])
        
        df = pd.DataFrame(ratings_data, columns=['user_id', 'movie_id', 'rating', 'movie_title'])
        return df
    
    def load_data(self, ratings_df=None):
        """
        Load and preprocess the ratings data.
        
        Args:
            ratings_df (pd.DataFrame): Ratings data. If None, creates synthetic data.
        """
        if ratings_df is None:
            print("Creating synthetic dataset...")
            ratings_df = self.create_synthetic_dataset()
            
        self.ratings_df = ratings_df
        print(f"Loaded {len(ratings_df)} ratings from {ratings_df['user_id'].nunique()} users and {ratings_df['movie_id'].nunique()} movies")
        
    def create_user_item_matrix(self):
        """
        Create user-item matrix from ratings data.
        Handles missing values by filling with 0.
        """
        # Create pivot table
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating', 
            fill_value=0
        )
        
        self.users = self.user_item_matrix.index.tolist()
        self.movies = self.user_item_matrix.columns.tolist()
        
        print(f"Created user-item matrix: {self.user_item_matrix.shape}")
        
    def compute_similarities(self):
        """
        Compute user-user and item-item similarity matrices using cosine similarity.
        """
        # User-based similarity
        self.user_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # Item-based similarity
        self.item_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix.T),
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        print("Computed user-user and item-item similarities")
        
    def user_based_recommend(self, user_id, n_recommendations=5):
        """
        Generate recommendations using user-based collaborative filtering.
        
        Args:
            user_id (int): Target user ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.Series: Top N movie recommendations with scores
        """
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found in dataset")
            
        # Get similarity scores for the target user
        user_similarities = self.user_similarity.loc[user_id]
        
        # Get movies not rated by the target user
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Calculate weighted average ratings for unrated movies
        recommendations = {}
        
        for movie in unrated_movies:
            # Get users who rated this movie
            movie_raters = self.user_item_matrix[movie][self.user_item_matrix[movie] > 0]
            
            if len(movie_raters) == 0:
                continue
                
            # Calculate weighted score
            numerator = sum(user_similarities[rater] * rating 
                          for rater, rating in movie_raters.items())
            denominator = sum(abs(user_similarities[rater]) 
                            for rater in movie_raters.index)
            
            if denominator > 0:
                recommendations[movie] = numerator / denominator
        
        # Sort and return top N
        recommendations = pd.Series(recommendations).sort_values(ascending=False)
        return recommendations.head(n_recommendations)
    
    def item_based_recommend(self, user_id, n_recommendations=5):
        """
        Generate recommendations using item-based collaborative filtering.
        
        Args:
            user_id (int): Target user ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.Series: Top N movie recommendations with scores
        """
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found in dataset")
            
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        recommendations = {}
        
        for movie in unrated_movies:
            # Calculate similarity-weighted score
            numerator = sum(self.item_similarity.loc[movie, rated_movie] * rating
                          for rated_movie, rating in rated_movies.items())
            denominator = sum(abs(self.item_similarity.loc[movie, rated_movie])
                            for rated_movie in rated_movies.index)
            
            if denominator > 0:
                recommendations[movie] = numerator / denominator
        
        recommendations = pd.Series(recommendations).sort_values(ascending=False)
        return recommendations.head(n_recommendations)
    
    def get_movie_title(self, movie_id):
        """Get movie title from movie ID."""
        movie_titles = self.ratings_df[self.ratings_df['movie_id'] == movie_id]['movie_title'].iloc[0]
        return movie_titles
    
    def evaluate_precision_at_k(self, test_ratings, k=5):
        """
        Evaluate recommendation system using Precision@K.
        
        Args:
            test_ratings (pd.DataFrame): Test set ratings
            k (int): Number of top recommendations to consider
            
        Returns:
            float: Average Precision@K across all users
        """
        precisions = []
        
        for user_id in test_ratings['user_id'].unique():
            if user_id not in self.users:
                continue
                
            # Get user's test movies (ground truth)
            user_test_movies = set(test_ratings[test_ratings['user_id'] == user_id]['movie_id'].values)
            
            # Get recommendations
            try:
                user_recs = self.user_based_recommend(user_id, k)
                recommended_movies = set(user_recs.index[:k])
                
                # Calculate precision
                if len(recommended_movies) > 0:
                    precision = len(user_test_movies & recommended_movies) / len(recommended_movies)
                    precisions.append(precision)
            except:
                continue
        
        return np.mean(precisions) if precisions else 0.0
    
    def display_recommendations(self, user_id, method='user_based', n_recs=5):
        """
        Display recommendations for a user in a formatted way.
        
        Args:
            user_id (int): Target user ID
            method (str): 'user_based' or 'item_based'
            n_recs (int): Number of recommendations
        """
        print(f"\n{'='*60}")
        print(f"MOVIE RECOMMENDATIONS FOR USER {user_id}")
        print(f"Method: {method.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        # Show user's rating history
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        print(f"\nUser's Rating History ({len(user_ratings)} movies):")
        print("-" * 40)
        for _, row in user_ratings.head(5).iterrows():
            print(f"  {row['movie_title']}: {row['rating']}/5")
        if len(user_ratings) > 5:
            print(f"  ... and {len(user_ratings) - 5} more")
        
        # Get recommendations
        if method == 'user_based':
            recs = self.user_based_recommend(user_id, n_recs)
        else:
            recs = self.item_based_recommend(user_id, n_recs)
        
        print(f"\nTop {n_recs} Recommendations:")
        print("-" * 40)
        for i, (movie_id, score) in enumerate(recs.items(), 1):
            movie_title = self.get_movie_title(movie_id)
            print(f"  {i}. {movie_title}")
            print(f"     Predicted Score: {score:.2f}")
        
        print(f"\n{'='*60}")

def main():
    """
    Main function to demonstrate the movie recommendation system.
    """
    print("🎬 Movie Recommendation System")
    print("=" * 50)
    
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Load data (synthetic in this case)
    recommender.load_data()
    
    # Preprocess data
    recommender.create_user_item_matrix()
    recommender.compute_similarities()
    
    # Split data for evaluation
    train_df, test_df = train_test_split(recommender.ratings_df, test_size=0.2, random_state=42)
    
    # Demonstrate recommendations for a few users
    sample_users = [1, 5, 10, 15]
    
    for user_id in sample_users:
        if user_id in recommender.users:
            # User-based recommendations
            recommender.display_recommendations(user_id, 'user_based', 5)
            
            # Item-based recommendations
            recommender.display_recommendations(user_id, 'item_based', 5)
    
    # Evaluate system
    print("\n🔍 SYSTEM EVALUATION")
    print("=" * 50)
    precision_k = recommender.evaluate_precision_at_k(test_df, k=5)
    print(f"Precision@5: {precision_k:.3f}")
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"- Total ratings: {len(recommender.ratings_df):,}")
    print(f"- Unique users: {recommender.ratings_df['user_id'].nunique():,}")
    print(f"- Unique movies: {recommender.ratings_df['movie_id'].nunique():,}")
    print(f"- Average rating: {recommender.ratings_df['rating'].mean():.2f}")
    print(f"- Rating distribution:")
    rating_dist = recommender.ratings_df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        print(f"  {rating} stars: {count:,} ({count/len(recommender.ratings_df)*100:.1f}%)")

if __name__ == "__main__":
    main()