"""
GUI Application for Movie Recommendation System
Simple Tkinter interface for the movie recommender.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from movie_recommender import MovieRecommender

class RecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Movie Recommendation System")
        self.root.geometry("800x600")
        
        # Initialize recommender
        self.recommender = MovieRecommender()
        self.setup_recommender()
        
        self.create_widgets()
        
    def setup_recommender(self):
        """Initialize the recommendation system."""
        try:
            self.recommender.load_data()  # Uses synthetic data
            self.recommender.create_user_item_matrix()
            self.recommender.compute_similarities()
            
            # Get available users
            self.available_users = self.recommender.users
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize recommender: {str(e)}")
    
    def create_widgets(self):
        """Create the GUI widgets."""
        # Title
        title_label = tk.Label(self.root, text="🎬 Movie Recommendation System", 
                              font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # User selection frame
        user_frame = ttk.Frame(self.root)
        user_frame.pack(pady=10, padx=20, fill='x')
        
        ttk.Label(user_frame, text="Select User ID:", font=("Arial", 12)).pack(side='left')
        
        self.user_var = tk.StringVar()
        user_combo = ttk.Combobox(user_frame, textvariable=self.user_var, 
                                 values=self.available_users, width=10)
        user_combo.pack(side='left', padx=10)
        user_combo.set(self.available_users[0] if self.available_users else "")
        
        # Method selection
        method_frame = ttk.Frame(self.root)
        method_frame.pack(pady=5, padx=20, fill='x')
        
        ttk.Label(method_frame, text="Recommendation Method:", font=("Arial", 12)).pack(side='left')
        
        self.method_var = tk.StringVar(value="user_based")
        method_radio1 = ttk.Radiobutton(method_frame, text="User-Based", 
                                       variable=self.method_var, value="user_based")
        method_radio2 = ttk.Radiobutton(method_frame, text="Item-Based", 
                                       variable=self.method_var, value="item_based")
        method_radio1.pack(side='left', padx=10)
        method_radio2.pack(side='left', padx=5)
        
        # Number of recommendations
        num_frame = ttk.Frame(self.root)
        num_frame.pack(pady=5, padx=20, fill='x')
        
        ttk.Label(num_frame, text="Number of Recommendations:", font=("Arial", 12)).pack(side='left')
        
        self.num_recs_var = tk.StringVar(value="5")
        num_spin = tk.Spinbox(num_frame, from_=1, to=10, textvariable=self.num_recs_var, width=5)
        num_spin.pack(side='left', padx=10)
        
        # Get recommendations button
        rec_button = ttk.Button(self.root, text="Get Recommendations", 
                               command=self.get_recommendations, style='Accent.TButton')
        rec_button.pack(pady=20)
        
        # Results area
        results_frame = ttk.Frame(self.root)
        results_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        ttk.Label(results_frame, text="Recommendations:", font=("Arial", 14, "bold")).pack(anchor='w')
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=20)
        self.results_text.pack(fill='both', expand=True, pady=5)
        
        # User history button
        history_button = ttk.Button(self.root, text="Show User's Rating History", 
                                   command=self.show_user_history)
        history_button.pack(pady=10)
        
    def get_recommendations(self):
        """Get and display recommendations for the selected user."""
        try:
            user_id = int(self.user_var.get())
            method = self.method_var.get()
            n_recs = int(self.num_recs_var.get())
            
            # Clear previous results
            self.results_text.delete('1.0', tk.END)
            
            # Get recommendations
            if method == "user_based":
                recs = self.recommender.user_based_recommend(user_id, n_recs)
            else:
                recs = self.recommender.item_based_recommend(user_id, n_recs)
            
            # Display results
            self.results_text.insert(tk.END, f"🎬 TOP {n_recs} RECOMMENDATIONS FOR USER {user_id}\n")
            self.results_text.insert(tk.END, f"Method: {method.replace('_', ' ').title()}\n")
            self.results_text.insert(tk.END, "=" * 60 + "\n\n")
            
            if len(recs) == 0:
                self.results_text.insert(tk.END, "No recommendations available for this user.\n")
            else:
                for i, (movie_id, score) in enumerate(recs.items(), 1):
                    movie_title = self.recommender.get_movie_title(movie_id)
                    self.results_text.insert(tk.END, f"{i}. {movie_title}\n")
                    self.results_text.insert(tk.END, f"   Predicted Score: {score:.3f}\n\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get recommendations: {str(e)}")
    
    def show_user_history(self):
        """Show the selected user's rating history."""
        try:
            user_id = int(self.user_var.get())
            
            user_ratings = self.recommender.ratings_df[
                self.recommender.ratings_df['user_id'] == user_id
            ].sort_values('rating', ascending=False)
            
            # Create new window for history
            history_window = tk.Toplevel(self.root)
            history_window.title(f"User {user_id} - Rating History")
            history_window.geometry("600x400")
            
            # History text
            history_text = scrolledtext.ScrolledText(history_window, wrap=tk.WORD)
            history_text.pack(fill='both', expand=True, padx=10, pady=10)
            
            history_text.insert(tk.END, f"RATING HISTORY FOR USER {user_id}\n")
            history_text.insert(tk.END, "=" * 50 + "\n\n")
            history_text.insert(tk.END, f"Total movies rated: {len(user_ratings)}\n")
            history_text.insert(tk.END, f"Average rating: {user_ratings['rating'].mean():.2f}\n\n")
            
            for _, row in user_ratings.iterrows():
                history_text.insert(tk.END, f"⭐ {row['rating']}/5 - {row['movie_title']}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show user history: {str(e)}")

def main():
    """Run the GUI application."""
    root = tk.Tk()
    app = RecommenderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()