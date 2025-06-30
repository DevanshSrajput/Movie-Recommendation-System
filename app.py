"""
Modern Streamlit Movie Recommendation System
A beautiful, intuitive web interface for movie recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from movie_recommender import MovieRecommender
import numpy as np

# Configure page
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .subheader {
        color: #4ecdc4;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .rating-display {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.5rem;
        border-radius: 5px;
        display: inline-block;
        color: white;
        font-weight: bold;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    """Load and initialize the movie recommender system."""
    with st.spinner("🔄 Initializing Movie Recommender..."):
        recommender = MovieRecommender()
        recommender.load_data()
        recommender.create_user_item_matrix()
        recommender.compute_similarities()
        return recommender

def display_movie_card(movie_title, score, rank):
    """Display a movie recommendation card."""
    st.markdown(f"""
    <div class="movie-card">
        <h3>#{rank} {movie_title}</h3>
        <p><span class="rating-display">⭐ {score:.3f}</span> Predicted Score</p>
    </div>
    """, unsafe_allow_html=True)

def create_rating_distribution_chart(ratings_df):
    """Create a rating distribution chart."""
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={'x': 'Rating', 'y': 'Count'},
        title="Rating Distribution",
        color=rating_counts.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Rating (Stars)",
        yaxis_title="Number of Ratings"
    )
    
    return fig

def create_user_activity_chart(ratings_df):
    """Create a user activity chart."""
    user_activity = ratings_df.groupby('user_id').size().reset_index(name='num_ratings')
    
    fig = px.histogram(
        user_activity,
        x='num_ratings',
        nbins=20,
        title="User Activity Distribution",
        labels={'num_ratings': 'Number of Ratings', 'count': 'Number of Users'}
    )
    
    fig.update_layout(height=400)
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Discover your next favorite movie with AI-powered recommendations</p>', unsafe_allow_html=True)
    
    # Initialize recommender
    try:
        recommender = load_recommender()
    except Exception as e:
        st.error(f"❌ Failed to initialize recommender: {str(e)}")
        return
    
    # Sidebar for controls
    st.sidebar.markdown("### 🎛️ Controls")
    
    # User selection
    available_users = recommender.users
    if available_users is None or len(available_users) == 0:
        st.error("No users available in the dataset.")
        return
    
    selected_user = st.sidebar.selectbox(
        "👤 Select User ID",
        options=available_users,
        format_func=lambda x: f"User {x}",
        help="Choose a user to get personalized recommendations"
    )
    
    # Method selection
    method = st.sidebar.radio(
        "🔍 Recommendation Method",
        options=["user_based", "item_based"],
        format_func=lambda x: x.replace("_", " ").title(),
        help="User-based: Find similar users\nItem-based: Find similar movies"
    )
    
    # Number of recommendations
    num_recs = st.sidebar.slider(
        "📊 Number of Recommendations",
        min_value=1,
        max_value=15,
        value=5,
        help="How many movie recommendations to show"
    )
    
    # Sidebar statistics
    st.sidebar.markdown("### 📈 Dataset Statistics")
    total_ratings = len(recommender.ratings_df)
    total_users = recommender.ratings_df['user_id'].nunique()
    total_movies = recommender.ratings_df['movie_id'].nunique()
    avg_rating = recommender.ratings_df['rating'].mean()
    
    st.sidebar.metric("Total Ratings", f"{total_ratings:,}")
    st.sidebar.metric("Total Users", f"{total_users:,}")
    st.sidebar.metric("Total Movies", f"{total_movies:,}")
    st.sidebar.metric("Average Rating", f"{avg_rating:.2f}/5.0")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f'<h2 class="subheader">🎯 Recommendations for User {selected_user}</h2>', unsafe_allow_html=True)
        
        # Get recommendations button
        if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
            try:
                with st.spinner(f"🔮 Generating {method.replace('_', ' ').title()} recommendations..."):
                    if method == "user_based":
                        recs = recommender.user_based_recommend(selected_user, num_recs)
                    else:
                        recs = recommender.item_based_recommend(selected_user, num_recs)
                    
                    if len(recs) == 0:
                        st.warning("😔 No recommendations available for this user.")
                    else:
                        st.success(f"✨ Found {len(recs)} great recommendations!")
                        
                        # Display recommendations
                        for i, (movie_id, score) in enumerate(recs.items(), 1):
                            movie_title = recommender.get_movie_title(movie_id)
                            display_movie_card(movie_title, score, i)
                        
                        # Store recommendations in session state for the chart
                        st.session_state.recommendations = recs
                        st.session_state.rec_method = method
                        
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
    
    with col2:
        st.markdown(f'<h2 class="subheader">📊 User Profile</h2>', unsafe_allow_html=True)
        
        # User rating history
        user_ratings = recommender.ratings_df[
            recommender.ratings_df['user_id'] == selected_user
        ].sort_values('rating', ascending=False)
        
        # User stats
        st.metric("Movies Rated", len(user_ratings))
        if len(user_ratings) > 0:
            st.metric("Average Rating", f"{user_ratings['rating'].mean():.2f}/5.0")
            st.metric("Highest Rating", f"{user_ratings['rating'].max()}/5.0")
        
        # Top rated movies by user
        st.markdown("#### 🌟 Your Top Rated Movies")
        top_movies = user_ratings.head(5)
        
        if len(top_movies) > 0:
            for _, movie in top_movies.iterrows():
                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 0.5rem; border-radius: 5px; margin: 0.25rem 0;">
                    <strong>{movie['movie_title']}</strong><br>
                    <span style="color: #ff6b6b;">{'⭐' * movie['rating']}</span> {movie['rating']}/5
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No ratings found for this user.")
    
    # Analytics section
    st.markdown("---")
    st.markdown('<h2 class="subheader">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Rating Distribution", "👥 User Activity", "🎬 Recommendation Analysis"])
    
    with tab1:
        st.plotly_chart(create_rating_distribution_chart(recommender.ratings_df), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_user_activity_chart(recommender.ratings_df), use_container_width=True)
    
    with tab3:
        if 'recommendations' in st.session_state:
            recs = st.session_state.recommendations
            method = st.session_state.rec_method
            
            # Create recommendation scores chart
            fig = px.bar(
                x=list(recs.values),
                y=[recommender.get_movie_title(mid) for mid in recs.index],
                orientation='h',
                title=f"{method.replace('_', ' ').title()} Recommendation Scores",
                labels={'x': 'Predicted Score', 'y': 'Movie'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Generate recommendations first to see the analysis!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🎬 Made with ❤️ using Streamlit | Powered by Collaborative Filtering</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
