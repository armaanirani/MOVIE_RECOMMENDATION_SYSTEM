import os
from dotenv import load_dotenv
import streamlit as st
from recommendation import df, get_recommendations
from omdb_utils import get_movie_info

# Load environment variables from .env file
load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon=":movie_camera:",
    layout="centered",
)

st.title("Movie Recommendation System")

movie_list = sorted(df['title'].dropna().unique())
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        recommendations = get_recommendations(selected_movie)
        if recommendations is None or recommendations.empty:
            st.error("No recommendations found.")
        else:
            st.write("Top 5 Recommended Movies:")
            for _, row in recommendations.iterrows():
                movie_title = row['title']
                plot, poster = get_movie_info(movie_title, OMDB_API_KEY)
                
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if poster != "N/A":
                            st.image(poster, width=100)
                        else:
                            st.write("No poster available")
                    with col2:
                        st.markdown(f"### {movie_title}")
                        st.markdown(f"*{plot}*" if plot != 'N/A' else 'No plot available')