import joblib
# import logging

# Set up logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s %(message)s',
#     handlers=[
#         logging.FileHandler('recommendation.log', encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )

# logging.info('Loading data...')
try:
    df = joblib.load('df_cleaned.pkl')
    cosine_sim = joblib.load('cosine_sim.pkl')
    # logging.info('Data loaded successfully.')
except Exception as e:
    # logging.error(f'Error loading data: {e}')
    raise e

def get_recommendations(movie_name, top_n=5):
    # logging.info(f'Getting recommendations for movie: {movie_name}')
    idx = df[df['title'].str.lower() == movie_name.lower()].index  # Get the index of the movie with the given title
    if len(idx) == 0:
        # logging.warning(f'Movie not found: {movie_name}')
        return None
    idx = idx[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))  # Get the similarity scores for the movie
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort the movies based on similarity scores
    sim_scores = sim_scores[1:top_n+1]  # Get the top N most similar movies (excluding the movie itself)
    
    movie_indices = [i[0] for i in sim_scores]  # Get the indices of the recommended movies
    # logging.info('Recommendations generated successfully.')
    
    # Create a DataFrame with recommended movie titles
    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = 'Sr.No.'
    
    return  result_df


