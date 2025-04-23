import requests

def get_movie_info(title, api_key):
    url = f"http://www.omdbapi.com/?t={title}&plot=full&apikey={api_key}"
    response = requests.get(url).json()
    
    if response.get("Response") == "True":
        result = response.get('Plot', 'N/A'), response.get('Poster', 'N/A')
        plot = result[0]
        poster = result[1]
        return plot, poster
    else:
        return "N/A", "N/A"
