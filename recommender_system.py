import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from streamlit_option_menu import option_menu
import openai
import json



# Streamlit App Configuration
st.set_page_config(page_title="Movie Recommender System", layout="wide")

# === 0 - Streamlit_Sidebar Navigation =================================
with st.sidebar:
    selected = option_menu(
        "Movie Recommender System",
        ["Exploratory Analysis", "Collaborative Filtering", "Content-Based Filtering", "Hybrid Filtering"],
        icons=["bar-chart", "people", "search", "layers"],
        default_index=0,
    )

# # Initialize session state for toggling additional information
# if "show_additional_info" not in st.session_state:
#     st.session_state.show_additional_info = False
    
    
# === 1 - Loading the data ============================================
os.chdir(r'C:\Users\USA\University\BU\IV TERM\CS688_Vasilkovski\Project\Term_project')
movie = pd.read_csv('movies.csv')
rating = pd.read_csv('ratings.csv')

print(movie.head(5))
print(rating.head(5))

# == 2 - Data Analysis ================================================
num_movie = movie.movieId.nunique()
num_rating = len(rating)
num_users = rating.userId.nunique()
mean_glob_rating = np.mean(rating.rating).round(1)

print(f"Number of unique movie titles: {num_movie}")
print(f"Number of ratings: {num_rating}")
print(f"Number of unique users: {num_users}")
print(f"The average global rating: {mean_glob_rating}")


# == RATING DISTRIBUTION =====================================
rating_group = rating.groupby('rating').count().reset_index()
x = rating_group.rating
freq = rating_group.userId
plt.bar(x, freq, color = 'red', width = 0.3)
plt.xticks(x)
plt.xlabel('rating')
plt.ylabel('number of ratings')
plt.title('Distribution of ratings')
plt.show()

# == THE MOST POPULAR GENRES (PIE CHART)============================
def split(df):
    """ The function converts string with delimiter '|'
        to the list """
    return df.split('|')

# transform the 'genres' format into list of genres
movie.genres = movie.genres.apply(split)

# count the frequency of different genres in the columns movie.genres
genres = []
for i in movie.genres:
    for e in i:
        genres.append(e)
genres = pd.DataFrame({'genre':genres, 'count' : [1]*len(genres)})
genres_freq = genres.groupby('genre').count().reset_index()
genres_freq = genres_freq.sort_values(by = 'count', ascending = False)[:-3]
print(genres_freq)

# pie chart
labels = genres_freq.genre
frequency = genres_freq['count']
plt.pie(frequency, labels = labels, wedgeprops = {"edgecolor" : "white",
                                                  'linewidth': 0.3,
                                                  'antialiased': True})
plt.title("Movie genre popularity")
plt.show()



# Create Pie Chart (for Streamlit)
fig = px.pie(genres_freq, values="count", names="genre")


# == MOST POPULAR MOVIES (BASED ON NUMBER OF RATINGS) ===================================
# add first the name of the movie to the rating dataset
rating_with_movie = rating.merge(movie)

# the movies with the highest number of ratings (the most) popular
most_popular_movies = rating_with_movie.title.value_counts().reset_index()
print(f"Top 10 most popular movies (based on number of ratings):\n{most_popular_movies.head(10)}")

# the 10 least popular movies
least_popular_movies = rating_with_movie.title.value_counts().reset_index().tail(10)
print(f"\nTop 10 least popular movies (based on number of ratings):\n{least_popular_movies}")

# since some movies have a very few number of rating, we'll keep only
# the movies with number of ratings equal or greater than average number of ratings per movie
average_number_of_ratings = num_rating/num_movie

# update most popular movies dataframe
most_popular_movies = most_popular_movies[most_popular_movies['count'] >= average_number_of_ratings]

# apply the list of movies with number of ratings >= 10 to our big dataset
rating_with_movie = rating_with_movie[rating_with_movie['title'].isin(most_popular_movies.title)]


def clean_title(title):
    """ The fucnction removes all unnessesary symbols
        except letters and numbers"""
    return re.sub("[^a-zA-Z0-9 ]", "", title)

# adding the column with a clean title to the dataset
movie['clean_title'] = movie.title.apply(clean_title)
movie.head(5)


def search(movie_title, min_similarity=0.1):
    """
    Search for the most similar movie title.

    Args:
        movie_title (str): The query title to search for.
        movie (DataFrame): A DataFrame containing a 'clean_title' and 'title' columns.
        min_similarity (float): Minimum similarity score to consider a match (default: 0.1).

    Returns:
        str: The most similar movie title, or None if no match exceeds min_similarity.
    """
    # Vectorize movie titles with unigrams and bigrams
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w']+\b", stop_words='english',             
                                 ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(movie.clean_title.str.lower())

    # Clean and vectorize the query title
    movie_title = clean_title(movie_title).lower()        # clean_title function is used
    movie_title_vec = vectorizer.transform([movie_title])

    # Calculate cosine similarity
    similarity_vec = cosine_similarity(movie_title_vec, tfidf_matrix).flatten()

    # Find the most similar title
    index = np.argmax(similarity_vec)
    max_similarity = similarity_vec[index]

    # Return result if similarity is above the threshold
    if max_similarity >= min_similarity:
        return movie.iloc[index].title
    else:
        return None  # No sufficiently similar title found

# the usage example
search('scary movie 5')

# == 3 - Collaborative Filtering ==============================================

def collaborative_filtering(movie_title, rating_with_movie, k = 10):

    """This function return k (by default 10) similar movies based on the
       ratings of the movies. kNN is used to find the nearest neigbors,
       cosine similarity is used as the distance metric for kNN method."""
    
    movie_title = search(movie_title)
    # number of unique users / mapping userId to user_indeces for sparse matrix
    u = rating_with_movie.userId.nunique()
    user_mapper = pd.DataFrame({'userId':rating_with_movie.userId.unique(),
                                'user_index':list(range(u))})

    # number of unique movies / mapping movieId to movies_indeces for sparse matrix
    m = rating_with_movie.movieId.nunique()
    movie_mapper = pd.DataFrame({'movieId':rating_with_movie.movieId.unique(),
                                'movie_index':list(range(m))})

    # adding indeces for a sparse matrix into rating_with_movie dataset
    rating_with_movie = pd.merge(rating_with_movie, user_mapper, on = 'userId')
    rating_with_movie = pd.merge(rating_with_movie, movie_mapper, on = 'movieId')
    

    # sparse matrix
    matrix = csr_matrix((rating_with_movie.rating, (rating_with_movie.user_index, rating_with_movie.movie_index)))

    # transpose the matrix, since we need to have movies as rows
    matrix = matrix.T

    # extracting the vector for 'movie_title' from the sparse matrix
    # first, extracting the sparse matrix index for the movie_title
    movie_index = rating_with_movie.loc[rating_with_movie.title == movie_title, 'movie_index'].unique()
    # 'movie_title' vector
    movie_title_vec = matrix[movie_index]

    # building a knn model (we use k+1 since the model returns 10 neighbors + the subject element itself)
    knn_model = NearestNeighbors(n_neighbors = k+1, algorithm = 'brute', metric = 'cosine')

    # fit sparse matrix into the model
    knn_model.fit(matrix)

    # return 'k' nearest neighbors (undeces) for out movie_title
    k_movies = knn_model.kneighbors(movie_title_vec, return_distance=True)
    k_movies_index = k_movies[1][0]

    # deleting the 'movie_title' from the list of neighbors
    k_movies_index = k_movies_index[k_movies_index != movie_index]

    # return the name of the movies
    recom_movies = rating_with_movie.loc[rating_with_movie.movie_index.isin(k_movies_index.tolist()),'title'].unique()
    recom_movies = pd.DataFrame(recom_movies, columns = ['movie_title'])


    # """The block below is used to extract data for the hybrid filtering function"""
    # data frame of movie indeces and distances
    global collab_with_distance
    collab_with_distance = pd.DataFrame({'movie_index': k_movies[1][0], 'distance':k_movies[0][0]})
    collab_with_distance = pd.merge(collab_with_distance, rating_with_movie[['title', 'movie_index']].drop_duplicates(), on = 'movie_index').drop(columns = ['movie_index'])
    if k<20:
        print(f"Because you watched {movie_title}:")

    return recom_movies


# == 4 - Content-based Filtering ==============================================
def content_based_filtering(movie_title, movie_df, k=10):

    """This function returns the list of k most similar movies from the dataset movie_df
       based on cosine similarity of the movies genres."""
       
    movie_df = movie_df.copy()
    movie_df = rating_with_movie[['title', 'genres']].drop_duplicates(subset = ['title']).reset_index().drop(columns=['index'])

    movie_title = search(movie_title)

    # get the set of the unique genre
    genre = {genre for e in movie_df.genres for genre in e}

    # since some movies are of many genres, we need to create separate columns
    # for each genre and assign 1, if the movie is of this genre, 0 - if not

    genre_matrix = pd.DataFrame()
    for e in genre:
        genre_matrix[e] = np.where(movie_df.genres.str.contains(e, regex = False), 1, 0)

    # create cosine similarity based on genre of each movie with each others
    cos_sim = cosine_similarity(genre_matrix, genre_matrix)

    # extract the vector of our 'movie_title' from the cos_sim matrix
    movie_index = movie_df[movie_df.title == movie_title].index

    # getting the array (vector) of similarity scores for our movie_title with all other movies in 'movie_df'
    movie_vec = cos_sim[movie_index][0]

    # mapping movie_vec with movie titles
    
    # """The block below is used to extract data for the hybrid filtering function"""
    global result_with_score
    result_with_score = pd.DataFrame({"title" : movie_df.title, 'score' : movie_vec}).sort_values(by = 'score', ascending=False)
    result = result_with_score.drop(columns = ['score'])
    result = result.reset_index().drop(columns=['index'])
    result.columns = ['movie_title']
    
    

    if k<20:
        print(f"Because you watched {movie_title}:")

    return result.head(k)


# == 4 - Hybrid Filtering ========================================================================================
def hybrid_filtering(movie_title, rating_with_movie, k =10):
    
    movie_title = search(movie_title)

    # n - number of unique movies (will be used to return all neighbors from collaborative-filtering function)
    n = rating_with_movie.title.nunique()

    # this two function we run to have an access to the global variables
    # 'collab_with_distance' and 'result_with_score' (they contain the movie title and corresponding
    # score or distance from collaborative-filtering and content-based-filtering functions
    collab_filtering = collaborative_filtering(movie_title, rating_with_movie, k = n)
    content_filtering = content_based_filtering('scary movie', rating_with_movie, k = n)

    # merge score and distance into one table
    df = pd.merge(result_with_score, collab_with_distance, on = 'title')

    # assign sequence from 1 (highest score and lowest distance) to n (lowest score and highest disctance) for all movies
    # first, replace 'score' values by sequence from 1 (highest score) to the n (lowest score)
    df.score = np.arange(n+1)

    # sort by distance (from smallest to highest), do the same
    df = df.sort_values(by='distance')
    df.distance = np.arange(n+1)

    # create a new column 'hybrid_score' which is sum of collaborative_filtering score and content_based distance
    df['hybrid_score'] = (df.score + df.distance)/2

    # sort by 'hybrid_score' in descending order
    df = df.sort_values(by='hybrid_score')
    df.rename(columns = {'score':'collaborative', 'distance':'content_based'}, inplace = True)
    print(f"Because you watched {movie_title}:")
    df_final = df.title.reset_index().drop(columns = 'index')
    df_final.columns = ['movie_title']
    return df_final.head(k)


# == 5 - Retrieving additional information with LLM (GPT-4o) ============================================================
def get_movie_details(movies):
    openai.api_key = "OPENAI_API_KEY"
    
    example_json = {"movie_title": "Ferris Bueller's Day Off",
                    "country": "USA",
                    "year": 1986,
                    "cast": ["Matthew Broderick", "Alan Ruck", "Mia Sara", "Jennifer Grey", "Jeffrey Jones"],
                    "description": "In this American teen comedy, high school senior Ferris Bueller skips school to embark on a one-day adventure through Chicago, creating an elaborate scheme to avoid getting caught by his suspicious principal."}
    
    data = []
    for movie in movies:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Provide output in valid JSON. The data schema should be like this: "+json.dumps(example_json)},
                {"role": "user", "content": f"Provide the following details for the movie '{movie}':\n"
                                            "1. Country of production\n"
                                            "2. Year of production\n"
                                            "3. Main cast (list up to 5 actors)\n"
                                            "4. Short description (1-2 sentences)\n"
                                            "Format the answer as a JSON object with keys: 'Movie_title', 'Country', 'Year', 'Cast', 'Logline'."}
            ]
        )
        
        details = response.choices[0].message.content
        details = json.loads(details)
        data.append(details)  # Converts the JSON-like response to a Python dictionary

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df


# === 6 - Putting everything in GUI (Streamlit) ===============================
# Page: Exploratory Analysis
if selected == "Exploratory Analysis":
    st.title("Exploratory Analysis")
    st.write("This page contains visualizations and insights about the dataset.")
    st.markdown(f"<p style='font-size:24px; color:black; font-weight:normal;'>Number of unique movies: {num_movie}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:24px; color:black; font-weight:normal;'>Number of ratings: {num_rating}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:24px; color:black; font-weight:normal;'>Number of unique users: {num_users}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:24px; color:black; font-weight:normal;'>Average global rating: {mean_glob_rating}</p>", unsafe_allow_html=True)

    # Example visualization
    st.subheader("Distribution of Ratings")
    st.bar_chart(rating_group.set_index("rating")["userId"])
    
    # Display Pie Chart
    st.subheader("Movie genre popularity")
    st.plotly_chart(fig)

# Filtering Methods
else:
    st.title(selected)
    st.write(f"This page is for {selected} recommendations.")
    
    # Add sublayers
    sublayer = st.sidebar.radio("Choose an option:", ["Recommended movies", "Recommended movies OpenAI"])
    st.write(f"Selected Sub-Option: {sublayer}")
    
    # Search Bar
    movie_query = st.text_input("Search for a movie:", "Terminator")
    movie_name = search(movie_query)

    if movie_query.strip():
        # Perform filtering based on the selected method and sublayer
        if selected == "Collaborative Filtering":
            if sublayer == "Recommended movies":
                results = collaborative_filtering(movie_query, rating_with_movie, k=10)
            else:
                movie_list = collaborative_filtering(movie_query, rating_with_movie, k=10)
                results = get_movie_details(movie_list.movie_title.to_list()[:5])
        elif selected == "Content-Based Filtering":
            if sublayer == "Recommended movies":
                results = content_based_filtering(movie_query, rating_with_movie, k=10)
            else:
                movie_list = content_based_filtering(movie_query, rating_with_movie, k=10)
                results = get_movie_details(movie_list.movie_title.to_list()[:5])
        else:  # Hybrid Filtering
            if sublayer == "Recommended movies":
                results = hybrid_filtering(movie_query, rating_with_movie, k=10)
            else:
                movie_list = hybrid_filtering(movie_query, rating_with_movie, k=10)
                results = get_movie_details(movie_list.movie_title.to_list()[:5])

        # Display recommendations
        if not results.empty:
            st.success(f"Top recommendations for '{movie_name}':")
            st.table(results)
        else:
            st.warning("No movies found. Try a different search query.")
    else:
        st.info("Enter a movie title in the search bar to get recommendations.")
