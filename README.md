# A-Movie-Recommender-System-with-LLM (GPT)
A Python-based application that provides movie recommendations using Collaborative Filtering, Content-Based Filtering, and Hybrid Filtering techniques. This application also integrates OpenAI's LLMs for additional movie information. The UI is built using Streamlit, making it easy to use and share.


# Features
#### Exploratory Analysis:

Visualize dataset insights like rating distribution and popular genres.
Analyze the top and least popular movies based on ratings.

#### Recommendation Techniques:

Collaborative Filtering: Recommends movies based on user behavior and ratings.
Content-Based Filtering: Recommends movies based on their features (e.g., genres).
Hybrid Filtering: Combines both collaborative and content-based approaches for enhanced recommendations.
OpenAI LLM Integration: Fetches additional movie information like the cast, production year, and logline.


## How to Run
#### Ensure you have Python 3.10 and the required libraries. Install dependencies using:

```python
pip install -r requirements.txt
```
#### Run the Application: Navigate to the directory containing recommender_system.py and execute:
```python
streamlit run recommender_system.py
```
#### Open in Browser: 
Streamlit will automatically open the app in your default browser. If not, access it via the URL provided in the terminal (e.g., http://localhost:8501).

## Usage
Select a page from the sidebar:

Exploratory Analysis: View dataset insights.
Collaborative Filtering, Content-Based Filtering, or Hybrid Filtering: Get movie recommendations.
Use the search bar to input a movie title.

Select a sub-option for recommendations:

Recommended Movies: Get a list of movie titles.
Recommended Movies OpenAI: Get detailed movie information using OpenAI.
Review the results in the main section of the app.

## Project Structure
recommender_system.py: Main application file.
movies.csv and ratings.csv: Dataset files.

## Built With
Python 3.10
Streamlit
Pandas
NumPy
Matplotlib and Plotly
scikit-learn
OpenAI AP

## License
This project is for educational purposes only and does not include licensing for datasets or models used.


