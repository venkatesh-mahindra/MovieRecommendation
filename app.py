import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Multilingual Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .movie-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    .movie-info {
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .movie-rating {
        font-size: 1.1rem;
        font-weight: bold;
        color: #FF9800;
    }
    .section-divider {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .filter-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .movie-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
    }
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try to load from the CSV file
        return pd.read_csv('movies_data.csv')
    except FileNotFoundError:
        # If file not found, create sample dataset
        return create_sample_dataset()

# Function to create sample dataset (same as in the notebook)
def create_sample_dataset():
    # Define sample data
    data = {
        'movie_id': list(range(1, 101)),
        'title': [
            # English movies
            'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump',
            'Inception', 'The Matrix', 'Interstellar', 'Fight Club', 'Goodfellas',
            'The Lord of the Rings: The Fellowship of the Ring', 'The Silence of the Lambs', 'Saving Private Ryan', 'The Green Mile', 'Schindler\'s List',
            'Gladiator', 'The Departed', 'The Prestige', 'The Lion King', 'Titanic',
            
            # Hindi movies
            'Dangal', 'PK', '3 Idiots', 'Bajrangi Bhaijaan', 'Lagaan',
            'Gully Boy', 'Dil Chahta Hai', 'Zindagi Na Milegi Dobara', 'Queen', 'Andhadhun',
            'Gangs of Wasseypur', 'Barfi!', 'Dil Dhadakne Do', 'Kabhi Khushi Kabhie Gham', 'Rang De Basanti',
            'Taare Zameen Par', 'Lage Raho Munna Bhai', 'Swades', 'Chak De! India', 'Drishyam',
            
            # Telugu movies
            'Baahubali: The Beginning', 'Baahubali 2: The Conclusion', 'Arjun Reddy', 'RRR', 'Pushpa: The Rise',
            'Ala Vaikunthapurramuloo', 'Magadheera', 'Eega', 'Sye Raa Narasimha Reddy', 'Jersey',
            'Mahanati', 'Rangasthalam', 'Fidaa', 'Pokiri', 'Athadu',
            'Okkadu', 'Nuvvu Nenu', 'Kushi', 'Bommarillu', 'Awe!',
            
            # Tamil movies
            'Vikram', 'Master', 'Vada Chennai', 'Super Deluxe', 'Asuran',
            'Kaithi', 'Pariyerum Perumal', '96', 'Aruvi', 'Visaranai',
            'Jai Bhim', 'Soorarai Pottru', 'Karnan', 'Kaithi', 'Peranbu',
            
            # Malayalam movies
            'Drishyam', 'Kumbalangi Nights', 'Premam', 'Bangalore Days', 'Maheshinte Prathikaaram',
            'Thondimuthalum Driksakshiyum', 'Ee.Ma.Yau', 'Jallikattu', 'Angamaly Diaries', 'Sudani from Nigeria',
            
            # Kannada movies
            'KGF: Chapter 1', 'KGF: Chapter 2', 'Kantara', 'Lucia', 'Ulidavaru Kandanthe',
            'Rangitaranga', 'U Turn', 'Kirik Party', 'Dia', 'Godhi Banna Sadharana Mykattu'
        ],
        'year': [random.randint(1990, 2023) for _ in range(100)],
        'genre': [
            random.choice(['Drama', 'Action', 'Comedy', 'Thriller', 'Romance', 'Sci-Fi', 'Adventure', 'Fantasy', 'Crime', 'Horror'])
            for _ in range(100)
        ],
        'language': [
            *['English'] * 20,
            *['Hindi'] * 20,
            *['Telugu'] * 20,
            *['Tamil'] * 15,
            *['Malayalam'] * 10,
            *['Kannada'] * 15
        ],
        'rating': [round(random.uniform(5.0, 10.0), 1) for _ in range(100)],
        'actor': [
            # English actors
            'Tom Hanks', 'Leonardo DiCaprio', 'Brad Pitt', 'Robert Downey Jr.', 'Denzel Washington',
            'Christian Bale', 'Tom Cruise', 'Johnny Depp', 'Morgan Freeman', 'Matt Damon',
            'Hugh Jackman', 'Will Smith', 'Matthew McConaughey', 'Ryan Gosling', 'Chris Hemsworth',
            'Chris Evans', 'Keanu Reeves', 'Joaquin Phoenix', 'Jake Gyllenhaal', 'Benedict Cumberbatch',
            
            # Hindi actors
            'Aamir Khan', 'Shah Rukh Khan', 'Salman Khan', 'Amitabh Bachchan', 'Ranbir Kapoor',
            'Ranveer Singh', 'Hrithik Roshan', 'Akshay Kumar', 'Irrfan Khan', 'Nawazuddin Siddiqui',
            'Rajkummar Rao', 'Ayushmann Khurrana', 'Vicky Kaushal', 'Pankaj Tripathi', 'Manoj Bajpayee',
            'Ajay Devgn', 'Varun Dhawan', 'Shahid Kapoor', 'Sushant Singh Rajput', 'Anil Kapoor',
            
            # Telugu actors
            'Prabhas', 'Allu Arjun', 'Ram Charan', 'Jr NTR', 'Mahesh Babu',
            'Nani', 'Vijay Deverakonda', 'Chiranjeevi', 'Nagarjuna', 'Venkatesh',
            'Ravi Teja', 'Naga Chaitanya', 'Rana Daggubati', 'Sai Dharam Tej', 'Nikhil Siddhartha',
            'Adivi Sesh', 'Sudheer Babu', 'Sharwanand', 'Naveen Polishetty', 'Vishwak Sen',
            
            # Tamil actors
            'Vijay', 'Ajith Kumar', 'Suriya', 'Dhanush', 'Vikram',
            'Kamal Haasan', 'Rajinikanth', 'Karthi', 'Sivakarthikeyan', 'Vijay Sethupathi',
            'Madhavan', 'Arya', 'Jayam Ravi', 'Simbu', 'Vishal',
            
            # Malayalam actors
            'Mohanlal', 'Mammootty', 'Fahadh Faasil', 'Dulquer Salmaan', 'Nivin Pauly',
            'Tovino Thomas', 'Prithviraj Sukumaran', 'Asif Ali', 'Kunchacko Boban', 'Biju Menon',
            
            # Kannada actors
            'Yash', 'Sudeep', 'Darshan', 'Puneeth Rajkumar', 'Upendra',
            'Rakshit Shetty', 'Ganesh', 'Shiva Rajkumar', 'Duniya Vijay', 'Dhruva Sarja',
            'Diganth', 'Prajwal Devaraj', 'Srimurali', 'Dhananjay', 'Rishabh Shetty'
        ],
        'actress': [
            # English actresses
            'Meryl Streep', 'Jennifer Lawrence', 'Scarlett']  [
            # English actresses
            'Meryl Streep', 'Jennifer Lawrence', 'Scarlett Johansson', 'Emma Stone', 'Natalie Portman',
            'Cate Blanchett', 'Anne Hathaway', 'Viola Davis', 'Nicole Kidman', 'Kate Winslet',
            'Charlize Theron', 'Emma Watson', 'Jessica Chastain', 'Amy Adams', 'Margot Robbie',
            'Saoirse Ronan', 'Brie Larson', 'Jennifer Aniston', 'Sandra Bullock', 'Angelina Jolie',
            
            # Hindi actresses
            'Deepika Padukone', 'Alia Bhatt', 'Priyanka Chopra', 'Kareena Kapoor', 'Katrina Kaif',
            'Vidya Balan', 'Kangana Ranaut', 'Anushka Sharma', 'Taapsee Pannu', 'Shraddha Kapoor',
            'Kiara Advani', 'Kriti Sanon', 'Sara Ali Khan', 'Janhvi Kapoor', 'Bhumi Pednekar',
            'Sonam Kapoor', 'Madhuri Dixit', 'Kajol', 'Rani Mukerji', 'Aishwarya Rai',
            
            # Telugu actresses
            'Samantha Ruth Prabhu', 'Anushka Shetty', 'Kajal Aggarwal', 'Pooja Hegde', 'Rashmika Mandanna',
            'Keerthy Suresh', 'Nayanthara', 'Tamannaah Bhatia', 'Rakul Preet Singh', 'Sai Pallavi',
            'Shruti Haasan', 'Nithya Menen', 'Ritu Varma', 'Raashi Khanna', 'Regina Cassandra',
            'Lavanya Tripathi', 'Eesha Rebba', 'Mehreen Pirzada', 'Nabha Natesh', 'Krithi Shetty',
            
            # Tamil actresses
            'Trisha Krishnan', 'Jyothika', 'Aishwarya Rajesh', 'Keerthy Suresh', 'Nayanthara',
            'Samantha Ruth Prabhu', 'Kajal Aggarwal', 'Tamannaah Bhatia', 'Shruti Haasan', 'Nithya Menen',
            'Aishwarya Rai', 'Amala Paul', 'Hansika Motwani', 'Anushka Shetty', 'Sai Pallavi',
            
            # Malayalam actresses
            'Manju Warrier', 'Parvathy Thiruvothu', 'Nazriya Nazim', 'Nayanthara', 'Nimisha Sajayan',
            'Anna Ben', 'Aishwarya Lekshmi', 'Aparna Balamurali', 'Grace Antony', 'Rajisha Vijayan',
            
            # Kannada actresses
            'Rachita Ram', 'Radhika Pandit', 'Rashmika Mandanna', 'Shanvi Srivastava', 'Shraddha Srinath',
            'Haripriya', 'Ashika Ranganath', 'Nabha Natesh', 'Srinidhi Shetty', 'Aditi Prabhudeva',
            'Meghana Raj', 'Ragini Dwivedi', 'Amulya', 'Malashri', 'Ramya'
        ],
        'poster_url': [f'https://picsum.photos/id/{i + 100}/300/450' for i in range(100)]
    }
    
    return pd.DataFrame(data)

# Load the movie data
movies_df = load_data()

# App header
st.markdown('<div class="main-header">🎬 Multilingual Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('Discover movies across Telugu, Hindi, Malayalam, English, Tamil, and Kannada languages')
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Sidebar for filters
st.sidebar.image("https://picsum.photos/id/1037/300/100", use_column_width=True)
st.sidebar.title("Movie Filters")

# Language filter
selected_language = st.sidebar.selectbox(
    "Select Language",
    ["All"] + sorted(movies_df['language'].unique().tolist())
)

# Genre filter
selected_genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + sorted(movies_df['genre'].unique().tolist())
)

# Actor filter
all_actors = sorted(movies_df['actor'].unique().tolist())
selected_actor = st.sidebar.selectbox(
    "Select Actor",
    ["All"] + all_actors
)

# Actress filter
all_actresses = sorted(movies_df['actress'].unique().tolist())
selected_actress = st.sidebar.selectbox(
    "Select Actress",
    ["All"] + all_actresses
)

# Year range filter
min_year = int(movies_df['year'].min())
max_year = int(movies_df['year'].max())
year_range = st.sidebar.slider(
    "Release Year",
    min_year, max_year, (min_year, max_year)
)

# Rating filter
min_rating = float(movies_df['rating'].min())
max_rating = float(movies_df['rating'].max())
rating_range = st.sidebar.slider(
    "Minimum Rating",
    min_rating, max_rating, min_rating
)

# Apply filters
filtered_movies = movies_df.copy()

if selected_language != "All":
    filtered_movies = filtered_movies[filtered_movies['language'] == selected_language]

if selected_genre != "All":
    filtered_movies = filtered_movies[filtered_movies['genre'] == selected_genre]

if selected_actor != "All":
    filtered_movies = filtered_movies[filtered_movies['actor'] == selected_actor]

if selected_actress != "All":
    filtered_movies = filtered_movies[filtered_movies['actress'] == selected_actress]

# Apply year range filter
filtered_movies = filtered_movies[
    (filtered_movies['year'] >= year_range[0]) & 
    (filtered_movies['year'] <= year_range[1])
]

# Apply rating filter
filtered_movies = filtered_movies[filtered_movies['rating'] >= rating_range]

# Sort by rating
filtered_movies = filtered_movies.sort_values(by='rating', ascending=False)

# Display statistics and visualizations
st.markdown('<div class="sub-header">📊 Movie Statistics</div>', unsafe_allow_html=True)

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Language Distribution", "Genre Distribution", "Top Actors"])

with tab1:
    # Language distribution
    st.subheader("Movies by Language")
    language_counts = movies_df['language'].value_counts()
    
    # Create a pie chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(language_counts, labels=language_counts.index, autopct='%1.1f%%', 
           startangle=90, colors=sns.color_palette('Set3', len(language_counts)))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    st.pyplot(fig)

with tab2:
    # Genre distribution
    st.subheader("Movies by Genre")
    genre_counts = movies_df['genre'].value_counts()
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    genre_counts.plot(kind='bar', ax=ax, color=sns.color_palette('Set2', len(genre_counts)))
    plt.title('Number of Movies by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab3:
    # Top actors
    st.subheader("Top Actors with Most Movies")
    actor_counts = movies_df['actor'].value_counts().head(10)
    
    # Create a horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    actor_counts.plot(kind='barh', ax=ax, color=sns.color_palette('viridis', len(actor_counts)))
    plt.title('Top 10 Actors with Most Movies')
    plt.xlabel('Number of Movies')
    plt.ylabel('Actor')
    st.pyplot(fig)

# Display recommended movies
st.markdown('<div class="sub-header">🎥 Recommended Movies</div>', unsafe_allow_html=True)

if len(filtered_movies) == 0:
    st.warning("No movies found with the selected filters. Please adjust your criteria.")
else:
    st.success(f"Found {len(filtered_movies)} movies matching your criteria")
    
    # Display movies in a grid
    cols = 3
    for i in range(0, len(filtered_movies), cols):
        row = st.columns(cols)
        for j in range(cols):
            if i + j < len(filtered_movies):
                movie = filtered_movies.iloc[i + j]
                with row[j]:
                    st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
                    
                    # Movie poster
                    try:
                        st.image(movie['poster_url'], use_column_width=True)
                    except:
                        st.image("https://picsum.photos/300/450", use_column_width=True)
                    
                    # Movie details
                    st.markdown(f'<div class="movie-title">{movie["title"]} ({movie["year"]})</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-info"><b>Genre:</b> {movie["genre"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-info"><b>Language:</b> {movie["language"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-info"><b>Actor:</b> {movie["actor"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-info"><b>Actress:</b> {movie["actress"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-rating">⭐ {movie["rating"]}/10</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('Developed with ❤️ by Movie Recommendation System Team')

# Add a random movie recommendation feature
if st.sidebar.button("Surprise Me! 🎲"):
    random_movie = movies_df.sample(1).iloc[0]
    
    st.sidebar.markdown("### Random Movie Pick")
    st.sidebar.image(random_movie['poster_url'], use_column_width=True)
    st.sidebar.markdown(f"**{random_movie['title']}** ({random_movie['year']})")
    st.sidebar.markdown(f"**Genre:** {random_movie['genre']}")
    st.sidebar.markdown(f"**Language:** {random_movie['language']}")
    st.sidebar.markdown(f"**Rating:** ⭐ {random_movie['rating']}/10")

# Add about section
with st.sidebar.expander("About This App"):
    st.write("""
    This Multilingual Movie Recommendation System helps you discover movies across multiple languages including Telugu, Hindi, Malayalam, English, Tamil, and Kannada.
    
    Use the filters to find movies based on language, genre, actors, and more!
    
    Data is generated for demonstration purposes.
    """)
