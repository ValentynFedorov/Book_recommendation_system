import pickle
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

model_path = 'artifacts/model1.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

df = model['dataframe']

df['description'] = df['description'].apply(lambda x: str(x) if isinstance(x, str) else '')

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description']) 
df['textual_representation'] = list(tfidf_matrix.toarray())  

def recommend_book(book_name):
    try:
        selected_book = df[df['title'].str.contains(book_name, case=False)].iloc[0]

        vector = selected_book['textual_representation']
        vectors = df['textual_representation'].tolist()

        similarity = cosine_similarity([vector], vectors)[0]
        similar_indices = np.argsort(-similarity)[1:5]

        recommendations = df.iloc[similar_indices]
        return recommendations
    except IndexError:
        st.error("Book not found. Please check the spelling or try another title.")
        return None

st.title('Books Recommender')

book_names = df['title'].tolist()
selected_book = st.text_input(
    "Type the name of a book or select from the dropdown",
    ""
)

if st.button('Show Recommendation'):
    if not selected_book:
        st.error("Please enter a book title.")
    else:
        recommendations = recommend_book(selected_book)

        if recommendations is not None:
            for _, row in recommendations.iterrows():
                col1, col2 = st.columns([1, 2])  

                with col1:
                    st.image(row['thumbnail'], width=150)

                with col2:
                    st.subheader(row['title'])
                    st.write(f"**Author:** {row['authors']}")
                    st.write(f"**Categories:** {row['categories']}")
                    st.write(f"**Publishing year:** {row['published_year']}")
                    st.write(f"**Average Rating:** {row['average_rating']}")
                    st.write(f"**Number of Pages:** {row['num_pages']}")
                    st.write("---")

                    st.text_area("Description", row['description'], height=200, max_chars=None)