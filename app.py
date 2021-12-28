import pandas as pd
import streamlit as st
import pickle
import re
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
from PIL import Image

st.title('Book Genre Prediction and  Recommendation System')



books = pickle.load(open('books.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# text cleaning,lowercase,removing special characters and stemming words. Note: We are handling stopwords directly using CountVectorizer's inbuilt stop_words functions.

ps = PorterStemmer()


def clean_summary(text):
    # removing everything other than alphabets and numbers with spaces
    text = re.sub('\W+', ' ', text)
    text = text.lower()  # converts all the text to lowercase
    # stemming words now.
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# getting the data from the user. It can be a description or summary of any book.

text_in = st.text_area('Please enter any summary/description of the book.')
rec = [clean_summary(text_in)]

# now once the user clicks on the predict and recommend button.The following will happen.
## Model will predict the genre of the description.
## the input text will be vectorized and then matched with outher rows in tags columns of Books df to get a similarity score.
## finally we will use the text entered the get the best cosine similarity score for 5 books and display them.

if st.button('Predict genre and Recommend books'):
    # predicting genre
    t = cv.transform(rec).toarray()
    le = preprocessing.LabelEncoder()
    le.fit_transform(books.genre)
    pr = le.inverse_transform(model.predict(t))
    st.write('The predicted genre is :', pr[0])

    # Recommending books now:
    tags = pd.DataFrame({'tags': rec})
    df_tags = books[['tags']]
    tags = tags.append(df_tags, ignore_index=True)
    tags_test_cv = cv.fit_transform(tags['tags']).toarray()
    similarity_tags = cosine_similarity(tags_test_cv)
    recommended_sorted_list = sorted(list(enumerate(similarity_tags[0])), reverse=True, key=lambda x: x[1])[1:6]
    recommended_books = []
    recommended_book_posters = []
    recommended_ar = []
    for i in recommended_sorted_list:
        recommended_books.append(books.iloc[i[0]].title)
        recommended_book_posters.append(books.iloc[i[0]].image_url)
        recommended_ar.append(books.iloc[i[0]].average_rating)

    st.markdown("<h4 style='text-align: center; color: red;'>5 Books that match the input summary/description.</h4>",
                unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_books[0])
        st.image(recommended_book_posters[0], caption=recommended_books[0],width=200)

    with col2:
        st.text(recommended_books[1])
        st.image(recommended_book_posters[1], caption=recommended_books[1],width=200)

    with col3:
        st.text(recommended_books[2])
        st.image(recommended_book_posters[2], caption=recommended_books[2],width=200)

    with col4:
        st.text(recommended_books[3])
        st.image(recommended_book_posters[3], caption=recommended_books[3],width=200)

    with col5:
        st.text(recommended_books[4])
        st.image(recommended_book_posters[4], caption=recommended_books[4],width=200)






