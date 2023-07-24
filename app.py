import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# laoding models

new_df=pd.read_csv("new.csv")
# similarity = pickle.load(open('similarity.pkl','rb'))

tfidvector = TfidfVectorizer()
matrix = tfidvector.fit_transform(new_df['features'][10000:])
similarity = cosine_similarity(matrix)


def recommendation(anime_df):
    idx = new_df[new_df['Name'] == anime_df].index[0]
    distances = sorted(list(enumerate(similarity[idx])),reverse=True,key=lambda x:x[1])
    
    anime = []
    for m_id in distances[1:21]:
        anime.append(new_df.iloc[m_id[0]].Name)
        
    return anime

list_anime=np.array(new_df["Name"])
option = st.selectbox(
"Select songs ",
(list_anime))


if st.button('Recommend Me'):
     st.write('anime Recomended for you are:')
     # st.write(anime_recommend(option),show_url(option))
     df = pd.DataFrame({
          'anime Recommended': recommendation(option),
     })

     st.table(df)
