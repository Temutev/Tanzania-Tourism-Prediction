import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from PIL import Image
from sklearn import metrics

def app():
    st.title('Further Research.')
    DATA_URL = ("train_df.csv")

    @st.cache(allow_output_mutation=True)
    def load_data():
        data = pd.read_csv(DATA_URL)
        return data

    # Load rows of data into the dataframe.
    df = load_data()
    
    
    st.image('cbc_output.png')
    st.write('The image above provides an explanation as to how catboost regressor was able to predict the total sales.')
    st.image('feature_imp_cbr_output.png')
    st.write('The image above provides the feature importance values of the catboost regressor model.')
    
    st.image('gradient_boosting_regressor_output.png')
    
    st.write('The image above provides information of how the gradient boosting regressor model overfits on unseen data. Further data is needed to improve the model.')
    st.image('gbr_explanation_output.png')
    st.write('The image above provides the feature importance values of the gradient boosting regressor model.')