from pyparsing import alphas
import streamlit as st

import numpy as np
import pandas as pd

import pydeck as pdk
import altair as alt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# To perform KMeans clustering 
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] =(25,25)


def app():
    st.title('Analysis.')
    DATA_URL = ("train_df.csv")

    @st.cache(allow_output_mutation=True)
    def load_data():
        data = pd.read_csv(DATA_URL)
        return data

    # Load rows of data into the dataframe.
    df = load_data()
    #st.write(df)

    # unique sales rep in the region 
    st.subheader('Tourists Origin Countries.')
    un =  df['country'].nunique()
    st.write('The number of unique countries are :',un)
    
    col1,col2,col3 = st.columns(3)
    col1.caption('Graphical Representation of Country Average Spending in Tshs.')
    country_avg_spending  = pd.DataFrame(df.groupby(['country'])['total_cost'].mean().sort_values(ascending=False)[:10])
    col1.bar_chart(country_avg_spending)
    
    col2.caption('Graphical Representation of Age Groups by  Average Spending in Tshs.')
    age_avg_spending  = pd.DataFrame(df.groupby(['age_group'])['total_cost'].mean().sort_values(ascending=False))
    col2.bar_chart(age_avg_spending)
    
    col3.caption('Graphical Representation of People Accompanied with by  Average Spending in Tshs.')
    travel_avg_spending  = pd.DataFrame(df.groupby(['travel_with'])['total_cost'].mean().sort_values(ascending=False))
    col3.bar_chart(travel_avg_spending)
    
    
    col4,col5,col6 = st.columns(3)
    col4.caption("Graphical Representation of Purpose by  Average Spending in Tshs.")
    purpose_avg_spending  = pd.DataFrame(df.groupby(['purpose'])['total_cost'].mean().sort_values(ascending=False))
    col4.bar_chart(purpose_avg_spending)
    
    
    col5.caption("Graphical Representation of Main Activity by  Average Spending in Tshs.")
    ma_avg_spending  = pd.DataFrame(df.groupby(['main_activity'])['total_cost'].mean().sort_values(ascending=False))
    col5.bar_chart(ma_avg_spending )
    
    
    col6.caption("Graphical Representation of Information Sources by  Average Spending in Tshs.")
    info_avg_spending  = pd.DataFrame(df.groupby(['info_source'])['total_cost'].mean().sort_values(ascending=False))
    col6.bar_chart(info_avg_spending )
    
    
    col7, col8 ,col9 = st.columns(3)
    col7.caption("Graphical Representation of Tour Arrangements by  Average Spending in Tshs.")
    ta_avg_spending  = pd.DataFrame(df.groupby(['tour_arrangement'])['total_cost'].mean().sort_values(ascending=False))
    col7.bar_chart(ta_avg_spending )
    
    
    col8.caption("Graphical Representation of Payment Mode by  Average Spending in Tshs.")
    pm_avg_spending  = pd.DataFrame(df.groupby(['payment_mode'])['total_cost'].mean().sort_values(ascending=False))
    col8.bar_chart(pm_avg_spending )
    
    col9.caption("Graphical Representation of Trips Made  by  Average Spending in Tshs.")
    ftz_avg_spending  = pd.DataFrame(df.groupby(['first_trip_tz'])['total_cost'].mean().sort_values(ascending=False))
    col9.bar_chart(ftz_avg_spending )
    
    
    st.caption("Graphical Representation of Most Impressing by  Average Spending in Tshs.")
    mi_avg_spending  = pd.DataFrame(df.groupby(['most_impressing'])['total_cost'].mean().sort_values(ascending=False))
    st.bar_chart(mi_avg_spending )
    