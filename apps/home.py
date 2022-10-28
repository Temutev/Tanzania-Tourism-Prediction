import streamlit as st
import numpy as np 
import pandas as pd 

def app():
    st.title("Tourism in Tanzania")

    st.header("Introduction")
    st.markdown(
        """
        The Tanzanian tourism sector plays a significant role in the Tanzanian economy, contributing about 17% to the country’s 
        GDP and 25% of all foreign exchange revenues. The sector, which provides direct employment for more than 600,000 people 
        and up to 2 million people indirectly, generated approximately $2.4 billion in 2018 according to government statistics. 
        Tanzania received a record 1.1 million international visitor arrivals in 2014, mostly from Europe, the US and Africa.
        Tanzania is the only country in the world which has allocated more than 25% of its total area for wildlife, national 
        parks, and protected areas.There are 16 national parks in Tanzania, 28 game reserves, 44 game-controlled areas, two 
        marine parks and one conservation area.Tanzania’s tourist attractions include the Serengeti plains, which hosts the largest 
        terrestrial mammal migration in the world; the Ngorongoro Crater, the world’s largest intact volcanic caldera and home to 
        the highest density of big game in Africa; Kilimanjaro, Africa’s highest mountain; and the Mafia Island marine park; among 
        many others. The scenery, topography, rich culture and very friendly people provide for excellent cultural tourism, beach 
        holidays, honeymooning, game hunting, historical and archaeological ventures – and certainly the best wildlife photography 
        safaris in the world.
    """
    )
    st.markdown(
        """
        As such, the data used is a representation of sales that have been done for different consumers 
        within Nairobi . 
        The purpose of this experiment is to provide insights on the data gathered.
        Here is an overview of the dataframe .
    """
    )
    DATA_URL = ("train_df.csv")


    @st.cache(allow_output_mutation=True)
    def load_data():
        data = pd.read_csv(DATA_URL)
        return data


    # Load rows of data into the dataframe.
    df = load_data()
    st.write(df.head())
