import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  MinMaxScaler
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor

final_mode = 'gradient_boosting_regressor_model.sav'
cbr_mode ='catboost_regressor_model.sav'
le = LabelEncoder()
scaler = MinMaxScaler(feature_range=(0, 1))

rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()
cbr = CatBoostRegressor()


df = pd.read_csv('train_df.csv')

train_df = df.copy()


df.drop(columns =['ID','package_transport_int','package_accomodation'	,
                    'package_food','package_transport_tz','package_sightseeing','package_guided_tour',
                    'package_insurance','most_impressing'] ,inplace=True)
    
df['age_group'] = le.fit_transform(df['age_group'])
df['first_trip_tz'] = le.fit_transform(df['first_trip_tz'])
df['country'] = le.fit_transform(df['country'])
df['travel_with'] = le.fit_transform(df['travel_with'])
df['purpose'] = le.fit_transform(df['purpose'])
df['main_activity'] = le.fit_transform(df['main_activity'])
df['info_source'] = le.fit_transform(df['info_source'])
df['tour_arrangement'] = le.fit_transform(df['tour_arrangement'])
df['payment_mode'] = le.fit_transform(df['payment_mode'])
    
  
train_df['total_female'] = train_df['total_female'].astype(int)
train_df['total_male'] = train_df['total_male'].astype(int)
train_df['night_mainland'] = train_df['night_mainland'].astype(int)
train_df['night_zanzibar'] = train_df['night_zanzibar'].astype(int)

target = df['total_cost']
feat_cols = df.drop(["total_cost"],1)
cols = feat_cols.columns
    
rfr.fit(df[cols],target)
gbr.fit(df[cols],target)


for i in range(10):
    ct=CatBoostRegressor(iterations=1000, 
                        loss_function='MAE',
                        logging_level='Silent',
                        depth = i
                        )
    ct.fit(df[cols], target)

def app():
    st.title('Prediction')
    
    #DATA_URL = ("train_df.csv")

    #@st.cache(allow_output_mutation=True)
    #def load_data():
    #    data = pd.read_csv(DATA_URL)
    #    return data

    # Load rows of data into the dataframe.
    #df = load_data()
    
    #train_df = load_data()
    
    
    

    
    #st.dataframe(df.head())
    
    st.write('Fill in the following information to find out how much a tourist is likely to spend in Tanzania.')
    
    
    col1,col2,col3 = st.columns(3)
    
    country = col1.selectbox('Country',train_df['country'].unique())
    
    age_group = col1.selectbox('Age Group',train_df['age_group'].unique())
    
    travel_with = col1.selectbox('Travel With',train_df['travel_with'].unique())

    total_male = col1.slider('Total Male',0,20)
    
    total_female = col1.slider('Total Female',0,20)
    
    
    purpose = col2.selectbox('Purpose of Visit',train_df['purpose'].unique())
    
    main_activity=  col2.selectbox('Main Activity',train_df['main_activity'].unique())
    
    info_source = col2.selectbox('Info Source',train_df['info_source'].unique())
    
    tour_arrangement= col2.selectbox('Tour Arrangement',train_df['tour_arrangement'].unique())
    
    night_zanzibar = col2.slider('Night Zanzibar',0,30)
    
    night_mainland = col3.slider('Night Mainland',0,30)
    
    payment_mode = col3.selectbox('Payment Mode',train_df['payment_mode'].unique())
    
    
    first_trip_tz = col3.radio('First Trip to Tanzania',train_df['first_trip_tz'].unique())
    
    
    
    
    if col3.button('Predict'):
    
        # Here we get two more features of the total number of people and the total number of nights spent
        total_persons = total_female + total_male
        total_nights_spent = night_mainland +night_zanzibar
        
        country = le.fit_transform(np.array(country).reshape(-1,1))
        age_group = le.fit_transform(np.array(age_group).reshape(-1,1))
        travel_with = le.fit_transform(np.array(travel_with).reshape(-1,1))
        
        purpose = le.fit_transform(np.array(purpose).reshape(-1,1))
        main_activity = le.fit_transform(np.array(main_activity).reshape(-1,1))
        info_source = le.fit_transform(np.array(info_source).reshape(-1,1))
        tour_arrangement = le.fit_transform(np.array(tour_arrangement).reshape(-1,1))
        first_trip_tz = le.fit_transform(np.array(first_trip_tz).reshape(-1,1))
        payment_mode = le.fit_transform(np.array(payment_mode).reshape(-1,1))
        
        usInf = pd.DataFrame({
            "country":country,
            "age_group":age_group,
            "travel_with":travel_with,
            "total_female":total_female,
            "total_male":total_male,
            "purpose":purpose,
            "main_activity":main_activity,
            "info_source":info_source,
            "tour_arrangement":tour_arrangement,
            "night_mainland":night_mainland,
            "night_zanzibar":night_zanzibar,
            "payment_mode":payment_mode,
            "first_trip_tz":first_trip_tz,
            "total_persons":total_persons,
            "total_nights_spent":total_nights_spent,
            
            
        }, index=[0])
        
        #usInf = scaler.fit_transform(usInf)
        
        gbr_loaded = joblib.load(final_mode)
        cbr_loaded =joblib.load(cbr_mode)
        
        
        gbr_result = gbr_loaded.predict(usInf)
        cbr_result = cbr_loaded.predict(usInf)
        
        rfr_preds = rfr.predict(usInf)
        gbr_preds = gbr.predict(usInf)
        cbr_preds = ct.predict(usInf)
        
        st.write('The predicted amount of money a tourist is likely to spend in Tanzania in USD using RandomForestRegressor is : ', rfr_preds[0]* 0.00043)
        
        st.write('The predicted amount of money a tourist is likely to spend in Tanzania in USD using GradientBoostingRegressor is: ',gbr_preds[0] *0.00043)
        
        st.write('The predicted amount of money a tourist is likely to spend in Tanzania in USD using CatBoostRegressor is: ',cbr_preds[0] *0.00043)
        
        
    
