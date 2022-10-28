import streamlit as st
from multiapp import MultiApp
from apps import home, analysis,shop, clusmap

st.set_page_config(layout="wide")


apps = MultiApp()

# Add all your application here

apps.add_app("Home", home.app)
apps.add_app("Analysis",analysis.app)
apps.add_app("Predictions", shop.app)
apps.add_app("Further Research", clusmap.app)
# The main app
apps.run()