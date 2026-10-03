import streamlit as st

dashboard = st.Page("reports/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True)
sentimen_prediction = st.Page("tools/sentiment_prediction.py", title="Sentimen Prediction", icon=":material/insights:")    

pg = st.navigation(
    {
        "Reports": [dashboard],
        "Tools": [sentimen_prediction]
    }
) 

pg.run()
