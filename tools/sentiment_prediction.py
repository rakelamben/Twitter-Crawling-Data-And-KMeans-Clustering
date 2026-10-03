import time

import streamlit as st
from transformers import AutoTokenizer, pipeline
import joblib
import re


COLORS = {"positive": "#8fe4b2", "neutral": "#f39c12", "negative": "#e74c3c"}
WELCOME = "Masukkan teks yang ingin dianalisis sentimennya. Misalnya: 'Saya sangat senang hari ini!'"

STEP_DELAY = 0.4

@st.cache_resource(show_spinner="Memuat model sentimen (pertama kali bisa agak lama)...")
def load_pipeline():
    model = joblib.load("models/model_sentiment_tweet.joblib")
    return model

def analyze_token_sentiment(docx, pipeline):
    pos_list = []
    neg_list = []
    neu_list = []
    
    for word in docx.split():
        # Predict sentiment for each token using the trained pipeline
        prediction = pipeline.predict([word])[0]
        
        # Check if the pipeline supports predict_proba
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba([word])
            proba_max = proba.max() # Get the highest probability
        else:
            proba_max = None # If predict_proba is not available, return None
            
        if prediction == 'positive':
            pos_list.append([word, proba_max])
        elif prediction == 'negative':
            neg_list.append([word, proba_max])
        else:
            neu_list.append([word, proba_max])
            
    result = {
        'positives': pos_list,
        'negatives': neg_list,
        'neutral': neu_list,
    }
    return result


def render_message(message):
    with st.chat_message(message["role"]):
        if message["role"] == "ai" and "score" in message:
            color = COLORS.get(message["content"], "#ffffff")
            st.markdown(
                f"<div style='color:{color}; font-weight:bold'>"
                f"Sentimen: {message['content'].title()}</div>",
                unsafe_allow_html=True,
            )
            st.write("Skor per token:", message["score"])
        else:
            st.write(message["content"])


def main():
    st.set_page_config(page_title="Analisis Sentimen Data Twitter", page_icon=":speech_balloon:")

    # Loading model: spinner muncul hanya saat model belum ada di cache
    sentiment_analysis = load_pipeline()
    if not st.session_state.get("model_ready"):
        st.toast("Model siap digunakan!", icon="✅")
        st.session_state.model_ready = True

    st.title("Analisis Sentimen Data Twitter")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": WELCOME}]

    if st.button("Reset Chat"):
        st.write("Menghapus semua pesan...")
        time.sleep(STEP_DELAY)
        st.session_state.messages = [{"role": "ai", "content": WELCOME}]
        st.rerun()

    st.divider()

    for message in st.session_state.messages:
        render_message(message)

    teks = st.chat_input("Masukkan teks di sini...")

    if teks and teks.strip():
        user_msg = {"role": "human", "content": f"Anda memasukkan: {teks}"}
        st.session_state.messages.append(user_msg)
        render_message(user_msg)  # tampil langsung, tidak menunggu rerun

        with st.chat_message("ai"):
            with st.status("Menganalisis sentimen...", expanded=True) as status:
                st.write("Memproses teks...")
                time.sleep(STEP_DELAY)

                st.write("Menjalankan model BERT...")
                result = sentiment_analysis.predict([teks])[0]
                time.sleep(STEP_DELAY)

                status.update(label="Analisis selesai", state="complete", expanded=False)

            ai_msg = {
                "role": "ai",
                "content": re.sub(r'[^a-z\s]', ' ', result),
                "score": analyze_token_sentiment(teks, sentiment_analysis)
            }
            st.session_state.messages.append(ai_msg)

            color = COLORS[ai_msg["content"].strip()]
            st.markdown(
                f"<div style='color:{color}; font-weight:bold'>"
                f"Sentimen: {ai_msg['content'].title()}</div>",
                unsafe_allow_html=True,
            )
            st.write("Skor per token:", ai_msg["score"])


if __name__ == "__main__":
    main()