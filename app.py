
import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

for pkg in ("vader_lexicon", "stopwords", "punkt", "wordnet"):
    nltk.download(pkg, quiet=True)

st.set_page_config(page_title="YouTube Comment Analyser", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@200;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Newsreader', serif !important;
        font-weight: 200 !important;
        background-color: #121212 !important;
        color: white !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .stApp, .main, .block-container, header, .css-18e3th9 {
        background-color: #121212 !important;
        padding-top: 0px !important;
        margin-top: 0px !important;
    }

    section[data-testid="stTextInput"] {
        width: 30%;
        margin: 0 auto;
        margin-top: 1px;
    }

    input {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        color: white !important;
        padding: 12px 20px !important;
        font-size: 18px !important;
        text-align: left;
        font-family: 'Newsreader', serif !important;
        font-weight: 200 !important;
        transition: all 0.3s ease;
    }

    input::placeholder {
        color: #cccccc;
        font-style: italic;
    }

    input:hover {
        box-shadow: 0 0 6px rgba(255, 255, 255, 0.4);
    }

    input:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    label {
        display: none;
        margin-bottom: 0px;
    }

    .stButton>button {
        background-color: white !important;
        color: black !important;
        border-radius: 10px !important;
        border: 1px solid white !important;
        font-family: 'Newsreader', serif !important;
        font-weight: 400 !important;
        padding: 10px 20px !important;
        transition: none !important;
        margin-top: 20px;
    }

    .stButton>button:hover {
        background-color: #FFFFFF !important;
        transform: scale(1.05);
    }

    .emotion-comment-box {
        border: 1px solid white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #1e1e1e;
        color: white;
        font-family: 'Newsreader', serif !important;
        font-weight: 300;
    }

    h1 {
        font-family: 'Newsreader', serif !important;
        font-weight: 500 !important;
        font-size: 70px;
        text-align: center;
    }

    .custom-label {
        font-size: 25px;
        font-family: 'Newsreader', serif;
        font-weight: 400;
        color: white;
        text-align: left;
        margin-bottom: -40px;
        margin-top: 10px;
    }

    label[for="url_input"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .analysis-card {
        background-color: #1e1e1e;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.6);
    }
    .analysis-card h2, .analysis-card h3, .analysis-card h4 {
        font-family: 'Newsreader', serif;
        font-weight: 500;
        color: white;
        margin-bottom: 15px;
    }
    .analysis-card hr {
        border: 0.5px solid #333333;
        margin: 10px 0 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
st.markdown("<h1>YouTube Comment Analyser</h1>", unsafe_allow_html=True)


st.markdown('<div class="custom-label">Enter YouTube Video URL:</div>', unsafe_allow_html=True)


url = st.text_input("", placeholder="Paste your link here...", key="url_input")


col1, col2, col3 = st.columns([1.5, 1, 1])
with col2:
    st.button("Analyze")




st.sidebar.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 24px;
        font-weight: 500;
        font-variant: small-caps;
        font-family: 'Newsreader', serif;
        color: #ffffff;
        margin-bottom: 30px;
    }

    .sidebar-description {
        font-size: 15px;
        color: #bbbbbb;
        font-family: 'Newsreader', serif;
        line-height: 1.6;
        margin-bottom: 30px;
    }

    .sidebar-section-title {
        font-size: 24px;
        font-weight: 500;
        font-variant: small-caps;
        font-family: 'Newsreader', serif;
        color: #ffffff;
        margin-bottom: 30px;
        margin-top: 0px;
    }

    .sidebar-step {
        font-size: 14px;
        color: #dddddd;
        font-family: 'Newsreader', serif;
        line-height: 1.8;
        margin-left: 5px;
    }

    .sidebar-footer {
        font-size: 12px;
        color: #777777;
        font-family: 'Newsreader', serif;
        line-height: 1.4;
        border-top: 0.5px solid #333333;
        padding-top: 15px;
    }

    .sidebar-hr {
        border: none;
        border-top: 0.5px solid #333333;
        margin: 25px 0;
    }
    </style>

    <div class="sidebar-title">YouTube Comment Analyzer</div>

    <div class="sidebar-description">
        Discover what your audience really feels.<br>
        Analyze sentiment, emotion, and extract insights in seconds.
    </div>

    <hr class="sidebar-hr">
    <div style="height: 50px;"></div> 

    <div class="sidebar-section-title">How to Use</div>
    <div class="sidebar-step">1. Paste a YouTube link</div>
    <div class="sidebar-step">2. Click Analyze</div>
    <div class="sidebar-step">3. See visual results and download CSV</div>

    <div style="height: 460px;"></div> <!-- Pushes footer down -->

    <div class="sidebar-footer">
        Developed with care by <b>Archa</b><br>
        Contact: <i>archasoman9505@gmail.com</i>
    </div>
    """,
    unsafe_allow_html=True
)





EMOTIONS  = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def extract_video_id(url):
    m = re.search(r"(?:v=|\.be/)([\w-]{11})", url)
    return m.group(1) if m else None

def get_comments(youtube, vid, max_comments=100):
    comments, timestamps, token = [], [], None

    while len(comments) < max_comments:
        resp = youtube.commentThreads().list(
            part="snippet", videoId=vid, maxResults=100,
            textFormat="plainText", pageToken=token).execute()

        for item in resp["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            published_at = item["snippet"]["topLevelComment"]["snippet"]["publishedAt"]

            comments.append(comment)
            timestamps.append(published_at)

            if len(comments) >= max_comments:
                break

        token = resp.get("nextPageToken")
        if not token:
            break

    return comments[:max_comments], timestamps[:max_comments]


def get_video_details(youtube, vid):
    request = youtube.videos().list(part="snippet,statistics", id=vid)
    response = request.execute()
    if not response["items"]:
        return None
    item = response["items"][0]
    return {
        "Title": item["snippet"]["title"],
        "Channel Name": item["snippet"]["channelTitle"],
        "Published Date": item["snippet"]["publishedAt"][:10],
        "Views": f'{int(item["statistics"].get("viewCount", 0)):,}',
        "Likes": f'{int(item["statistics"].get("likeCount", 0)):,}',
        "Comments": f'{int(item["statistics"].get("commentCount", 0)):,}',
    }


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def clean(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text).lower()
    words = [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)
             if w not in stop_words and w not in string.punctuation]
    return " ".join(words)


sent_analyzer = SentimentIntensityAnalyzer()
emotion_model = pipeline("text-classification",
                         model="j-hartmann/emotion-english-distilroberta-base",
                         return_all_scores=True)


API_KEY = st.secrets["API_KEY"]

yt = build("youtube", "v3", developerKey=API_KEY)

if url:
    vid = extract_video_id(url)
    if not vid:
        st.error("❌ Invalid YouTube URL")
        st.stop()

    video_details = get_video_details(yt, vid)
    if not video_details:
        st.error("❌ Could not fetch video details. Please check the URL.")
        st.stop()

    with st.spinner("Fetching comments…"):
        comments, timestamps = get_comments(yt, vid)

    if not comments:
        st.warning("😕 This video has no public comments to analyze.")
        st.stop()
        
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    cleaned = [clean(c) for c in comments]
    scores = [sent_analyzer.polarity_scores(t)["compound"] for t in cleaned]
    sent_lbl = ["Positive" if s > .05 else "Negative" if s < -.05 else "Neutral" for s in scores]
    
    with st.spinner("Running emotion model…"):
        raw_emotions = emotion_model(cleaned, truncation=True, batch_size=32)

    st.markdown(
    """
    <div style="
        background-color: white;
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 25px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        text-align: center;
        display: inline-block;
        width: 100%;
    ">
        <h2 style="
            font-family: 'Newsreader', serif;
            font-weight: 500;
            font-size: 32px;
            margin: 0;
            color: black;
        ">
            Here's Your Detailed Analysis
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)


    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .info-card {
            background-color: #1e1e1e;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            margin: 5px;
            height: 100px;
        }
        .info-card h3 {
            font-family: 'Newsreader', serif;
            font-weight: 400;
            font-size: 16px;
            margin: 0;
            color: #bbbbbb;
        }
        .info-card p {
            font-family: 'Newsreader', serif;
            font-weight: 500;
            font-size: 20px;
            margin: 8px 0 0 0;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
<style>
.analysis-card {
    background-color: #1e1e1e;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 25px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.6);
}
.analysis-card h3 {
    color: white;
    margin-top: 0;
}
</style>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="info-card">
                <h3>Channel</h3>
                <p>{video_details['Channel Name']}</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="info-card">
                <h3>Title</h3>
                <p>{video_details['Title'][:25] + "..." if len(video_details['Title']) > 25 else video_details['Title']}</p>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="info-card">
                <h3>Published</h3>
                <p>{video_details['Published Date']}</p>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="info-card">
                <h3>Comments</h3>
                <p>{video_details['Comments']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

  


    emo_rows = []
    for res in raw_emotions:
        row = {e['label']: e['score'] for e in res}
        emo_rows.append({emo: row.get(emo, 0.0) for emo in EMOTIONS})
    emotion_df = pd.DataFrame(emo_rows)

    df = pd.DataFrame({
        "Original Comment": comments,
        "Cleaned Comment" : cleaned,
        "Sentiment"       : sent_lbl
    }).join(emotion_df)
    df["Timestamp"] = pd.to_datetime(timestamps)
    df["Date"] = df["Timestamp"].dt.date
    

    st.markdown("""
    <style>
    .analysis-card {
        background-color: #1e1e1e;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.6);
    }
    .analysis-card h3 {
        color: #f0f2f6;
        margin-top: 0;
        margin-bottom: 15px;
    }
    .stPlotlyChart {
        background-color: #1e1e1e;
        border-radius: 14px;
    }
    .stSelectbox > div {
        background-color: #3b3b3b;
        border-radius: 8px;
    }
    .stMarkdown p, .stMarkdown strong {
        color: #c9c9c9 !important;
    }
    .js-plotly-plot {
        background-color: #1e1e1e;
        border-radius: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

   
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown("<div class='analysis-card'><h3> ⚪ Sentiment Distribution</h3>", unsafe_allow_html=True)
            cnts = df["Sentiment"].value_counts().reset_index()
            cnts.columns = ["Sentiment", "Count"]
            fig = px.pie(
                cnts, names="Sentiment", values="Count",
                color="Sentiment",
                color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"}
            )
            fig.update_layout(
                paper_bgcolor="#1e1e1e",
                font_color="#f0f2f6"
            )
            fig.update_traces(pull=[0.05] * len(cnts))
            st.plotly_chart(fig, use_container_width=True, key="sentiment_chart")
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown("<div class='analysis-card'><h3>⚪ Average Emotion Intensity</h3>", unsafe_allow_html=True)
            means = emotion_df.mean().reset_index()
            means.columns = ["Emotion", "Average Score"]
            fig2 = px.bar(means, x="Emotion", y="Average Score", color="Emotion")
            fig2.update_layout(
                paper_bgcolor="#1e1e1e",
                plot_bgcolor="#1e1e1e",
                font_color="#f0f2f6"
            )
            st.plotly_chart(fig2, use_container_width=True, key="emotion_chart")
            st.markdown("</div>", unsafe_allow_html=True)

    
    with st.container():
        st.markdown("<div class='analysis-card'><h3> ⚪Sentiment Trend Over Time</h3>", unsafe_allow_html=True)
        
        sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
        df["Sentiment_Score"] = df["Sentiment"].map(sentiment_map)
        sentiment_trend = df.groupby("Date")["Sentiment_Score"].mean().reset_index()
        fig3 = px.line(sentiment_trend, x="Date", y="Sentiment_Score", markers=True)
        fig3.update_layout(
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font_color="#f0f2f6",
            xaxis_title="Date", yaxis_title="Avg Sentiment Score",
            yaxis=dict(
                tickmode="array",
                tickvals=[-1, 0, 1],
                ticktext=["Negative", "Neutral", "Positive"]
            )
        )
        st.plotly_chart(fig3, use_container_width=True, key="trend_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown("<div class='analysis-card'><h3>⚪Word Cloud</h3>", unsafe_allow_html=True)
            wc = WordCloud(width=800, height=400, background_color="#1e1e1e", colormap="viridis", max_words=200).generate(" ".join(df["Cleaned Comment"]))
            fig_wc, ax = plt.subplots(figsize=(10, 5), facecolor="#1e1e1e")
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown(
            """
            <div class='analysis-card'>
                <h3>⚪ Download Your Analysis</h3>
            """,
            unsafe_allow_html=True
        )

        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report", csv, "analysis_report.csv", "text/csv")

       
        st.markdown("</div>", unsafe_allow_html=True)

    

           
  
        
  

        
   

    with col2:
        with st.container():
            st.markdown("<div class='analysis-card'><h3>⚪Top Comments by Emotion</h3>", unsafe_allow_html=True)
            chosen = st.selectbox("Choose an emotion", EMOTIONS, index=0)
            top_ids = emotion_df[chosen].nlargest(5).index
            for i in top_ids:
                st.markdown(f"""
                <div style="background:#1e1e1e; border-radius:8px; padding:10px; margin:8px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.6);">
                    <p style="color:#c9c9c9; margin:0;"><strong>• {df['Original Comment'][i]}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

