import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
from datetime import date


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Social Media Data Analysis System",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# ADITYA UNIVERSITY STYLE COLOURS & UI
# ============================================================

st.markdown("""
<style>

    /* PAGE BACKGROUND */
    .stApp {
        background-color: #F5F9FD;
    }

    /* MAIN TITLE */
    h1 {
        color: #004F9E !important;
        text-align: center !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
    }

    /* SUB HEADINGS */
    h2, h3 {
        color: #0056A6 !important;
        font-weight: 700 !important;
    }

    /* NORMAL TEXT */
    p, label {
        color: #222222;
    }

    /* HOME PAGE BOXES */
    div[data-testid="column"] {
        background-color: #FFFFFF;
        border: 2px solid #D7E8F7;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0px 4px 12px rgba(0, 79, 158, 0.10);
    }

    /* ALL BUTTONS - ORANGE */
    .stButton > button {
        width: 100%;
        background-color: #F58220 !important;
        color: #FFFFFF !important;
        border: 2px solid #F58220 !important;
        border-radius: 8px;
        font-weight: 700;
        padding: 10px 20px;
        transition: 0.2s;
    }

    /* BUTTON HOVER */
    .stButton > button:hover {
        background-color: #D96B0B !important;
        color: #FFFFFF !important;
        border-color: #D96B0B !important;
    }

    /* TEXT INPUT */
    .stTextInput input {
        border: 2px solid #0056A6;
        border-radius: 8px;
        background-color: #FFFFFF;
        color: #222222 !important;
    }

    .stTextInput input::placeholder {
        color: #777777 !important;
        opacity: 1;
    }

    .stTextInput input:focus {
        border-color: #F58220;
        box-shadow: 0 0 0 1px #F58220;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border: 2px solid #D7E8F7;
        border-radius: 10px;
        padding: 10px;
    }

    /* DIVIDERS */
    hr {
        border-color: #F58220;
    }

    /* METRIC BOXES */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-left: 5px solid #F58220;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0px 3px 8px rgba(0, 79, 158, 0.08);
    }

    /* ALERT BOXES */
    [data-testid="stAlert"] {
        border-radius: 8px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "home",
    "analysis_done": False,
    "selected_result": None,
    "word_frequency": Counter(),
    "sentiments": [],
    "categories": [],
    "processed_data": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "the", "is", "a", "an", "and", "or", "of", "to", "in",
    "on", "for", "with", "this", "that", "it", "are", "was",
    "were", "be", "as", "at", "by", "from", "has", "have",
    "had", "will", "would", "can", "could", "should", "you",
    "your", "we", "our", "they", "their", "i", "me", "my",
    "he", "she", "his", "her", "but", "not", "so", "if",
    "about", "into", "than", "then", "also", "very", "more",
    "these", "those", "such", "its", "which", "who",
    "what", "when", "where", "how", "been", "being",
    "there", "here", "may", "one", "two", "many", "some",
    "other", "used", "use", "using"
}


# ============================================================
# SENTIMENT WORD LISTS
# ============================================================

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome",
    "happy", "love", "like", "best", "wonderful",
    "success", "successful", "beautiful", "nice",
    "positive", "enjoy", "enjoyed", "helpful", "perfect",
    "benefit", "benefits", "improve", "improved",
    "improvement", "support", "supported", "valuable",
    "effective", "interesting", "popular", "strong",
    "opportunity", "opportunities", "advantage",
    "advantages", "progress", "growth"
}


NEGATIVE_WORDS = {
    "bad", "worst", "poor", "hate", "sad", "angry",
    "terrible", "awful", "problem", "problems", "fail",
    "failed", "negative", "boring", "difficult",
    "disappointed", "disappointing", "wrong", "issue",
    "issues", "harm", "harmful", "danger", "dangerous",
    "risk", "risks", "threat", "threats", "abuse",
    "violence", "crime", "criminal", "fake", "false",
    "misinformation", "controversy", "controversial",
    "criticism", "criticized", "damage", "damaging",
    "loss", "losses", "failure"
}


# ============================================================
# CATEGORY KEYWORDS
# ============================================================

CATEGORY_KEYWORDS = {

    "Technology": {
        "technology", "internet", "software", "platform",
        "computer", "digital", "online", "website",
        "application", "applications", "app", "apps",
        "mobile", "smartphone", "network", "data",
        "algorithm", "artificial", "intelligence",
        "device", "devices"
    },

    "Business": {
        "business", "marketing", "advertising",
        "company", "companies", "brand", "brands",
        "market", "commerce", "sales", "customer",
        "customers", "economic", "economy", "profit",
        "businesses", "industry"
    },

    "Society": {
        "society", "social", "community", "people",
        "users", "user", "communication", "interaction",
        "culture", "public", "population", "relationship",
        "relationships", "citizens", "human"
    },

    "Politics": {
        "politics", "political", "government", "election",
        "elections", "party", "parties", "president",
        "minister", "policy", "policies", "democracy",
        "vote", "voting", "parliament", "law", "laws"
    },

    "Health": {
        "health", "medical", "medicine", "disease",
        "diseases", "mental", "hospital", "doctor",
        "doctors", "healthcare", "wellness", "treatment",
        "virus", "infection", "patients", "patient"
    },

    "Education": {
        "education", "school", "schools", "college",
        "university", "universities", "student",
        "students", "teacher", "teachers", "learning",
        "learn", "academic", "course", "courses",
        "study", "research"
    },

    "Entertainment": {
        "entertainment", "music", "movie", "movies",
        "film", "films", "video", "videos", "game",
        "games", "gaming", "celebrity", "celebrities",
        "television", "tv", "artist", "artists",
        "sports", "sport"
    }
}


# ============================================================
# PLATFORMS THAT BLOCK WEB SCRAPING
# ============================================================

BLOCKED_PLATFORMS = [
    "instagram.com",
    "facebook.com",
    "fb.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "reddit.com",
    "linkedin.com",
    "snapchat.com",
    "threads.net"
]


SCRAPING_MESSAGE = (
    "The Web Scraping module collects publicly accessible textual "
    "content from webpages. Direct scraping of major social-media "
    "platforms is restricted. Please use the Dataset option to "
    "analyze social-media data."
)


def is_blocked_platform(url):

    host = (
        url.lower()
        .split("//")[-1]
        .split("/")[0]
        .split(":")[0]
    )

    return any(
        host == domain
        or host.endswith("." + domain)
        for domain in BLOCKED_PLATFORMS
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Remove citation marks such as [1], [23]
    text = re.sub(r"\[\d+\]", " ", text)

    # Remove [edit]
    text = re.sub(
        r"\[edit\]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# SPLIT TEXT INTO SENTENCES
# ============================================================

def split_into_sentences(text):

    text = clean_text(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence.split()) >= 5:
            cleaned_sentences.append(sentence)

    return cleaned_sentences


# ============================================================
# WORD FREQUENCY
# ============================================================

def calculate_word_frequency(text):

    text = clean_text(text)

    words = re.findall(
        r"[a-zA-Z]+",
        text.lower()
    )

    words = [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 2
    ]

    return Counter(words)


# ============================================================
# SENTIMENT CLASSIFICATION
# ============================================================

def sentiment_category(text):

    words = re.findall(
        r"[a-zA-Z]+",
        str(text).lower()
    )

    positive_count = sum(
        1
        for word in words
        if word in POSITIVE_WORDS
    )

    negative_count = sum(
        1
        for word in words
        if word in NEGATIVE_WORDS
    )

    if positive_count > negative_count:
        return "Positive"

    elif negative_count > positive_count:
        return "Negative"

    else:
        return "Neutral"


# ============================================================
# CATEGORY CLASSIFICATION
# ============================================================

def category_classification(text):

    words = set(
        re.findall(
            r"[a-zA-Z]+",
            str(text).lower()
        )
    )

    category_scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():

        score = len(
            words.intersection(keywords)
        )

        category_scores[category] = score

    highest_category = max(
        category_scores,
        key=category_scores.get
    )

    highest_score = category_scores[
        highest_category
    ]

    if highest_score == 0:
        return "General"

    return highest_category


# ============================================================
# CREATE STRUCTURED SCRAPED DATA
# ============================================================

def create_scraped_dataset(text):

    sentences = split_into_sentences(text)

    rows = []

    for index, sentence in enumerate(
        sentences,
        start=1
    ):

        rows.append({
            "Post_ID": index,
            "Text": sentence,
            "Category": category_classification(sentence),
            "Date": str(date.today())
        })

    return pd.DataFrame(rows)


# ============================================================
# WEB SCRAPING
# ============================================================

def scrape_webpage(url):

    try:

        headers = {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unwanted elements
        for element in soup.find_all(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav",
                "aside",
                "table",
                "form"
            ]
        ):
            element.decompose()

        # Remove Wikipedia-specific unwanted elements
        for element in soup.select(
            ".mw-editsection, "
            ".reference, "
            ".reflist, "
            ".navbox, "
            ".vertical-navbox, "
            ".metadata"
        ):
            element.decompose()

        # Get paragraphs
        paragraphs = soup.find_all("p")

        paragraph_text = []

        for paragraph in paragraphs:

            text = paragraph.get_text(
                " ",
                strip=True
            )

            if len(text.split()) >= 5:
                paragraph_text.append(text)

        text = " ".join(paragraph_text)

        # Fallback
        if not text:

            text = soup.get_text(
                separator=" ",
                strip=True
            )

        return clean_text(text)

    except Exception:

        st.warning(
            SCRAPING_MESSAGE
        )

        return ""


# ============================================================
# FIND TEXT COLUMN
# ============================================================

def find_text_column(data):

    possible_names = [
        "Text",
        "text",
        "TEXT",
        "Post",
        "post",
        "Content",
        "content",
        "Comment",
        "comment",
        "Tweet",
        "tweet",
        "Caption",
        "caption"
    ]

    for column in data.columns:

        if str(column).strip() in possible_names:
            return column

    # Flexible search
    for column in data.columns:

        name = str(column).lower()

        if any(
            word in name
            for word in [
                "text",
                "post",
                "content",
                "comment",
                "tweet",
                "caption"
            ]
        ):
            return column

    # First object/string column
    object_columns = data.select_dtypes(
        include=["object"]
    ).columns

    if len(object_columns) > 0:
        return object_columns[0]

    return None


# ============================================================
# FIND CATEGORY COLUMN
# ============================================================

def find_category_column(data):

    possible_names = [
        "Category",
        "category",
        "CATEGORY",
        "Type",
        "type",
        "Topic",
        "topic"
    ]

    for column in data.columns:

        if str(column).strip() in possible_names:
            return column

    for column in data.columns:

        name = str(
            column
        ).lower().strip()

        if (
            "category" in name
            or "type" in name
            or "topic" in name
        ):
            return column

    return None


# ============================================================
# RESULT BUTTONS
# ============================================================

def show_result_options():

    st.subheader("Analysis Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "📊 Bar Chart",
            use_container_width=True,
            key="result_bar_chart"
        ):

            st.session_state.selected_result = "bar"

    with col2:

        if st.button(
            "🥧 Pie Chart",
            use_container_width=True,
            key="result_category_pie"
        ):

            st.session_state.selected_result = "category"

    with col3:

        if st.button(
            "💬 Sentiment",
            use_container_width=True,
            key="result_sentiment"
        ):

            st.session_state.selected_result = "sentiment"


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_selected_result():

    result = st.session_state.selected_result

    if result is None:
        return


    # ========================================================
    # BAR CHART
    # ========================================================

    if result == "bar":

        st.subheader(
            "Top 10 Most Frequent Words"
        )

        word_frequency = (
            st.session_state.word_frequency
        )

        if not word_frequency:

            st.info(
                "No words available for analysis."
            )

        else:

            top_words = (
                word_frequency
                .most_common(10)
            )

            words = [
                item[0]
                for item in top_words
            ]

            counts = [
                item[1]
                for item in top_words
            ]

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            bars = ax.bar(
                words,
                counts
            )

            ax.set_xlabel(
                "Words"
            )

            ax.set_ylabel(
                "Frequency / Count"
            )

            ax.set_title(
                "Top 10 Most Frequent Words"
            )

            plt.xticks(
                rotation=45
            )

            # Display count above every bar
            for bar, count in zip(
                bars,
                counts
            ):

                ax.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    bar.get_height(),
                    str(count),
                    ha="center",
                    va="bottom",
                    fontweight="bold"
                )

            plt.tight_layout()

            st.pyplot(fig)


    # ========================================================
    # CATEGORY PIE CHART
    # ========================================================

    elif result == "category":

        st.subheader(
            "Category Distribution"
        )

        categories = (
            st.session_state.categories
        )

        if not categories:

            st.warning(
                "No category information is available."
            )

        else:

            category_counts = Counter(
                categories
            )

            labels = list(
                category_counts.keys()
            )

            values = list(
                category_counts.values()
            )

            fig, ax = plt.subplots(
                figsize=(8, 8)
            )

            ax.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.68,
                labeldistance=1.08,
                textprops={
                    "fontsize": 10,
                    "fontweight": "bold"
                }
            )

            ax.set_title(
                "Category Distribution"
            )

            plt.tight_layout()

            st.pyplot(fig)

            st.write("### Category Counts")

            category_table = pd.DataFrame({
                "Category": labels,
                "Count": values
            })

            st.dataframe(
                category_table,
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # SENTIMENT
    # ========================================================

    elif result == "sentiment":

        st.subheader(
            "Sentiment Analysis"
        )

        sentiments = (
            st.session_state.sentiments
        )

        if not sentiments:

            st.info(
                "No sentiment data available."
            )

        else:

            sentiment_counts = Counter(
                sentiments
            )

            labels = [
                "Positive",
                "Neutral",
                "Negative"
            ]

            values = [
                sentiment_counts.get(
                    "Positive",
                    0
                ),
                sentiment_counts.get(
                    "Neutral",
                    0
                ),
                sentiment_counts.get(
                    "Negative",
                    0
                )
            ]

            # Remove zero-value categories
            chart_labels = []
            chart_values = []

            for label, value in zip(
                labels,
                values
            ):

                if value > 0:

                    chart_labels.append(
                        label
                    )

                    chart_values.append(
                        value
                    )

            if chart_values:

                fig, ax = plt.subplots(
                    figsize=(8, 8)
                )

                ax.pie(
                    chart_values,
                    labels=chart_labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    pctdistance=0.68,
                    labeldistance=1.08,
                    textprops={
                        "fontsize": 10,
                        "fontweight": "bold"
                    }
                )

                ax.set_title(
                    "Sentiment Distribution"
                )

                plt.tight_layout()

                st.pyplot(fig)

            st.write("### Sentiment Counts")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Positive",
                    sentiment_counts.get(
                        "Positive",
                        0
                    )
                )

            with col2:

                st.metric(
                    "Neutral",
                    sentiment_counts.get(
                        "Neutral",
                        0
                    )
                )

            with col3:

                st.metric(
                    "Negative",
                    sentiment_counts.get(
                        "Negative",
                        0
                    )
                )

            overall_sentiment = max(
                sentiment_counts,
                key=sentiment_counts.get
            )

            st.info(
                f"Overall Sentiment: "
                f"**{overall_sentiment}**"
            )


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    # Centered title
    st.markdown(
        "<h1 style='text-align:center;'>"
        "SOCIAL MEDIA DATA ANALYSIS SYSTEM"
        "</h1>",
        unsafe_allow_html=True
    )

    # Centered description
    st.markdown(
        """
        <p style="
            text-align:center;
            font-size:17px;
            margin-bottom:25px;
        ">
        Analyze social media text using data cleaning,
        word frequency analysis, sentiment categorization
        and visualization.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Centered heading
    st.markdown(
        """
        <h2 style="
            text-align:center;
            color:#0056A6;
        ">
        Choose an Analysis Method
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ========================================================
    # TWO HOME PAGE BOXES
    # ========================================================

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # WEB SCRAPING BOX
    # --------------------------------------------------------

    with col1:

        st.subheader("🌐 Web Scraping")

        st.write(
            "Enter a webpage URL and analyze "
            "the available textual content."
        )

        if st.button(
            "Web Scraping",
            use_container_width=True,
            key="home_web_scraping"
        ):

            st.session_state.page = "web"
            st.session_state.analysis_done = False
            st.session_state.selected_result = None

            st.rerun()


    # --------------------------------------------------------
    # DATASET BOX
    # --------------------------------------------------------

    with col2:

        st.subheader("📁 Dataset")

        st.write(
            "Upload a social media dataset and "
            "analyze its textual content."
        )

        if st.button(
            "Dataset",
            use_container_width=True,
            key="home_dataset"
        ):

            st.session_state.page = "dataset"
            st.session_state.analysis_done = False
            st.session_state.selected_result = None

            st.rerun()


# ============================================================
# WEB SCRAPING PAGE
# ============================================================

def web_scraping_page():

    if st.button(
        "← Home",
        key="web_home"
    ):

        st.session_state.page = "home"
        st.session_state.analysis_done = False
        st.session_state.selected_result = None

        st.rerun()

    st.title(
        "WEB SCRAPING"
    )

    st.divider()

    st.subheader(
        "Enter Webpage URL"
    )

    url = st.text_input(
        "URL",
        placeholder="https://example.com",
        key="web_url"
    )

    st.caption(
        "Works best on publicly accessible pages such as news sites, "
        "blogs, forums and Wikipedia."
    )

    if st.button(
        "Analyze",
        use_container_width=True,
        key="web_analyze"
    ):

        if not url.strip():

            st.warning(
                "Please enter a URL."
            )

        elif is_blocked_platform(url):

            st.warning(
                SCRAPING_MESSAGE
            )

        else:

            with st.spinner(
                "Scraping and analyzing webpage..."
            ):

                scraped_text = scrape_webpage(
                    url
                )

                if scraped_text:

                    # ----------------------------------------
                    # CREATE STRUCTURED DATASET
                    # ----------------------------------------

                    processed_data = (
                        create_scraped_dataset(
                            scraped_text
                        )
                    )

                    if processed_data.empty:

                        st.warning(
                            SCRAPING_MESSAGE
                        )

                    else:

                        st.session_state.processed_data = (
                            processed_data
                        )

                        # ----------------------------------------
                        # WORD FREQUENCY
                        # ----------------------------------------

                        all_text = " ".join(
                            processed_data["Text"]
                            .tolist()
                        )

                        st.session_state.word_frequency = (
                            calculate_word_frequency(
                                all_text
                            )
                        )

                        # ----------------------------------------
                        # SENTIMENT
                        # ----------------------------------------

                        st.session_state.sentiments = (
                            processed_data["Text"]
                            .apply(
                                sentiment_category
                            )
                            .tolist()
                        )

                        # ----------------------------------------
                        # CATEGORY
                        # ----------------------------------------

                        st.session_state.categories = (
                            processed_data["Category"]
                            .tolist()
                        )

                        st.session_state.analysis_done = True
                        st.session_state.selected_result = None

                        st.success(
                            "Webpage scraped and analyzed successfully."
                        )

                        st.info(
                            f"Processed "
                            f"{len(processed_data)} "
                            f"text records."
                        )


    # --------------------------------------------------------
    # RESULTS ON SAME PAGE
    # --------------------------------------------------------

    if st.session_state.analysis_done:

        show_result_options()

        display_selected_result()


# ============================================================
# DATASET PAGE
# ============================================================

def dataset_page():

    if st.button(
        "← Home",
        key="dataset_home"
    ):

        st.session_state.page = "home"
        st.session_state.analysis_done = False
        st.session_state.selected_result = None

        st.rerun()

    st.title(
        "DATASET ANALYSIS"
    )

    st.divider()

    st.subheader(
        "Upload Dataset"
    )

    uploaded_file = st.file_uploader(
        "Upload your social media dataset",
        type=[
            "csv",
            "xlsx",
            "xls",
            "json",
            "txt"
        ],
        key="dataset_uploader"
    )

    if st.button(
        "Analyze",
        use_container_width=True,
        key="dataset_analyze"
    ):

        if uploaded_file is None:

            st.warning(
                "Please upload a dataset first."
            )

        else:

            try:

                file_name = (
                    uploaded_file.name.lower()
                )

                # ----------------------------------------
                # READ FILE
                # ----------------------------------------

                if file_name.endswith(".csv"):

                    data = pd.read_csv(
                        uploaded_file
                    )

                elif file_name.endswith(
                    (".xlsx", ".xls")
                ):

                    data = pd.read_excel(
                        uploaded_file
                    )

                elif file_name.endswith(".json"):

                    data = pd.read_json(
                        uploaded_file
                    )

                else:

                    content = (
                        uploaded_file
                        .read()
                        .decode(
                            "utf-8",
                            errors="ignore"
                        )
                    )

                    data = pd.DataFrame({
                        "Text":
                        content.splitlines()
                    })


                # ----------------------------------------
                # CLEAN DATASET
                # ----------------------------------------

                data = data.drop_duplicates()

                if data.empty:

                    st.error(
                        "The uploaded dataset is empty."
                    )

                else:

                    text_column = find_text_column(
                        data
                    )

                    if text_column is None:

                        st.error(
                            "No textual column was found."
                        )

                    else:

                        data = data.dropna(
                            subset=[
                                text_column
                            ]
                        )


                        # --------------------------------
                        # WORD FREQUENCY
                        # --------------------------------

                        all_text = " ".join(
                            data[text_column]
                            .astype(str)
                            .tolist()
                        )

                        st.session_state.word_frequency = (
                            calculate_word_frequency(
                                all_text
                            )
                        )


                        # --------------------------------
                        # SENTIMENT
                        # --------------------------------

                        st.session_state.sentiments = [
                            sentiment_category(text)
                            for text in data[text_column]
                        ]


                        # --------------------------------
                        # CATEGORY
                        # --------------------------------

                        category_column = (
                            find_category_column(
                                data
                            )
                        )

                        if category_column is not None:

                            category_data = (
                                data[category_column]
                                .dropna()
                                .astype(str)
                                .str.strip()
                            )

                            category_data = (
                                category_data[
                                    category_data != ""
                                ]
                            )

                            st.session_state.categories = (
                                category_data.tolist()
                            )

                        else:

                            # If dataset has no Category column,
                            # create categories automatically.

                            st.session_state.categories = [

                                category_classification(text)

                                for text in data[text_column]

                            ]


                        # --------------------------------
                        # SAVE DATA IN SESSION
                        # --------------------------------

                        st.session_state.processed_data = data

                        st.session_state.analysis_done = True

                        st.session_state.selected_result = None

                        st.success(
                            "Dataset analyzed successfully."
                        )

            except Exception as e:

                st.error(
                    f"Error while processing dataset: {e}"
                )


    # --------------------------------------------------------
    # RESULTS ON SAME PAGE
    # --------------------------------------------------------

    if st.session_state.analysis_done:

        show_result_options()

        display_selected_result()


# ============================================================
# PAGE ROUTING
# ============================================================

if st.session_state.page == "home":

    home_page()

elif st.session_state.page == "web":

    web_scraping_page()

elif st.session_state.page == "dataset":

    dataset_page()