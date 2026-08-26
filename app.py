import streamlit as st
import pandas as pd
import numpy as np
import re
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SkinCare Match 🎀",
    page_icon="🎀",
    layout="wide",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 5% 5%,
                #ffe3eb 0%,
                transparent 23%
            ),
            radial-gradient(
                circle at 95% 8%,
                #ffdae6 0%,
                transparent 23%
            ),
            linear-gradient(
                180deg,
                #fff8fa 0%,
                #fffafb 100%
            );
    }

    .main .block-container {
        max-width: 1050px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ------------------------------------------
       HERO
       ------------------------------------------ */

    .hero {
        text-align: center;
        padding: 5px 0 25px 0;
    }

    .hero-bow {
        font-size: 48px;
        margin-bottom: 8px;
    }

    .hero-title {
        font-family: Georgia, serif;
        font-size: 52px;
        font-weight: 700;
        color: #8c3d57;
    }

    .hero-subtitle {
        color: #ae7184;
        font-size: 17px;
        margin-top: 8px;
    }

    .hero-line {
        width: 150px;
        height: 3px;
        background: #e7a5b8;
        border-radius: 10px;
        margin: 18px auto 0;
    }

    /* ------------------------------------------
       INPUT HEADING
       ------------------------------------------ */

    .input-heading {
        text-align: center;
        font-family: Georgia, serif;
        color: #8c3d57;
        font-size: 28px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .input-subheading {
        text-align: center;
        color: #ad7083;
        font-size: 14px;
        margin-bottom: 22px;
    }

    /* ------------------------------------------
       STREAMLIT LABELS
       ------------------------------------------ */

    label {
        color: #835064 !important;
        font-weight: 600 !important;
    }

    /* ------------------------------------------
       INPUTS
       ------------------------------------------ */

    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #fffafb !important;
        border: 1px solid #e9b5c5 !important;
        border-radius: 14px !important;
    }

    [data-baseweb="tag"] {
        background: #f5bfd0 !important;
        color: #74364b !important;
        border-radius: 12px !important;
    }

    /* ------------------------------------------
       BUTTON
       ------------------------------------------ */

    .stButton > button {
        background:
            linear-gradient(
                135deg,
                #e78ba7,
                #d86f91
            ) !important;

        color: white !important;
        border: none !important;
        border-radius: 999px !important;

        min-height: 52px;

        font-size: 17px !important;
        font-weight: 700 !important;

        box-shadow:
            0 8px 20px rgba(216,111,145,0.25);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }

    /* ------------------------------------------
       WELCOME
       ------------------------------------------ */

    .welcome {
        text-align: center;
        padding: 45px 20px 55px;
    }

    .welcome-bow {
        font-size: 58px;
    }

    .welcome-title {
        font-family: Georgia, serif;
        font-size: 28px;
        color: #8c3d57;
        font-weight: 700;
        margin-top: 10px;
    }

    .welcome-text {
        color: #ad7184;
        font-size: 15px;
        line-height: 1.7;
        margin-top: 8px;
    }

    /* ------------------------------------------
       RESULT HEADING
       ------------------------------------------ */

    .results-heading {
        text-align: center;
        font-family: Georgia, serif;
        color: #8c3d57;
        font-size: 31px;
        font-weight: 700;
        margin-top: 32px;
    }

    .results-subtitle {
        text-align: center;
        color: #ae7184;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .latency {
        text-align: center;
        color: #b27a8d;
        font-size: 12px;
        margin-bottom: 25px;
    }

    /* ------------------------------------------
       PRODUCT CARD
       ------------------------------------------ */

    .product-card {
        background: rgba(255,255,255,0.97);
        border: 1px solid #efc5d1;
        border-radius: 25px;
        padding: 22px;
        margin-bottom: 22px;
        box-shadow:
            0 8px 25px rgba(153,70,101,0.08);
    }

    .product-name {
        font-family: Georgia, serif;
        color: #843b54;
        font-size: 23px;
        font-weight: 700;
        line-height: 1.3;
    }

    .match-text {
        color: #9c4966;
        background: #ffe0ea;
        display: inline-block;
        border-radius: 999px;
        padding: 7px 14px;
        font-weight: 700;
        margin: 10px 0 12px;
    }

    .reason-pill {
        background: #fff1f5;
        border: 1px solid #f1c3d0;
        color: #8d5367;
        border-radius: 999px;
        padding: 6px 11px;
        margin: 3px;
        display: inline-block;
        font-size: 12px;
    }

    .product-image img {
        border-radius: 20px !important;
        border: 1px solid #f0c4d0 !important;
    }

    .footer-text {
        text-align: center;
        color: #b27a8d;
        font-size: 12px;
        margin-top: 45px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "Skinpro - Skinpro (3).csv"
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


df = load_data()


# ============================================================
# NORMALIZE SKIN TYPE
# ============================================================

def normalize_skin_type(value):

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()
    value = value.replace("-", " ")

    if "combination" in value:
        return "Combination"

    if "oily" in value:
        return "Oily"

    if "dry" in value:
        return "Dry"

    if "sensitive" in value:
        return "Sensitive"

    if "normal" in value:
        return "Normal"

    return value.title()


df["skin_type_clean"] = (
    df["skin_type"]
    .apply(normalize_skin_type)
)


# ============================================================
# NORMALIZE CONCERNS
# ============================================================

def normalize_concern(value):

    value = value.lower().strip()

    mappings = {
        "whitehead": "Whiteheads",
        "whiteheads": "Whiteheads",
        "blackhead": "Blackheads",
        "blackheads": "Blackheads",
        "acne": "Acne",
        "hydration": "Hydration",
        "dryness": "Dryness",
        "dark spots": "Dark Spots",
        "darkspot": "Dark Spots",
        "sun protection": "Sun Protection",
        "sun damage": "Sun Protection",
        "anti aging": "Anti-Aging",
        "anti-aging": "Anti-Aging",
        "aging": "Anti-Aging",
        "wrinkles": "Anti-Aging",
        "redness": "Redness",
        "dullness": "Dullness",
    }

    return mappings.get(
        value,
        value.title()
    )


def clean_concerns(value):

    if pd.isna(value):
        return []

    parts = re.split(
        r",|/|;|\|",
        str(value).lower().strip()
    )

    return list(
        set(
            normalize_concern(part.strip())
            for part in parts
            if part.strip()
        )
    )


df["concerns_list"] = (
    df["concern"]
    .apply(clean_concerns)
)


# ============================================================
# CATEGORY EXTRACTION
# ============================================================

def extract_category(product_name):

    name = str(product_name).lower()

    category_keywords = {

        "Cleanser": [
            "cleanser",
            "face wash",
            "facial wash",
            "cleansing",
        ],

        "Moisturizer": [
            "moisturizer",
            "moisturiser",
            "moisturizing",
            "moisturising",
            "cream",
            "lotion",
        ],

        "Serum": [
            "serum",
        ],

        "Sunscreen": [
            "sunscreen",
            "sun screen",
            "spf",
        ],

        "Face Mask": [
            "face mask",
            "mask",
        ],

        "Toner": [
            "toner",
        ],

        "Face Oil": [
            "face oil",
            "facial oil",
        ],

        "Eye Care": [
            "eye cream",
            "eye gel",
            "under eye",
        ],
    }

    for category, keywords in category_keywords.items():

        for keyword in keywords:

            if keyword in name:
                return category

    return "Other"


df["category"] = (
    df["product"]
    .apply(extract_category)
)


# ============================================================
# PRODUCT PROFILE
# ============================================================

df["concerns_text"] = (
    df["concerns_list"]
    .apply(lambda x: " ".join(x))
)

df["product_profile"] = (
    "skin_"
    + df["skin_type_clean"]
    .str.lower()
    .str.replace(" ", "_")
    + " concern_"
    + df["concerns_text"]
    .str.lower()
    .str.replace(" ", "_")
    + " category_"
    + df["category"]
    .str.lower()
    .str.replace(" ", "_")
)


# ============================================================
# TF-IDF MODEL
# ============================================================

@st.cache_resource
def build_model(product_profiles):

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    product_vectors = (
        tfidf.fit_transform(product_profiles)
    )

    return tfidf, product_vectors


tfidf, product_vectors = build_model(
    df["product_profile"]
)


# ============================================================
# USER PROFILE
# ============================================================

def create_user_profile(
    skin_type,
    concerns,
    category
):

    skin_text = (
        "skin_"
        + skin_type.lower().replace(" ", "_")
    )

    concern_text = " ".join(
        "concern_"
        + c.lower().replace(" ", "_")
        for c in concerns
    )

    category_text = ""

    if category != "All":

        category_text = (
            "category_"
            + category.lower().replace(" ", "_")
        )

    return (
        f"{skin_text} "
        f"{concern_text} "
        f"{category_text}"
    )


# ============================================================
# SCORE FUNCTIONS
# ============================================================

def calculate_skin_score(
    product_skin,
    user_skin
):

    return float(
        product_skin == user_skin
    )


def calculate_concern_score(
    product_concerns,
    user_concerns
):

    product_set = set(product_concerns)
    user_set = set(user_concerns)

    if not user_set:
        return 0.0

    return (
        len(
            product_set.intersection(user_set)
        )
        / len(user_set)
    )


def calculate_category_score(
    product_category,
    user_category
):

    if user_category == "All":
        return 0.5

    return float(
        product_category == user_category
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend_products(
    skin_type,
    concerns,
    category,
    top_k
):

    user_profile = create_user_profile(
        skin_type,
        concerns,
        category
    )

    user_vector = tfidf.transform(
        [user_profile]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        product_vectors
    ).flatten()

    results = df.copy()

    results["similarity_score"] = (
        similarity_scores
    )

    results["skin_score"] = (
        results["skin_type_clean"]
        .apply(
            lambda x:
            calculate_skin_score(
                x,
                skin_type
            )
        )
    )

    results["concern_score"] = (
        results["concerns_list"]
        .apply(
            lambda x:
            calculate_concern_score(
                x,
                concerns
            )
        )
    )

    results["category_score"] = (
        results["category"]
        .apply(
            lambda x:
            calculate_category_score(
                x,
                category
            )
        )
    )

    results["final_score"] = (
        0.30 * results["similarity_score"]
        + 0.30 * results["skin_score"]
        + 0.30 * results["concern_score"]
        + 0.10 * results["category_score"]
    )

    if category != "All":

        results = results[
            results["category"] == category
        ]

    return (
        results
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(top_k)
    )


# ============================================================
# EXPLANATIONS
# ============================================================

def generate_explanation(
    row,
    user_skin_type,
    user_concerns,
    user_category
):

    reasons = []

    if row["skin_type_clean"] == user_skin_type:

        reasons.append(
            f"Matches your {user_skin_type.lower()} skin"
        )

    matched_concerns = (
        set(row["concerns_list"])
        .intersection(
            set(user_concerns)
        )
    )

    for concern in matched_concerns:

        reasons.append(
            f"Targets {concern.lower()}"
        )

    if (
        user_category != "All"
        and row["category"] == user_category
    ):

        reasons.append(
            f"Matches your {user_category.lower()} preference"
        )

    if not reasons:

        reasons.append(
            "Similar to your selected preferences"
        )

    return reasons


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">
        <div class="hero-bow">🎀</div>

        <div class="hero-title">
            SkinCare Match
        </div>

        <div class="hero-subtitle">
            your little skincare matchmaker ♡
        </div>

        <div class="hero-line"></div>
    </div>
    """
)


# ============================================================
# INPUT INTRO
# ============================================================

st.html(
    """
    <div class="input-heading">
        Tell me about your skin ♡
    </div>

    <div class="input-subheading">
        We'll find the products that match you best.
    </div>
    """
)


# ============================================================
# OPTIONS
# ============================================================

skin_types = sorted(
    df["skin_type_clean"]
    .dropna()
    .unique()
)

categories = (
    ["All"]
    + sorted(
        df["category"]
        .dropna()
        .unique()
    )
)

all_concerns = sorted(
    set(
        concern
        for concerns in df["concerns_list"]
        for concern in concerns
    )
)


# ============================================================
# INPUTS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    selected_skin = st.selectbox(
        "🌸 Skin type",
        skin_types
    )

with col2:

    selected_category = st.selectbox(
        "🎀 Product type",
        categories
    )


selected_concerns = st.multiselect(
    "♡ What are your skin concerns?",
    all_concerns,
    placeholder="Choose one or more concerns..."
)


top_k = st.slider(
    "♡ Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)


# ============================================================
# BUTTON
# ============================================================

st.write("")

recommend_button = st.button(
    "🎀 Find My Perfect Matches ♡",
    type="primary",
    use_container_width=True
)


# ============================================================
# WELCOME
# ============================================================

if not recommend_button:

    st.html(
        """
        <div class="welcome">

            <div class="welcome-bow">
                🎀
            </div>

            <div class="welcome-title">
                Ready to find your skincare match?
            </div>

            <div class="welcome-text">
                Select your skin type and concerns above,<br>
                then let your little skincare matchmaker
                do the rest ♡
            </div>

        </div>
        """
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

if recommend_button:

    if not selected_concerns:

        st.warning(
            "🎀 Pick at least one concern so I can "
            "find a better match for you."
        )

    else:

        start_time = time.perf_counter()

        recommendations = recommend_products(
            skin_type=selected_skin,
            concerns=selected_concerns,
            category=selected_category,
            top_k=top_k
        )

        latency = (
            time.perf_counter()
            - start_time
        ) * 1000


        # ----------------------------------------------------
        # RESULTS HEADER
        # ----------------------------------------------------

        st.html(
            """
            <div class="results-heading">
                🎀 Your Pretty Little Matches
            </div>

            <div class="results-subtitle">
                hand-matched to your skin profile ♡
            </div>
            """
        )

        st.caption(
            f"♡ Recommendations generated in "
            f"{latency:.2f} ms"
        )


        # ----------------------------------------------------
        # EMPTY RESULTS
        # ----------------------------------------------------

        if recommendations.empty:

            st.warning(
                "No matching products were found "
                "for this combination."
            )


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        else:

            for _, row in recommendations.iterrows():

                score = min(
                    max(
                        row["final_score"] * 100,
                        0
                    ),
                    100
                )

                reasons = generate_explanation(
                    row,
                    selected_skin,
                    selected_concerns,
                    selected_category
                )


                # ============================================
                # PRODUCT CONTAINER
                # ============================================

                with st.container():

                    # ----------------------------------------
                    # PRODUCT IMAGE + INFO
                    # ----------------------------------------

                    img_col, info_col = st.columns(
                        [1, 1.7]
                    )


                    # ----------------------------------------
                    # IMAGE
                    # ----------------------------------------

                    with img_col:

                        image_url = row.get(
                            "product_pic",
                            None
                        )

                        if (
                            pd.notna(image_url)
                            and
                            str(image_url).startswith("http")
                        ):

                            st.image(
                                image_url,
                                use_container_width=True
                            )

                        else:

                            st.html(
                                """
                                <div style="
                                    height:220px;
                                    display:flex;
                                    align-items:center;
                                    justify-content:center;
                                    background:#fff0f5;
                                    border-radius:20px;
                                    border:1px solid #f0c4d0;
                                    color:#b06a80;
                                    text-align:center;
                                ">
                                    🎀<br>
                                    Product image unavailable
                                </div>
                                """
                            )


                    # ----------------------------------------
                    # PRODUCT DETAILS
                    # ----------------------------------------

                    with info_col:

                        st.html(
                            f"""
                            <div class="product-name">
                                {row["product"]}
                            </div>

                            <div class="match-text">
                                🎀 {score:.0f}% Match
                            </div>
                            """
                        )

                        st.write(
                            f"**Skin type:** "
                            f"{row['skin_type_clean']}"
                        )

                        st.write(
                            f"**Product type:** "
                            f"{row['category']}"
                        )

                        st.write(
                            "**Concerns:** "
                            + ", ".join(
                                row["concerns_list"]
                            )
                        )

                        st.write(
                            "**Why this was picked for you ♡**"
                        )

                        # ------------------------------------
                        # REASONS
                        # ------------------------------------

                        for reason in reasons:

                            st.html(
                                f"""
                                <span class="reason-pill">
                                    ♡ {reason}
                                </span>
                                """
                            )

                        # ------------------------------------
                        # LINK
                        # ------------------------------------

                        product_url = row.get(
                            "product_url",
                            None
                        )

                        if (
                            pd.notna(product_url)
                            and
                            str(product_url).startswith("http")
                        ):

                            st.link_button(
                                "♡ View Product",
                                product_url
                            )

                    st.divider()


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer-text">
        🎀 Made with skincare data,
        machine learning & a little love ♡
    </div>
    """
)