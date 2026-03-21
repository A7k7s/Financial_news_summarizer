"""
Financial News Summarizer
=========================
Lightweight extractive text summarizer using:
  - NLTK sentence tokenization
  - scikit-learn TF-IDF vectorization
  - Cosine similarity sentence scoring

No deep learning / Hugging Face models used.
Runs fast on modest hardware (i5, 8 GB RAM).
"""

import re
import string
import io

import streamlit as st
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# One-time NLTK data download (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def download_nltk_data():
    """Download required NLTK corpora once and cache the result."""
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass  # silently skip if already present

download_nltk_data()

# ---------------------------------------------------------------------------
# NLP utility functions
# ---------------------------------------------------------------------------

# Patterns that reliably indicate website boilerplate (nav links, footers, etc.)
_BOILERPLATE_PATTERNS = re.compile(
    r"^("
    r"skip to|sign in|register|log in|subscribe|"
    r"home|news|sport|business|technology|health|culture|"
    r"arts|travel|earth|audio|video|live|weather|"
    r"follow (us|bbc)|share|save|terms of use|privacy policy|"
    r"cookie|copyright|all rights reserved|accessibility|"
    r"contact|advertise|bbc shop|britbox|"
    r"related|more from|hrs? ago|days? ago|hours? ago"
    r")$",
    re.IGNORECASE,
)


def clean_web_content(text: str) -> tuple[str, bool]:
    """
    Remove common website boilerplate lines (navbars, footers, cookie notices,
    "Related" section headers, timestamps, etc.) that get copy-pasted alongside
    the real article text.

    Returns
    -------
    cleaned_text : str
        Text with boilerplate lines removed and paragraphs re-joined.
    was_cleaned  : bool
        True if any lines were actually stripped.
    """
    lines = text.splitlines()
    kept, removed = [], 0

    for line in lines:
        stripped = line.strip()

        # Drop empty lines (keep once for paragraph breaks)
        if not stripped:
            kept.append("")
            continue

        # Drop very short lines that are almost certainly nav items / labels
        # (real sentences are rarely under 40 characters)
        if len(stripped) < 40 and not stripped.endswith("."):
            if _BOILERPLATE_PATTERNS.search(stripped) or len(stripped.split()) <= 5:
                removed += 1
                continue

        # Drop lines that look like image captions from BBC/Reuters
        # e.g. "Getty Images People carrying shopping bags…"
        if stripped.startswith(("Getty Images", "Reuters ", "AFP ", "AP ")):
            removed += 1
            continue

        kept.append(stripped)

    # Collapse multiple blank lines into one, then join
    result_lines, prev_blank = [], False
    for ln in kept:
        if ln == "":
            if not prev_blank:
                result_lines.append("")
            prev_blank = True
        else:
            result_lines.append(ln)
            prev_blank = False

    return "\n".join(result_lines).strip(), removed > 0


def preprocess_text(text: str) -> tuple[list[str], list[str]]:
    """
    Tokenize *text* into sentences and produce cleaned versions for scoring.

    Parameters
    ----------
    text : str
        Raw article text.

    Returns
    -------
    sentences : list[str]
        Original (un-modified) sentence strings.
    cleaned   : list[str]
        Lower-cased, stop-word-free, punctuation-stripped sentences used
        for TF-IDF computation.
    """
    # --- sentence splitting ---
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    stop_words = set(stopwords.words("english"))

    cleaned = []
    for sent in sentences:
        # lowercase
        lower = sent.lower()
        # remove punctuation
        no_punct = lower.translate(str.maketrans("", "", string.punctuation))
        # tokenize words, remove stop-words and short tokens
        tokens = [
            w for w in word_tokenize(no_punct)
            if w not in stop_words and len(w) > 1
        ]
        cleaned.append(" ".join(tokens))

    return sentences, cleaned


def compute_sentence_scores(sentences: list[str], cleaned: list[str]) -> np.ndarray:
    """
    Score every sentence by its average cosine similarity to all other sentences
    in the TF-IDF space.  Sentences that are *most similar* to the overall
    document represent central/important ideas.

    Parameters
    ----------
    sentences : list[str]
        Original sentence strings (used only for length).
    cleaned   : list[str]
        Pre-processed sentences for vectorization.

    Returns
    -------
    scores : np.ndarray
        1-D array of float scores, one per sentence.
    """
    if len(cleaned) == 0:
        return np.array([])

    # Build TF-IDF matrix  (shape: n_sentences × n_terms)
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        # All cleaned sentences are empty (e.g. input was only stop-words)
        return np.ones(len(cleaned))

    # Pairwise cosine similarity  (n_sentences × n_sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Score = mean similarity to all other sentences (excluding self)
    np.fill_diagonal(sim_matrix, 0)
    scores = sim_matrix.mean(axis=1)

    return scores


def generate_summary(
    text: str, num_sentences: int = 3
) -> tuple[str, list[int], bool]:
    """
    Build an extractive summary by selecting the top-scoring sentences and
    returning them in their *original* document order.

    Parameters
    ----------
    text          : str   Raw article text.
    num_sentences : int   Number of sentences to include in the summary.

    Returns
    -------
    summary        : str        Joined summary text.
    selected_idxs  : list[int]  Indices of selected sentences (original order).
    was_cleaned    : bool       Whether boilerplate was stripped before scoring.
    """
    # Strip website boilerplate first
    clean_text, was_cleaned = clean_web_content(text)

    sentences, cleaned = preprocess_text(clean_text)

    if len(sentences) == 0:
        return "", [], was_cleaned

    # Clamp num_sentences to the number of available sentences
    n = min(num_sentences, len(sentences))

    scores = compute_sentence_scores(sentences, cleaned)

    # Rank sentence indices by score (descending)
    ranked_idxs = np.argsort(scores)[::-1]

    # Take the top-n and restore original order
    top_idxs = sorted(ranked_idxs[:n].tolist())

    summary = " ".join(sentences[i] for i in top_idxs)
    return summary, top_idxs, was_cleaned


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Financial News Summarizer",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS – clean, professional look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(90deg, #1d4ed8, #0ea5e9);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(14,165,233,0.25);
    }
    .header-banner h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        margin: 6px 0 0;
        font-size: 0.95rem;
        color: #bae6fd;
    }

    /* ── Section cards ── */
    .card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(99,179,237,0.18);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }
    .card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #7dd3fc;
        margin: 0 0 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Highlighted summary sentence ── */
    .summary-sentence {
        background: linear-gradient(90deg, rgba(14,165,233,0.15), rgba(29,78,216,0.10));
        border-left: 4px solid #0ea5e9;
        border-radius: 6px;
        padding: 10px 16px;
        margin: 8px 0;
        color: #e2e8f0;
        font-size: 0.97rem;
        line-height: 1.6;
        transition: background 0.2s;
    }

    /* ── Metric boxes ── */
    .metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 16px;
    }
    .metric-box {
        flex: 1;
        background: rgba(14,165,233,0.12);
        border: 1px solid rgba(14,165,233,0.3);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-box .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-box .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* ── Streamlit widget polish ── */
    .stTextArea textarea {
        background: #0f172a !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    .stSlider > div > div > div > div {
        background: #0ea5e9 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1d4ed8, #0ea5e9) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 2rem !important;
        font-size: 1rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
    }
    div[data-testid="stFileUploader"] {
        background: #0f172a;
        border: 1px dashed #334155;
        border-radius: 8px;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="header-banner">
        <h1>📊 Financial News Summarizer</h1>
        <p>Extractive AI summarization powered by TF-IDF &amp; Cosine Similarity — no heavy models, blazing fast.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar – controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    num_sentences = st.slider(
        "Summary length (sentences)",
        min_value=2,
        max_value=7,
        value=3,
        step=1,
        help="How many top-ranked sentences to include in the summary.",
    )
    st.markdown("---")
    st.markdown(
        """
        **How it works**
        1. Text is tokenized into sentences.
        2. TF-IDF vectors are built for each sentence.
        3. Cosine similarity ranks central sentences.
        4. Top-N are returned in original order.
        """,
        unsafe_allow_html=False,
    )

# ---------------------------------------------------------------------------
# Main layout – two columns
# ---------------------------------------------------------------------------

col_input, col_output = st.columns([1, 1], gap="large")

# ── Left column: input ──────────────────────────────────────────────────────
with col_input:
    st.markdown('<div class="card"><h3>📥 Input Article</h3>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a .txt file (optional)",
        type=["txt"],
        help="Upload a plain-text news article. Content will auto-populate the text area below.",
    )

    file_text = ""
    if uploaded_file is not None:
        try:
            file_text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error("❌ Could not decode file. Please upload a UTF-8 encoded .txt file.")

    article_text = st.text_area(
        "Or paste article text here",
        value=file_text,
        height=340,
        placeholder="Paste a financial news article here…",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    generate_btn = st.button("🚀 Generate Summary", use_container_width=True)

# ── Right column: output ────────────────────────────────────────────────────
with col_output:
    st.markdown('<div class="card"><h3>📤 Summary Output</h3>', unsafe_allow_html=True)

    if generate_btn:
        raw_text = article_text.strip()

        # ── Validation ──────────────────────────────────────────────────────
        if not raw_text:
            st.error("⚠️ Please provide some article text before generating a summary.")
        else:
            # Quick sentence count check
            probe_sentences, _ = preprocess_text(raw_text)
            if len(probe_sentences) < 2:
                st.warning(
                    "⚠️ The article is too short to summarize meaningfully. "
                    "Please provide at least 2–3 sentences."
                )
            else:
                with st.spinner("Analyzing and ranking sentences…"):
                    summary, selected_idxs, was_cleaned = generate_summary(raw_text, num_sentences)
                    # Use the same cleaned text for sentence display
                    clean_for_display, _ = clean_web_content(raw_text)
                    all_sentences, _ = preprocess_text(clean_for_display)

                if was_cleaned:
                    st.info(
                        "🧹 **Web boilerplate detected and removed** (nav bars, footers, "
                        "cookie notices, etc.) before summarizing."
                    )

                # ── Word count metrics (based on cleaned text so compression is meaningful) ──
                orig_words = len(clean_for_display.split())
                summ_words = len(summary.split())
                compression = round((1 - summ_words / orig_words) * 100, 1) if orig_words else 0

                st.markdown(
                    f"""
                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="metric-value">{orig_words}</div>
                            <div class="metric-label">Original words</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{summ_words}</div>
                            <div class="metric-label">Summary words</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{compression}%</div>
                            <div class="metric-label">Compression</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Highlighted summary sentences ────────────────────────────
                st.markdown("**Selected sentences (highlighted):**")
                for idx in selected_idxs:
                    sent = all_sentences[idx]
                    # Escape any HTML in the sentence for safe rendering
                    safe_sent = (
                        sent.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                    )
                    st.markdown(
                        f'<div class="summary-sentence">🔹 {safe_sent}</div>',
                        unsafe_allow_html=True,
                    )

                # ── Download button ──────────────────────────────────────────
                summary_bytes = summary.encode("utf-8")
                st.download_button(
                    label="⬇️ Download Summary (.txt)",
                    data=io.BytesIO(summary_bytes),
                    file_name="financial_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    else:
        st.info("👈 Enter an article and click **Generate Summary** to begin.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <hr style="border-color:#334155; margin-top:32px;">
    <p style="text-align:center; color:#475569; font-size:0.82rem;">
        Financial News Summarizer · Extractive NLP · NLTK + scikit-learn · No LLMs · Fast &amp; Local
    </p>
    """,
    unsafe_allow_html=True,
)
