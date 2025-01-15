import streamlit as st
import pandas as pd
import openai
import json
import io
import base64
import numpy as np
from openpyxl import Workbook
from datetime import datetime

###############################################################################
# Helper functions
###############################################################################
def generate_sample_excel_template():
    """
    Generates an in-memory Excel file containing the required headers:
    URL, H1, Meta Description.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Pages"

    # Write headers
    ws["A1"] = "URL"
    ws["B1"] = "H1"
    ws["C1"] = "Meta Description"

    # (Optional) Populate sample rows for demonstration
    ws["A2"] = "https://example.com/sample-page"
    ws["B2"] = "Sample Page Title"
    ws["C2"] = "This is a sample meta description for demonstration."

    # Save to a BytesIO buffer
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def get_download_link(file_bytes: bytes, filename: str, file_label: str) -> str:
    """
    Generates an HTML download link for a file in memory (bytes).
    """
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{file_label}</a>'


def parse_openai_response_strict(response_text: str):
    """
    Attempt to parse the model response as JSON. Return an empty list if parsing fails.
    """
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            # In case the model accidentally returns a single JSON object
            data = [data]
        return data
    except json.JSONDecodeError:
        return []


def build_prompt(mode: str, topic: str, keyword: str, target_page: dict, candidate_pages: pd.DataFrame) -> str:
    """
    Construct the prompt text for ChatCompletion.
    """
    if mode == "brand_new":
        # Brand new page / no existing URL
        target_url = "N/A (brand-new page, not published yet)"
        target_h1 = f"{topic} (New Topic)"
        target_meta = f"A future page about {topic}."
        brand_new_msg = (
            "This is a brand-new topic or page that doesn't exist yet.\n"
            "Recommend internal links from the pages below that are thematically relevant\n"
            "and provide anchor text suggestions that are semantically aligned with\n"
            "the new topic.\n"
        )
    else:
        target_url = target_page["URL"]
        target_h1 = target_page["H1"]
        target_meta = target_page["Meta Description"]
        brand_new_msg = ""

    pages_info = []
    for _, row in candidate_pages.iterrows():
        pages_info.append({
            "URL": row["URL"],
            "H1": row["H1"],
            "Meta Description": row["Meta Description"]
        })

    prompt = f"""
You are an SEO assistant. The user wants internal link recommendations.

Mode: {mode}
Topic: {topic}
Target Keyword (optional): {keyword}

{brand_new_msg}

Target Page Information:
- Target Page URL: {target_url}
- Target Page H1: {target_h1}
- Target Page Meta Description: {target_meta}

Here is a list of other pages (candidate link targets) on the site in JSON format:

{json.dumps(pages_info, indent=2)}

Instructions:
1. Identify the most semantically relevant pages from this list.
2. For each recommended link, provide an optimized anchor text (5-10 words ideally) 
   that is user-friendly and semantically relevant to the topic.
3. Return your response strictly in JSON. No extra text. The JSON must be an array of objects,
   each object having:
   - "target_url": The URL to link FROM
   - "anchor_text": The anchor text

Example JSON format:
[
  {{
    "target_url": "https://example.com/page-1",
    "anchor_text": "Semantic anchor text example"
  }},
  {{
    "target_url": "https://example.com/page-2",
    "anchor_text": "Another relevant anchor text"
  }}
]
""".strip()
    return prompt


def generate_internal_links(openai_api_key: str, mode: str, topic: str, keyword: str, target_page: dict, candidate_pages: pd.DataFrame):
    """
    Calls the OpenAI ChatCompletion API to generate internal link recommendations.
    Returns a list of {target_url, anchor_text} dicts.
    """
    openai.api_key = openai_api_key
    prompt = build_prompt(mode, topic, keyword, target_page, candidate_pages)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # e.g. "gpt-4o-2024-05-13", etc.
            messages=[
                {"role": "system", "content": "You are a helpful SEO assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.0  # Lower temperature => more predictable, JSON-friendly
        )
        response_text = response.choices[0].message["content"].strip()

        # Attempt to parse strictly as JSON
        links = parse_openai_response_strict(response_text)

        return links

    except Exception as e:
        st.error(f"Error in OpenAI API call: {e}")
        return []

def convert_to_excel(df: pd.DataFrame) -> bytes:
    """
    Converts a pandas DataFrame to an Excel file in memory (as bytes).
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="InternalLinkSuggestions")
    return buffer.getvalue()

###############################################################################
# Vectorization, Batch Embedding, and Semantic Search
###############################################################################
def embed_text_batch(openai_api_key: str, texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a batch of texts in a single API call.
    Returns a list of embeddings, one per input text.
    """
    openai.api_key = openai_api_key
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=texts
        )
        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings
    except Exception as e:
        st.error(f"Error generating batch embeddings: {e}")
        return [[] for _ in texts]

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

def embed_pages_in_batches(df: pd.DataFrame, openai_api_key: str, batch_size: int = 10) -> pd.DataFrame:
    st.info("Embedding pages in batches. Please wait...")

    texts = []
    for _, row in df.iterrows():
        combined_text = f"{row['URL']} {row['H1']} {row['Meta Description']}"
        texts.append(combined_text)

    all_embeddings = []
    n = len(texts)
    for start_idx in range(0, n, batch_size):
        end_idx = start_idx + batch_size
        batch_texts = texts[start_idx:end_idx]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

    df["embedding"] = all_embeddings
    return df

def embed_query(query: str, openai_api_key: str) -> np.ndarray:
    results = embed_text_batch(openai_api_key, [query])
    return np.array(results[0]) if results else np.array([])

def semantic_search(df: pd.DataFrame, query: str, openai_api_key: str, top_k: int = 10) -> pd.DataFrame:
    query_embedding = embed_query(query, openai_api_key)
    if query_embedding.size == 0:
        return df.head(0)

    similarities = []
    for _, row in df.iterrows():
        page_embedding = row["embedding"]
        similarity = compute_cosine_similarity(query_embedding, np.array(page_embedding))
        similarities.append(similarity)

    df["similarity"] = similarities
    df_sorted = df.sort_values(by="similarity", ascending=False)
    return df_sorted.head(top_k)

###############################################################################
# Streamlit App
###############################################################################
def main():
    st.title("Internal Link Recommender Tool (Two-Step Flow)")
    st.write("""
    This version separates the workflow:
    1) Upload file, choose brand-new or existing, embed pages.
    2) If existing, pick your target page. Then generate links.
    """)

    # --- SAMPLE TEMPLATE DOWNLOAD ---
    st.subheader("Step A: Download Sample Template (Optional)")
    template_bytes = generate_sample_excel_template()
    template_link = get_download_link(template_bytes, "sample_template.xlsx", "Download Sample Excel Template")
    st.markdown(template_link, unsafe_allow_html=True)

    # --- FILE UPLOAD ---
    st.subheader("Step B: Upload Your CSV or Excel")
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel with columns: URL, H1, Meta Description",
        type=["csv", "xlsx"]
    )

    # --- MODE SELECT ---
    st.subheader("Step C: Pick Mode (Brand-New or Existing)")
    mode_option = st.radio(
        "Internal Linking Mode",
        ["Brand-New Topic/Page", "Optimize Existing Page"]
    )

    # --- TOPIC + KEYWORD ---
    topic = st.text_input("Topic", help="e.g., 'Best Coffee Beans'")
    keyword = st.text_input("Optional Keyword", help="Refine semantic focus")

    # --- OPENAI KEY ---
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    # --- EMBEDDING BATCH SIZE ---
    batch_size = st.slider("Batch Size for Embeddings", min_value=1, max_value=50, value=10)

    # Create placeholders to store data after embedding
    st.write("Click 'Embed Pages' first before generating links.")
    embed_button = st.button("Embed Pages")

    if embed_button:
        if not uploaded_file:
            st.error("Please upload a CSV or Excel file first.")
            st.stop()

        if not topic:
            st.error("Please enter a topic.")
            st.stop()

        if not openai_api_key:
            st.error("Please provide an OpenAI API key.")
            st.stop()

        # Read file
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        # Validate columns
        required_cols = {"URL", "H1", "Meta Description"}
        if not required_cols.issubset(df.columns):
            st.error(f"Please ensure these columns exist: {', '.join(required_cols)}")
            st.stop()

        # Embed pages
        df = embed_pages_in_batches(df, openai_api_key, batch_size)
        st.session_state["embedded_df"] = df
        st.session_state["mode"] = mode_option
        st.session_state["topic"] = topic
        st.session_state["keyword"] = keyword
        st.session_state["api_key"] = openai_api_key
        st.success("Pages embedded successfully! Now proceed to Step D below.")

    # Now Step D: If the pages are embedded, we can choose the existing URL (if mode is existing)
    if "embedded_df" in st.session_state:
        st.subheader("Step D: Generate Internal Links")

        df_after_embedding = st.session_state["embedded_df"]
        mode = st.session_state["mode"]
        final_topic = st.session_state["topic"]
        final_keyword = st.session_state["keyword"]
        final_api_key = st.session_state["api_key"]

        if mode == "Optimize Existing Page":
            st.write("Select your target page URL for optimization:")
            url_options = df_after_embedding["URL"].unique().tolist()
            selected_url = st.selectbox("Target Page URL", url_options)
            target_page_data = df_after_embedding.loc[df_after_embedding["URL"] == selected_url].iloc[0].to_dict()
            internal_mode = "existing_page"
        else:
            # brand-new
            st.write("You chose a brand-new topic/page. No URL to select.")
            target_page_data = {
                "URL": "N/A",
                "H1": f"(New) {final_topic}",
                "Meta Description": f"Future page about {final_topic}"
            }
            internal_mode = "brand_new"

        # Button to run the link generation
        if st.button("Generate Links"):
            candidate_df = df_after_embedding.copy()
            if internal_mode == "existing_page":
                candidate_df = candidate_df.loc[candidate_df["URL"] != target_page_data["URL"]]

            # Perform semantic search
            query_text = final_topic if not final_keyword else f"{final_topic} {final_keyword}"
            top_k = 20
            best_matches = semantic_search(candidate_df, query_text, final_api_key, top_k=top_k)

            # Filter by similarity
            threshold = 0.50
            filtered = best_matches[best_matches["similarity"] >= threshold].copy()

            # Generate GPT-based recommendations
            links = generate_internal_links(
                final_api_key,
                internal_mode,
                final_topic,
                final_keyword,
                target_page_data,
                filtered[["URL", "H1", "Meta Description"]]
            )

            if not links:
                st.warning("No recommendations were returned or the JSON was invalid.\n"
                           "Try adjusting your topic or keyword, or reduce temperature further.")
            else:
                # Convert to DataFrame
                results_df = pd.DataFrame(links)
                # Merge similarity
                merged_df = pd.merge(
                    results_df,
                    filtered[["URL", "similarity"]],
                    left_on="target_url",
                    right_on="URL",
                    how="left"
                )
                merged_df.drop(columns=["URL"], inplace=True, errors="ignore")
                if "similarity" in merged_df.columns:
                    merged_df.rename(columns={"similarity": "similarity_score"}, inplace=True)

                st.dataframe(merged_df)

                # Download options
                csv_data = merged_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download as CSV",
                    csv_data,
                    file_name=f"internal_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                excel_data = convert_to_excel(merged_df)
                st.download_button(
                    "Download as Excel",
                    excel_data,
                    file_name=f"internal_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
