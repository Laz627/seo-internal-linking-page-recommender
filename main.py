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


def parse_openai_response(response_text: str):
    """
    Given the raw response text from the OpenAI ChatCompletion,
    attempt to parse it as JSON. Returns a list of dictionaries
    with keys: 'target_url' and 'anchor_text'.

    If parsing fails, return an empty list.
    """
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            # In case the model accidentally returns a JSON object
            data = [data]
        return data
    except json.JSONDecodeError:
        return []


def parse_openai_response_with_warning(response_text: str):
    """
    Same as parse_openai_response, but warns the user if parsing fails.
    """
    parsed = parse_openai_response(response_text)
    if not parsed:
        st.warning("Unable to parse the model response as valid JSON.")
    return parsed


def build_prompt(
    mode: str, 
    topic: str, 
    keyword: str, 
    target_page: dict, 
    candidate_pages: pd.DataFrame
) -> str:
    """
    Construct the prompt text that will be sent to the OpenAI ChatCompletion API.

    :param mode: "brand_new" or "existing_page"
    :param topic: The main topic input by the user
    :param keyword: Optional target keyword
    :param target_page: dict with "URL", "H1", "Meta Description" 
                       (only used if mode == "existing_page")
    :param candidate_pages: DataFrame of pages to consider for internal linking
    """

    if mode == "brand_new":
        # For a brand-new topic/page, we have no real "target URL" in the dataset
        # We'll pass a placeholder URL and H1 to GPT, but clarify it's brand-new
        target_url = "N/A (brand-new page, not published yet)"
        target_h1 = f"{topic} (New Topic)"
        target_meta = f"A future page about {topic}."
        brand_new_info = """
This is a brand-new topic or page that doesn't exist yet.
Recommend internal links from the pages below that are thematically relevant
and provide anchor text suggestions that are semantically aligned with
the new topic.
"""
    else:
        # Existing page scenario
        target_url = target_page["URL"]
        target_h1 = target_page["H1"]
        target_meta = target_page["Meta Description"]
        brand_new_info = ""

    pages_info = []
    for _, row in candidate_pages.iterrows():
        pages_info.append({
            "URL": row["URL"],
            "H1": row["H1"],
            "Meta Description": row["Meta Description"]
        })

    # Provide a more explicit instruction for strict JSON output
    prompt = f"""
You are an SEO assistant. The user wants to generate internal link recommendations.

Mode: {mode}
Topic: {topic}
Target Keyword (optional): {keyword}

{brand_new_info}

Target Page Information:
- Target Page URL: {target_url}
- Target Page H1: {target_h1}
- Target Page Meta Description: {target_meta}

Below is a list of other pages (candidate link targets) on the site, in JSON format:

{json.dumps(pages_info, indent=2)}

Please propose a list of internal links (FROM the pages listed above) that are most semantically relevant
to this topic. For each recommended link, provide an optimized anchor text that is user-friendly, 
semantically relevant, and rich with context.

Return the result as valid JSON (strictly no extra text). The JSON must be an array of objects,
where each object has these keys:
- "target_url": The URL of the page to link FROM
- "anchor_text": The anchor text for the link

Example JSON response:
[
  {{
    "target_url": "https://example.com/page-1",
    "anchor_text": "Relevant anchor text"
  }},
  {{
    "target_url": "https://example.com/page-2",
    "anchor_text": "Another relevant anchor text"
  }}
]
"""
    return prompt


def generate_internal_links(
    openai_api_key: str, 
    mode: str,
    topic: str, 
    keyword: str, 
    target_page: dict, 
    candidate_pages: pd.DataFrame
):
    """
    Calls the OpenAI ChatCompletion API to generate internal link recommendations.
    Mode can be "brand_new" or "existing_page".
    """
    # Set up the OpenAI API key
    openai.api_key = openai_api_key

    # Build the prompt
    prompt = build_prompt(mode, topic, keyword, target_page, candidate_pages)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-4o-2024-05-13", etc.
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful AI assistant specialized in SEO recommendations."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        response_text = response.choices[0].message["content"].strip()
        links = parse_openai_response_with_warning(response_text)
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
    Generates embeddings for a batch of texts in a single API call using OpenAI's Embeddings API.
    Returns a list of embeddings, one per input text.
    """
    openai.api_key = openai_api_key
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=texts
        )
        # The embeddings come back in "data" in the same order as the input
        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings
    except Exception as e:
        st.error(f"Error generating batch embeddings: {e}")
        return [[] for _ in texts]


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.
    """
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def embed_pages_in_batches(df: pd.DataFrame, openai_api_key: str, batch_size: int = 10) -> pd.DataFrame:
    """
    Compute embeddings for each page in batches (reducing the number of API calls).
    Adds a new column "embedding" to the DataFrame.
    """
    st.info("Embedding pages in batches. Please wait...")

    # We'll combine URL + H1 + Meta Description
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
    """
    Embed a single query string (topic + keyword).
    """
    results = embed_text_batch(openai_api_key, [query])
    return np.array(results[0]) if results else np.array([])


def semantic_search(df: pd.DataFrame, query: str, openai_api_key: str, top_k: int = 10) -> pd.DataFrame:
    """
    Given a DataFrame of pages (with "embedding" column),
    embed the query, compute similarity, and return top_k results (sorted by similarity).
    """
    query_embedding = embed_query(query, openai_api_key)
    if query_embedding.size == 0:
        return df.head(0)  # Return empty if something went wrong

    similarities = []
    for idx, row in df.iterrows():
        page_embedding = row["embedding"]
        similarity = compute_cosine_similarity(
            query_embedding, 
            np.array(page_embedding)
        )
        similarities.append(similarity)

    df["similarity"] = similarities
    # Sort by descending similarity
    df_sorted = df.sort_values(by="similarity", ascending=False)
    # Return top_k
    return df_sorted.head(top_k)

###############################################################################
# Streamlit App
###############################################################################
def main():
    st.title("Internal Link Recommender Tool (Batch Embedding + Brand New / Existing Mode)")
    st.write("""
    This application helps SEO professionals generate internal link recommendations
    for a specific topic/keyword.
    
    You can choose:
    - A brand-new topic with no existing page
    - An existing page in your dataset
    """)

    # Provide a link to download the sample Excel template
    st.subheader("1. Download Sample Excel Template")
    template_bytes = generate_sample_excel_template()
    template_link = get_download_link(
        template_bytes,
        "sample_template.xlsx",
        "Click here to download the sample Excel template."
    )
    st.markdown(template_link, unsafe_allow_html=True)

    # File Uploader
    st.subheader("2. Upload Your Website Pages CSV or Excel File")
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file with columns: URL, H1, Meta Description",
        type=["csv", "xlsx"]
    )

    # Mode Selector
    st.subheader("3. Choose Topic Mode")
    mode = st.radio(
        "Is this a brand-new topic (no existing URL) or do you want to optimize an existing page?",
        options=["Brand-New Topic/Page", "Optimize Existing Page"]
    )

    # Text inputs: Topic and optional Target Keyword
    st.subheader("4. Enter Your Topic and Optional Target Keyword")
    topic = st.text_input("Topic", help="Enter the main topic for which you want to generate internal links.")
    keyword = st.text_input("Target Keyword (optional)", help="Enter a target keyword to fine-tune semantic relevance.")

    # OpenAI API Key input (type='password' for security)
    st.subheader("5. Enter Your OpenAI API Key")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your OpenAI API key. Get yours at https://platform.openai.com/."
    )

    # Batch Size selection
    st.subheader("Optional: Set Batch Size for Embedding")
    batch_size = st.slider("Number of pages to embed per request", min_value=1, max_value=50, value=10)

    # Create a button to trigger link generation
    if st.button("Generate Links"):
        if not uploaded_file:
            st.error("Please upload a CSV or Excel file.")
            return
        
        if not topic:
            st.error("Please enter a topic.")
            return
        
        if not openai_api_key:
            st.error("Please enter your OpenAI API key.")
            return

        # Read the uploaded file into a pandas DataFrame
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        # Check for required columns
        required_columns = {"URL", "H1", "Meta Description"}
        if not required_columns.issubset(df.columns):
            st.error(f"Uploaded file must contain columns: {', '.join(required_columns)}")
            return

        # Build embeddings for each page in batches
        try:
            df = embed_pages_in_batches(df, openai_api_key, batch_size=batch_size)
        except Exception as e:
            st.error(f"Error building embeddings: {e}")
            return

        # If user picked brand-new topic:
        if mode == "Brand-New Topic/Page":
            # We have no target page in the dataset. We'll pass a dummy dictionary.
            target_page_data = {
                "URL": "N/A",
                "H1": "New Topic Page (not published)",
                "Meta Description": f"Future page about {topic}"
            }
            internal_links_mode = "brand_new"
        else:
            # Optimize existing page
            st.write("**Select the Target Page** from the list below:")
            target_pages = df["URL"].unique().tolist()
            selected_url = st.selectbox(
                "Choose a URL to serve as the target page for new internal links:",
                options=target_pages
            )
            target_page_data = df.loc[df["URL"] == selected_url].iloc[0].to_dict()
            internal_links_mode = "existing_page"

        # Exclude the target page from candidate pages if we have an actual URL
        if internal_links_mode == "existing_page":
            candidate_df = df.loc[df["URL"] != target_page_data["URL"]].copy()
        else:
            # brand_new case: all pages are candidates
            candidate_df = df.copy()

        # Combine the topic and keyword into a query for semantic search
        query_text = topic if not keyword else f"{topic} {keyword}"

        # Find top K most relevant pages using semantic search
        top_k = 20  
        best_matches = semantic_search(candidate_df, query_text, openai_api_key, top_k=top_k)

        # Apply a similarity threshold of 0.50
        threshold = 0.50
        filtered_matches = best_matches[best_matches["similarity"] >= threshold].copy()

        # Now we pass these filtered matches to the GPT prompt
        links = generate_internal_links(
            openai_api_key,
            internal_links_mode,
            topic,
            keyword,
            target_page_data,
            filtered_matches[["URL", "H1", "Meta Description"]]
        )

        # Display results
        if links:
            # Convert links to a DataFrame
            results_df = pd.DataFrame(links)

            # Merge the similarity scores for each URL
            # "target_url" in GPT results corresponds to "URL" in df
            # We'll add the similarity column to results
            merged_df = pd.merge(
                results_df,
                filtered_matches[["URL", "similarity"]],
                left_on="target_url",
                right_on="URL",
                how="left"
            )

            # Clean up DataFrame (remove duplicate column)
            merged_df.drop(columns=["URL"], inplace=True, errors="ignore")

            # Rename 'similarity' column for clarity
            if "similarity" in merged_df.columns:
                merged_df.rename(columns={"similarity": "similarity_score"}, inplace=True)

            st.subheader("Internal Link Recommendations")
            if merged_df.empty:
                st.warning("The model returned data, but we couldn't merge any similarity scores. "
                           "This could indicate the model recommended URLs outside your dataset.")
            else:
                st.dataframe(merged_df)

            # Download options
            st.subheader("Download Your Results")
            # 1) CSV
            csv_data = merged_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name=f"internal_link_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            # 2) Excel
            excel_data = convert_to_excel(merged_df)
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name=f"internal_link_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No recommendations were returned. Please try again or adjust your topic/keyword.")

if __name__ == "__main__":
    main()
