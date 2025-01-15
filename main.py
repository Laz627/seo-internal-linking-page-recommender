import streamlit as st
import pandas as pd
import openai
import json
import io
import base64
import numpy as np
from openpyxl import Workbook
from datetime import datetime
from docx import Document  # <-- NEW: python-docx for parsing .docx files

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
            data = [data]
        return data
    except json.JSONDecodeError:
        return []

###############################################################################
# Parsing Word Documents (NEW)
###############################################################################
def parse_word_doc(file_obj: io.BytesIO) -> list[dict]:
    """
    Given a BytesIO object for a .docx file, extracts headings and paragraphs.
    Returns a list of dict, each containing:
      - 'type': 'heading' or 'paragraph'
      - 'content': the text
    """
    doc = Document(file_obj)
    results = []

    for block in doc.paragraphs:
        text = block.text.strip()
        if not text:
            continue
        # We can treat paragraphs as 'paragraph' or detect if block.style.name == 'Heading x'
        style_name = block.style.name.lower() if block.style else ""
        if "heading" in style_name:
            results.append({"type": "heading", "content": text})
        else:
            results.append({"type": "paragraph", "content": text})
    return results

###############################################################################
# Prompt Construction with Anchor Text Best Practices
###############################################################################
def build_prompt(
    mode: str,
    topic: str,
    keyword: str,
    target_data: dict,
    candidate_pages: pd.DataFrame
) -> str:
    """
    Construct the prompt text for the ChatCompletion, 
    incorporating anchor text best practices.
    """

    # If mode is "analyze_word_doc", interpret the doc content as the "target" 
    # which we want to create links TO or FROM. We'll assume we want suggestions
    # for linking FROM this doc content TO existing site pages. 
    # Adjust logic as needed for your scenario.

    if mode == "brand_new":
        target_url = "N/A (brand-new page, not published yet)"
        target_h1 = f"{topic} (New Topic)"
        target_meta = f"A future page about {topic}."
        brand_new_msg = (
            "This is a brand-new topic/page that doesn't exist yet. "
            "Recommend internal links from the pages below that are thematically relevant, "
            "providing anchor text suggestions that follow best practices.\n"
        )
    elif mode == "analyze_word_doc":
        target_url = "N/A (Word doc content, not published yet)"
        # We'll store combined doc content in `target_data["content"]`
        target_h1 = f"(Word Doc) {topic}"
        target_meta = target_data.get("content", "No doc content?")[:200]
        brand_new_msg = (
            "You have content from a Word document (not published yet). "
            "Recommend the best existing pages to link to from this doc, "
            "using anchor text best practices.\n"
        )
    else:
        # existing_page
        target_url = target_data["URL"]
        target_h1 = target_data["H1"]
        target_meta = target_data["Meta Description"]
        brand_new_msg = ""

    pages_info = []
    for _, row in candidate_pages.iterrows():
        pages_info.append({
            "URL": row["URL"],
            "H1": row["H1"],
            "Meta Description": row["Meta Description"]
        })

    anchor_text_guidelines = """
Anchor Text Best Practices:
- Descriptive, concise, and relevant to both the linking page and the linked page.
- Avoid overly generic text like 'click here' or 'read more'.
- Avoid overly long or keyword-stuffed anchor text.
- The text should read naturally, make sense on its own, and reflect the linked page's content.
- Surrounding text also matters; do not chain multiple links with no context.
"""

    prompt = f"""
You are an SEO assistant. The user wants internal link recommendations.

Mode: {mode}
Topic: {topic}
Target Keyword (optional): {keyword}

{brand_new_msg}

Target Content:
- Target URL: {target_url}
- Target H1: {target_h1}
- Target Meta/Excerpt: {target_meta}

Below is a list of other pages (candidate link targets) in JSON format:

{json.dumps(pages_info, indent=2)}

Please follow these anchor text best practices:
{anchor_text_guidelines}

TASK:
1. Identify the most semantically relevant pages from the above list.
2. For each recommended link, provide an optimized anchor text that is:
   - descriptive and concise
   - relevant to the doc or brand-new topic
   - does not unnecessarily stuff keywords
   - reads naturally
3. Return your response strictly in JSON. No extra text.
   The JSON must be an array of objects, each object having:
     - "target_url": The URL to link FROM
     - "anchor_text": The anchor text

Example JSON:
[
  {{
    "target_url": "https://example.com/page-1",
    "anchor_text": "Concise descriptive anchor text example"
  }},
  {{
    "target_url": "https://example.com/page-2",
    "anchor_text": "Another well-structured anchor text"
  }}
]
""".strip()
    return prompt


def generate_internal_links(
    openai_api_key: str,
    mode: str,
    topic: str,
    keyword: str,
    target_data: dict,
    candidate_pages: pd.DataFrame
):
    """
    Calls the OpenAI ChatCompletion API to generate internal link recommendations
    (brand-new, existing, or doc analysis).
    """
    openai.api_key = openai_api_key
    prompt = build_prompt(mode, topic, keyword, target_data, candidate_pages)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-4o-2024-05-13"
            messages=[
                {"role": "system", "content": "You are a helpful SEO assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.0
        )
        response_text = response.choices[0].message["content"].strip()
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


def embed_pages_in_batches(df: pd.DataFrame, openai_api_key: str, batch_size: int = 10):
    """
    Compute embeddings for each page in batches, showing a progress bar to the user.
    """
    st.info("Embedding pages (website pages) in batches. Please wait...")

    texts = []
    for _, row in df.iterrows():
        combined_text = f"{row['URL']} {row['H1']} {row['Meta Description']}"
        texts.append(combined_text)

    all_embeddings = []
    n = len(texts)

    progress_bar = st.progress(0)
    progress_label = st.empty()

    total_batches = (n + batch_size - 1) // batch_size
    for batch_index, start_idx in enumerate(range(0, n, batch_size), start=1):
        end_idx = start_idx + batch_size
        batch_texts = texts[start_idx:end_idx]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        # Update progress
        progress_val = int(batch_index / total_batches * 100)
        progress_bar.progress(progress_val)
        progress_label.write(f"Processed batch {batch_index} / {total_batches}")

    df["embedding"] = all_embeddings
    return df


def embed_doc_in_batches(doc_data: list[dict], openai_api_key: str, batch_size: int = 10):
    """
    Similar to embed_pages_in_batches, but for the Word doc content (headings/paragraphs).
    We'll combine them into a single text per item.
    Returns a DataFrame with columns:
      - 'type': heading or paragraph
      - 'content': the actual text
      - 'embedding': the embedding vector
    """
    st.info("Embedding content from Word document. Please wait...")

    # Prepare text for embedding
    texts = []
    for item in doc_data:
        # Combine "type" + content if desired, or just content
        combined_text = f"[{item['type'].upper()}] {item['content']}"
        texts.append(combined_text)

    all_embeddings = []
    n = len(texts)

    progress_bar = st.progress(0)
    progress_label = st.empty()

    total_batches = (n + batch_size - 1) // batch_size
    for batch_index, start_idx in enumerate(range(0, n, batch_size), start=1):
        end_idx = start_idx + batch_size
        batch_texts = texts[start_idx:end_idx]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        # Update progress
        progress_val = int(batch_index / total_batches * 100)
        progress_bar.progress(progress_val)
        progress_label.write(f"Doc content batch {batch_index} / {total_batches}")

    # Build a DataFrame
    df_doc = pd.DataFrame(doc_data)  # has 'type', 'content'
    df_doc["embedding"] = all_embeddings
    return df_doc


def semantic_search(df: pd.DataFrame, query: str, openai_api_key: str, top_k: int = 50) -> pd.DataFrame:
    query_embedding = embed_text_batch(openai_api_key, [query])
    if not query_embedding or not query_embedding[0]:
        return df.head(0)
    query_vector = np.array(query_embedding[0])

    similarities = []
    for _, row in df.iterrows():
        page_embedding = np.array(row["embedding"])
        sim = compute_cosine_similarity(query_vector, page_embedding)
        similarities.append(sim)

    df["similarity"] = similarities
    df_sorted = df.sort_values(by="similarity", ascending=False)
    return df_sorted.head(top_k)

###############################################################################
# Streamlit App
###############################################################################
def main():
    st.title("Internal Link Recommender (with Word Doc Analysis)")
    st.write("""
    This tool now supports three scenarios:
    1) **Brand-New Topic/Page** (no existing URL),
    2) **Optimize Existing Page** (select from your site list),
    3) **Analyze Word Document** (parse headings/paragraphs for recommended links).

    Follow the steps below.
    """)

    # Step 1: (Optional) Download sample template
    st.subheader("1) Optional: Download Sample Template for Site Pages")
    template_bytes = generate_sample_excel_template()
    template_link = get_download_link(
        template_bytes,
        "sample_template.xlsx",
        "Download Sample Excel Template"
    )
    st.markdown(template_link, unsafe_allow_html=True)

    # Step 2: Upload your site CSV/Excel
    st.subheader("2) Upload Site Pages CSV or Excel")
    uploaded_file = st.file_uploader(
        "Upload a CSV/Excel with columns: URL, H1, Meta Description",
        type=["csv", "xlsx"]
    )

    df = None
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Step 3: Choose Mode
    st.subheader("3) Choose Mode")
    mode_option = st.radio(
        "Mode:",
        ["Brand-New Topic/Page", "Optimize Existing Page", "Analyze Word Document"]
    )

    # Additional file uploader for Word doc if "Analyze Word Document"
    docx_data = None
    doc_paragraphs = []
    if mode_option == "Analyze Word Document":
        st.subheader("3a) Upload Word Document (.docx)")
        uploaded_doc = st.file_uploader(
            "Upload .docx file to parse headings/paragraphs",
            type=["docx"]
        )
        if uploaded_doc is not None:
            try:
                docx_data = uploaded_doc.read()
                # parse doc
                doc_paragraphs = parse_word_doc(io.BytesIO(docx_data))
                st.success(f"Parsed {len(doc_paragraphs)} text blocks from the Word doc.")
            except Exception as e:
                st.error(f"Error parsing Word document: {e}")

    # If existing page, user picks the URL
    selected_url = None
    if mode_option == "Optimize Existing Page":
        if df is not None and {"URL", "H1", "Meta Description"}.issubset(df.columns):
            url_options = df["URL"].unique().tolist()
            selected_url = st.selectbox("Target Page URL to Optimize", url_options)
        else:
            st.info("Please upload a valid CSV/Excel with the required columns first.")

    # Step 4: Topic/Keyword
    st.subheader("4) Topic & Optional Keyword")
    topic = st.text_input("Topic (required)", help="e.g., 'Best Coffee Beans'")
    keyword = st.text_input("Optional Keyword", help="e.g., 'dark roast', etc.")

    # Step 5: OpenAI API Key
    st.subheader("5) Enter Your OpenAI API Key")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Get yours at https://platform.openai.com/"
    )

    # Step 6: Embedding Batch Size
    st.subheader("6) Set Batch Size for Embeddings")
    batch_size = st.slider(
        "Number of pages (or doc paragraphs) to embed per request",
        min_value=1, max_value=50, value=10
    )

    # Step 7: Embed Data
    st.subheader("7) Embed Data")
    st.write("""
    **For brand-new or existing page modes**, we'll embed your site pages.  
    **For Word document**, we'll embed site pages + doc content to do cross-referencing.
    """)
    embed_button = st.button("Embed Now")

    if embed_button:
        if not openai_api_key:
            st.error("Please provide an OpenAI API key.")
            st.stop()

        if mode_option in ["Brand-New Topic/Page", "Optimize Existing Page"]:
            if not uploaded_file or df is None:
                st.error("Please upload a valid CSV/Excel with site pages first.")
                st.stop()
            # Validate columns
            required_cols = {"URL", "H1", "Meta Description"}
            if not required_cols.issubset(df.columns):
                st.error(f"File missing columns: {', '.join(required_cols)}")
                st.stop()

            if not topic:
                st.error("Please enter a topic.")
                st.stop()

            # Embed site pages
            df_embedded = embed_pages_in_batches(df.copy(), openai_api_key, batch_size)
            st.session_state["site_df_embedded"] = df_embedded
            st.session_state["doc_df_embedded"] = None  # not used in brand-new or existing
            st.session_state["mode"] = mode_option
            st.session_state["topic"] = topic
            st.session_state["keyword"] = keyword
            st.session_state["selected_url"] = selected_url
            st.success("Embedding for site pages complete. Proceed to generate links below.")

        elif mode_option == "Analyze Word Document":
            if not uploaded_file or df is None:
                st.error("Please upload a site pages CSV/Excel to compare them with your doc.")
                st.stop()

            if not doc_paragraphs:
                st.error("Please upload a Word document to parse.")
                st.stop()

            required_cols = {"URL", "H1", "Meta Description"}
            if not required_cols.issubset(df.columns):
                st.error(f"Site file missing columns: {', '.join(required_cols)}")
                st.stop()

            if not topic:
                st.error("Please enter a topic to help guide GPT.")
                st.stop()

            # 1) Embed site pages
            df_embedded = embed_pages_in_batches(df.copy(), openai_api_key, batch_size)
            # 2) Embed doc paragraphs
            df_doc = embed_doc_in_batches(doc_paragraphs, openai_api_key, batch_size)

            st.session_state["site_df_embedded"] = df_embedded
            st.session_state["doc_df_embedded"] = df_doc
            st.session_state["mode"] = "analyze_word_doc"  # we'll treat it as a special mode
            st.session_state["topic"] = topic
            st.session_state["keyword"] = keyword
            st.success("Embedding complete for both site pages and Word document. Generate links below.")

    # Step 8: Generate Links
    st.subheader("8) Generate Links")
    if st.button("Generate Links"):
        if "mode" not in st.session_state:
            st.error("Please embed data first.")
            st.stop()

        final_mode = st.session_state["mode"]
        final_topic = st.session_state["topic"]
        final_keyword = st.session_state["keyword"]
        final_url = st.session_state.get("selected_url", None)
        site_df_embedded = st.session_state.get("site_df_embedded", None)
        doc_df_embedded = st.session_state.get("doc_df_embedded", None)

        if final_mode in ["Brand-New Topic/Page", "Optimize Existing Page"]:
            # We'll do the same process as before
            if site_df_embedded is None:
                st.error("No embedded site data found.")
                st.stop()

            if final_mode == "Optimize Existing Page" and final_url:
                # existing
                try:
                    target_row = site_df_embedded.loc[site_df_embedded["URL"] == final_url].iloc[0].to_dict()
                except IndexError:
                    st.error("Selected URL not found in embedded dataset.")
                    st.stop()
                internal_mode = "existing_page"
                # exclude it from candidates
                candidate_df = site_df_embedded.loc[site_df_embedded["URL"] != final_url].copy()

                # combine topic + keyword
                query = final_topic if not final_keyword else f"{final_topic} {final_keyword}"
                best_matches = semantic_search(candidate_df, query, openai_api_key, top_k=50)
                threshold = 0.50
                filtered = best_matches[best_matches["similarity"] >= threshold].copy()

                # GPT
                links = generate_internal_links(
                    openai_api_key,
                    internal_mode,
                    final_topic,
                    final_keyword,
                    target_row,
                    filtered[["URL", "H1", "Meta Description"]]
                )

            else:
                # brand_new
                internal_mode = "brand_new"
                # all site pages are candidates
                candidate_df = site_df_embedded.copy()

                query = final_topic if not final_keyword else f"{final_topic} {final_keyword}"
                best_matches = semantic_search(candidate_df, query, openai_api_key, top_k=50)
                threshold = 0.50
                filtered = best_matches[best_matches["similarity"] >= threshold].copy()

                # dummy target data
                target_data = {
                    "URL": "N/A",
                    "H1": f"(New) {final_topic}",
                    "Meta Description": f"Future page about {final_topic}"
                }
                links = generate_internal_links(
                    openai_api_key,
                    internal_mode,
                    final_topic,
                    final_keyword,
                    target_data,
                    filtered[["URL", "H1", "Meta Description"]]
                )

        else:
            # analyze_word_doc
            if site_df_embedded is None or doc_df_embedded is None:
                st.error("We need both site and doc embeddings, but they're missing.")
                st.stop()

            # We'll treat the entire doc as "the new target"
            # We can combine all doc content into a single text snippet or pick certain paragraphs.
            # For simplicity, let's combine doc paragraphs into one big string.
            combined_doc_content = "\n".join(
                [item["content"] for item in doc_paragraphs]
            )
            target_data = {
                "content": combined_doc_content
            }

            # The user is basically “embedding a doc, wanting to find relevant site pages to link to”
            # -> We'll do semantic search with the doc as "query."
            # However, if doc is large, you might want to break it up. 
            # For now, let's do a single query with the doc content + topic+keyword as a summary.

            doc_query = f"{combined_doc_content[:1000]} {final_topic} {final_keyword}"
            # We'll limit to first 1000 chars of doc for the query. 
            # A more advanced approach might average doc embeddings or do multiple queries.

            # semantic search among site pages
            best_matches = semantic_search(site_df_embedded.copy(), doc_query, openai_api_key, top_k=50)
            threshold = 0.50
            filtered = best_matches[best_matches["similarity"] >= threshold].copy()

            # call GPT
            links = generate_internal_links(
                openai_api_key,
                "analyze_word_doc",
                final_topic,
                final_keyword,
                target_data,
                filtered[["URL", "H1", "Meta Description"]]
            )

        # Show results
        if not links:
            st.warning("No recommendations or invalid JSON. Try adjusting doc content, topic, or threshold.")
        else:
            results_df = pd.DataFrame(links)
            # Merge similarity
            if 'filtered' in locals() and not filtered.empty:
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
            else:
                merged_df = results_df.copy()

            st.write("**Recommendations**")
            st.dataframe(merged_df)
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
