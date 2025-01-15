import streamlit as st
import pandas as pd
import openai
import json
import io
import base64
import numpy as np
from openpyxl import Workbook
from datetime import datetime
from docx import Document
import re

###############################################################################
# 1. Helper Functions
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

    # (Optional) Sample row
    ws["A2"] = "https://example.com/sample-page"
    ws["B2"] = "Sample Page Title"
    ws["C2"] = "A sample meta description for demonstration."

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def get_download_link(file_bytes: bytes, filename: str, link_label: str) -> str:
    """
    Generates an HTML download link for a file in memory (bytes).
    """
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_label}</a>'


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

###############################################################################
# 2. Word Doc Parsing (Paragraphs only)
###############################################################################
def parse_word_doc_paragraphs_only(file_obj: io.BytesIO) -> list[dict]:
    """
    Extracts ONLY paragraphs from the .docx file, skipping headings.

    Returns a list of dict:
      [
        { "type": "paragraph", "content": "...text..." },
        ...
      ]
    """
    doc = Document(file_obj)
    paragraphs_only = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name.lower() if para.style else ""
        if "heading" not in style_name:
            paragraphs_only.append({"type": "paragraph", "content": text})
    return paragraphs_only

###############################################################################
# 3. Embedding Utilities
###############################################################################
def embed_text_batch(openai_api_key: str, texts: list[str], model="text-embedding-ada-002"):
    """
    Embeds a batch of strings using the OpenAI Embeddings API in one request.
    """
    openai.api_key = openai_api_key
    try:
        response = openai.Embedding.create(
            model=model,
            input=texts
        )
        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings
    except Exception as e:
        st.error(f"Error generating batch embeddings: {e}")
        return [[] for _ in texts]


def embed_site_pages_in_batches(df: pd.DataFrame, openai_api_key: str, batch_size: int = 10):
    """
    For each page (row), embed URL + H1 + Meta Description.
    """
    st.info("Embedding site pages. Please wait...")

    texts = []
    for _, row in df.iterrows():
        combined_text = f"{row['URL']} {row['H1']} {row['Meta Description']}"
        texts.append(combined_text)

    all_embeddings = []
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    bar = st.progress(0)
    label = st.empty()

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_texts = texts[start_idx:end_idx]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        progress_val = int((batch_idx + 1) / total_batches * 100)
        bar.progress(progress_val)
        label.write(f"Processed batch {batch_idx+1}/{total_batches}")

    df["embedding"] = all_embeddings
    return df


def embed_doc_paragraphs_in_batches(paragraphs: list[dict], openai_api_key: str, batch_size: int = 10):
    """
    Embeds each paragraph's text. Returns a DataFrame with columns:
      - type: "paragraph"
      - content: actual paragraph text
      - embedding: the embedding vector
    """
    st.info("Embedding doc paragraphs. Please wait...")

    texts = [p["content"] for p in paragraphs]

    all_embeddings = []
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    bar = st.progress(0)
    label = st.empty()

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_texts = texts[start_idx:end_idx]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        progress_val = int((batch_idx + 1) / total_batches * 100)
        bar.progress(progress_val)
        label.write(f"Paragraph batch {batch_idx+1}/{total_batches}")

    df_doc = pd.DataFrame(paragraphs)  # columns: type, content
    df_doc["embedding"] = all_embeddings
    return df_doc

def embed_sentences_in_paragraph(paragraph_text: str, openai_api_key: str):
    """
    Split a single paragraph into sentences, embed each sentence,
    return a list of (sentence, embedding).
    """
    # naive sentence split, can be improved
    # e.g. re.split(r'(?<=[.!?])\s+', paragraph_text)
    sentences = re.split(r'(?<=[.!?])\s+', paragraph_text.strip())
    # remove empty strings
    sentences = [s.strip() for s in sentences if s.strip()]

    # embed all at once
    embeddings = embed_text_batch(openai_api_key, sentences)
    return list(zip(sentences, embeddings))

###############################################################################
# 4. Brand-New / Existing Page Logic (GPT-based)
###############################################################################
def build_prompt_for_mode(mode: str, topic: str, keyword: str, target_data: dict, candidate_pages: pd.DataFrame) -> str:
    anchor_text_guidelines = """
Anchor Text Best Practices:
- Descriptive, concise, and relevant
- Avoid 'click here', 'read more'
- Avoid keyword stuffing or overly long anchor text
- Should read naturally and reflect the linked page
"""

    if mode == "brand_new":
        brand_new_msg = (
            "Brand-new topic/page that doesn't exist yet.\n"
            "Recommend internal links from these site pages.\n"
        )
        target_url = "N/A"
        target_h1 = f"{topic} (New Topic)"
        target_meta = f"A future page about {topic}"
    else:
        brand_new_msg = ""
        target_url = target_data["URL"]
        target_h1 = target_data["H1"]
        target_meta = target_data["Meta Description"]

    pages_info = []
    for _, row in candidate_pages.iterrows():
        pages_info.append({
            "URL": row["URL"],
            "H1": row["H1"],
            "Meta Description": row["Meta Description"]
        })

    prompt = f"""
You are an SEO assistant. Mode: {mode}
Topic: {topic}
Keyword (optional): {keyword}

{brand_new_msg}

Target Page:
URL: {target_url}
H1: {target_h1}
Meta Description: {target_meta}

Site Pages (JSON):
{json.dumps(pages_info, indent=2)}

Please follow anchor text best practices:
{anchor_text_guidelines}

Return strictly JSON (no extra text). The JSON must be an array of objects:
[
  {{
    "target_url": "...",
    "anchor_text": "..."
  }},
  ...
]
""".strip()

    return prompt


def generate_internal_links(openai_api_key: str, mode: str, topic: str, keyword: str,
                            target_data: dict, candidate_pages: pd.DataFrame):
    """
    For brand-new or existing mode, calls GPT for a single JSON response.
    """
    openai.api_key = openai_api_key
    prompt = build_prompt_for_mode(mode, topic, keyword, target_data, candidate_pages)

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful SEO assistant. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        text_out = resp.choices[0].message["content"].strip()
        data = json.loads(text_out)
        if isinstance(data, dict):
            data = [data]
        return data
    except Exception as e:
        st.error(f"Error calling GPT: {e}")
        return []

###############################################################################
# 5. Streamlit App
###############################################################################
def main():
    st.title("Internal Link Recommender (3 Modes, No GPT for Word Doc)")

    st.write("""
    **Modes**:
    1) **Brand-New Topic/Page** (requires a Topic/Keyword, calls GPT).
    2) **Optimize Existing Page** (requires a Topic/Keyword, calls GPT).
    3) **Analyze Word Document** (skips headings, no GPT calls, just embeddings):
       - For each site page, find the single doc paragraph with highest similarity.
       - Then within that paragraph, pick the best matching sentence to serve as the anchor text.
       - The result is one recommended anchor substring per site page.
    """)

    # Step 1: Download sample
    st.subheader("Step 1: (Optional) Download Site Template")
    sample_xlsx = generate_sample_excel_template()
    link = get_download_link(sample_xlsx, "sample_template.xlsx", "Download Sample Excel Template")
    st.markdown(link, unsafe_allow_html=True)

    # Step 2: Upload site pages
    st.subheader("Step 2: Upload Site Pages (CSV/Excel)")
    pages_file = st.file_uploader("Upload site CSV/Excel (URL, H1, Meta Description)", type=["csv", "xlsx"])
    df_pages = None
    if pages_file:
        try:
            if pages_file.name.endswith(".csv"):
                df_pages = pd.read_csv(pages_file)
            else:
                df_pages = pd.read_excel(pages_file)
        except Exception as e:
            st.error(f"Error reading pages file: {e}")

    # Step 3: Mode
    st.subheader("Step 3: Choose Mode")
    mode_option = st.radio(
        "Pick one:",
        ["Brand-New Topic/Page", "Optimize Existing Page", "Analyze Word Document"]
    )

    # If existing page, user picks
    selected_url = None
    if mode_option == "Optimize Existing Page" and df_pages is not None:
        # check columns
        if {"URL", "H1", "Meta Description"}.issubset(df_pages.columns):
            selected_url = st.selectbox("Target Page URL", df_pages["URL"].unique().tolist())
        else:
            st.warning("Please upload valid CSV/Excel with the required columns.")

    # If Word doc analysis, parse paragraphs only
    doc_file = None
    doc_paragraphs = []
    if mode_option == "Analyze Word Document":
        st.subheader("Step 3A: Upload Word Document (.docx, skipping headings)")
        doc_file = st.file_uploader("Upload your .docx", type=["docx"])
        if doc_file:
            try:
                docx_data = doc_file.read()
                doc_paragraphs = parse_word_doc_paragraphs_only(io.BytesIO(docx_data))
                st.success(f"Found {len(doc_paragraphs)} paragraphs (no headings).")
            except Exception as e:
                st.error(f"Error parsing Word doc: {e}")

    # Step 4: Topic + Keyword (only needed for brand-new / existing)
    st.subheader("Step 4: Topic & (Optional) Keyword")
    if mode_option in ["Brand-New Topic/Page", "Optimize Existing Page"]:
        topic = st.text_input("Topic (required)")
        keyword = st.text_input("Optional Keyword")
    else:
        # For Word doc, we skip forcing topic/keyword
        topic = ""
        keyword = ""

    # Step 5: OpenAI Key
    st.subheader("Step 5: OpenAI API Key (for embeddings, GPT in brand-new/existing)")
    openai_api_key = st.text_input("OpenAI Key", type="password")

    # Step 6: Batch Size
    st.subheader("Step 6: Set Batch Size for Embedding")
    batch_size = st.slider("Batch size", min_value=1, max_value=50, value=10)

    # Step 7: Embed
    st.subheader("Step 7: Embed Data")
    if st.button("Embed Now"):
        if not openai_api_key:
            st.error("Please provide an OpenAI key.")
            st.stop()

        # Validate site pages
        if not df_pages:
            st.error("Please upload site pages first.")
            st.stop()
        required_cols = {"URL", "H1", "Meta Description"}
        if not required_cols.issubset(df_pages.columns):
            st.error(f"File missing columns: {', '.join(required_cols)}")
            st.stop()

        df_pages_emb = embed_site_pages_in_batches(df_pages.copy(), openai_api_key, batch_size)

        # If brand-new / existing, we do nothing more
        if mode_option in ["Brand-New Topic/Page", "Optimize Existing Page"]:
            if mode_option == "Brand-New Topic/Page":
                if not topic:
                    st.error("Please enter a topic.")
                    st.stop()
            elif mode_option == "Optimize Existing Page":
                if not topic:
                    st.error("Please enter a topic (for GPT).")
                    st.stop()
                if not selected_url:
                    st.error("Please select a target URL.")
                    st.stop()

            st.session_state["mode"] = mode_option
            st.session_state["topic"] = topic
            st.session_state["keyword"] = keyword
            st.session_state["df_pages"] = df_pages_emb
            st.session_state["selected_url"] = selected_url
            st.session_state["doc_paragraphs"] = None
            st.success("Site pages embedded. Proceed to Generate Links.")

        else:
            # analyze_word_doc
            if not doc_paragraphs:
                st.error("No doc paragraphs found.")
                st.stop()

            # 1) embed doc paragraphs
            df_doc_emb = embed_doc_paragraphs_in_batches(doc_paragraphs, openai_api_key, batch_size)
            st.session_state["df_pages"] = df_pages_emb
            st.session_state["df_doc_paragraphs"] = df_doc_emb
            st.session_state["mode"] = "analyze_word_doc"
            st.success("Site pages + doc paragraphs embedded. Proceed to Generate Links.")

    # Step 8: Generate Links
    st.subheader("Step 8: Generate Links")
    if st.button("Generate Links"):
        if "mode" not in st.session_state:
            st.error("Please embed data first.")
            st.stop()

        final_mode = st.session_state["mode"]
        if final_mode in ["Brand-New Topic/Page", "Optimize Existing Page"]:
            # GPT-based approach
            final_topic = st.session_state["topic"]
            final_keyword = st.session_state["keyword"]
            df_pages_embedded = st.session_state["df_pages"]
            final_url = st.session_state.get("selected_url", None)

            if final_mode == "Optimize Existing Page" and final_url:
                # gather top pages for GPT
                # exclude the target page
                target_row = df_pages_embedded.loc[df_pages_embedded["URL"] == final_url].iloc[0].to_dict()
                candidate_df = df_pages_embedded.loc[df_pages_embedded["URL"] != final_url]

                # do optional semantic search based on topic
                if final_topic:
                    # embed query
                    combined_query = final_topic + " " + final_keyword if final_keyword else final_topic
                    q_emb = embed_text_batch(openai_api_key, [combined_query])
                    if q_emb and q_emb[0]:
                        sims = []
                        for i, row in candidate_df.iterrows():
                            sim_val = compute_cosine_similarity(np.array(q_emb[0]), np.array(row["embedding"]))
                            sims.append(sim_val)
                        candidate_df["similarity"] = sims
                        candidate_df = candidate_df.sort_values("similarity", ascending=False).head(50)
                        threshold = 0.50
                        candidate_df = candidate_df[candidate_df["similarity"] >= threshold].copy()

                final_links = generate_internal_links(
                    openai_api_key,
                    "existing_page",
                    final_topic,
                    final_keyword,
                    target_row,
                    candidate_df[["URL", "H1", "Meta Description"]]
                )

                if not final_links:
                    st.warning("No recommendations returned or invalid JSON.")
                else:
                    df_out = pd.DataFrame(final_links)
                    st.dataframe(df_out)
                    st.download_button(
                        "Download CSV",
                        df_out.to_csv(index=False).encode("utf-8"),
                        file_name=f"existing_page_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )

            else:
                # brand-new
                final_topic = st.session_state["topic"]
                final_keyword = st.session_state["keyword"]
                df_pages_embedded = st.session_state["df_pages"]

                if final_topic:
                    combined_query = final_topic + " " + final_keyword if final_keyword else final_topic
                    q_emb = embed_text_batch(openai_api_key, [combined_query])
                    if q_emb and q_emb[0]:
                        sims = []
                        for i, row in df_pages_embedded.iterrows():
                            sim_val = compute_cosine_similarity(np.array(q_emb[0]), np.array(row["embedding"]))
                            sims.append(sim_val)
                        df_pages_embedded["similarity"] = sims
                        df_pages_embedded = df_pages_embedded.sort_values("similarity", ascending=False).head(50)
                        threshold = 0.50
                        df_pages_embedded = df_pages_embedded[df_pages_embedded["similarity"] >= threshold].copy()

                target_data = {
                    "URL": "N/A",
                    "H1": f"(New) {final_topic}",
                    "Meta Description": f"Future page about {final_topic}"
                }
                final_links = generate_internal_links(
                    openai_api_key,
                    "brand_new",
                    final_topic,
                    final_keyword,
                    target_data,
                    df_pages_embedded[["URL", "H1", "Meta Description"]]
                )
                if not final_links:
                    st.warning("No recommendations returned or invalid JSON.")
                else:
                    df_out = pd.DataFrame(final_links)
                    st.dataframe(df_out)
                    st.download_button(
                        "Download CSV",
                        df_out.to_csv(index=False).encode("utf-8"),
                        file_name=f"brand_new_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )

        else:
            # analyze_word_doc -> purely embeddings
            df_pages_emb = st.session_state["df_pages"]
            df_doc_emb = st.session_state["df_doc_paragraphs"]

            # We'll pick for each site page the single doc paragraph with highest similarity
            # then from that paragraph, we pick the best sentence
            results = []
            threshold = 0.50

            for i, page_row in df_pages_emb.iterrows():
                page_vec = np.array(page_row["embedding"])
                best_para_idx = -1
                best_para_sim = -1.0

                # 1) find best paragraph
                for idx_para, doc_row in df_doc_emb.iterrows():
                    para_vec = np.array(doc_row["embedding"])
                    sim_val = compute_cosine_similarity(page_vec, para_vec)
                    if sim_val > best_para_sim:
                        best_para_sim = sim_val
                        best_para_idx = idx_para

                if best_para_sim < threshold:
                    # skip if below threshold
                    continue

                # 2) best paragraph
                best_para = df_doc_emb.loc[best_para_idx]
                paragraph_text = best_para["content"]

                # 3) split paragraph into sentences, embed them, pick best
                #    This ensures we pick a substring from the paragraph text
                sents = re.split(r'(?<=[.!?])\s+', paragraph_text.strip())
                sents = [s.strip() for s in sents if s.strip()]

                if not sents:
                    continue
                sent_embeddings = embed_text_batch(openai_api_key, sents)
                best_sent_idx = -1
                best_sent_sim = -1.0
                for s_idx, s_emb in enumerate(sent_embeddings):
                    sim_val = compute_cosine_similarity(page_vec, np.array(s_emb))
                    if sim_val > best_sent_sim:
                        best_sent_sim = sim_val
                        best_sent_idx = s_idx

                if best_sent_sim < threshold:
                    # no good sentence match
                    continue
                anchor_substring = sents[best_sent_idx]

                results.append({
                    "page_url": page_row["URL"],
                    "page_title": page_row["H1"],
                    "paragraph_index": best_para_idx,
                    "paragraph_text": paragraph_text,
                    "chosen_sentence": anchor_substring,
                    "similarity_paragraph": best_para_sim,
                    "similarity_sentence": best_sent_sim
                })

            if not results:
                st.warning("No link suggestions found (all below threshold).")
            else:
                final_df = pd.DataFrame(results)
                st.write("**Recommended Anchor Substrings (One per Site Page)**")
                st.dataframe(final_df)

                csv_data = final_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name=f"doc_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
