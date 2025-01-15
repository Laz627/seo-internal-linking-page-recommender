import streamlit as st
import pandas as pd
import openai
import json
import io
import base64
import numpy as np
import re
from docx import Document
from datetime import datetime
from openpyxl import Workbook

###############################################################################
# 1. Utility Functions
###############################################################################
def generate_sample_excel_template():
    """
    Create an Excel file in memory with columns: URL, H1, Meta Description.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Pages"

    ws["A1"] = "URL"
    ws["B1"] = "H1"
    ws["C1"] = "Meta Description"

    # Example row
    ws["A2"] = "https://example.com/sample-page"
    ws["B2"] = "Sample Page Title"
    ws["C2"] = "Sample meta description content."

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def get_download_link(file_bytes: bytes, filename: str, link_label: str) -> str:
    """
    Generate a download link for an in-memory file (BytesIO).
    """
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_label}</a>'

def compute_cosine_similarity(vec_a, vec_b) -> float:
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

###############################################################################
# 2. Parsing Word Doc (Paragraphs Only)
###############################################################################
def parse_word_doc_paragraphs_only(doc_bytes: bytes):
    """
    Parse .docx, skip headings. Return a list of dicts:
    [ { "content": "...paragraph text...", "embedding": None, "sentences": None }, ... ]
    We'll add "embedding" and "sentences" fields later.
    """
    doc = Document(io.BytesIO(doc_bytes))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name.lower() if para.style else ""
        # skip if recognized as heading
        if "heading" not in style_name:
            paragraphs.append({
                "content": text
            })
    return paragraphs

###############################################################################
# 3. Embedding Calls
###############################################################################
def embed_text_batch(openai_api_key: str, texts: list[str], model="text-embedding-ada-002"):
    """
    Batch call to OpenAI Embeddings API for a list of strings.
    """
    openai.api_key = openai_api_key
    try:
        response = openai.Embedding.create(model=model, input=texts)
        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return [[] for _ in texts]

def embed_site_pages(df: pd.DataFrame, openai_api_key: str, batch_size: int = 10):
    """
    For each row, combine URL+H1+Meta Desc, embed them in BATCHES,
    store in df["embedding"].
    """
    st.info("Embedding site pages in batches...")
    rows = len(df)
    texts = []
    for _, row in df.iterrows():
        combined_text = f"{row['URL']} {row['H1']} {row['Meta Description']}"
        texts.append(combined_text)

    all_embeddings = []
    total_batches = (rows + batch_size - 1) // batch_size
    progress_bar = st.progress(0)
    label = st.empty()

    idx = 0
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        idx += 1
        progress_val = int((idx / total_batches) * 100)
        progress_bar.progress(progress_val)
        label.write(f"Site pages batch {idx}/{total_batches}")

    df["embedding"] = all_embeddings
    return df

def embed_doc_paragraphs(paragraphs: list[dict], openai_api_key: str, batch_size: int = 10):
    """
    Each paragraph is embedded as a single chunk. We'll also embed each paragraph's sentences, once, up front.
    Return a list of dicts: [ { "content": str, "embedding": <vec>, "sentences": [ (sentence, embedding), ... ] }, ... ]
    """
    st.info("Embedding doc paragraphs in batches...")

    # Step 1: embed each paragraph
    texts = [p["content"] for p in paragraphs]
    all_embeddings = []
    total = len(paragraphs)
    total_batches = (total + batch_size - 1) // batch_size
    bar = st.progress(0)
    label = st.empty()

    offset = 0
    for b_idx in range(total_batches):
        start = b_idx * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        progress_val = int(((b_idx+1)/total_batches)*100)
        bar.progress(progress_val)
        label.write(f"Paragraphs batch {b_idx+1}/{total_batches}")

    # Step 2: store paragraph embeddings
    for i, emb in enumerate(all_embeddings):
        paragraphs[i]["embedding"] = emb

    # Step 3: embed each paragraph's sentences
    # We'll do a single API call per paragraph's sentences
    st.info("Embedding sentences inside each paragraph...this should be quick overall.")
    total_paras = len(paragraphs)
    bar_sents = st.progress(0)
    for i, p in enumerate(paragraphs):
        bar_sents.progress(int(((i+1)/total_paras)*100))
        # naive sentence split
        sents = re.split(r'(?<=[.!?])\s+', p["content"].strip())
        sents = [s.strip() for s in sents if s.strip()]

        if len(sents) == 0:
            p["sentences"] = []
            continue

        emb_sents = embed_text_batch(openai_api_key, sents)
        p["sentences"] = list(zip(sents, emb_sents))

    return paragraphs

###############################################################################
# 4. Streamlit App
###############################################################################
def main():
    st.title("Internal Link Recommender (Optimized Word Doc Mode)")

    st.write("""
    This app demonstrates a 3-mode approach:
    1. **Brand-New** or **Existing** page (GPT-based).
    2. **Analyze Word Document** (paragraphs only) with **no GPT calls** at generation time.
       We do all embeddings up front, so link generation should only take **30-60 seconds** 
       even for thousands of pages.

    **Why It's Fast**: 
    - Each paragraph is embedded *once*. 
    - Each paragraph's sentences are also embedded *once*. 
    - For each page, we do an in-memory match to find the best paragraph & best sentence. 
      No extra API calls per page.
    """)

    # Sample site template
    st.subheader("Step 1: (Optional) Download Sample Site Template")
    sample_xlsx = generate_sample_excel_template()
    link = get_download_link(sample_xlsx, "sample_template.xlsx", "Download Sample Excel Template")
    st.markdown(link, unsafe_allow_html=True)

    # Upload site pages
    st.subheader("Step 2: Upload Site Pages CSV/Excel")
    pages_file = st.file_uploader("Must have columns: URL, H1, Meta Description", type=["csv", "xlsx"])
    df_pages = None
    if pages_file:
        try:
            if pages_file.name.endswith(".csv"):
                df_pages = pd.read_csv(pages_file)
            else:
                df_pages = pd.read_excel(pages_file)
        except Exception as e:
            st.error(f"Error reading site pages: {e}")

    # Mode select
    st.subheader("Step 3: Choose Mode")
    mode_option = st.radio(
        "Mode:",
        ["Brand-New Topic/Page", "Optimize Existing Page", "Analyze Word Document"]
    )

    # If existing, let user pick the target page
    selected_url = None
    if mode_option == "Optimize Existing Page" and df_pages is not None:
        if {"URL", "H1", "Meta Description"}.issubset(df_pages.columns):
            all_urls = df_pages["URL"].unique().tolist()
            selected_url = st.selectbox("Select Target Page", all_urls)
        else:
            st.warning("Please upload valid site pages first.")

    # If analyzing Word doc, parse paragraphs only
    doc_paragraphs = []
    if mode_option == "Analyze Word Document":
        st.subheader("3A: Upload Word Document (paragraphs only, skipping headings)")
        doc_file = st.file_uploader("Upload .docx", type=["docx"])
        if doc_file:
            try:
                docx_data = doc_file.read()
                doc_paragraphs = parse_word_doc_paragraphs_only(docx_data)
                st.success(f"Found {len(doc_paragraphs)} paragraphs (excluding headings).")
            except Exception as e:
                st.error(f"Error parsing Word doc: {e}")

    # For brand-new/existing, user can supply topic/keyword
    if mode_option in ["Brand-New Topic/Page", "Optimize Existing Page"]:
        st.subheader("Step 4: Enter Topic & Optional Keyword")
        topic = st.text_input("Topic (required for GPT-based modes)")
        keyword = st.text_input("Keyword (optional)")
    else:
        topic = ""
        keyword = ""

    st.subheader("Step 5: Enter OpenAI API Key")
    openai_api_key = st.text_input("OpenAI Key", type="password")

    st.subheader("Step 6: Set Batch Size")
    batch_size = st.slider("Batch size for embeddings", min_value=1, max_value=50, value=10)

    # EMBED button
    st.subheader("Step 7: Embed Data")
    if st.button("Embed Now"):
        if not openai_api_key:
            st.error("Please provide an OpenAI API key.")
            st.stop()

        if df_pages is None or df_pages.empty:
            st.error("Please upload a CSV/Excel with site pages first.")
            st.stop()

        # Basic column check
        required_cols = {"URL", "H1", "Meta Description"}
        if not required_cols.issubset(df_pages.columns):
            st.error(f"Missing columns: {', '.join(required_cols)}")
            st.stop()

        # 1) Embed site pages
        df_pages_emb = embed_site_pages(df_pages.copy(), openai_api_key, batch_size)

        # 2) If doc mode, embed doc paragraphs
        if mode_option == "Analyze Word Document":
            if not doc_paragraphs:
                st.error("No doc paragraphs found or doc not uploaded.")
                st.stop()
            doc_data = embed_doc_paragraphs(doc_paragraphs, openai_api_key, batch_size)
            st.session_state["doc_data"] = doc_data
        else:
            st.session_state["doc_data"] = None

        # store everything
        st.session_state["df_pages"] = df_pages_emb
        st.session_state["mode"] = mode_option
        st.session_state["topic"] = topic
        st.session_state["keyword"] = keyword
        st.session_state["selected_url"] = selected_url

        st.success("Embedding complete! Proceed to Generate Links.")

    # GENERATE LINKS
    st.subheader("Step 8: Generate Links")
    if st.button("Generate Links"):
        # retrieve from session
        if "df_pages" not in st.session_state:
            st.error("Please embed data first.")
            st.stop()

        df_pages_emb = st.session_state["df_pages"]
        mode = st.session_state["mode"]
        final_topic = st.session_state["topic"]
        final_keyword = st.session_state["keyword"]
        selected_url = st.session_state["selected_url"]
        doc_data = st.session_state["doc_data"]

        if mode in ["Brand-New Topic/Page", "Optimize Existing Page"]:
            # GPT-based
            if mode == "Optimize Existing Page":
                # gather candidate
                try:
                    target_row = df_pages_emb.loc[df_pages_emb["URL"] == selected_url].iloc[0].to_dict()
                except Exception as e:
                    st.error("Target page not found in dataset.")
                    st.stop()
                candidate_df = df_pages_emb.loc[df_pages_emb["URL"] != selected_url]
                # optional semantic filter
                if final_topic:
                    combined_query = final_topic + " " + final_keyword if final_keyword else final_topic
                    q_emb = embed_text_batch(openai_api_key, [combined_query])
                    sims = []
                    for i, row in candidate_df.iterrows():
                        sim_val = compute_cosine_similarity(np.array(q_emb[0]), np.array(row["embedding"]))
                        sims.append(sim_val)
                    candidate_df["similarity"] = sims
                    candidate_df = candidate_df.sort_values("similarity", ascending=False).head(50)
                    threshold = 0.50
                    candidate_df = candidate_df[candidate_df["similarity"] >= threshold]

                final_links = generate_internal_links(
                    openai_api_key,
                    "existing_page",
                    final_topic,
                    final_keyword,
                    target_row,
                    candidate_df[["URL", "H1", "Meta Description"]]
                )
                if not final_links:
                    st.warning("No recommendations or invalid JSON.")
                else:
                    df_out = pd.DataFrame(final_links)
                    st.dataframe(df_out)
                    st.download_button(
                        "Download CSV",
                        df_out.to_csv(index=False).encode("utf-8"),
                        file_name=f"existing_page_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
            else:
                # brand_new
                if final_topic:
                    combined_query = final_topic + " " + final_keyword if final_keyword else final_topic
                    q_emb = embed_text_batch(openai_api_key, [combined_query])
                    if q_emb[0]:
                        sims = []
                        for i, row in df_pages_emb.iterrows():
                            sim_val = compute_cosine_similarity(np.array(q_emb[0]), np.array(row["embedding"]))
                            sims.append(sim_val)
                        df_pages_emb["similarity"] = sims
                        df_pages_emb = df_pages_emb.sort_values("similarity", ascending=False).head(50)
                        df_pages_emb = df_pages_emb[df_pages_emb["similarity"] >= 0.50]

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
                    df_pages_emb[["URL","H1","Meta Description"]]
                )
                if not final_links:
                    st.warning("No recommendations or invalid JSON.")
                else:
                    df_out = pd.DataFrame(final_links)
                    st.dataframe(df_out)
                    st.download_button(
                        "Download CSV",
                        df_out.to_csv(index=False).encode("utf-8"),
                        file_name=f"brand_new_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )

        else:
            # analyze_word_doc -> no GPT calls, purely embeddings
            if doc_data is None:
                st.error("No doc paragraphs found. Please embed the doc first.")
                st.stop()

            # We'll produce a link for each site page. 
            # The progress bar ensures we see updates for each page => 30-60 seconds total for thousands of pages
            results = []
            threshold = 0.50

            pages_count = len(df_pages_emb)
            progress_bar = st.progress(0)
            label = st.empty()

            for idx_page, (_, page_row) in enumerate(df_pages_emb.iterrows()):
                progress_val = int(((idx_page+1)/pages_count)*100)
                progress_bar.progress(progress_val)
                label.write(f"Generating link for page {idx_page+1}/{pages_count}")

                page_vec = np.array(page_row["embedding"])
                best_para_idx = -1
                best_para_sim = -1.0

                # 1) find best paragraph
                for p_idx, p_dict in enumerate(doc_data):
                    para_emb = np.array(p_dict["embedding"])
                    sim_val = compute_cosine_similarity(page_vec, para_emb)
                    if sim_val > best_para_sim:
                        best_para_sim = sim_val
                        best_para_idx = p_idx

                if best_para_sim < threshold:
                    # skip
                    continue

                chosen_para = doc_data[best_para_idx]
                # pick best sentence
                best_sent_idx = -1
                best_sent_sim = -1.0
                for s_idx, (sent_text, sent_emb) in enumerate(chosen_para["sentences"]):
                    sim_val = compute_cosine_similarity(page_vec, np.array(sent_emb))
                    if sim_val > best_sent_sim:
                        best_sent_sim = sim_val
                        best_sent_idx = s_idx

                if best_sent_sim < threshold:
                    # skip if no sentence is above threshold
                    continue

                anchor_substring = chosen_para["sentences"][best_sent_idx][0]
                results.append({
                    "page_url": page_row["URL"],
                    "page_title": page_row["H1"],
                    "paragraph_index": best_para_idx,
                    "paragraph_text": chosen_para["content"],
                    "anchor_substring": anchor_substring,
                    "similarity_paragraph": best_para_sim,
                    "similarity_sentence": best_sent_sim
                })

            if not results:
                st.warning("No links found or all below threshold.")
            else:
                final_df = pd.DataFrame(results)
                st.write("**Paragraph-Level Link Recommendations (One per Page)**")
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
