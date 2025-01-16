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
    Creates an Excel file (in memory) with columns: URL, H1, Meta Description
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Pages"

    ws["A1"] = "URL"
    ws["B1"] = "H1"
    ws["C1"] = "Meta Description"

    # Example data row
    ws["A2"] = "https://example.com/sample"
    ws["B2"] = "Sample Title"
    ws["C2"] = "Sample Meta Description"

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def get_download_link(file_bytes: bytes, filename: str, link_label: str) -> str:
    """
    Returns an HTML link to download 'file_bytes' as 'filename'.
    """
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_label}</a>'


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray)->float:
    """
    Compute cosine similarity between two vectors.
    """
    if np.linalg.norm(vec_a)==0 or np.linalg.norm(vec_b)==0:
        return 0.0
    return float(np.dot(vec_a, vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b)))


###############################################################################
# 2. Word Doc Parsing
###############################################################################
def parse_word_doc_paragraphs_only(doc_bytes: bytes):
    """
    Parse .docx, skipping headings, returning a list of dicts:
      [ { "content": "...paragraph text...", "embedding": None }, ... ]
    """
    doc = Document(io.BytesIO(doc_bytes))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name.lower() if para.style else ""
        if "heading" not in style_name:
            paragraphs.append({"content": text})
    return paragraphs


###############################################################################
# 3. Embedding Site Pages & Doc Paragraphs
###############################################################################
def embed_text_batch(openai_api_key: str, texts: list[str], model="text-embedding-ada-002"):
    """
    Batch call to OpenAI's Embeddings API for a list of strings.
    """
    openai.api_key = openai_api_key
    try:
        response = openai.Embedding.create(model=model, input=texts)
        return [item["embedding"] for item in response["data"]]
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return [[] for _ in texts]


def embed_site_pages(df: pd.DataFrame, openai_api_key: str, batch_size: int=10)->pd.DataFrame:
    """
    For each site page (row), embed (URL + H1 + Meta Description).
    """
    st.info("Embedding site pages in batches...")

    texts = []
    for _, row in df.iterrows():
        combined = f"{row['URL']} {row['H1']} {row['Meta Description']}"
        texts.append(combined)

    n = len(texts)
    all_embeddings = []
    total_batches = (n + batch_size -1)//batch_size
    bar = st.progress(0)
    label = st.empty()
    idx=0

    for batch_idx in range(total_batches):
        start = batch_idx*batch_size
        end = start+batch_size
        batch_texts = texts[start:end]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        idx+=1
        bar.progress(int(idx/total_batches*100))
        label.write(f"Site pages batch {idx}/{total_batches}")

    df["embedding"] = all_embeddings
    return df


def embed_doc_paragraphs(paragraphs: list[dict], openai_api_key: str, batch_size: int=10)->list[dict]:
    """
    Embed each doc paragraph. Optionally embed sentences if you like.
    """
    st.info("Embedding doc paragraphs in batches...")
    texts = [p["content"] for p in paragraphs]

    n = len(texts)
    all_embeddings = []
    total_batches = (n+batch_size-1)//batch_size
    bar = st.progress(0)
    label = st.empty()
    idx=0

    for b_idx in range(total_batches):
        start = b_idx*batch_size
        end = start+batch_size
        batch_texts = texts[start:end]
        batch_embeds = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeds)

        idx+=1
        bar.progress(int(idx/total_batches*100))
        label.write(f"Paragraphs batch {idx}/{total_batches}")

    for i, emb in enumerate(all_embeddings):
        paragraphs[i]["embedding"] = emb

    # (Optional) embed sentences if you want finer detail. We'll skip for brevity.
    return paragraphs


###############################################################################
# 4. Streamlit App
###############################################################################
def main():
    st.title("Internal Linking Generator For Content Drafts")
    st.write("""
    **What this tool does**:
    1. Upload a list of your site pages (CSV/Excel) that include your site's URLs, H1s, and meta descriptions.
    2. Upload a Word document (.docx) of paragraphs (skipping headings).
    3. Each paragraph picks exactly **one** best-match site page to internally link to.
       (≥ 80% similarity), then **removes** that page from the pool
       so it can’t be used again.
    4. You can review the matches and download a CSV of results to add to your content draft

    **How to Use**:
    1. (Optional) Download the sample site pages template.
    2. Upload your site pages CSV/Excel.
    3. Upload your Word .docx paragraphs.
    4. Enter your OpenAI API key.
    5. Embed the data (both site pages and doc paragraphs).
    6. Generate links: each paragraph’s best page is shown in an expander.
    """)

    # Step 1: sample template
    st.subheader("Step 1: (Optional) Download Sample Site Template")
    sample_xlsx = generate_sample_excel_template()
    dl_link = get_download_link(sample_xlsx, "sample_template.xlsx", "Download Sample Template")
    st.markdown(dl_link, unsafe_allow_html=True)

    # Step 2: Upload site pages
    st.subheader("Step 2: Upload Site Pages CSV/Excel")
    up_file = st.file_uploader("Needs columns: URL, H1, Meta Description", type=["csv","xlsx"])
    df_pages = None
    if up_file:
        try:
            if up_file.name.endswith(".csv"):
                df_pages = pd.read_csv(up_file)
            else:
                df_pages = pd.read_excel(up_file)
        except Exception as e:
            st.error(f"Error reading site pages: {e}")

    # Step 3: Upload Word doc
    st.subheader("Step 3: Upload Word Document (.docx)")
    doc_file = st.file_uploader("Paragraphs only, skipping headings", type=["docx"])
    doc_paragraphs = []
    if doc_file:
        try:
            raw_bytes = doc_file.read()
            doc_paragraphs = parse_word_doc_paragraphs_only(raw_bytes)
            st.success(f"Found {len(doc_paragraphs)} paragraphs.")
        except Exception as e:
            st.error(f"Error parsing Word doc: {e}")

    # Step 4: OpenAI API Key
    st.subheader("Step 4: Enter OpenAI API Key")
    openai_key = st.text_input("OpenAI API Key", type="password")

    # Step 5: Embedding batch size
    st.subheader("Step 5: Set Batch Size for Embedding")
    batch_sz = st.slider("Batch size", 1, 50, 10)

    # EMBED
    st.subheader("Step 6: Embed Now")
    if st.button("Embed Data"):
        if df_pages is None or df_pages.empty:
            st.error("Please upload site pages first or ensure your CSV/Excel isn't empty.")
            st.stop()
        required_cols = {"URL","H1","Meta Description"}
        if not required_cols.issubset(df_pages.columns):
            st.error(f"Missing columns: {', '.join(required_cols)}")
            st.stop()

        if not doc_paragraphs:
            st.error("Please upload a Word doc with paragraphs first.")
            st.stop()

        if not openai_key:
            st.error("Please enter an OpenAI API key.")
            st.stop()

        # 1) embed site pages
        df_pages_emb = embed_site_pages(df_pages.copy(), openai_key, batch_sz)
        # 2) embed doc paragraphs
        doc_data = embed_doc_paragraphs(doc_paragraphs, openai_key, batch_sz)

        st.session_state["df_pages"] = df_pages_emb
        st.session_state["doc_data"] = doc_data
        st.success("Embedding complete. Proceed to Generate Links.")

    # GENERATE
    st.subheader("Step 7: Generate Links")
    if st.button("Generate Links"):
        if "df_pages" not in st.session_state or "doc_data" not in st.session_state:
            st.error("Please embed data first.")
            st.stop()

        df_pages_emb = st.session_state["df_pages"]
        doc_data = st.session_state["doc_data"]

        # The "inverse" approach: paragraphs in order, each picks a unique page if >= 80% similarity
        threshold = 0.80
        doc_df = pd.DataFrame(doc_data)  # easier to iterate
        pages_copy = df_pages_emb.copy()
        results = []

        paragraphs_count = len(doc_df)
        bar = st.progress(0)
        label = st.empty()

        st.write("**Each paragraph picks exactly one unique page ≥80% similarity.**")

        for p_idx, p_row in doc_df.iterrows():
            progress_val = int(((p_idx+1)/paragraphs_count)*100)
            bar.progress(progress_val)
            label.write(f"Paragraph {p_idx+1}/{paragraphs_count}")

            para_vec = np.array(p_row["embedding"])
            best_sim = -1.0
            best_idx = -1

            for i, page_row in pages_copy.iterrows():
                page_vec = np.array(page_row["embedding"])
                sim_val = compute_cosine_similarity(para_vec, page_vec)
                if sim_val > best_sim:
                    best_sim = sim_val
                    best_idx = i

            if best_idx != -1 and best_sim >= threshold:
                chosen_page = pages_copy.loc[best_idx]
                results.append({
                    "paragraph_index": p_idx,
                    "paragraph_text": p_row["content"],
                    "page_url": chosen_page["URL"],
                    "page_title": chosen_page["H1"],
                    "similarity_score": best_sim
                })
                # Remove that page so it won't be reused
                pages_copy.drop(index=best_idx, inplace=True)

            if pages_copy.empty:
                break  # no more pages left, done

        if not results:
            st.warning("No paragraphs matched any page above 80%. Or no pages left.")
        else:
            st.subheader("Paragraph-to-Page Results")
            # Show in expanders
            for item in results:
                with st.expander(f"Paragraph #{item['paragraph_index']+1}"):
                    st.markdown(f"> **Paragraph Text**:\n> {item['paragraph_text']}")
                    st.markdown(f"- **Page Link**: [{item['page_title']}]({item['page_url']})")
                    st.markdown(f"- **Similarity Score**: {round(item['similarity_score']*100,2)}%")

            # Download CSV
            df_final = pd.DataFrame(results)
            st.subheader("Download CSV of Matches")
            csv_data = df_final.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv_data,
                file_name=f"doc_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


if __name__=="__main__":
    main()
