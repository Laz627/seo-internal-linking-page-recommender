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
    Parse .docx, skipping headings. Return a list of dicts:
    [
      { "content": "...paragraph text...", "embedding": None, "sentences": None },
      ...
    ]
    We'll add 'embedding' and 'sentences' after we embed them.
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
            paragraphs.append({"content": text})
    return paragraphs


###############################################################################
# 3. Embedding Calls
###############################################################################
def embed_text_batch(openai_api_key: str, texts: list[str], model="text-embedding-ada-002"):
    """
    Batch call to OpenAI's Embeddings API for a list of strings.
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
    For each row in df, embed (URL + H1 + Meta Description) in batches => store in df["embedding"].
    """
    st.info("Embedding site pages in batches...")
    rows = len(df)
    texts = [
        f"{row['URL']} {row['H1']} {row['Meta Description']}"
        for _, row in df.iterrows()
    ]

    all_embeddings = []
    total_batches = (rows + batch_size - 1) // batch_size
    progress_bar = st.progress(0)
    label = st.empty()

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        progress_val = int((batch_idx + 1) / total_batches * 100)
        progress_bar.progress(progress_val)
        label.write(f"Site pages batch {batch_idx+1}/{total_batches}")

    df["embedding"] = all_embeddings
    return df


def embed_doc_paragraphs(paragraphs: list[dict], openai_api_key: str, batch_size: int = 10):
    """
    Each paragraph is embedded as a single chunk, plus we embed each paragraph's sentences.
    Return a list of dicts: 
      [
        {
          "content": str, 
          "embedding": <vec>, 
          "sentences": [ (sentence_text, sentence_embedding), ... ]
        },
        ...
      ]
    """
    st.info("Embedding doc paragraphs in batches...")

    texts = [p["content"] for p in paragraphs]
    all_embeddings = []
    total = len(paragraphs)
    total_batches = (total + batch_size - 1) // batch_size
    bar = st.progress(0)
    label = st.empty()

    for b_idx in range(total_batches):
        start = b_idx * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        progress_val = int((b_idx + 1) / total_batches * 100)
        bar.progress(progress_val)
        label.write(f"Paragraph batch {b_idx+1}/{total_batches}")

    for i, emb in enumerate(all_embeddings):
        paragraphs[i]["embedding"] = emb

    st.info("Embedding paragraph sentences...")
    total_paras = len(paragraphs)
    bar_sents = st.progress(0)
    for i, p in enumerate(paragraphs):
        bar_sents.progress(int((i+1)/total_paras * 100))

        # naive sentence split
        sents = re.split(r'(?<=[.!?])\s+', p["content"].strip())
        sents = [s.strip() for s in sents if s.strip()]

        if not sents:
            p["sentences"] = []
            continue

        s_embs = embed_text_batch(openai_api_key, sents)
        p["sentences"] = list(zip(sents, s_embs))

    return paragraphs


###############################################################################
# 4. GPT-based Logic for Brand-New / Existing
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
    Then merges the results with 'similarity' if present in candidate_pages.
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
        # We'll store results in a DataFrame for easy merging with similarity
        results_df = pd.DataFrame(data)

        # If candidate_pages had a 'similarity' column, we can merge by URL
        if "similarity" in candidate_pages.columns:
            merged = results_df.merge(
                candidate_pages[["URL", "similarity"]],
                left_on="target_url",
                right_on="URL",
                how="left"
            )
            merged.drop(columns=["URL"], inplace=True, errors="ignore")
            return merged.to_dict("records")
        else:
            return results_df.to_dict("records")

    except Exception as e:
        st.error(f"Error calling GPT: {e}")
        return []


###############################################################################
# 5. Streamlit App
###############################################################################
def main():
    st.title("Internal Link Recommender (3 Modes)")

    st.write("""
    **Modes**:
    1) **Brand-New Topic/Page** (GPT-based)  
    2) **Optimize Existing Page** (GPT-based)  
    3) **Analyze Word Document** (embeddings only, no GPT):
       - Each paragraph picks exactly **1** site page (highest similarity ≥ 80%)
       - UI displays the paragraph text, plus the single recommended link and similarity score
    """)

    # Step 1: Download sample site template
    st.subheader("Step 1: (Optional) Download Sample Site Template")
    sample_xlsx = generate_sample_excel_template()
    link = get_download_link(sample_xlsx, "sample_template.xlsx", "Download Sample Excel Template")
    st.markdown(link, unsafe_allow_html=True)

    # Step 2: Upload site pages
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

    # Step 3: Mode selection
    st.subheader("Step 3: Choose Mode")
    mode_option = st.radio(
        "Mode:",
        ["Brand-New Topic/Page", "Optimize Existing Page", "Analyze Word Document"]
    )

    # If existing page, user picks the URL
    selected_url = None
    if mode_option == "Optimize Existing Page" and df_pages is not None:
        req_cols = {"URL", "H1", "Meta Description"}
        if req_cols.issubset(df_pages.columns):
            all_urls = df_pages["URL"].unique().tolist()
            selected_url = st.selectbox("Select Target Page URL", all_urls)
        else:
            st.warning("Please upload valid site pages with URL, H1, Meta Description first.")

    # If Word doc mode, parse paragraphs
    doc_paragraphs = []
    if mode_option == "Analyze Word Document":
        st.subheader("3A: Upload Word Document (paragraphs only)")
        doc_file = st.file_uploader("Upload .docx", type=["docx"])
        if doc_file:
            try:
                docx_data = doc_file.read()
                doc_paragraphs = parse_word_doc_paragraphs_only(docx_data)
                st.success(f"Found {len(doc_paragraphs)} paragraphs.")
            except Exception as e:
                st.error(f"Error parsing Word doc: {e}")

    # Step 4: Topic + Keyword (only relevant for GPT-based modes)
    if mode_option in ["Brand-New Topic/Page", "Optimize Existing Page"]:
        st.subheader("Step 4: Enter Topic & Optional Keyword")
        topic = st.text_input("Topic (required for GPT-based modes)")
        keyword = st.text_input("Keyword (optional)")
    else:
        topic = ""
        keyword = ""

    # Step 5: OpenAI Key
    st.subheader("Step 5: Enter OpenAI API Key")
    openai_api_key = st.text_input("OpenAI Key", type="password")

    # Step 6: Batch size
    st.subheader("Step 6: Set Batch Size for Embedding")
    batch_size = st.slider("Batch size for embeddings", min_value=1, max_value=50, value=10)

    # EMBED
    st.subheader("Step 7: Embed Data")
    if st.button("Embed Now"):
        if not openai_api_key:
            st.error("Please provide an OpenAI API key.")
            st.stop()

        # Validate site pages
        if df_pages is None or df_pages.empty:
            st.error("Please upload site pages or ensure your CSV/Excel isn't empty.")
            st.stop()

        required_cols = {"URL", "H1", "Meta Description"}
        if not required_cols.issubset(df_pages.columns):
            st.error(f"Site file missing columns: {', '.join(required_cols)}")
            st.stop()

        # 1) embed site pages
        df_embedded = embed_site_pages(df_pages.copy(), openai_api_key, batch_size)

        doc_data = None
        if mode_option == "Analyze Word Document":
            if not doc_paragraphs:
                st.error("Please upload a doc with paragraphs.")
                st.stop()
            doc_data = embed_doc_paragraphs(doc_paragraphs, openai_api_key, batch_size)

        st.session_state["df_pages"] = df_embedded
        st.session_state["doc_data"] = doc_data
        st.session_state["mode"] = mode_option
        st.session_state["topic"] = topic
        st.session_state["keyword"] = keyword
        st.session_state["selected_url"] = selected_url

        st.success("Embedding complete. Proceed to Generate Links.")

    # GENERATE
    st.subheader("Step 8: Generate Links")
    if st.button("Generate Links"):
        if "df_pages" not in st.session_state:
            st.error("Please embed data first.")
            st.stop()

        df_pages_emb = st.session_state["df_pages"]
        doc_data = st.session_state["doc_data"]
        final_mode = st.session_state["mode"]
        final_topic = st.session_state["topic"]
        final_keyword = st.session_state["keyword"]
        final_url = st.session_state.get("selected_url", None)

        # 80% threshold
        threshold = 0.80

        if final_mode in ["Brand-New Topic/Page", "Optimize Existing Page"]:
            # GPT-based approach
            if final_mode == "Optimize Existing Page":
                try:
                    target_row = df_pages_emb.loc[df_pages_emb["URL"] == final_url].iloc[0].to_dict()
                except IndexError:
                    st.error("Selected URL not found.")
                    st.stop()

                # Exclude target from candidates
                candidate_df = df_pages_emb.loc[df_pages_emb["URL"] != final_url].copy()

                # Optional semantic filter
                if final_topic:
                    query_str = final_topic + " " + final_keyword if final_keyword else final_topic
                    q_emb = embed_text_batch(openai_api_key, [query_str])
                    if q_emb and q_emb[0]:
                        sims = []
                        for i, row in candidate_df.iterrows():
                            sim_val = compute_cosine_similarity(np.array(q_emb[0]), np.array(row["embedding"]))
                            sims.append(sim_val)
                        candidate_df["similarity"] = sims
                        candidate_df = candidate_df.sort_values("similarity", ascending=False)
                        # We only keep pages above 80% if we want to be consistent
                        candidate_df = candidate_df[candidate_df["similarity"] >= threshold]

                final_links = generate_internal_links(
                    openai_api_key,
                    "existing_page",
                    final_topic,
                    final_keyword,
                    target_row,
                    candidate_df[["URL", "H1", "Meta Description", "similarity"]]
                )

                if not final_links:
                    st.warning("No recommendations or GPT returned invalid JSON.")
                else:
                    # Display in a more user-friendly UI
                    st.subheader("Optimize Existing Page Results")
                    for link_item in final_links:
                        st.markdown(f"- **Page Link**: [{link_item.get('anchor_text','(No Title)')}]({link_item.get('target_url','')})")
                        # We interpret 'anchor_text' as the "Proposed Anchor Text" from GPT
                        # If we have 'similarity', show it
                        similarity_val = link_item.get("similarity", None)
                        if similarity_val is not None:
                            st.markdown(f"  - **Similarity Score**: {round(similarity_val*100,2)}%")

            else:
                # Brand-New
                if final_topic:
                    query_str = final_topic + " " + final_keyword if final_keyword else final_topic
                    q_emb = embed_text_batch(openai_api_key, [query_str])
                    if q_emb and q_emb[0]:
                        sims = []
                        for i, row in df_pages_emb.iterrows():
                            sim_val = compute_cosine_similarity(np.array(q_emb[0]), np.array(row["embedding"]))
                            sims.append(sim_val)
                        df_pages_emb["similarity"] = sims
                        df_pages_emb = df_pages_emb.sort_values("similarity", ascending=False)
                        df_pages_emb = df_pages_emb[df_pages_emb["similarity"] >= threshold]

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
                    df_pages_emb[["URL","H1","Meta Description","similarity"]]
                )
                if not final_links:
                    st.warning("No recommendations or GPT returned invalid JSON.")
                else:
                    st.subheader("Brand-New Topic Results")
                    for link_item in final_links:
                        st.markdown(f"- **Page Link**: [{link_item.get('anchor_text','(No Title)')}]({link_item.get('target_url','')})")
                        sim_val = link_item.get("similarity", None)
                        if sim_val is not None:
                            st.markdown(f"  - **Similarity Score**: {round(sim_val*100,2)}%")

        else:
            # Word Doc Mode => each paragraph picks exactly 1 link above 80% similarity
            if not doc_data:
                st.error("No doc paragraphs found. Please embed the doc first.")
                st.stop()

            # We do top_k=1
            paragraphs_count = len(doc_data)
            st.write(f"**Generating 1 link per paragraph (threshold {threshold}, top_k=1).**")

            final_output = []
            progress_bar = st.progress(0)
            label = st.empty()

            for p_idx, p_dict in enumerate(doc_data):
                progress_val = int(((p_idx+1)/paragraphs_count)*100)
                progress_bar.progress(progress_val)
                label.write(f"Processing Paragraph {p_idx+1}/{paragraphs_count}")

                para_vec = np.array(p_dict["embedding"])
                paragraph_text = p_dict["content"]

                # find the single best site page
                best_sim = -1.0
                best_page_row = None
                for i, page_row in df_pages_emb.iterrows():
                    page_vec = np.array(page_row["embedding"])
                    sim_val = compute_cosine_similarity(para_vec, page_vec)
                    if sim_val > best_sim:
                        best_sim = sim_val
                        best_page_row = page_row

                if best_sim < threshold:
                    # skip
                    continue

                # We have a best page above 80%
                # We'll display in a more digestible format: highlight the paragraph, show the page link, similarity
                final_output.append({
                    "paragraph_index": p_idx,
                    "paragraph_text": paragraph_text,
                    "page_url": best_page_row["URL"],
                    "page_title": best_page_row["H1"],
                    "similarity_score": best_sim
                })

            # Show the results in expanders
            if not final_output:
                st.warning("No paragraphs found a site page above 80% similarity.")
            else:
                st.subheader("Word Doc Linking Results (One Link per Paragraph)")
                for item in final_output:
                    p_idx = item["paragraph_index"]
                    p_text = item["paragraph_text"]
                    p_url = item["page_url"]
                    p_title = item["page_title"]
                    p_sim = item["similarity_score"]
                    # Output using an expander
                    with st.expander(f"Paragraph #{p_idx+1}"):
                        st.markdown(f"> **Paragraph Text:**  \n> {p_text}\n")
                        st.markdown(f"- **Proposed Link**: [{p_title}]({p_url})")
                        st.markdown(f"- **Similarity Score**: {round(p_sim*100,2)}%")

                # Optionally produce a CSV
                df_final = pd.DataFrame(final_output)
                st.subheader("Download CSV of Word Doc Results")
                csv_data = df_final.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name=f"word_doc_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )

if __name__ == "__main__":
    main()
