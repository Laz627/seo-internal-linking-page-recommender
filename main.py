import streamlit as st
import pandas as pd
import openai
import json
import io
import base64
import numpy as np
from datetime import datetime
from openpyxl import Workbook
from docx import Document  # for parsing .docx

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

    # (Optional) Populate sample rows for demonstration
    ws["A2"] = "https://example.com/sample-page"
    ws["B2"] = "Sample Page Title"
    ws["C2"] = "This is a sample meta description for demonstration."

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


def parse_openai_json(response_text: str):
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
# 2. Word Doc Parsing
###############################################################################
def parse_word_doc(file_obj: io.BytesIO):
    """
    Extracts each paragraph or heading from the .docx file, returns a list of dicts:
        [
          { "type": "heading"|"paragraph", "content": "...some text..." },
          ...
        ]
    """
    doc = Document(file_obj)
    blocks = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name.lower() if para.style else ""
        if "heading" in style_name:
            blocks.append({"type": "heading", "content": text})
        else:
            blocks.append({"type": "paragraph", "content": text})
    return blocks

###############################################################################
# 3. Embeddings + Semantic Search
###############################################################################
def embed_text_batch(openai_api_key: str, texts: list[str], model="text-embedding-ada-002"):
    """
    Embeds a batch of texts in a single API call using OpenAI's Embeddings API.
    Returns a list of embedding vectors, one per input text.
    """
    openai.api_key = openai_api_key
    try:
        response = openai.Embedding.create(model=model, input=texts)
        # Each item in response["data"] corresponds to an embedding for each input
        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings
    except Exception as e:
        st.error(f"Error generating batch embeddings: {e}")
        return [[] for _ in texts]


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def batch_embed_rows(df: pd.DataFrame, openai_api_key: str, text_col: str, batch_size: int = 10):
    """
    Embeds each row in df by taking the text from `text_col`. 
    Stores the result in a new column "embedding".
    Uses a progress bar to indicate batch progress.
    """
    st.info(f"Embedding {len(df)} rows. Please wait...")

    # Prepare the texts
    texts = df[text_col].tolist()

    all_embeddings = []
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    progress_bar = st.progress(0)
    progress_label = st.empty()

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]

        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        # Update progress
        percentage = int((batch_idx + 1) / total_batches * 100)
        progress_bar.progress(percentage)
        progress_label.write(f"Batch {batch_idx+1}/{total_batches} done.")

    df["embedding"] = all_embeddings
    return df


def semantic_search(query_embedding: np.ndarray, df: pd.DataFrame, top_k: int = 5):
    """
    Compare the query_embedding to each row's 'embedding', compute similarity,
    sort by descending similarity, return top_k results as a DataFrame.
    """
    similarities = []
    for idx, row in df.iterrows():
        page_vec = row["embedding"]
        sim = compute_cosine_similarity(query_embedding, np.array(page_vec))
        similarities.append(sim)
    df["similarity"] = similarities
    df_sorted = df.sort_values(by="similarity", ascending=False).head(top_k)
    return df_sorted


###############################################################################
# 4. GPT Prompt + Internal Link Generation
###############################################################################
def build_prompt_for_paragraph(
    doc_snippet: dict, 
    candidate_pages: pd.DataFrame, 
    topic: str, 
    keyword: str
):
    """
    Build a GPT prompt that references a single doc snippet (paragraph/heading)
    and the top candidate site pages. We want GPT to produce anchor text from
    the snippet to these pages.
    """

    anchor_text_guidelines = """
Anchor Text Best Practices:
- Be descriptive, concise, and relevant.
- Avoid overly generic text ("click here", "read more").
- Avoid keyword stuffing or extremely long anchor text.
- Anchor text should make sense on its own and reflect the linked page's content.
"""

    # doc_snippet has { "type": "paragraph"/"heading", "content": "...", ... }
    doc_content = doc_snippet["content"]
    doc_type = doc_snippet["type"]

    pages_info = []
    for _, row in candidate_pages.iterrows():
        pages_info.append({
            "URL": row["URL"],
            "H1": row["H1"],
            "Meta Description": row["Meta Description"]
        })

    prompt = f"""
You are an SEO assistant. We have a doc snippet of type [{doc_type.upper()}]:
---
{doc_content}
---

Topic: {topic}
Keyword (optional): {keyword}

We also have these candidate site pages in JSON form:

{json.dumps(pages_info, indent=2)}

TASK:
1. Recommend internal links (FROM the site pages above) to be referenced in the doc snippet.
2. Provide an optimized anchor text for each link, following anchor text best practices:
{anchor_text_guidelines}

Return strictly JSON (no extra text). The JSON must be an array of objects:
[
  {{
    "target_url": "https://example.com/page-1",
    "anchor_text": "concise anchor text"
  }},
  ...
]
""".strip()

    return prompt


def gpt_internal_links(
    openai_api_key: str,
    doc_snippet: dict,
    candidate_pages: pd.DataFrame,
    topic: str,
    keyword: str,
    model="gpt-4o-mini"
):
    """
    Calls GPT to get anchor text linking from doc_snippet to each candidate page.
    """
    openai.api_key = openai_api_key
    prompt = build_prompt_for_paragraph(doc_snippet, candidate_pages, topic, keyword)

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful SEO assistant. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1500
        )
        text_out = response.choices[0].message["content"].strip()
        parsed = parse_openai_json(text_out)
        return parsed
    except Exception as e:
        st.error(f"Error from GPT: {e}")
        return []

###############################################################################
# 5. Streamlit App
###############################################################################
def main():
    st.title("Paragraph-Level Embedding for Word Docs + Site Pages")
    st.write("""
    This app demonstrates:
    1. Uploading a CSV/Excel of site pages (one row per page).
    2. Uploading a Word document, extracting paragraphs & headings.
    3. Embedding each site page row (individually).
    4. Embedding each doc paragraph/heading (individually).
    5. For **each** paragraph, performing a semantic search to find top site pages,
       then generating recommended anchor texts via GPT.
    """)

    # Download sample for site pages
    st.subheader("Step 1: (Optional) Download Sample Site Template")
    sample_xlsx = generate_sample_excel_template()
    link = get_download_link(sample_xlsx, "sample_template.xlsx", "Download Sample Excel Template")
    st.markdown(link, unsafe_allow_html=True)

    # Upload site CSV/Excel
    st.subheader("Step 2: Upload Site Pages CSV/Excel")
    pages_file = st.file_uploader(
        "Site Pages CSV/Excel (Must have columns: URL, H1, Meta Description)",
        type=["csv", "xlsx"]
    )

    # Upload Word Doc
    st.subheader("Step 3: Upload Word Document (.docx)")
    doc_file = st.file_uploader("Word Doc with paragraphs/headings", type=["docx"])

    # Topic + Keyword
    st.subheader("Step 4: Enter Topic & Optional Keyword")
    topic = st.text_input("Topic (Required)", help="E.g. 'Egress Basement Windows'")
    keyword = st.text_input("Optional Keyword", help="E.g. 'sliding windows'")

    # OpenAI key
    st.subheader("Step 5: Enter Your OpenAI API Key")
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    # Batch size
    st.subheader("Step 6: Set Batch Size for Embedding")
    batch_size = st.slider("Batch Size", min_value=1, max_value=50, value=10)

    # Button to embed
    st.subheader("Step 7: Click to Embed Content")
    if st.button("Embed Site Pages & Doc"):
        if not pages_file:
            st.error("Please upload your site pages file first.")
            st.stop()
        if not doc_file:
            st.error("Please upload a Word document first.")
            st.stop()
        if not topic:
            st.error("Please enter a topic.")
            st.stop()
        if not openai_api_key:
            st.error("Please enter your OpenAI API key.")
            st.stop()

        # Read pages file
        try:
            if pages_file.name.endswith(".csv"):
                df_pages = pd.read_csv(pages_file)
            else:
                df_pages = pd.read_excel(pages_file)
        except Exception as e:
            st.error(f"Error reading site pages: {e}")
            st.stop()

        # Validate columns
        required_cols = {"URL", "H1", "Meta Description"}
        if not required_cols.issubset(df_pages.columns):
            st.error(f"File missing columns: {', '.join(required_cols)}")
            st.stop()

        # Parse doc
        try:
            doc_bytes = doc_file.read()
            doc_paragraphs = parse_word_doc(io.BytesIO(doc_bytes))
            if not doc_paragraphs:
                st.error("No valid paragraphs/headings found in the Word doc.")
                st.stop()
        except Exception as e:
            st.error(f"Error parsing doc: {e}")
            st.stop()

        # Build DataFrame for doc. Each row = 1 paragraph/heading
        df_doc = pd.DataFrame(doc_paragraphs)  # columns: [type, content]

        # Add a "text_for_embedding" column for the site pages
        # We'll just combine URL+H1+Meta Desc for each row
        df_pages["text_for_embedding"] = (
            df_pages["URL"] + " " + df_pages["H1"] + " " + df_pages["Meta Description"]
        )

        # For doc paragraphs, embed the "content" directly
        df_doc["text_for_embedding"] = df_doc["type"].str.upper() + ": " + df_doc["content"]

        # Embed pages
        st.write("**Embedding site pages**")
        df_pages_embedded = batch_embed_rows(
            df=df_pages.copy(), 
            openai_api_key=openai_api_key, 
            text_col="text_for_embedding", 
            batch_size=batch_size
        )

        # Embed doc paragraphs
        st.write("**Embedding Word doc paragraphs/headings**")
        df_doc_embedded = batch_embed_rows(
            df=df_doc.copy(),
            openai_api_key=openai_api_key,
            text_col="text_for_embedding",
            batch_size=batch_size
        )

        st.session_state["df_pages_embedded"] = df_pages_embedded
        st.session_state["df_doc_embedded"] = df_doc_embedded
        st.session_state["topic"] = topic
        st.session_state["keyword"] = keyword
        st.session_state["openai_api_key"] = openai_api_key
        st.success("All embeddings complete! Proceed to 'Generate Links' below.")

    st.subheader("Step 8: Generate Links for Each Paragraph")
    if st.button("Generate Paragraph-Level Links"):
        if "df_pages_embedded" not in st.session_state or "df_doc_embedded" not in st.session_state:
            st.error("Please embed data first.")
            st.stop()

        df_pages_embedded = st.session_state["df_pages_embedded"]
        df_doc_embedded = st.session_state["df_doc_embedded"]
        final_topic = st.session_state["topic"]
        final_keyword = st.session_state["keyword"]
        final_api_key = st.session_state["openai_api_key"]

        results_list = []

        # For each paragraph in the doc, let's do a semantic search among site pages
        for idx, doc_row in df_doc_embedded.iterrows():
            snippet = {
                "type": doc_row["type"],
                "content": doc_row["content"]
            }
            doc_embedding = np.array(doc_row["embedding"])

            # 1) Compare doc snippet to each site page
            sims = []
            for i, page_row in df_pages_embedded.iterrows():
                page_embedding = np.array(page_row["embedding"])
                sim = compute_cosine_similarity(doc_embedding, page_embedding)
                sims.append(sim)
            df_pages_embedded["similarity"] = sims

            # 2) Sort by descending similarity, take top K
            top_k = 5  # adjust as needed
            best_matches = df_pages_embedded.sort_values("similarity", ascending=False).head(top_k)

            # 3) GPT call with snippet + these top pages
            links_data = gpt_internal_links(
                openai_api_key=final_api_key,
                doc_snippet=snippet,
                candidate_pages=best_matches[["URL", "H1", "Meta Description"]],
                topic=final_topic,
                keyword=final_keyword
            )
            # links_data => e.g. [ { "target_url": "...", "anchor_text": "..." }, ...]

            # We store the results along with snippet info
            for link_item in links_data:
                results_list.append({
                    "paragraph_index": idx,
                    "paragraph_type": doc_row["type"],
                    "paragraph_content": doc_row["content"],
                    "target_url": link_item.get("target_url", ""),
                    "anchor_text": link_item.get("anchor_text", ""),
                    "similarity_score": float(best_matches.loc[
                        best_matches["URL"] == link_item.get("target_url", ""),
                        "similarity"
                    ].max()) if link_item.get("target_url", "") in best_matches["URL"].values else None
                })

        if not results_list:
            st.warning("No recommendations were returned.")
        else:
            final_df = pd.DataFrame(results_list)
            st.write("**Paragraph-Level Recommendations**")
            st.dataframe(final_df)

            # Download
            csv_data = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv_data,
                file_name=f"paragraph_link_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            # Optionally convert to Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                final_df.to_excel(writer, index=False, sheet_name="Recommendations")
            excel_data = excel_buffer.getvalue()
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"paragraph_link_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
