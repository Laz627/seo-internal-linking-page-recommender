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

    # (Optional) Populate a sample row
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


def parse_openai_json(response_text: str):
    """
    Attempt to parse GPT response as JSON. Return an empty list if parsing fails.
    """
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            data = [data]
        return data
    except json.JSONDecodeError:
        return []

###############################################################################
# 2. Word Document Parsing (Paragraphs Only)
###############################################################################
def parse_word_doc_paragraphs_only(file_obj: io.BytesIO):
    """
    Extracts ONLY paragraphs from the .docx file, skipping any headings.

    Returns a list of dicts:
      [
        { "type": "paragraph", "content": "...some text..." },
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
            # skip if it's recognized as heading
            paragraphs_only.append({"type": "paragraph", "content": text})
    return paragraphs_only

###############################################################################
# 3. Embeddings + Similarity
###############################################################################
def embed_text_batch(openai_api_key: str, texts: list[str], model="text-embedding-ada-002"):
    """
    Batch-embed a list of texts using OpenAI's Embeddings API.
    """
    openai.api_key = openai_api_key
    try:
        response = openai.Embedding.create(model=model, input=texts)
        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings
    except Exception as e:
        st.error(f"Error generating batch embeddings: {e}")
        return [[] for _ in texts]


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def embed_site_pages_in_batches(df: pd.DataFrame, openai_api_key: str, batch_size: int = 10):
    """
    For each row (page) in df, embed URL + H1 + Meta Description as a single chunk.
    """
    st.info("Embedding site pages. Please wait...")

    # Prepare texts
    texts = []
    for _, row in df.iterrows():
        combined_text = f"{row['URL']} {row['H1']} {row['Meta Description']}"
        texts.append(combined_text)

    all_embeddings = []
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    bar = st.progress(0)
    label = st.empty()

    idx = 0
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)

        all_embeddings.extend(batch_embeddings)

        idx += 1
        percentage = int(idx / total_batches * 100)
        bar.progress(percentage)
        label.write(f"Site pages batch {idx}/{total_batches}")

    df["embedding"] = all_embeddings
    return df


def embed_doc_paragraphs_in_batches(paragraphs: list[dict], openai_api_key: str, batch_size: int = 10):
    """
    For each paragraph in the doc, embed the paragraph text.
    Return a DataFrame with columns: type, content, embedding
    """
    st.info("Embedding doc paragraphs (no headings). Please wait...")

    # combine text
    texts = []
    for p in paragraphs:
        texts.append(p["content"])

    all_embeddings = []
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    bar = st.progress(0)
    label = st.empty()

    idx = 0
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        idx += 1
        percentage = int(idx / total_batches * 100)
        bar.progress(percentage)
        label.write(f"Doc paragraphs batch {idx}/{total_batches}")

    # Build final DF
    df_doc = pd.DataFrame(paragraphs)  # has columns type, content
    df_doc["embedding"] = all_embeddings
    return df_doc

###############################################################################
# 4. GPT Prompt: "Pick substring from paragraph"
###############################################################################
def build_substring_prompt(paragraph_text: str, site_page: dict):
    """
    We ask GPT to identify a substring within the paragraph_text that best
    matches the site_page content. The user wants to place the link on that substring,
    rather than GPT inventing a brand new anchor text.

    site_page has { "URL", "H1", "Meta Description" }
    """
    instructions = f"""
You are an SEO assistant. 
We have a paragraph of text:
---
{paragraph_text}
---

We want to link TO this site page:
URL: {site_page["URL"]}
H1: {site_page["H1"]}
Meta Description: {site_page["Meta Description"]}

Instead of inventing a new anchor text, find a substring **exactly** from the paragraph 
that best describes or correlates with this site page. 
We want to place the hyperlink on that exact substring in the paragraph.
Make sure the substring is short (up to ~10 words) and is relevant to the page content. 
If there's no good substring, just return an empty array.

Return strictly JSON, no extra text. Format:
[
  {{
    "anchor_substring": "substring from the paragraph"
  }}
]
"""
    return instructions.strip()


def pick_substring_for_link(
    openai_api_key: str,
    paragraph_text: str,
    site_page: dict
):
    """
    Calls GPT with build_substring_prompt. 
    Expects GPT to return JSON with a key "anchor_substring" 
    taken from the paragraph.
    """
    openai.api_key = openai_api_key
    prompt = build_substring_prompt(paragraph_text, site_page)

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful SEO assistant. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.0
        )
        text_out = resp.choices[0].message["content"].strip()
        parsed = parse_openai_json(text_out)
        return parsed
    except Exception as e:
        st.error(f"Error from GPT: {e}")
        return []

###############################################################################
# 5. Brand-New / Existing Page Logic (unchanged)
###############################################################################
def build_prompt_for_mode(
    mode: str,
    topic: str,
    keyword: str,
    target_data: dict,
    candidate_pages: pd.DataFrame
) -> str:
    anchor_text_guidelines = """
Anchor Text Best Practices:
- Descriptive, concise, and relevant
- Avoid overly generic text like 'click here', 'read more'
- Avoid overly long or keyword-stuffed anchor text
- The text should read naturally and reflect the linked page's content
"""

    if mode == "brand_new":
        brand_new_msg = (
            "Brand-new topic/page that doesn't exist yet.\n"
            "Recommend internal links from the site pages below.\n"
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


def generate_internal_links(
    openai_api_key: str,
    mode: str,
    topic: str,
    keyword: str,
    target_data: dict,
    candidate_pages: pd.DataFrame
):
    """
    For brand-new or existing mode, calls GPT once for the entire set of candidate pages.
    """
    openai.api_key = openai_api_key
    prompt = build_prompt_for_mode(mode, topic, keyword, target_data, candidate_pages)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful SEO assistant. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        text_out = response.choices[0].message["content"].strip()
        parsed = parse_openai_json(text_out)
        return parsed
    except Exception as e:
        st.error(f"Error in GPT call: {e}")
        return []

###############################################################################
# 6. The Streamlit App
###############################################################################
def main():
    st.title("Internal Link Recommender (3 Modes, Paragraph-Only Word Doc)")

    st.write("""
    **Modes**:
    1) **Brand-New Topic/Page**: no existing URL  
    2) **Optimize Existing Page**: pick an existing URL  
    3) **Analyze Word Document**: skip headings, parse paragraphs only,  
       and for each site page, we place the link in exactly one paragraph (the best match).
    """)

    # Step 1: Download sample
    st.subheader("Step 1: (Optional) Download Site Template")
    template_bytes = generate_sample_excel_template()
    link = get_download_link(template_bytes, "sample_template.xlsx", "Download Sample Excel Template")
    st.markdown(link, unsafe_allow_html=True)

    # Step 2: Upload site CSV/Excel
    st.subheader("Step 2: Upload Site Pages")
    pages_file = st.file_uploader("Site CSV/Excel (URL, H1, Meta Description)", type=["csv", "xlsx"])
    df_pages = None
    if pages_file:
        try:
            if pages_file.name.endswith(".csv"):
                df_pages = pd.read_csv(pages_file)
            else:
                df_pages = pd.read_excel(pages_file)
        except Exception as e:
            st.error(f"Error reading pages: {e}")

    # Step 3: Mode
    st.subheader("Step 3: Pick Mode")
    mode_option = st.radio(
        "Choose one:",
        ["Brand-New Topic/Page", "Optimize Existing Page", "Analyze Word Document"]
    )

    # If existing page, user picks
    selected_url = None
    if mode_option == "Optimize Existing Page" and df_pages is not None:
        if {"URL", "H1", "Meta Description"}.issubset(df_pages.columns):
            all_urls = df_pages["URL"].unique().tolist()
            selected_url = st.selectbox("Existing Page URL", all_urls)
        else:
            st.warning("Please upload a valid CSV/Excel with the required columns first.")

    # If Word doc analysis, user uploads doc
    doc_file = None
    doc_paragraphs = []
    if mode_option == "Analyze Word Document":
        st.subheader("Step 3A: Upload Word Document (paragraphs only)")
        doc_file = st.file_uploader("Upload .docx to parse paragraphs", type=["docx"])
        if doc_file:
            try:
                doc_bytes = doc_file.read()
                doc_paragraphs = parse_word_doc_paragraphs_only(io.BytesIO(doc_bytes))
                st.success(f"Found {len(doc_paragraphs)} paragraphs (no headings).")
            except Exception as e:
                st.error(f"Error parsing Word doc: {e}")

    # Step 4: Topic, Keyword
    st.subheader("Step 4: Topic & Optional Keyword")
    topic = st.text_input("Topic (required)")
    keyword = st.text_input("Optional Keyword")

    # Step 5: OpenAI Key
    st.subheader("Step 5: OpenAI API Key")
    openai_api_key = st.text_input("OpenAI Key", type="password")

    # Step 6: Batch size
    st.subheader("Step 6: Batch Size for Embedding")
    batch_size = st.slider("Batch size", min_value=1, max_value=50, value=10)

    # Step 7: Embed
    st.subheader("Step 7: Embed Data")
    if st.button("Embed Now"):
        if not openai_api_key:
            st.error("Please provide an OpenAI key.")
            st.stop()

        if mode_option in ["Brand-New Topic/Page", "Optimize Existing Page"]:
            if df_pages is None:
                st.error("Please upload site pages first.")
                st.stop()

            required_cols = {"URL", "H1", "Meta Description"}
            if not required_cols.issubset(df_pages.columns):
                st.error(f"File missing columns: {', '.join(required_cols)}")
                st.stop()

            if not topic:
                st.error("Please enter a topic.")
                st.stop()

            # Embed site pages
            df_emb = df_pages.copy()
            df_emb = embed_site_pages_in_batches(df_emb, openai_api_key, batch_size)

            st.session_state["site_pages"] = df_emb
            st.session_state["doc_paragraphs"] = None
            st.session_state["mode"] = mode_option
            st.session_state["topic"] = topic
            st.session_state["keyword"] = keyword
            st.session_state["selected_url"] = selected_url

            st.success("Site pages embedded. Proceed to Generate Links.")

        else:
            # analyze_word_doc
            if df_pages is None:
                st.error("Need site pages to compare with doc.")
                st.stop()
            if not doc_paragraphs:
                st.error("No paragraphs found or doc not uploaded.")
                st.stop()

            required_cols = {"URL", "H1", "Meta Description"}
            if not required_cols.issubset(df_pages.columns):
                st.error(f"Site file missing columns: {', '.join(required_cols)}")
                st.stop()

            if not topic:
                st.error("Topic is required.")
                st.stop()

            # Embed site pages
            df_pages_emb = df_pages.copy()
            df_pages_emb = embed_site_pages_in_batches(df_pages_emb, openai_api_key, batch_size)

            # Embed doc paragraphs
            df_doc_emb = embed_doc_paragraphs_in_batches(doc_paragraphs, openai_api_key, batch_size)

            st.session_state["site_pages"] = df_pages_emb
            st.session_state["doc_paragraphs"] = df_doc_emb.to_dict("records")
            st.session_state["mode"] = "analyze_word_doc"
            st.session_state["topic"] = topic
            st.session_state["keyword"] = keyword

            st.success("Site pages + doc paragraphs embedded. Proceed to Generate Links.")

    # Step 8: Generate links
    st.subheader("Step 8: Generate Links")
    if st.button("Generate Links"):
        if "mode" not in st.session_state:
            st.error("Please embed data first.")
            st.stop()

        final_mode = st.session_state["mode"]
        final_topic = st.session_state["topic"]
        final_keyword = st.session_state["keyword"]
        final_url = st.session_state.get("selected_url", None)
        df_site = st.session_state.get("site_pages", None)
        doc_paragraph_dicts = st.session_state.get("doc_paragraphs", None)

        if final_mode in ["Brand-New Topic/Page", "Optimize Existing Page"]:
            # Same logic as before
            if df_site is None:
                st.error("No embedded site data found.")
                st.stop()

            if final_mode == "Optimize Existing Page" and final_url:
                try:
                    target_row = df_site.loc[df_site["URL"] == final_url].iloc[0].to_dict()
                except IndexError:
                    st.error("Selected URL not found in embedded dataset.")
                    st.stop()

                # exclude from candidate
                candidate_df = df_site.loc[df_site["URL"] != final_url].copy()

                # do similarity
                query_str = final_topic if not final_keyword else f"{final_topic} {final_keyword}"
                one_batch = embed_text_batch(openai_api_key, [query_str])
                if not one_batch or not one_batch[0]:
                    st.error("Query embedding failed.")
                    st.stop()
                q_vec = np.array(one_batch[0])

                sims = []
                for i, row in candidate_df.iterrows():
                    page_vec = np.array(row["embedding"])
                    sims.append(compute_cosine_similarity(q_vec, page_vec))
                candidate_df["similarity"] = sims
                candidate_df_sorted = candidate_df.sort_values("similarity", ascending=False).head(50)
                threshold = 0.50
                filtered = candidate_df_sorted[candidate_df_sorted["similarity"] >= threshold].copy()

                final_links = generate_internal_links(
                    openai_api_key,
                    "existing_page",
                    final_topic,
                    final_keyword,
                    target_row,
                    filtered[["URL", "H1", "Meta Description"]]
                )

            else:
                # brand_new
                query_str = final_topic if not final_keyword else f"{final_topic} {final_keyword}"
                one_batch = embed_text_batch(openai_api_key, [query_str])
                if not one_batch or not one_batch[0]:
                    st.error("Query embedding failed.")
                    st.stop()
                q_vec = np.array(one_batch[0])

                sims = []
                for i, row in df_site.iterrows():
                    page_vec = np.array(row["embedding"])
                    sims.append(compute_cosine_similarity(q_vec, page_vec))
                df_site["similarity"] = sims
                df_site_sorted = df_site.sort_values("similarity", ascending=False).head(50)
                threshold = 0.50
                filtered = df_site_sorted[df_site_sorted["similarity"] >= threshold].copy()

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
                    filtered[["URL", "H1", "Meta Description"]]
                )

            if not final_links:
                st.warning("No recommendations returned or invalid JSON.")
            else:
                res_df = pd.DataFrame(final_links)
                if 'filtered' in locals() and not filtered.empty:
                    merged = pd.merge(
                        res_df,
                        filtered[["URL", "similarity"]],
                        left_on="target_url",
                        right_on="URL",
                        how="left"
                    )
                    merged.drop(columns=["URL"], inplace=True, errors="ignore")
                    if "similarity" in merged.columns:
                        merged.rename(columns={"similarity": "similarity_score"}, inplace=True)
                else:
                    merged = res_df.copy()

                st.write("**Recommendations**")
                st.dataframe(merged)
                csv_data = merged.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name=f"links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )

        else:
            # analyze_word_doc with paragraphs only
            if df_site is None or not doc_paragraph_dicts:
                st.error("Missing data for pages or doc paragraphs.")
                st.stop()

            # For each SITE PAGE, we find which doc paragraph is the best match.
            # Then we call GPT to pick a substring from that paragraph text.

            # We'll produce one link suggestion per site page (so each page is unique).
            results = []
            for i, site_row in df_site.iterrows():
                site_vec = np.array(site_row["embedding"])

                # Find best paragraph
                best_sim = -1
                best_idx = -1
                for idx, para_dict in enumerate(doc_paragraph_dicts):
                    para_vec = np.array(para_dict["embedding"])
                    sim = compute_cosine_similarity(site_vec, para_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = idx

                if best_sim < 0.50:
                    # If no paragraph above threshold, skip
                    continue

                # We have the best paragraph for this site page
                chosen_para = doc_paragraph_dicts[best_idx]
                # Now call GPT to pick a substring
                snippet_text = chosen_para["content"]
                chosen_page = {
                    "URL": site_row["URL"],
                    "H1": site_row["H1"],
                    "Meta Description": site_row["Meta Description"]
                }

                substring_resp = pick_substring_for_link(openai_api_key, snippet_text, chosen_page)
                if not substring_resp:
                    # GPT returned nothing or invalid
                    continue

                anchor_substring = substring_resp[0].get("anchor_substring", "")
                if not anchor_substring:
                    # no good substring
                    continue

                results.append({
                    "page_url": chosen_page["URL"],
                    "paragraph_index": best_idx,
                    "paragraph_text": snippet_text,
                    "anchor_substring": anchor_substring,
                    "similarity_score": best_sim
                })

            if not results:
                st.warning("No link suggestions found or all below threshold.")
            else:
                final_df = pd.DataFrame(results)
                st.write("**One Link per Site Page (Word Doc Paragraph)**")
                st.dataframe(final_df)

                csv_data = final_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name=f"word_doc_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
