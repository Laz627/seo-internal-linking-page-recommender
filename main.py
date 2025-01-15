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
    Generates an in-memory Excel file containing columns:
    URL, H1, Meta Description.
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
    ws["C2"] = "This is a sample meta description for demonstration."

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def get_download_link(file_bytes: bytes, filename: str, file_label: str) -> str:
    """
    Returns an HTML link to download the given bytes as a file.
    """
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{file_label}</a>'

def compute_cosine_similarity(vec_a, vec_b) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a)*np.linalg.norm(vec_b)))

###############################################################################
# 2. Parsing Word Doc (Paragraphs Only)
###############################################################################
def parse_word_doc_paragraphs_only(doc_bytes: bytes):
    """
    Parse .docx, skipping headings. Return a list of dicts:
      [{ 'content': "...paragraph text...", 'embedding': None, 'sentences': None }, ...]
    We'll add 'embedding' after we embed them.
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
# 3. Embedding Calls
###############################################################################
def embed_text_batch(openai_api_key: str, texts: list[str], model="text-embedding-ada-002"):
    """
    Batch call to OpenAI's Embeddings API for a list of strings.
    """
    openai.api_key = openai_api_key
    try:
        resp = openai.Embedding.create(model=model, input=texts)
        embeddings = [item["embedding"] for item in resp["data"]]
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return [[] for _ in texts]

def embed_site_pages(df: pd.DataFrame, openai_api_key: str, batch_size: int=10):
    """
    For each row, embed (URL + H1 + Meta Description) in batches => df['embedding'].
    """
    st.info("Embedding site pages in batches...")
    texts = []
    for _, row in df.iterrows():
        combined_text = f"{row['URL']} {row['H1']} {row['Meta Description']}"
        texts.append(combined_text)

    n = len(texts)
    all_embeddings = []
    total_batches = (n + batch_size - 1)//batch_size
    bar = st.progress(0)
    label = st.empty()

    idx = 0
    for batch_idx in range(total_batches):
        start = batch_idx*batch_size
        end = start+batch_size
        batch_texts = texts[start:end]
        batch_embeddings = embed_text_batch(openai_api_key, batch_texts)
        all_embeddings.extend(batch_embeddings)

        idx+=1
        bar.progress(int(idx/total_batches*100))
        label.write(f"Processing site pages batch {idx}/{total_batches}")

    df["embedding"] = all_embeddings
    return df

def embed_doc_paragraphs(paragraphs: list[dict], openai_api_key: str, batch_size: int=10):
    """
    Embed each paragraph's text, plus (optionally) each paragraph's sentences.
    """
    st.info("Embedding doc paragraphs in batches...")
    texts = [p["content"] for p in paragraphs]
    n = len(texts)
    all_embeddings = []
    total_batches = (n + batch_size - 1)//batch_size

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
        label.write(f"Paragraphs batch {idx}/{total_batches}")

    for i,emb in enumerate(all_embeddings):
        paragraphs[i]["embedding"] = emb

    # If you want to embed sentences too:
    st.info("Embedding paragraph sentences as well...")
    bar_sents = st.progress(0)
    for i, p in enumerate(paragraphs):
        bar_sents.progress(int((i+1)/n*100))
        raw_txt = p["content"].strip()
        sents = re.split(r'(?<=[.!?])\s+', raw_txt)
        sents = [s.strip() for s in sents if s.strip()]
        if not sents:
            p["sentences"] = []
            continue
        s_embs = embed_text_batch(openai_api_key, sents)
        p["sentences"] = list(zip(sents,s_embs))

    return paragraphs

###############################################################################
# 4. GPT-based Internal Links (Brand-New or Existing)
###############################################################################
def build_prompt(mode:str, topic:str, keyword:str, target_data:dict, candidate_pages:pd.DataFrame)->str:
    # anchor text best practices
    anchor_text_guidelines = """
Anchor Text Best Practices:
- Descriptive, concise, relevant
- Avoid "click here", "read more", or stuffing keywords
- Anchor text should read naturally and make sense on its own
"""

    if mode=="brand_new":
        brand_new_msg = "This is a brand-new topic/page that doesn't exist yet.\n"
        target_url = "N/A"
        target_h1 = f"(New) {topic}"
        target_meta = f"A future page about {topic}"
    else:
        brand_new_msg=""
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

    prompt=f"""
You are an SEO assistant. The user wants internal link recommendations.

Mode: {mode}
Topic: {topic}
Keyword: {keyword}
{brand_new_msg}

Target Page Info:
URL: {target_url}
H1: {target_h1}
Meta Description: {target_meta}

Candidate Pages in JSON:
{json.dumps(pages_info, indent=2)}

Please return only valid JSON (array of objects). No disclaimers or extra text.

Each object: 
  "target_url" : "URL to link FROM"
  "anchor_text": "Proposed Anchor Text"

Anchor Text Best Practices:
{anchor_text_guidelines}
""".strip()
    return prompt

def generate_internal_links(openai_api_key:str, mode:str, topic:str, keyword:str,
                            target_data:dict, candidate_pages:pd.DataFrame):
    openai.api_key = openai_api_key
    prompt_text = build_prompt(mode, topic, keyword, target_data, candidate_pages)

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role":"system",
                    "content":"You are a helpful SEO assistant. Return valid JSON only. No disclaimers."
                },
                {
                    "role":"user",
                    "content":prompt_text
                }
            ],
            max_tokens=8000,
            temperature=0.0
        )
        raw_response = resp.choices[0].message["content"].strip()
        if not raw_response:
            st.warning("GPT returned empty response. Possibly an error or out of tokens.")
            return []
        # Attempt to parse
        try:
            data=json.loads(raw_response)
            if isinstance(data, dict):
                data=[data]
            return data
        except json.JSONDecodeError:
            st.warning(f"GPT returned invalid JSON:\n{raw_response}")
            return []
    except Exception as e:
        st.error(f"Error calling GPT: {e}")
        return []

###############################################################################
# 5. Main Streamlit App
###############################################################################
def main():
    st.title("Internal Link Recommender (3 Modes)")

    st.write("""
    This script merges:
    1) **Brand-New Topic/Page** (GPT-based)
    2) **Optimize Existing Page** (GPT-based)
    3) **Analyze Word Document** (Pure embeddings, paragraphs claim pages).
    
    We add strict instructions for GPT to hopefully avoid invalid JSON.
    """)

    # Step 1: Download sample template
    st.subheader("Step 1: (Optional) Download Site Template")
    sample_bytes = generate_sample_excel_template()
    dl_link = get_download_link(sample_bytes,"sample_template.xlsx","Download Sample Template")
    st.markdown(dl_link, unsafe_allow_html=True)

    # Step 2: Upload site pages
    st.subheader("Step 2: Upload Site Pages CSV/Excel")
    up_file=st.file_uploader("Columns: URL, H1, Meta Description", type=["csv","xlsx"])
    df=None
    if up_file:
        try:
            if up_file.name.endswith(".csv"):
                df=pd.read_csv(up_file)
            else:
                df=pd.read_excel(up_file)
        except Exception as e:
            st.error(f"Error reading site pages: {e}")

    # Step 3: Mode
    st.subheader("Step 3: Choose Mode")
    mode_opts=["Brand-New Topic/Page","Optimize Existing Page","Analyze Word Document"]
    mode_choice = st.radio("Mode:", mode_opts)

    selected_url=None
    doc_paragraphs=[]
    if mode_choice=="Optimize Existing Page" and df is not None:
        req_cols={"URL","H1","Meta Description"}
        if req_cols.issubset(df.columns):
            all_urls=df["URL"].unique().tolist()
            selected_url=st.selectbox("Select Existing Page URL", all_urls)
        else:
            st.warning("Please upload valid CSV with URL,H1,Meta Description first.")
    elif mode_choice=="Analyze Word Document":
        st.subheader("3A: Upload Word Doc (.docx)")
        doc_file=st.file_uploader("Upload .docx", type=["docx"])
        if doc_file:
            try:
                raw_bytes=doc_file.read()
                doc_paragraphs=parse_word_doc_paragraphs_only(raw_bytes)
                st.success(f"Found {len(doc_paragraphs)} paragraphs (excluding headings).")
            except Exception as e:
                st.error(f"Error parsing Word doc: {e}")

    # Step 4: Topic & Optional Keyword (for GPT-based modes)
    if mode_choice in ["Brand-New Topic/Page","Optimize Existing Page"]:
        st.subheader("Step 4: Enter Topic & Optional Keyword")
        topic=st.text_input("Topic (Required)")
        keyword=st.text_input("Keyword (Optional)")
    else:
        topic=""
        keyword=""

    # Step 5: OpenAI key
    st.subheader("Step 5: Enter OpenAI API Key")
    openai_key=st.text_input("OpenAI Key",type="password")

    # Step 6: Batch size
    st.subheader("Step 6: Set Batch Size for Embedding")
    batch_sz=st.slider("Batch size",1,50,10)

    # EMBED
    st.subheader("Step 7: Embed Data")
    if st.button("Embed Now"):
        if not openai_key:
            st.error("Please provide OpenAI API key.")
            st.stop()
        if df is None or df.empty:
            st.error("Please upload site pages first.")
            st.stop()
        needed_cols={"URL","H1","Meta Description"}
        if not needed_cols.issubset(df.columns):
            st.error(f"Missing columns: {', '.join(needed_cols)}")
            st.stop()

        df_emb=embed_site_pages(df.copy(), openai_key,batch_sz)

        doc_data=None
        if mode_choice=="Analyze Word Document":
            if not doc_paragraphs:
                st.error("No paragraphs found in doc.")
                st.stop()
            doc_data=embed_doc_paragraphs(doc_paragraphs, openai_key,batch_sz)

        st.session_state["df_pages"]=df_emb
        st.session_state["doc_data"]=doc_data
        st.session_state["mode"]=mode_choice
        st.session_state["topic"]=topic
        st.session_state["keyword"]=keyword
        st.session_state["selected_url"]=selected_url
        st.success("Embedding done. Proceed to generate links below.")

    # Step 8: Generate Links
    st.subheader("Step 8: Generate Links")
    if st.button("Generate Links"):
        if "df_pages" not in st.session_state:
            st.error("Please embed data first.")
            st.stop()

        final_mode=st.session_state["mode"]
        df_pages_emb=st.session_state["df_pages"]
        doc_data=st.session_state["doc_data"]
        final_topic=st.session_state["topic"]
        final_keyword=st.session_state["keyword"]
        final_url=st.session_state.get("selected_url",None)

        threshold=0.80  # for semantic filtering

        if final_mode in ["Brand-New Topic/Page","Optimize Existing Page"]:
            if final_mode=="Optimize Existing Page":
                # get target row
                try:
                    target_row=df_pages_emb.loc[df_pages_emb["URL"]==final_url].iloc[0].to_dict()
                except IndexError:
                    st.error("Selected URL not found in dataset.")
                    st.stop()
                # exclude that row from candidates
                candidate_df=df_pages_emb.loc[df_pages_emb["URL"]!=final_url].copy()

                # if user typed a topic => semantic filter
                if final_topic:
                    combo_query=final_topic+" "+final_keyword if final_keyword else final_topic
                    # embed that query
                    from_embed=embed_text_batch(openai_key,[combo_query])
                    if from_embed and from_embed[0]:
                        q_vec=np.array(from_embed[0])
                        sims=[]
                        for i,row in candidate_df.iterrows():
                            page_vec=np.array(row["embedding"])
                            sim_val=compute_cosine_similarity(q_vec,page_vec)
                            sims.append(sim_val)
                        candidate_df["similarity"]=sims
                        # keep only >=80% for now
                        candidate_df=candidate_df[candidate_df["similarity"]>=threshold]

                # Call GPT
                links=generate_internal_links(
                    openai_key,
                    "existing_page",
                    final_topic,
                    final_keyword,
                    target_row,
                    candidate_df[["URL","H1","Meta Description","similarity"]] if "similarity" in candidate_df.columns else candidate_df[["URL","H1","Meta Description"]]
                )

                if not links:
                    st.warning("No recommendations or invalid JSON from GPT.")
                else:
                    st.subheader("Optimize Existing Page Results")
                    for item in links:
                        # anchor_text, target_url, similarity if present
                        anchor_txt=item.get("anchor_text","(No anchor)")
                        url=item.get("target_url","")
                        st.markdown(f"- **Page Link**: [{anchor_txt}]({url})")
                        sim_val=item.get("similarity",None)
                        if sim_val is not None:
                            st.markdown(f"  - **Similarity Score**: {round(sim_val*100,2)}%")

            else:
                # brand_new
                if final_topic:
                    combo_query=final_topic+" "+final_keyword if final_keyword else final_topic
                    q_embed=embed_text_batch(openai_key,[combo_query])
                    if q_embed and q_embed[0]:
                        q_vec=np.array(q_embed[0])
                        sims=[]
                        for i,row in df_pages_emb.iterrows():
                            page_vec=np.array(row["embedding"])
                            sim_val=compute_cosine_similarity(q_vec,page_vec)
                            sims.append(sim_val)
                        df_pages_emb["similarity"]=sims
                        df_pages_emb=df_pages_emb[df_pages_emb["similarity"]>=threshold]

                # dummy target data
                new_target={
                    "URL": "N/A",
                    "H1": f"(New) {final_topic}",
                    "Meta Description": f"Future page about {final_topic}"
                }
                links=generate_internal_links(
                    openai_key,
                    "brand_new",
                    final_topic,
                    final_keyword,
                    new_target,
                    df_pages_emb[["URL","H1","Meta Description","similarity"]] if "similarity" in df_pages_emb.columns else df_pages_emb[["URL","H1","Meta Description"]]
                )
                if not links:
                    st.warning("No recommendations or invalid JSON from GPT.")
                else:
                    st.subheader("Brand-New Topic Results")
                    for item in links:
                        anchor_txt=item.get("anchor_text","(No anchor)")
                        url=item.get("target_url","")
                        st.markdown(f"- **Page Link**: [{anchor_txt}]({url})")
                        sim_val=item.get("similarity",None)
                        if sim_val is not None:
                            st.markdown(f"  - **Similarity Score**: {round(sim_val*100,2)}%")

        else:
            # Word Doc mode => each paragraph picks exactly one page if >= 80%, removing that page from the pool.
            if not doc_data:
                st.error("No doc data found. Please embed paragraphs first.")
                st.stop()

            doc_df=pd.DataFrame(doc_data)
            pages_copy=df_pages_emb.copy()
            results=[]
            paragraphs_count=len(doc_df)
            bar=st.progress(0)
            label=st.empty()

            st.write("Each paragraph claims one unique page above 80% similarity. Once used, page is removed.")

            for p_idx, p_row in doc_df.iterrows():
                progress_val=int(((p_idx+1)/paragraphs_count)*100)
                bar.progress(progress_val)
                label.write(f"Paragraph {p_idx+1}/{paragraphs_count}")

                para_vec=np.array(p_row["embedding"])
                best_sim=-1.0
                best_idx=-1
                for i, page_row in pages_copy.iterrows():
                    pv=np.array(page_row["embedding"])
                    sim_val=compute_cosine_similarity(para_vec,pv)
                    if sim_val>best_sim:
                        best_sim=sim_val
                        best_idx=i
                if best_sim>=0.80 and best_idx!=-1:
                    chosen_page=pages_copy.loc[best_idx]
                    results.append({
                        "paragraph_index": p_idx,
                        "paragraph_text": p_row["content"],
                        "page_url": chosen_page["URL"],
                        "page_title": chosen_page["H1"],
                        "similarity":best_sim
                    })
                    pages_copy.drop(index=best_idx,inplace=True)
                if pages_copy.empty:
                    break

            if not results:
                st.warning("No paragraphs matched pages above 80%. Or no pages left.")
            else:
                st.subheader("Word Doc Mode Results (Paragraph => Unique Page)")
                for item in results:
                    with st.expander(f"Paragraph #{item['paragraph_index']+1}"):
                        st.markdown(f"> **Paragraph Text**:\n> {item['paragraph_text']}")
                        st.markdown(f"- **Page Link**: [{item['page_title']}]({item['page_url']})")
                        st.markdown(f"- **Similarity Score**: {round(item['similarity']*100,2)}%")

                df_final=pd.DataFrame(results)
                st.subheader("Download CSV")
                csv_bytes=df_final.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv_bytes, f"doc_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")


if __name__=="__main__":
    main()
