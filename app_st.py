import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from mistralai import Mistral
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import plotly.graph_objects as go

# ========================= CONFIG =========================
st.set_page_config(
    page_title="AI Resume-JD Matcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 AI Resume & JD Matcher")
st.markdown("**Powered by Mistral AI + LangChain + FAISS**  •  Get instant match % + tailored interview questions")

# --------------------- Sidebar ---------------------
st.sidebar.header("🔑 Configuration")

load_dotenv()
api_key = os.getenv("MISTRAL_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Mistral API Key", type="password", help="Get it from https://console.mistral.ai")

if not api_key:
    st.sidebar.error("Please provide your Mistral API Key")
    st.stop()

client = Mistral(api_key=api_key)
MODEL = "mistral-small-latest"

st.sidebar.success("✅ API Key loaded")

# ========================= PROMPTS =========================
MATCH_PROMPT = f"""
    You are a Senior ATS Recruiter. Your task is to perform a strict 'Phase 1' screening.

    ### CRITICAL FILTERING RULES:
    1. ROLE ALIGNMENT: Identify the target job title in the JD and the candidate's professional identity in the Resume. 
       - If the candidate is a 'Graphic Designer' applying for 'Data Scientist', REJECT (Score < 10).
       - Do not allow 'transferable skills' to bypass a complete lack of core domain experience.
    2. EXPERIENCE GAP: 
       - If the JD requires 5+ years and the candidate has < 2 years, REJECT (Score < 30).
    3. SKILL SYNERGY:
       - Match must-have tools (e.g., Python, AWS, Docker). If the primary stack is missing, REJECT.

    ### SCORING SYSTEM:
    - 0-30: Total Mismatch (Wrong role or zero relevant experience)
    - 31-60: Weak Match (Right domain, but missing 50% of core tools/seniority)
    - 61-85: Strong Match (Has 80% of skills and correct seniority)
    - 86-100: Perfect Match (All skills + exact industry experience)

    RESUME: {resume_text}
    JD: {jd_text}

    OUTPUT: Provide ONLY the numerical score (0-100).
    """

QUESTION_PROMPT = """
You are a Senior Technical Interviewer. Your goal is to conduct a deep-dive technical assessment.

TASK:
Based on the provided CONTEXT (which includes both the Job Description and the Candidate's Resume), generate exactly 10 high-quality interview questions.

RULES:
- Be specific: Reference actual technologies and project names found in the context.
- No Fluff: Do not provide answers, introductions, or feedback.
- Difficulty: Senior-level. Focus on "Why" and "How" rather than "What is".

STRUCTURE:

### 1. Technical/Skill-Based Questions (5 questions)
Focus on the intersection of the JD requirements and the candidate's stated expertise. Challenge their understanding of the tools they claim to know.

### 2. Project-Based Questions (3 questions)
Select the most relevant projects from the resume. Ask about architecture, technical trade-offs, or specific challenges mentioned.

### 3. Scenario-Based Questions (2 questions)
Create hypothetical technical hurdles the candidate would face *in this specific role* based on the JD's responsibilities.

CONTEXT:
{context}
"""

# ========================= HELPER FUNCTIONS =========================
def load_pdf(uploaded_file):
    """Load PDF from Streamlit UploadedFile using temporary file"""
    if uploaded_file is None:
        return []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        return loader.load()
    finally:
        os.unlink(tmp_path)


def get_match_percentage(retriever):
    docs = retriever.invoke("resume job description match")
    context = "\n\n".join([d.page_content for d in docs])

    response = client.chat.complete(
        model=MODEL,
        messages=[{"role": "user", "content": MATCH_PROMPT.format(context=context)}]
    )
    content = response.choices[0].message.content.strip()
    try:
        return float(content)
    except ValueError:
        return 50.0  # fallback


def generate_questions(retriever):
    
    jd_docs     = retriever.invoke("core job responsibilities key requirements must-have skills technologies")
    resume_docs = retriever.invoke("candidate projects experiences achievements tools used skills demonstrated")
    combined    = jd_docs[:4] + resume_docs[:4]   # bias toward 4+4 or adjust
    context = "\n\n".join([d.page_content for d in combined])

    response = client.chat.complete(
        model=MODEL,
        messages=[{"role": "user", "content": QUESTION_PROMPT.format(context=context)}]
    )
    return response.choices[0].message.content


def create_match_gauge(percentage: float):
    color = "#22c55e" if percentage >= 60 else "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Match Score", 'font': {'size': 28}},
        delta={'reference': 60, 'increasing': {'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 40], 'color': "#991b1b"},
                {'range': [40, 60], 'color': "#b45309"},
                {'range': [60, 80], 'color': "#166534"},
                {'range': [80, 100], 'color': "#052e16"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 6},
                'thickness': 0.8,
                'value': 60
            }
        }
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "system-ui"}
    )
    return fig


def rag_pipeline(resume_docs, jd_docs):
    documents = resume_docs + jd_docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    match_pct = get_match_percentage(retriever)

    if match_pct < 60:
        return match_pct, None, "❌ Match below 60% – Candidate rejected"
    
    questions = generate_questions(retriever)
    return match_pct, questions, "✅ Strong match! Interview questions ready"


# ========================= MAIN UI =========================
col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader(
        "📄 Upload Candidate Resume (PDF)",
        type="pdf",
        help="PDF format only"
    )

with col2:
    jd_file = st.file_uploader(
        "📋 Upload Job Description (PDF)",
        type="pdf",
        help="PDF format only"
    )

if st.button("🔥 Analyze Resume vs JD", type="primary", use_container_width=True):
    if not resume_file or not jd_file:
        st.error("⚠️ Please upload both the Resume and Job Description")
        st.stop()

    with st.spinner("📚 Loading PDFs • 🧠 Chunking • 🔍 Embedding • ⚡ Calling Mistral AI..."):
        try:
            resume_docs = load_pdf(resume_file)
            jd_docs = load_pdf(jd_file)

            match_pct, questions, status_msg = rag_pipeline(resume_docs, jd_docs)

            # --------------------- RESULTS ---------------------
            st.divider()
            st.subheader("📊 Match Analysis")

            gauge_col, info_col = st.columns([2, 1])
            
            with gauge_col:
                fig = create_match_gauge(match_pct)
                st.plotly_chart(fig, use_container_width=True)

            with info_col:
                st.metric(
                    label="Match Percentage",
                    value=f"{match_pct:.1f}%",
                    delta=None
                )
                if match_pct >= 80:
                    st.success("🌟 Outstanding Match – Highly Recommended")
                elif match_pct >= 60:
                    st.success("✅ Good Match – Proceed to Interview")
                else:
                    st.error("❌ Low Match – Consider rejecting")

            st.info(status_msg)

            if questions:
                st.divider()
                st.subheader("🗣️ Tailored Interview Questions")
                st.markdown(questions)

                st.caption("💡 Questions are generated based on the actual content of the resume and JD.")

        except Exception as e:
            st.error(f"❌ Something went wrong: {str(e)}")
            st.info("Tip: Make sure PDFs are not password-protected and contain text (not scanned images).")

# ========================= FOOTER =========================
st.caption("Made with ❤️ using Mistral AI • LangChain • FAISS • Streamlit • Plotly")