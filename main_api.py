# api.py
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
import tempfile

load_dotenv()

app = FastAPI(
    title="Resume ↔ Job Description Match & Question Generator",
    description="Upload resume + JD → get match % and interview questions",
    version="0.1.0"
)

@app.get("/")
async def root():
    return """
    <h1>Resume ↔ Job Description AI Matcher</h1>
    <p>API is running successfully!</p>
    <ul>
        <li>Interactive docs → <a href="/docs">/docs</a></li>
        <li>Try the frontend → open <code>index.html</code> in your browser</li>
    </ul>
    """

api_key = os.getenv("MISTRAL_KEY")
if not api_key:
    raise ValueError("MISTRAL_KEY missing in .env")

client = Mistral(api_key=api_key)
MODEL = "mistral-small-latest"

# ── Prompts (same as yours) ────────────────────────────────────────
MATCH_PROMPT = """You are an ATS system.

Given the CONTEXT below (resume + job description):
1. Calculate percentage match between resume and JD.
2. Consider skills, experience, tools, projects.
3. Output ONLY a number between 0 and 100.

CONTEXT:
{context}
"""

QUESTION_PROMPT = """You are a technical interviewer.

Using the CONTEXT:
- Job description requirements
- Skills mentioned in resume
- Projects done by candidate

Generate:
1. 5 technical questions on the job description
2. 3 project-based questions on the projects done by candidate
3. 2 skill-based questions on the skills mentioned in resume
4. 2 scenario-based questions based on the job description

Return ONLY the list of questions, nicely formatted.

CONTEXT:
{context}
"""

# ── Models for response ─────────────────────────────────────────────
class AnalysisResult(BaseModel):
    match_percentage: float
    status: str                        # "accepted" | "rejected"
    questions: str | None = None       # only when accepted
    message: str

# ── Helpers ─────────────────────────────────────────────────────────
def pdf_to_docs(uploaded_file: UploadFile) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        os.unlink(tmp_path)           # clean up

    return docs


def build_vectorstore(docs: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


def get_match_percentage(retriever):
    docs = retriever.invoke("resume job description match percentage skills experience")
    context = "\n\n".join(d.page_content for d in docs)

    resp = client.chat.complete(
        model=MODEL,
        messages=[{"role": "user", "content": MATCH_PROMPT.format(context=context)}]
    )
    try:
        return float(resp.choices[0].message.content.strip())
    except:
        return 0.0


def generate_questions(retriever):
    docs = retriever.invoke("skills projects job requirements")
    context = "\n\n".join(d.page_content for d in docs)

    resp = client.chat.complete(
        model=MODEL,
        messages=[{"role": "user", "content": QUESTION_PROMPT.format(context=context)}]
    )
    return resp.choices[0].message.content.strip()


# ── API Endpoints ───────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_resume_jd(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...)
):
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Resume must be a PDF")
    if not job_description.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Job description must be a PDF")

    try:
        resume_docs  = pdf_to_docs(resume)
        jd_docs      = pdf_to_docs(job_description)

        all_docs = resume_docs + jd_docs

        vectorstore = build_vectorstore(all_docs)
        retriever   = vectorstore.as_retriever(search_kwargs={"k": 7})

        match_pct = get_match_percentage(retriever)

        result = {
            "match_percentage": match_pct,
            "status": "accepted" if match_pct >= 60 else "rejected",
            "message": f"Match: {match_pct:.1f}%"
        }

        if match_pct >= 60:
            questions = generate_questions(retriever)
            result["questions"] = questions
        else:
            result["questions"] = None
            result["message"] += " → below threshold, no questions generated."

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL}