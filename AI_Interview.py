
# %%
import os
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# %%
load_dotenv()               # reads .env file

# %%
api_key = os.getenv("MISTRAL_KEY")

if not api_key:
    raise ValueError("Missing MISTRAL_KEY in .env")
else:
    print('Key fetched')

# %%
client = Mistral(api_key=api_key)
MODEL = "mistral-small-latest"

# %%
# --------- Load PDFs ----------
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

# %%
def rag_impl(resume_docs, jd_docs):
  ''' Implements RAGs for the input resume and JD description'''

  documents = resume_docs + jd_docs

  # --------- Chunking ----------
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=100
  )
  chunks = splitter.split_documents(documents)

  # --------- Embeddings ----------
  embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2"
  )

  vectorstore = FAISS.from_documents(chunks, embeddings)
  retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

  match_pct = get_match_percentage(retriever)
  print(f"\nResume–JD Match: {match_pct}%")

  if match_pct < 60:
    print("❌ Match below 60%. Candidate rejected.")
  else:
    print("✅ Match above 60%. Generating interview questions...\n")
    questions = generate_questions(retriever)
    print(questions)

# %%
# --------- Match Percentage Prompt ----------
MATCH_PROMPT = """
You are an ATS system.

Given the CONTEXT below (resume + job description):
1. Calculate percentage match between resume and JD.
2. Consider skills, experience, tools, projects.
3. Output ONLY a number between 0 and 100.

CONTEXT:
{context}
"""

# %%
def get_match_percentage(retriever):
    docs = retriever.invoke("resume job description match")
    context = "\n".join([d.page_content for d in docs])

    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "user", "content": MATCH_PROMPT.format(context=context)}
        ]
    )
    return float(response.choices[0].message.content.strip())

# %%
# --------- Question Generation Prompt ----------
QUESTION_PROMPT = """
You are a technical interviewer.

Using the CONTEXT:
- Job description requirements
- Skills mentioned in resume
- Projects done by candidate

Generate:
1. 5 technical questions on the job description
2. 3 project-based questions on the projects done by candidate
3. 2 skill-based questions on the skills mentioned in resume
3. 2 scenario-based questions based on the job description

CONTEXT:
{context}
"""

# %%
def generate_questions(retriever):
    docs = retriever.invoke("skills projects requirements")
    context = "\n".join([d.page_content for d in docs])

    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "user", "content": QUESTION_PROMPT.format(context=context)}
        ]
    )
    return response.choices[0].message.content

# %%
# --------- Pipeline ----------
if __name__ == "__main__":
    
    print('in main')

  # Get file paths from user
    resume_path = r'C:\Users\Admin\Downloads\Git_Clone\AI-Tools\AI_Interview\data\Rajat__Sharma_AI_ML.pdf' # input("Enter resume PDF path (e.g. Rajat__Sharma_AI_ML.pdf): ").strip()
    jd_path = r'C:\Users\Admin\Downloads\Git_Clone\AI-Tools\AI_Interview\data\JD_ML.pdf' # input("Enter Job Description PDF path: ").strip()

    # Loading pdf files
    resume_docs = load_pdf(resume_path)
    jd_docs = load_pdf(jd_path)

    print('Documents fetched!')

    # Calling
    rag_impl(resume_docs, jd_docs)

    print("\nProcessing...\n")

# %%



