# AI Resume Builder (demo)

## Run it
    pip install -r requirements.txt
    streamlit run app.py

Opens at http://localhost:8501. Click "Load sample data" then "Analyze".
Files: engine.py (parsing, scoring, ATS audit, rewriting) | app.py (Streamlit UI)
Optional: choose "LLM Mode" in the sidebar and paste an OpenAI key (falls back to offline Demo Mode on error).

## Retrieval (RAG) layer
rag.py chunks the resume (one bullet per chunk, section heading attached), indexes it with TF-IDF and retrieves the
top-3 chunks by cosine similarity. These chunks are shown in the "Retrieved Evidence" tab and used to ground the LLM prompt.
