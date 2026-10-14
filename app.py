import streamlit as st
import engine as E
import rag

st.set_page_config(page_title="AI Resume Builder", page_icon="📄", layout="wide")

SAMPLE_RESUME = """Kavya K
kavya@email.com | +91 98765 43210 | linkedin.com/in/kavya | github.com/kavya

SUMMARY
Final-year CS student with 2 years of hands-on experience building Python and machine learning projects.

SKILLS
Python, SQL, Pandas, NumPy, Scikit-learn, Git, Streamlit, Problem Solving, Teamwork

EXPERIENCE
Data Science Intern, TechCorp (Jun 2024 - Dec 2024)
- Responsible for cleaning datasets and building dashboards
- Worked on a churn prediction model using Scikit-learn
- Built an internal tool with Streamlit used by 25 analysts
- Improved report generation speed by 40%

Machine Learning Project (2023 - 2024)
- Developed an NLP resume classifier with 88% accuracy
- Helped to deploy the model using Flask and Git

EDUCATION
B.Tech in Computer Science, 2021 - 2025
"""

SAMPLE_JD = """Machine Learning Engineer

Requirements:
- 2+ years of experience with Python and machine learning
- Strong SQL and Pandas skills
- Experience with NLP and LLMs
- Docker and Git for deployment
- Bachelor's degree in Computer Science

Nice to have:
- AWS or Azure
- Streamlit or Flask
- Unit testing and CI/CD
"""

st.title("📄 AI Resume Builder")
st.caption("LLM-based job matching, ATS audit and resume improvement - hybrid SLM/LLM pipeline")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Model mode", ["Demo Mode (offline)", "LLM Mode (OpenAI)"])
    api_key = st.text_input("OpenAI API key", type="password") if mode.startswith("LLM") else ""
    prompting = st.radio("Prompting strategy", ["Few-shot", "Zero-shot"])
    st.divider()
    use_sample = st.button("Load sample data")
    st.caption("Demo Mode never sends your resume anywhere.")

if use_sample:
    st.session_state["resume_box"], st.session_state["jd_box"] = SAMPLE_RESUME, SAMPLE_JD

c1, c2 = st.columns(2)
with c1:
    st.subheader("1. Your resume")
    up = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
    if up:
        st.session_state["resume_box"] = E.extract_text(up)
    resume = st.text_area("Resume text", height=300, key="resume_box")
with c2:
    st.subheader("2. Job description")
    jd = st.text_area("Paste the job description", height=370, key="jd_box")

if st.button("Analyze", type="primary", use_container_width=True):
    if not resume.strip() or not jd.strip():
        st.warning("Please provide both a resume and a job description (or click 'Load sample data').")
        st.stop()

    result = E.match_score(resume, jd)
    audit = E.ats_audit(resume, jd)
    passed = sum(a["pass"] for a in audit)

    t1, t2, t3, t4 = st.tabs(["🎯 Match Score", "✅ ATS Audit", "✨ Improved Resume", "🔎 Retrieved Evidence"])
    index = rag.ResumeIndex(resume)
    jd_skills = sorted(E.parse_jd(jd)["required"] | E.parse_jd(jd)["preferred"])
    evidence = rag.gather_evidence(index, jd_skills)

    with t1:
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall match", f"{result['total']} / 100")
        m2.metric("Skills matched", len(result["matched"]))
        m3.metric("Skills missing", len(result["missing"]))
        st.markdown("**Score breakdown (40 / 20 / 20 / 10 / 10)**")
        for name, got, mx in result["breakdown"]:
            st.write(f"{name}: **{got} / {mx}**")
            st.progress(min(1.0, got / mx))
        a, b = st.columns(2)
        a.success("**Matched:** " + (", ".join(result["matched"]) or "none"))
        b.error("**Missing:** " + (", ".join(result["missing"]) or "none"))
        if result["missing_required"]:
            st.info("Priority: add evidence of these **required** skills if you have them - "
                    + ", ".join(result["missing_required"]))

    with t2:
        st.metric("ATS checks passed", f"{passed} / {len(audit)}")
        st.progress(passed / len(audit))
        for a in audit:
            if a["pass"]:
                st.markdown(f"✅ {a['check']}")
            else:
                st.markdown(f"❌ **{a['check']}** - _{a['fix']}_")

    with t3:
        with st.spinner("Improving resume..."):
            if mode.startswith("LLM") and api_key:
                improved, changes = E.llm_rewrite(resume, jd, api_key, few_shot=(prompting == "Few-shot"),
                                                  evidence_text=rag.evidence_block(evidence))
            else:
                improved, changes = E.demo_rewrite(resume, result["missing"])
        if changes:
            st.markdown("**Changes made**")
            for old, new in changes:
                st.markdown(f"- ~~{old}~~  →  {new}")
        st.text_area("Improved resume", improved, height=350)
        d1, d2 = st.columns(2)
        d1.download_button("Download .txt", improved, "improved_resume.txt")
        d2.download_button("Download .docx", E.to_docx_bytes(improved), "improved_resume.docx")
        st.caption("Review the output before use - only keep skills and claims that are true for you.")

    with t4:
        st.caption(f"Resume split into {len(index.chunks)} chunks; top-{rag.TOP_K} retrieved per job skill "
                   "(TF-IDF + cosine similarity). These chunks ground the LLM prompt.")
        for skill, res in evidence.items():
            if res["supported"]:
                st.markdown(f"**{skill}** - evidence found")
                for chunk, score in res["hits"]:
                    st.code(f"[{score:.2f}] {chunk}", language=None)
            else:
                st.markdown(f"**{skill}** - ❌ no supporting evidence in resume")
        st.markdown("**Ask a question about the resume**")
        q = st.text_input("e.g. Where did the candidate use Streamlit?", key="rag_q")
        if q:
            r = index.answer(q)
            if r["supported"]:
                for chunk, score in r["hits"]:
                    st.code(f"[{score:.2f}] {chunk}", language=None)
            else:
                st.info("No supporting evidence found in the resume.")
