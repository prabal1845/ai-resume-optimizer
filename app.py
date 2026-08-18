import streamlit as st
import google.generativeai as genai
import PyPDF2
import re
import os
from io import BytesIO
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="AI Resume Optimizer", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .keyword-found { color: green; font-weight: bold; }
    .keyword-missing { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Optimizer")
st.subheader("Rewrite your resume bullets to match any job description")
st.divider()

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📁 Upload your Resume (PDF)", type="pdf")
with col2:
    job_desc = st.text_area("📋 Paste the Job Description here", height=200)


def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Section headers this tool recognizes when splitting a resume.
# Add more synonyms here if a resume uses a header not in this list.
SECTION_HEADERS = [
    "PROFESSIONAL SUMMARY", "SUMMARY", "OBJECTIVE", "CAREER OBJECTIVE",
    "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE", "EXPERIENCE",
    "EDUCATION",
    "TECHNICAL SKILLS", "KEY SKILLS", "SKILLS",
    "PROJECTS", "ACADEMIC PROJECTS",
    "CERTIFICATIONS", "CERTIFICATES",
    "EXTRACURRICULAR ACTIVITIES", "EXTRACURRICULAR",
    "ACHIEVEMENTS", "AWARDS",
]

# Sections that are meaningful to score against a job description.
# Education/Certifications/Extracurricular are excluded — comparing them
# to a JD via keyword overlap doesn't produce a meaningful signal.
SCORABLE_SECTIONS = {
    "PROFESSIONAL SUMMARY", "SUMMARY", "OBJECTIVE", "CAREER OBJECTIVE",
    "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE", "EXPERIENCE",
    "TECHNICAL SKILLS", "KEY SKILLS", "SKILLS",
    "PROJECTS", "ACADEMIC PROJECTS",
}


def split_into_sections(text, headers=SECTION_HEADERS):
    """
    Splits resume text into sections based on common header lines.
    Works when a header appears alone on its own line (true for the
    vast majority of standard resume templates and PDF text extraction).
    Returns {} if no recognized headers are found, so callers should
    fall back to whole-resume scoring in that case.
    """
    pattern = r'(?im)^\s*(' + '|'.join(re.escape(h) for h in headers) + r')\s*$'
    matches = list(re.finditer(pattern, text))

    sections = {}
    for i, match in enumerate(matches):
        header = match.group(1).upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[header] = content
    return sections


def get_score_for_text(text, vectorizer, jd_vector, min_sim, max_sim, jd_keyword_weights=None):
    """
    Shared scoring logic: TF-IDF cosine similarity normalized to 0-100.

    For short sections (e.g. Projects, Skills), pure cosine similarity
    against the JD often collapses to ~0 even when the section legitimately
    names several JD-relevant keywords, since there just isn't enough text
    to build up vocabulary overlap. To avoid those sections reading as a
    flat 0%, we blend in a keyword-overlap score when jd_keyword_weights
    is provided (used for section-level scoring, not overall scoring).

    jd_keyword_weights should be the TOP N (kw, tfidf_weight) pairs from the
    JD only — not the full ~200-word vectorizer vocabulary. Using the full
    vocabulary as the denominator was the earlier bug: it buries real
    keyword hits under hundreds of low-signal words, so even a section that
    names 8-10 core JD terms scores only a few percent. Weighting by each
    keyword's TF-IDF importance (instead of counting every match equally)
    also means matching a JD's most emphasized terms counts more than
    matching a word the JD only mentions once.
    """
    vec = vectorizer.transform([text])
    raw = cosine_similarity(vec, jd_vector)[0][0]
    normalized = (raw - min_sim) / (max_sim - min_sim)
    cosine_score = max(0, min(normalized, 1)) * 100

    if jd_keyword_weights:
        text_lower = text.lower()
        total_weight = sum(w for _, w in jd_keyword_weights)
        matched_weight = sum(w for kw, w in jd_keyword_weights if kw in text_lower)
        keyword_score = (matched_weight / total_weight) * 100 if total_weight > 0 else 0
        return int(0.4 * cosine_score + 0.6 * keyword_score)

    return int(cosine_score)


def get_ats_score(resume_text, job_desc):
    """
    Computes:
      1. An overall ATS score for the whole resume.
      2. A per-section breakdown (Summary, Experience, Skills, Projects)
         when recognizable section headers are found.
      3. Matched/missing keywords based on the JD's own vocabulary.

    Calibration bounds were derived empirically by testing this scoring
    function against resume/JD pairs of known quality (see comments below
    for the reference points used):

    OVERALL resume bounds (min=0.10, max=0.95):
      - Unrelated resume vs. ML/Data Science JD:        raw sim ~0.11
      - General CS resume, light on ML:                  raw sim ~0.54
      - ML-focused resume w/ relevant internships:        raw sim ~0.61
      - Resume that closely mirrors the JD's language:    raw sim ~0.95

    SECTION-level bounds (min=0.0, max=0.65) — sections are much shorter
    than a full resume, so they carry less vocabulary overlap by nature.
    Weak/moderate/strong test sections (Summary, Skills, Experience,
    Projects) topped out around raw sim 0.55-0.67 for genuinely strong
    matches, and 0.0 for sections sharing no vocabulary with the JD at all.

    Section scores are further blended with direct keyword-hit-rate
    (see get_score_for_text) so that short sections like Projects aren't
    unfairly flattened to 0% just for lacking broad phrasing overlap.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
    vectorizer.fit([job_desc])
    jd_vector = vectorizer.transform([job_desc])

    overall_score = get_score_for_text(resume_text, vectorizer, jd_vector,
                                        min_sim=0.10, max_sim=0.95)

    feature_names = vectorizer.get_feature_names_out()
    jd_array = jd_vector.toarray()[0]
    jd_keyword_scores = [(kw, jd_array[i]) for i, kw in enumerate(feature_names) if jd_array[i] > 0]
    jd_keyword_scores.sort(key=lambda x: x[1], reverse=True)

    # Only the JD's top, highest-weight keywords are used for section-level
    # keyword-overlap scoring — using the full ~200-word vocabulary here
    # was the earlier bug (real matches got buried under the denominator).
    top_jd_keywords = jd_keyword_scores[:25]

    sections = split_into_sections(resume_text)
    section_scores = {}
    section_debug = {}
    for header, content in sections.items():
        if header in SCORABLE_SECTIONS:
            section_scores[header] = get_score_for_text(
                content, vectorizer, jd_vector,
                min_sim=0.0, max_sim=0.65, jd_keyword_weights=top_jd_keywords
            )
            content_lower = content.lower()
            hit_kws = [kw for kw, _ in top_jd_keywords if kw in content_lower]
            miss_kws = [kw for kw, _ in top_jd_keywords if kw not in content_lower]
            section_debug[header] = {
                "content_preview": content[:300],
                "top_keywords_hit": hit_kws,
                "top_keywords_missed": miss_kws,
            }

    resume_lower = resume_text.lower()
    matched = [kw for kw, _ in jd_keyword_scores if kw in resume_lower][:15]
    missing = [kw for kw, _ in jd_keyword_scores if kw not in resume_lower][:15]

    return overall_score, section_scores, matched, missing, section_debug


def create_pdf(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=inch * 0.8,
        leftMargin=inch * 0.8,
        topMargin=inch * 0.8,
        bottomMargin=inch * 0.8
    )

    styles = getSampleStyleSheet()
    style_normal = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=4)
    style_h1 = ParagraphStyle('CustomH1', parent=styles['Heading1'], fontSize=16, spaceAfter=8)
    style_h2 = ParagraphStyle('CustomH2', parent=styles['Heading2'], fontSize=13, spaceAfter=6)
    style_h3 = ParagraphStyle('CustomH3', parent=styles['Heading3'], fontSize=11, spaceAfter=4)

    story = []

    for line in text.split('\n'):
        line = line.strip()

        if line == '':
            story.append(Spacer(1, 6))
            continue

        line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        line = line.replace('*', '')

        if line.startswith('### '):
            story.append(Paragraph(line[4:], style_h3))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], style_h2))
        elif line.startswith('# '):
            story.append(Paragraph(line[2:], style_h1))
        elif line.startswith('- ') or line.startswith('• '):
            story.append(Paragraph(f"• {line[2:]}", style_normal))
        else:
            story.append(Paragraph(line, style_normal))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def score_label(score):
    if score >= 70:
        return "🟢 Strong"
    elif score >= 40:
        return "🟡 Moderate"
    else:
        return "🔴 Weak"


if st.button("🚀 Optimize My Resume", use_container_width=True):
    if not uploaded_file:
        st.error("Please upload your resume PDF")
    elif not job_desc:
        st.error("Please paste the job description")
    else:
        resume_text = extract_pdf_text(uploaded_file)

        with st.spinner("Calculating ATS Score..."):
            overall_score, section_scores, matched_kw, missing_kw, section_debug = get_ats_score(resume_text, job_desc)

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Overall ATS Score", f"{overall_score}%")
            if overall_score >= 70:
                st.success("Strong match!")
            elif overall_score >= 40:
                st.warning("Moderate match")
            else:
                st.error("Weak match")

        with col2:
            st.markdown("**✅ Keywords Found:**")
            for kw in matched_kw:
                st.markdown(f'<span class="keyword-found">✓ {kw}</span>', unsafe_allow_html=True)

        with col3:
            st.markdown("**❌ Missing Keywords:**")
            for kw in missing_kw:
                st.markdown(f'<span class="keyword-missing">✗ {kw}</span>', unsafe_allow_html=True)

        st.divider()

        if section_scores:
            st.markdown("### 📊 Section-by-Section Breakdown")
            st.caption("Shows which parts of your resume align well with the job description, and which need work.")
            sec_cols = st.columns(len(section_scores))
            for col, (header, score) in zip(sec_cols, section_scores.items()):
                with col:
                    st.metric(header.title(), f"{score}%")
                    st.caption(score_label(score))
            st.divider()
        else:
            st.info("Couldn't detect standard section headers (Summary, Skills, Experience, Projects) in your resume, "
                     "so only the overall score is shown. Make sure each section has a clear header on its own line.")
            st.divider()

        with st.spinner("AI is optimizing your resume..."):
            prompt = f"""
            You are an expert resume writer.
            Resume: {resume_text}
            Job Description: {job_desc}
            Tasks:
            1. Rewrite resume bullet points to match the job description
            2. Add impact metrics where possible
            3. Use strong action verbs
            4. Add these missing keywords naturally: {missing_kw}
            5. Keep it ATS friendly and professional
            Format clearly with sections: Summary, Skills, Projects, Experience
            """
            response = model.generate_content(prompt)
            optimized_text = response.text

        st.success("✅ Done! Here is your optimized resume:")
        st.markdown(optimized_text)
        st.divider()

        pdf_data = create_pdf(optimized_text)
        st.download_button(
            label="📥 Download Optimized Resume (PDF)",
            data=pdf_data,
            file_name="optimized_resume.pdf",
            mime="application/pdf",
            use_container_width=True
        )