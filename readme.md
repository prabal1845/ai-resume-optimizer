# AI Resume Optimizer

Live demo: https://ai-resume-optimizer-tyg3awr2j3m2zcdtydtuwe.streamlit.app/

A tool that compares your resume against a job description and tells you how well it matches. Upload a resume PDF and paste in a job description, and it calculates an ATS-style score, shows which keywords from the JD are missing from your resume, and uses Google Gemini to rewrite your resume so it's more aligned with the role.

## What it does

- Upload a resume as PDF
- Get an overall ATS match score against the job description
- See which JD keywords are already in your resume and which ones aren't
- Get a per-section breakdown (Summary, Skills, Experience, Projects) if the resume has clear section headers
- Get an AI-rewritten version of the resume with stronger action verbs, added metrics, and missing keywords worked in
- Download the rewritten resume as a PDF

## How the scoring works

Both the resume and job description are converted into TF-IDF vectors, and cosine similarity is used to measure how close they are. Raw similarity scores tend to fall in a fairly narrow range, so they're rescaled to a 0-100 scale that's easier to read.

Section-level scores work a bit differently — since sections like Skills or Projects are short, cosine similarity alone often underscores them even when they contain the right keywords. So section scores blend in a keyword-overlap check as well, weighted by how important each keyword is in the job description.

## Tech used

Python, Streamlit, Google Gemini, Scikit-learn, PyPDF2, ReportLab, python-dotenv

## Running it locally

```
git clone https://github.com/prabal1845/ai-resume-optimizer.git
cd ai-resume-optimizer
pip install -r requirements.txt
```

Create a `.env` file with:
```
GEMINI_API_KEY=your_api_key_here
```

Then run:
```
python -m streamlit run app.py
```

## Project structure

```
ai-resume-optimizer/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Why I built this

Applying to SDE and ML roles means tailoring your resume for every JD, which gets tedious fast. This automates the "does my resume actually match this posting" check and gives concrete suggestions instead of guessing.

## Possible improvements

- Better formatting control for the generated resume
- Support for DOCX uploads, not just PDF
- Resume version history
- More advanced section detection for non-standard resume formats
