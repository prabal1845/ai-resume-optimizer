AI Resume Optimizer

 **Live Demo:** https://ai-resume-optimizer-tyg3awr2j3m2zcdtydtuwe.streamlit.app/

An AI-powered resume optimization tool that analyzes a resume against a job description, calculates an ATS-style match score, identifies important keywords, and generates an optimized version of the resume using Google Gemini.

FEATURES

* Upload resumes in PDF format
* Calculate an ATS-style resume match score
* Identify keywords found in the resume
* Identify important missing keywords
* Analyze resume sections individually
* Rewrite and optimize resume content using Google Gemini
* Download the optimized resume as a PDF
* Tailor resume content to a specific job description

TECHNOLOGIES USED

Python
Streamlit
Google Gemini
Scikit-learn
PyPDF2
ReportLab
python-dotenv

HOW IT WORKS

Resume PDF
↓
Extract Resume Text
↓
Job Description
↓
TF-IDF and Cosine Similarity
↓
ATS Score and Keyword Analysis
↓
Missing Keyword Detection
↓
Google Gemini
↓
Optimized Resume
↓
Download PDF

ATS SCORING

The application uses TF-IDF vectorization and cosine similarity to compare the resume with the job description.

It also performs keyword analysis to identify important job-description terms that appear in or are missing from the resume.

The application provides individual scores for recognizable resume sections such as:

Summary
Experience
Skills
Projects

AI RESUME OPTIMIZATION

Google Gemini is used to:

Rewrite resume bullet points
Use stronger action verbs
Add impact metrics where possible
Naturally incorporate missing keywords
Keep the resume professional and ATS-friendly

PDF GENERATION

The optimized resume can be converted into a downloadable PDF directly from the application.

INSTALLATION

1. Clone the repository

git clone https://github.com/YOUR-USERNAME/ai-resume-optimizer.git

cd ai-resume-optimizer

2. Install dependencies

pip install -r requirements.txt

3. Configure your Gemini API key

Create a .env file in the project directory.

GEMINI_API_KEY=your_api_key_here

Never commit your .env file or API key to GitHub.

4. Run the application

python -m streamlit run app.py

The application will open locally in your browser.

PROJECT STRUCTURE

ai-resume-optimizer/

app.py
requirements.txt
.gitignore
README.md

USE CASE

This project helps job seekers understand how closely their resume matches a specific job description and improve their resume accordingly.

It can be useful for:

Software Development Engineer roles
Machine Learning Engineer roles
AI/ML internships
Data Science roles
Other technical positions

SECURITY

API credentials are stored using environment variables and excluded from version control using .gitignore.

Never share or commit your Gemini API key.

FUTURE IMPROVEMENTS

Resume formatting improvements
Job-description recommendations
Multiple resume templates
More advanced ATS analysis
Resume version history
Support for additional document formats
Deployment to a public cloud platform

PROJECT

Built as a practical AI/ML application combining natural language processing, machine learning, and generative AI.
