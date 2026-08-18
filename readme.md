\# 📄 AI Resume Optimizer



An AI-powered resume optimization tool that analyzes a resume against a job description, calculates an ATS-style match score, identifies important keywords, and generates an optimized version of the resume using Google Gemini.



\## 🚀 Features



\* 📄 Upload resumes in PDF format

\* 🎯 Calculate an ATS-style resume match score

\* ✅ Identify keywords found in the resume

\* ❌ Identify important missing keywords

\* 📊 Analyze resume sections individually

\* 🤖 Rewrite and optimize resume content using Google Gemini

\* 📥 Download the optimized resume as a PDF

\* 💼 Tailor resume content to a specific job description



\## 🛠️ Tech Stack



\* \*\*Python\*\*

\* \*\*Streamlit\*\* — Web application interface

\* \*\*Google Gemini\*\* — AI-powered resume optimization

\* \*\*Scikit-learn\*\* — TF-IDF and cosine similarity for ATS scoring

\* \*\*PyPDF2\*\* — PDF text extraction

\* \*\*ReportLab\*\* — PDF generation

\* \*\*python-dotenv\*\* — Environment variable management



\## 🧠 How It Works



The application follows this workflow:



```text

Resume PDF

&#x20;    ↓

Extract Resume Text

&#x20;    ↓

Job Description

&#x20;    ↓

TF-IDF + Cosine Similarity

&#x20;    ↓

ATS Score + Keyword Analysis

&#x20;    ↓

Missing Keyword Detection

&#x20;    ↓

Google Gemini

&#x20;    ↓

Optimized Resume

&#x20;    ↓

Download PDF

```



\## 📊 ATS Scoring



The application uses \*\*TF-IDF vectorization and cosine similarity\*\* to compare the resume with the job description.



It also performs keyword analysis to identify important job-description terms that appear in or are missing from the resume.



For recognizable resume sections, the application provides individual scores for areas such as:



\* Summary

\* Experience

\* Skills

\* Projects



\## 🤖 AI Resume Optimization



After analyzing the resume, Google Gemini is used to:



\* Rewrite resume bullet points

\* Use stronger action verbs

\* Add impact metrics where possible

\* Naturally incorporate missing keywords

\* Keep the resume professional and ATS-friendly



\## 📥 PDF Generation



The optimized resume can be converted into a downloadable PDF directly from the application.



\## ⚙️ Installation



\### 1. Clone the repository



```bash

git clone https://github.com/YOUR-USERNAME/ai-resume-optimizer.git

cd ai-resume-optimizer

```



\### 2. Install dependencies



```bash

pip install -r requirements.txt

```



\### 3. Configure your Gemini API key



Create a `.env` file in the project directory:



```text

GEMINI\_API\_KEY=your\_api\_key\_here

```



\*\*Never commit your `.env` file or API key to GitHub.\*\*



\### 4. Run the application



```bash

python -m streamlit run app.py

```



The application will open locally in your browser.



\## 📁 Project Structure



```text

ai-resume-optimizer/

│

├── app.py

├── requirements.txt

├── .gitignore

└── README.md

```



\## 🎯 Use Case



This project is designed to help job seekers understand how closely their resume matches a specific job description and improve their resume accordingly.



It can be particularly useful when applying for:



\* Software Development Engineer roles

\* Machine Learning Engineer roles

\* AI/ML internships

\* Data Science roles

\* Other technical positions



\## 🔐 Security



API credentials are stored using environment variables and excluded from version control using `.gitignore`.



Never share or commit your Gemini API key.



\## 🔮 Future Improvements



Potential future improvements include:



\* Resume formatting improvements

\* Job-description recommendations

\* Multiple resume templates

\* More advanced ATS analysis

\* Resume version history

\* Support for additional document formats

\* Deployment to a public cloud platform



\## 👨‍💻 Project



Built as a practical AI/ML application combining natural language processing, machine learning, and generative AI.



