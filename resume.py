import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with all resume texts
    documents = [job_description] + resumes
    
    # Convert text to numerical vectors using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # First vector is job description, rest are resumes
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]

    # Compute cosine similarity between job description and resumes
    similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return similarities

# Streamlit UI
st.title("AI-Powered Resume Screening & Ranking System")

# Input Job Description
job_description = st.text_area("Enter Job Description", height=150)

# Upload Resumes (PDFs)
uploaded_files = st.file_uploader("Upload Resumes (PDF only)", accept_multiple_files=True, type=['pdf'])

if st.button("Rank Resumes"):
    if not job_description.strip():
        st.error("Please enter the job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        # Extract text from each uploaded PDF resume
        resume_texts = [extract_text_from_pdf(file) for file in uploaded_files]
        resume_names = [file.name for file in uploaded_files]

        # Rank resumes
        scores = rank_resumes(job_description, resume_texts)

        # Combine names and scores into DataFrame
        results_df = pd.DataFrame({
            'Resume': resume_names,
            'Match Score': scores
        }).sort_values(by='Match Score', ascending=False)

        # Display Results
        st.write("### Ranking Results:")
        st.write(results_df)

        # Optional: Highlight Top Candidate
        top_candidate = results_df.iloc[0]
        st.success(f"**Top Candidate: {top_candidate['Resume']} with Score: {top_candidate['Match Score']:.2f}**")
