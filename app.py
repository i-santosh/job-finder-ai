import streamlit as st
import PyPDF2
from resume_classifier_module import predict_resume_category
import requests
# Set page configuration
st.set_page_config(
    page_title="Resume Based Job Finder",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #424242;
            margin-bottom: 1rem;
        }
        .result-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Main Streamlit app
def main():
    local_css()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/resume.png", width=100)
        st.markdown("## Navigation")
        page = st.radio("Navigation Menu", ["Home", "About", "How It Works"])
    
    if page == "Home":
        st.markdown("<h1 class='main-header'>Resume Based Job Finder</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Upload a resume PDF or paste resume text to find the best job for you</p>", unsafe_allow_html=True)
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        resume_text = None
        
        # Create tabs for different input methods
        with col1:
            st.markdown("### Input Methods")
            tab1, tab2 = st.tabs(["📤 Upload PDF", "✏️ Paste Text"])
            
            with tab1:
                uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
                if uploaded_file is not None:
                    with st.spinner("Extracting text from PDF..."):
                        resume_text = extract_text_from_pdf(uploaded_file)
                        with st.expander("View Extracted Text"):
                            st.text_area("Extracted Text", resume_text, height=200, label_visibility="collapsed")
            
            with tab2:
                text_input = st.text_area("Paste resume text here", height=300)
                if text_input:
                    resume_text = text_input
        
        with col2:
            st.markdown("### Results")
            if resume_text:
                if st.button("Classify Resume", key="classify_btn"):
                    with st.spinner("Analyzing resume..."):
                        category, confidence = predict_resume_category(resume_text)
                        st.markdown(f"""
                        <div class='result-box'>
                            <h3>Analysis Complete</h3>
                            <p>Based on your resume content, we recommend:</p>
                            <h2 style='color:#1E88E5; text-align:center;'>{category}</h2>
                            <p style='text-align:center; font-size:1rem; font-weight:bold; background-color: #1E88E5; color: white; padding: 5px; border-radius: 5px; width: 30%; margin: 0 auto;'>Confidence: {int(confidence*100)}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info("❇️ Available Jobs for This Category")
                        
                        available_jobs = get_jobs_for_category(category)
                        if available_jobs:
                            for job in available_jobs:
                                with st.expander(f"{job['title']} at {job['company_name']}"):
                                    st.markdown(f"**Company:** {job['company_name']}")
                                    if job.get('company_logo'):
                                        st.image(job['company_logo'], width=100)
                                    st.markdown(f"**Job Type:** {job.get('job_type', 'Not specified')}")
                                    st.markdown(f"**Location:** {job.get('candidate_required_location', 'Remote')}")
                                    if job.get('salary'):
                                        st.markdown(f"**Salary:** {job['salary']}")
                                    st.markdown(f"**Tags:** {', '.join(job.get('tags', []))}")
                                    st.markdown("**Description:**")
                                    st.markdown(job.get('description', 'No description available'), unsafe_allow_html=True)
                                    st.markdown(f"👉[Apply for this job]({job.get('url', '#')})", unsafe_allow_html=True)
                        else:
                            st.warning("No jobs found for this category.")
            else:
                st.info("Please upload a resume or paste text to get started")
                
    elif page == "About":
        st.markdown("<h1 class='main-header'>About</h1>", unsafe_allow_html=True)
        st.write("""
        This application uses machine learning to analyze resumes and find the most suitable job categories and positions.
        
        The classifier has been trained on thousands of real-world resumes across various industries and job functions.
        
        This tool can help:
        - Job seekers discover the best job matches based on their skills and experience
        - Identify potential career paths based on your resume content
        - Get recommendations for specific job titles that match your profile
        """)
        
    elif page == "How It Works":
        st.markdown("<h1 class='main-header'>How It Works</h1>", unsafe_allow_html=True)
        st.write("""
        1. **Upload or paste your resume**: The system extracts and processes the text
        2. **Text processing**: We clean and prepare the text for analysis
        3. **Classification**: Our machine learning model analyzes your skills and experience
        4. **Job Matching**: The system identifies the best job categories and specific positions for your profile
        """)
        
        with st.expander("Technical Details"):
            st.write("""
            The classification model uses natural language processing techniques to analyze resume content.
            Features extracted from the text include skills, experience, education, and specific keywords.
            """)

# Helper function to get job titles based on category
def get_job_titles_for_category(category):
    job_titles_map = {
        "Data Science": ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "AI Researcher"],
        "HR": ["HR Manager", "Talent Acquisition Specialist", "HR Business Partner", "Recruiter"],
        "Advocate": ["Legal Counsel", "Corporate Lawyer", "Legal Advisor", "Attorney"],
        "Arts": ["Graphic Designer", "Creative Director", "Content Creator", "Artist"],
        "Web Designing": ["UI/UX Designer", "Frontend Developer", "Web Developer", "WordPress Developer"],
        "Accounts": ["Accountant", "Financial Analyst", "Auditor", "Bookkeeper"],
        "Sales": ["Sales Representative", "Business Development Manager", "Account Executive", "Sales Manager"],
        "Health and fitness": ["Fitness Trainer", "Nutritionist", "Health Coach", "Wellness Coordinator"],
        "Aviation": ["Pilot", "Flight Attendant", "Aviation Technician", "Air Traffic Controller"],
        "Teaching": ["Teacher", "Professor", "Curriculum Developer", "Education Consultant"],
        "Engineering": ["Software Engineer", "Mechanical Engineer", "Civil Engineer", "Electrical Engineer"],
        "Automobile": ["Automotive Engineer", "Mechanical Designer", "Vehicle Technician", "Production Manager"],
        "Business-Development": ["Business Development Manager", "Strategic Partnership Director", "Growth Specialist", "Market Expansion Lead"],
        "Chef": ["Head Chef", "Sous Chef", "Pastry Chef", "Culinary Director"],
        "Finance": ["Financial Analyst", "Investment Banker", "Portfolio Manager", "Financial Planner"],
        "Healthcare": ["Physician", "Nurse", "Medical Technician", "Healthcare Administrator"],
        "Construction": ["Project Manager", "Civil Engineer", "Construction Supervisor", "Architect"],
        "Digital-Media": ["Social Media Manager", "Digital Marketing Specialist", "Content Strategist", "SEO Expert"],
        "BPO": ["Call Center Representative", "Customer Service Agent", "Process Associate", "Team Leader"],
        "Agriculture": ["Agronomist", "Farm Manager", "Agricultural Engineer", "Food Technologist"],
        "Media": ["Journalist", "Content Producer", "Public Relations Specialist", "Communications Manager"],
        "Consultant": ["Management Consultant", "Strategy Consultant", "IT Consultant", "HR Consultant"],
        "Designer": ["Product Designer", "Interior Designer", "Fashion Designer", "Industrial Designer"],
        "Public-Relations": ["PR Manager", "Communications Specialist", "Media Relations", "Brand Ambassador"],
        "Banking": ["Bank Manager", "Loan Officer", "Investment Banker", "Financial Advisor"]
    }
    
    return job_titles_map.get(category, ["No specific job titles available for this category"])

# Helper function to get jobs for category
def get_jobs_for_category(category):
    REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"

    url = f"{REMOTIVE_API_URL}?search={category}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        jobs = data.get("jobs", [])
        if jobs:
            return jobs[:5]
        else:
            return []
    return []

if __name__ == "__main__":
    main()