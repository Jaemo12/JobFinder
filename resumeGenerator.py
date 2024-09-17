import json
import os
import re
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.colors import black, grey, HexColor
import google.generativeai as genai
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def extract_keywords(text, top_n=20):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    custom_stopwords = {'experience', 'year', 'skill', 'ability', 'work', 'job', 'candidate', 'position'}
    stop_words.update(custom_stopwords)
    
    # Add software development related terms to be preserved
    software_dev_terms = {'python', 'java', 'javascript', 'c++', 'react', 'angular', 'vue', 'node.js', 'express', 
                          'django', 'flask', 'spring', 'docker', 'kubernetes', 'aws', 'azure', 'git', 'agile', 
                          'scrum', 'ci/cd', 'rest', 'api', 'microservices', 'database', 'sql', 'nosql', 'mongodb', 
                          'postgresql', 'machine learning', 'ai', 'cloud', 'devops', 'frontend', 'backend', 'fullstack'}
    
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens 
                       if (word.isalnum() and word not in stop_words) or word in software_dev_terms]
    
    bigrams = list(nltk.bigrams(filtered_tokens))
    trigrams = list(nltk.trigrams(filtered_tokens))
    
    all_terms = (filtered_tokens + 
                 [' '.join(bg) for bg in bigrams] + 
                 [' '.join(tg) for tg in trigrams])
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(all_terms)])
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    sorted_terms = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    
    # Prioritize software development terms
    job_keywords = sorted(
        [term for term, score in sorted_terms if len(term) > 2],
        key=lambda x: x in software_dev_terms,
        reverse=True
    )
    
    return job_keywords[:top_n]

def generate_summary(job_description, personal_data):
    prompt = f"""
    Job Description: {job_description}

    Candidate Information:
    {json.dumps(personal_data, indent=2)}

    Task: Create a powerful and concise and short professional summary in 2 short sentences that showcases the candidate's most relevant skills and experiences for the target job.

    Guidelines:
    1. Analyze the job description and identify 3-5 key requirements or skills.
    2. Match these requirements with the candidate's most relevant experiences and achievements.
    3. Include specific technical skills and methodologies mentioned in both the job description and candidate's information.
    4. Incorporate 1-2 quantifiable achievements that directly relate to the job requirements.
    5. Use industry-specific terminology relevant to the position.
    6. Ensure the language is clear, direct, and free of vague generalities.
    7. Avoid overused phrases like "results-driven," "team player," or "passionate about."
    8. Focus on unique value propositions that set the candidate apart.
    9. Use active voice and strong action verbs.
    10. Ensure the summary is ATS-friendly by naturally incorporating key job-specific terms.
    11. dont use personal pronouns like "I," "my," or "we."

    Output Format: A single paragraph (2 sentences) that serves as a compelling, job-specific, and ATS-friendly professional summary. The summary should be concrete, achievement-oriented, and directly relevant to the target position.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

def optimize_content(content, job_keywords, content_type, used_verbs):
    prompt = f"""
    {content_type}: {json.dumps(content)}
    Job Keywords: {', '.join(job_keywords)}
    Previously used verbs: {', '.join(used_verbs)}

    Rewrite the provided {content_type.lower()} for a software developer role, following these guidelines:

    1. Create {'3-4' if content_type == 'Project' else '4-5'} high-impact bullet points.
    2. Use active voice, powerful action verb not previously used in this resume or listed in the 'Previously used verbs'.
    3. Naturally incorporate specific job keywords throughout the text, especially those related to software development.
    4. Include a quantifiable achievement in EVERY bullet point (use specific percentages, numbers, time frames).
    5. Highlight technical skills, programming languages, and methodologies relevant to software development.
    6. Emphasize problem-solving abilities, innovative solutions, and technical challenges overcome.
    7. Showcase collaboration in development teams and technical leadership experiences when applicable.
    8. Ensure each point demonstrates clear value and impact to the organization's software development efforts.
    9. Use concise, direct language without repetition, filler words, or clichés.
    10. Tailor the content to showcase software development skills and experiences most relevant to the target job.
    11. Ensure the content is ATS-friendly by naturally incorporating key job-specific terms.
    12.Ensure the bullet points are concise and short and direct
    11. dont use personal pronouns like "I," "my," or "we."
    12. do not repeat any words use synonyms or something similar

    Prioritize technical accuracy, readability, and impact. Each bullet point should clearly communicate a distinct software development accomplishment or skill.

    Output: A list of {'3' if content_type == 'Project' else '4'} bullet points, optimized for ATS parsing and software development roles. Do not include any introductory or explanatory text.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    # Clean up unwanted characters and phrases
    cleaned_response = response.text.strip()

     # Clean up unwanted characters and phrases
    cleaned_response = response.text.replace('#', '').replace('*', '').strip()
    
    # Remove bullet points, numbers, and other symbols at the start of each line, including *, # signs
    cleaned_response = re.sub(r'^[\s•\-\d.#*]+', '', cleaned_response, flags=re.MULTILINE)
    
    # Remove any remaining introductory text
    cleaned_response = re.sub(r'^.*?(?=•|\d\.|\w+:)', '', cleaned_response, flags=re.MULTILINE).strip()
    
    # Remove extra newlines and clean up spacing
    cleaned_response = re.sub(r'\s*\n\s*', '\n', cleaned_response).strip()
    
    # Split the response into bullet points
    points = [point.strip() for point in cleaned_response.split('\n') if point.strip()]
    
    # Add bullet points to the beginning of each point
    points = [f"• {point}" for point in points]
    
    # Get the first word of each point as the new action verb
    new_verbs = [point.split()[1] for point in points if len(point.split()) > 1]
    
    return points, new_verbs


def create_ats_friendly_resume_pdf(job_description, personal_data, output_file):
    doc = SimpleDocTemplate(output_file, pagesize=letter,
                            rightMargin=0.55*inch, leftMargin=0.55*inch,
                            topMargin=0.55*inch, bottomMargin=0.55*inch)
    story = []

    # Customizing styles for enhanced look
    styles = {
        'Heading1': ParagraphStyle(
            'Heading1',
            fontSize=14,
            textColor=HexColor("#0070C0"),
            spaceAfter=10,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT
        ),
        'Heading2': ParagraphStyle(
            'Heading2',
            fontSize=12,
            textColor=HexColor("#4F4F4F"),
            spaceAfter=6,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT
        ),
        'Normal': ParagraphStyle(
            'Normal',
            fontSize=10,
            leading=14,
            textColor=black,
            spaceAfter=4,
            fontName="Helvetica"
        ),
        'JustifiedParagraph': ParagraphStyle(
            'JustifiedParagraph',
            fontSize=10,
            leading=14,
            textColor=black,
            alignment=TA_JUSTIFY,
            fontName="Helvetica"
        ),
        'Link': ParagraphStyle(
            'Link',
            parent=ParagraphStyle('Normal'),
            textColor=HexColor("#0000FF"),
        )
    }

    # Personal Information
    story.append(Paragraph(f"{personal_data['name']}", styles['Heading1']))
    
    # Create a visually appealing contact information layout
    contact_info = [
        f'<link href="mailto:{personal_data["email"]}">{personal_data["email"]}</link>',
        personal_data['phone'],
        f'<link href="{personal_data["linkedin"]}">LinkedIn</link>',
        f'<link href="{personal_data["github"]}">GitHub</link>',
        f'<link href="{personal_data["portfolio"]}">Portfolio</link>'
    ]
    contact_line = " | ".join(contact_info)
    story.append(Paragraph(contact_line, styles['Link']))
    story.append(Paragraph(personal_data['address'], styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Professional Summary
    story.append(Paragraph("Professional Summary", styles['Heading2']))
    summary = generate_summary(job_description, personal_data)
    story.append(Paragraph(summary, styles['JustifiedParagraph']))
    story.append(Spacer(1, 0.2*inch))

    # Extract keywords from the job description
    job_keywords = set(extract_keywords(job_description))

     # Skills Section
    story.append(Paragraph("Skills", styles['Heading2']))
    
    # Define skill categories based on the structure in personal_data
    skill_categories = {
        "Programming Languages": personal_data['skills']['programming_languages'],
        "Databases": personal_data['skills']['databases'],
        "Frontend Development": personal_data['skills']['frontend_development'],
        "Version Control & Cloud": personal_data['skills']['version_control_and_cloud'],
        "Backend Development": personal_data['skills']['backend_development']
    }
    
    # Extract keywords from the job description
    job_keywords = set(extract_keywords(job_description))
    
    # Prioritize and display skills
    for category, skills in skill_categories.items():
        # Sort skills based on whether they appear in job keywords
        prioritized_skills = sorted(skills, key=lambda x: x.lower() in job_keywords, reverse=True)
        if prioritized_skills:
            story.append(Paragraph(f"<b>{category}:</b> {', '.join(prioritized_skills)}", styles['Normal']))
    
    story.append(Spacer(1, 0.2*inch))

    # Work Experience Section
    story.append(Paragraph("Work Experience", styles['Heading2']))
    used_verbs = []
    for exp in personal_data['experience']:
        story.append(Paragraph(f"<b>{exp['job_title']} - {exp['company']}</b>, {exp['startDate']} to {exp['endDate']}", styles['Normal']))
        optimized_content, new_verbs = optimize_content(exp['responsibilities'], job_keywords, 'Work Experience', used_verbs)
        used_verbs.extend(new_verbs)
        for resp in optimized_content:
            story.append(Paragraph(resp, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

    # Projects Section
    story.append(Paragraph("Projects", styles['Heading2']))
    for project in personal_data['projects']:
        story.append(Paragraph(f"<b>{project['name']}</b> - Skills: {', '.join(project['skills'])}", styles['Normal']))
        optimized_content, new_verbs = optimize_content(project['highlights'], job_keywords, 'Project', used_verbs)
        used_verbs.extend(new_verbs)
        for highlight in optimized_content:
            story.append(Paragraph(highlight, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

    # Education Section
    story.append(Paragraph("Education", styles['Heading2']))
    for edu in personal_data['education']:
        story.append(Paragraph(f"<b>{edu['degree']}</b> - {edu['institution']}, {edu['startDate']} to {edu['endDate']}", styles['Normal']))

    # Build the PDF
    doc.build(story)

def load_job_description(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def load_personal_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def main():
    job_description_file = "job_description.txt"
    personal_data_file = "details.json"
    output_pdf_file = "generated_resume.pdf"

    job_description = load_job_description(job_description_file)
    personal_data = load_personal_data(personal_data_file)

    create_ats_friendly_resume_pdf(job_description, personal_data, output_pdf_file)
    print(f"Resume generated: {output_pdf_file}")

if __name__ == "__main__":
    main()