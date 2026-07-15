from flask import Flask, request, render_template
import joblib
import os
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load or create model
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    joblib.dump(model, MODEL_PATH)

# Skills list
SKILLS = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "SQL",
    "Power BI",
    "Flask",
    "TensorFlow",
    "PyTorch",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "Data Analysis",
    "Data Visualization",
    "Git",
    "Docker",
    "AWS"
]

# Home Page
@app.route("/")
def index():
    return render_template("index.html")


# Resume Analysis
@app.route("/predict", methods=["POST"])
def predict():
    # Get uploaded PDF and Job Description
    resume = request.files["resume"]
    job_description = request.form["job_description"]

    # Extract text from PDF
    pdf = fitz.open(stream=resume.read(), filetype="pdf")
    resume_text = ""

    for page in pdf:
        resume_text += page.get_text()

    # Generate embeddings
    resume_embedding = model.encode(resume_text)
    job_embedding = model.encode(job_description)

    # Calculate similarity
    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    match_percentage = round(similarity * 100, 2)

    # Extract skills
    resume_skills = [
        skill for skill in SKILLS
        if skill.lower() in resume_text.lower()
    ]

    job_skills = [
        skill for skill in SKILLS
        if skill.lower() in job_description.lower()
    ]

    matched_skills = [
        skill for skill in resume_skills
        if skill in job_skills
    ]

    missing_skills = [
        skill for skill in job_skills
        if skill not in resume_skills
    ]

    # Render result page
    return render_template(
        "result.html",
        score=match_percentage,
        match_skills=matched_skills,
        missing_skills=missing_skills
    )


if __name__ == "__main__":
    app.run(debug=True)