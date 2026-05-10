# AI Course & Career Advisor

A Streamlit application that uses sentence embeddings and semantic retrieval to recommend relevant skills, job roles, and courses for analytics students.

## Problem
Analytics students often know a goal such as "I want to become a data analyst" or a skill such as "Python," but may not know which roles match that goal or which courses to take next.

## What the app does
Users can enter:
- a career goal: `I want to become a data analyst`
- a job title: `business analyst`
- a skill: `Python`
- a course interest: `machine learning`

The system returns:
- relevant skills
- suggested jobs
- recommended courses

## Technical approach
1. Load a structured dataset of jobs, courses, and skills.
2. Convert each row into a searchable text document.
3. Generate sentence embeddings using `all-MiniLM-L6-v2`.
4. Compute semantic similarity between the user query and dataset rows.
5. Return ranked recommendations by category.
6. Display results in an interactive Streamlit app.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Future improvements
- Expand the dataset with Kaggle job datasets and company career pages.
- Add Fordham course catalog or professor-provided course files.
- Extract structured skills, responsibilities, and prerequisites from real job descriptions.
- Add evaluation metrics for retrieval quality.
- Personalize recommendations based on student background and completed courses.
