import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Course & Career Advisor", page_icon="🎓", layout="wide")

st.title("AI Course & Career Advisor")
st.caption("Powered by sentence embeddings + semantic retrieval")

DATA_PATH = Path("course_career_data_expanded.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def build_documents(data):
    docs = []
    for _, row in data.iterrows():
        text = f"""
Title: {row['title']}
Type: {row['type']}
Description: {row['description']}
Skills: {row['skills']}
Source: {row['source']}
"""
        docs.append(text.strip())
    return docs

@st.cache_data
def compute_embeddings(_model_name_placeholder, docs):
    model = load_model()
    return model.encode(docs, normalize_embeddings=True)


def detect_query_type(query: str):
    q = query.lower().strip()
    if any(word in q for word in ["course", "class", "learn", "study", "take"]):
        return "course-oriented"
    if any(word in q for word in ["job", "role", "career", "analyst", "scientist", "engineer", "consultant"]):
        return "career-oriented"
    if any(word in q for word in ["python", "sql", "tableau", "power bi", "excel", "machine learning", "statistics", "nlp"]):
        return "skill-oriented"
    return "general"


def search_by_type(query, df, documents, embeddings, model, item_type, k=4):
    type_indices = df.index[df["type"] == item_type].tolist()
    if not type_indices:
        return []

    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    type_embeddings = embeddings[type_indices]
    scores = np.dot(type_embeddings, query_embedding)
    ranked_local = np.argsort(scores)[::-1][:k]

    results = []
    for local_idx in ranked_local:
        original_idx = type_indices[local_idx]
        row = df.iloc[original_idx]
        results.append({
            "title": row["title"],
            "type": row["type"],
            "description": row["description"],
            "skills": row["skills"],
            "source": row["source"],
            "score": float(scores[local_idx])
        })
    return results


def render_cards(results):
    for item in results:
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            st.write(item["description"])
            st.write(f"Skills: {item['skills']}")
            st.caption(f"Similarity score: {item['score']:.3f} | Source: {item['source']}")


df = load_data()
model = load_model()
documents = build_documents(df)
embeddings = compute_embeddings("all-MiniLM-L6-v2", documents)

with st.sidebar:
    st.header("About this prototype")
    st.write("This app helps analytics students connect career goals, skills, and course planning.")
    st.metric("Dataset rows", len(df))
    st.write("Dataset types:")
    st.dataframe(df["type"].value_counts().reset_index().rename(columns={"index": "type", "type": "count"}), hide_index=True)
    st.write("Try prompts like:")
    st.code("I want to become a data analyst")
    st.code("Python")
    st.code("What courses should I take for machine learning?")

query = st.text_input("Enter a career goal, job title, course interest, or skill:", placeholder="e.g. Python / Data Analyst / I want to work in AI analytics")

if query:
    query_type = detect_query_type(query)
    st.info(f"Detected query type: {query_type}")

    skills = search_by_type(query, df, documents, embeddings, model, "skill", k=3)
    jobs = search_by_type(query, df, documents, embeddings, model, "job", k=4)
    courses = search_by_type(query, df, documents, embeddings, model, "course", k=4)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Relevant Skills")
        render_cards(skills)

    with col2:
        st.subheader("Suggested Jobs")
        render_cards(jobs)

    with col3:
        st.subheader("Recommended Courses")
        render_cards(courses)

    st.divider()
    st.subheader("Recommendation Summary")
    top_job = jobs[0]["title"] if jobs else "N/A"
    top_course = courses[0]["title"] if courses else "N/A"
    top_skill = skills[0]["title"] if skills else "N/A"
    st.write(
        f"Based on your input, the closest skill area is **{top_skill}**, "
        f"the most relevant role is **{top_job}**, and a strong course starting point is **{top_course}**."
    )
else:
    st.write("Enter a prompt above to receive skill, job, and course recommendations.")
    st.write("The final version supports three input styles: career goals, job titles, and skills.")
