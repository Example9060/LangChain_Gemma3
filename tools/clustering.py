from langchain_core.tools import tool
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from langchain_community.chat_models import ChatOllama
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import tool

@tool(description="Класстиризует новости")
def cluster_articles(dummy_input: str = "") -> str:

    df = pd.read_csv("data/data1.csv")
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["text"])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/data_with_clusters.csv", index=False)
    return f"Кластеризация на {3} кластеров завершена. Сохранено в data/data_with_clusters.csv"

@tool
def summarize_clusters(dummy_input: str = "") -> str:
    """Пишет краткое резюме для каждого кластера"""

    df = pd.read_csv("data/data_with_clusters.csv")
    llm = ChatOllama(model="gemma3")

    summaries = []
    for cluster_id in sorted(df['cluster'].unique()):
        texts = df[df["cluster"] == cluster_id]["text"].tolist()
        combined_text = " ".join(texts[:10])
        summary = llm.invoke(f"Сделай краткое резюме по следующим статьям:\n{combined_text[:3000]}")
        summaries.append(f"Кластер {cluster_id}:\n{summary}\n")

    with open("data/cluster_summaries.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summaries))

    return f"Сгенерированы резюме для {len(summaries)} кластеров"
