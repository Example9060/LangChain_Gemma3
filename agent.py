from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
from ollama import generate
from tools.article_loader_cleaner import fetch_and_clean_articles
from tools.clustering import cluster_articles,summarize_clusters
from tools.generate_cluster_summaries import generate_article

llm = Ollama(model="gemma3")

tools = [fetch_and_clean_articles, summarize_clusters, cluster_articles, generate_article]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
