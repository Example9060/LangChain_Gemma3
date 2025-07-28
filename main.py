from agent import tools
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOllama
from tools.article_loader_cleaner import fetch_and_clean_articles
from tools.clustering import cluster_articles, summarize_clusters
from tools.generate_cluster_summaries import generate_article
llm = ChatOllama(model="gemma3")
tools = [
    fetch_and_clean_articles,
    cluster_articles,
    summarize_clusters,
    generate_article,
]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.invoke("Загрузи новости, очисти их, класстеризуй и напиши резюме для каждого класстера и потом создай мета статью")

