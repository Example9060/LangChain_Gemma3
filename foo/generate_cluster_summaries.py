from langchain_core.tools import tool
from langchain_community.chat_models import ChatOllama

@tool
def generate_article(dummy_input: str = "") -> str:
    """Создает финальную мета-статью из cluster_summaries.txt"""

    with open("data/cluster_summaries.txt", "r", encoding="utf-8") as f:
        summaries = f.read()

    llm = ChatOllama(model="gemma3")
    prompt = f"""На основе следующих кратких резюме кластеров напиши единую мета-статью на тему "AI в здравоохранении":

{summaries}

Требования:
- объем от 5000 до 7000 слов;
- единый связный стиль;
- структура: введение, ключевые темы, выводы;
- стиль: научно-популярный, понятный, профессиональный."""

    result = llm.invoke(prompt)

    with open("data/meta_article.txt", "w", encoding="utf-8") as f:
        f.write(result.content)

    return "Мета-статья успешно создана и сохранена в data/meta_article.txt"
