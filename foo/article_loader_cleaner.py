from langchain_core.tools import tool
import os
import re
import json
import time
import pandas as pd
from gnews import GNews
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

@tool(description="Загружает свежие статьи по теме AI")
def fetch_and_clean_articles(dummy_input: str = "") -> str:

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    queries = [
        "AI in healthcare",
        "artificial intelligence in medicine",
        "machine learning in healthcare"
    ]

    seen_urls = set()
    articles = []
    MIN_WORDS = 50
    gnews = GNews(language='en', country='US', max_results=100)
    max_articles: int = 100
    def extract_text_with_selenium(url):
        try:
            driver.get(url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text() for p in paragraphs)
            return text.strip()
        except Exception as e:
            print(f"️ Ошибка парсинга {url}: {e}")
            return None

    for query in queries:
        results = gnews.get_news(query)
        for result in results:
            url = result.get("url")
            title = result.get("title", "No title")

            if not url or url in seen_urls:
                continue

            text = extract_text_with_selenium(url)
            if text and len(text.split()) >= MIN_WORDS:
                articles.append({
                    "title": title,
                    "url": url,
                    "text": text
                })
                seen_urls.add(url)
                print(f"[+] {title[:60]} ({len(text.split())} слов)")

            if len(articles) >= max_articles:
                break
        if len(articles) >= max_articles:
            break

    driver.quit()

    # Очистка текста
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    cleaned_data = []
    for article in articles:
        cleaned = clean_text(article["text"])
        if len(cleaned.split()) >= 100:
            cleaned_data.append({
                "title": article["title"],
                "url": article["url"],
                "text": cleaned
            })

    os.makedirs("data", exist_ok=True)
    with open("data/articles_raw.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(cleaned_data)
    df.to_csv("data/data1.csv", index=False, encoding="utf-8")

    return f" Загружено {len(articles)} статей, очищено и сохранено {len(df)} "
