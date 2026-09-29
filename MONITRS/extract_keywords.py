import csv
import json
import requests
from bs4 import BeautifulSoup
from google import genai


INPUT_CSV = "Data/articles_combined.csv"
OUTPUT_JSON = "Data/keywords.json"

MODEL_NAME = "gemini-3.5-flash-lite"
BLACK_LIST = ['google', 'wikipedia', 'youtube', 'twitter', 'facebook', 'instagram',
              'linkedin', 'pinterest', 'reddit', 'quora', 'tiktok', 'tumblr']


def get_article_content(url):
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ''
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            return title, content
    except Exception:
        pass
    return None, None

def scrape_articles(links):
    articles = []
    for link in links:
        if any(b in link for b in BLACK_LIST):
            continue
        title, article_content = get_article_content(link)
        if article_content and len(article_content) > 100:
            articles.append({
                "url": link,
                "title": title or "",
                "content": article_content
            })

    return articles


def extract_keywords(gemini_client, title, body):
    prompt = f"""
You are extracting search vocabulary from a local natural-disaster
news article.

The goal is to identify words and phrases that can later be used to
search for OTHER local news coverage of natural disasters.

IMPORTANT:
- Preserve terms in their ORIGINAL LANGUAGE.
- Do NOT translate terms into English.
- Prefer words and phrases actually appearing in the article.
- Include synonyms or related expressions ONLY when they are explicitly
  present in the article.
- Do not invent event names.
- Do not include generic words such as "news", "report", "people",
  "area", "today", etc.
- Focus on vocabulary useful for discovering local disaster coverage.
- Return ONLY valid JSON.

TITLE:
{title}

ARTICLE:
{body}

Extract:

1. "title_keywords"
   Important keywords and phrases from the title.

2. "local_terms"
   Local-language words/phrases that would be useful for searching
   for similar disaster coverage.

3. "disaster_terms"
   Terms directly describing the disaster itself.

4. "weather_terms"
   Weather or environmental conditions associated with the event.

5. "impact_terms"
   Terms describing consequences or impacts, such as evacuations,
   damaged homes, road closures, rescues, missing people, casualties,
   power outages, etc.

6. "locations"
   Cities, towns, villages, provinces, regions, rivers, mountains,
   roads, or other geographic names mentioned.

7. "search_phrases"
   Useful combinations of the extracted terms that could be sent
   directly to a local-language news search engine.

Return exactly:

{{
  "title_keywords": [],
  "local_terms": [],
  "disaster_terms": [],
  "weather_terms": [],
  "impact_terms": [],
  "locations": [],
  "search_phrases": []
}}
"""

    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        # ignore non-text content
        text = next(
            part.text
            for part in response.candidates[0].content.parts
            if part.text
        ).strip()

        # Remove markdown JSON fences if Gemini adds them.
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()

        return json.loads(text)

    except Exception as e:
        print(f"Gemini extraction failed: {e}")
        return None


def main():
    gemini_client = genai.Client()
    # Group links by DisNo.
    events = {}

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # for now, only process 15
        for i, row in enumerate(reader):
            if i >= 15:
                break
            dis_no = row["DisNo."]
            language = row["lang"]
            url = row["url"]

            if dis_no not in events:
                events[dis_no] = []

            events[dis_no].append({
                "language": language,
                "url": url
            })

    results = {}

    for dis_no, links in events.items():
        print(f"\nProcessing {dis_no}")
        print(f"Found {len(links)} links")
        urls = [x["url"] for x in links]
        articles = scrape_articles(urls)

        if not articles:
            print("No articles successfully scraped")
            continue

        results[dis_no] = {
            "articles": []
        }

        for article in articles:
            # Find language from original CSV
            language = next(
                (
                    x["language"]
                    for x in links
                    if x["url"] == article["url"]
                ),
                None
            )

            print(f"  Extracting: {article['url']}")
            keywords = extract_keywords(
                gemini_client,
                article["title"],
                article["content"]
            )

            if keywords is None:
                continue

            results[dis_no]["articles"].append({
                "language": language,
                "url": article["url"],
                "title": article["title"],
                "keywords": keywords
            })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()