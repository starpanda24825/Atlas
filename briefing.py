# briefing.py
from datetime import datetime
import feedparser
from langchain_ollama import OllamaLLM
from config import RSS_FEEDS, OLLAMA_BASE_URL, MAIN_MODEL, AGENT_NAME


class BriefingSystem:
    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def fetch_headlines(self, max_per_feed: int = 4) -> dict:
        """Pull headlines from each configured RSS feed."""
        all_headlines: dict = {}

        for category, feed_urls in RSS_FEEDS.items():
            items = []
            for url in feed_urls:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:max_per_feed]:
                        summary = getattr(entry, "summary", "") or ""
                        # Strip HTML tags
                        import re
                        summary = re.sub(r"<[^>]+>", "", summary)[:250]
                        items.append({
                            "title":   entry.get("title", ""),
                            "summary": summary,
                        })
                except Exception as e:
                    print(f"[Briefing] Could not fetch {url}: {e}")
            if items:
                all_headlines[category] = items

        return all_headlines

    def generate_briefing(self) -> str:
        """Produce a natural spoken briefing from current headlines."""
        print("[Briefing] Fetching headlines...")
        headlines = self.fetch_headlines()

        if not headlines:
            return "I was unable to retrieve news for today. Please check your internet connection."

        news_block = ""
        for category, items in headlines.items():
            news_block += f"\n{category}:\n"
            for item in items:
                news_block += f"  - {item['title']}. {item['summary']}\n"

        now    = datetime.now()
        prompt = f"""You are {AGENT_NAME}, a personal AI assistant giving a morning briefing to your user.

Today is {now.strftime("%A, %B %d, %Y")} and the time is {now.strftime("%I:%M %p")}.

Here are the latest headlines:
{news_block}

Deliver a concise, natural morning briefing of about 250 to 300 words.
Rules:
- Write as if speaking aloud — no bullet points, no markdown, no asterisks
- Highlight the 3 to 4 most significant stories across categories
- Keep financial news brief but include it
- End with a one-sentence motivational or observational remark
- Address the user directly as "you"
"""
        print("[Briefing] Generating summary with Ollama...")
        return self.llm.invoke(prompt).strip()
