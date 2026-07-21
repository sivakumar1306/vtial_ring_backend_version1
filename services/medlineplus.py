import httpx
import xml.etree.ElementTree as ET
import re

MEDLINEPLUS_API = "https://wsearch.nlm.nih.gov/ws/query"

def clean_html(text: str) -> str:
    """Strip HTML tags from text."""
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

async def fetch_medlineplus(query: str, max_results: int = 3) -> list[dict]:
    """
    Fetch health topic content from MedlinePlus API.
    Returns list of {title, content, url} dicts.
    """
    try:
        params = {
            "db": "healthTopics",
            "term": query,
            "retmax": max_results,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(MEDLINEPLUS_API, params=params)
            response.raise_for_status()

        root = ET.fromstring(response.text)
        results = []

        for doc in root.findall('.//document'):
            title = ''
            content_parts = []
            url = doc.get('url', '')

            for content in doc.findall('content'):
                name = content.get('name', '')
                text = clean_html(content.text or '')
                if not text:
                    continue
                if name == 'title':
                    title = text
                elif name in ('FullSummary', 'snippet', 'altTitle'):
                    content_parts.append(text)

            if title and content_parts:
                results.append({
                    'title': title,
                    'content': f"{title}\n\n" + "\n\n".join(content_parts),
                    'url': url
                })

        return results

    except Exception as e:
        print(f"[MedlinePlus] Fetch failed for '{query}': {e}")
        return []