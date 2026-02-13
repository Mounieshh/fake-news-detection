import requests
from bs4 import BeautifulSoup

class URLHandler:
    def get_content(self, url: str) -> str:
        """Fetch and extract main text content from a URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove clutter
            for element in soup(["script", "style", "nav", "footer", "aside", "iframe", "noscript"]):
                element.decompose()
            
            # prioritize article text
            article = soup.find('article')
            if article:
                text = article.get_text(separator=' ', strip=True)
            else:
                text = soup.body.get_text(separator=' ', strip=True)
                
            return text[:20000] # Limit for API
        except Exception as e:
            return f"Error fetching URL: {str(e)}"
