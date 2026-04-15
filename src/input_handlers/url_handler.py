import time
import requests
from bs4 import BeautifulSoup

TRANSIENT_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
RETRY_DELAY = 2   # seconds
TIMEOUT = 15      # seconds per attempt
MIN_CONTENT_LENGTH = 50


class URLHandler:
    def get_content(self, url: str) -> str:
        """Fetch and extract main text content from a URL."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.get(url, headers=headers, timeout=TIMEOUT)
                if response.status_code in TRANSIENT_CODES and attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return "Error fetching URL: request timed out after multiple attempts"
            except requests.exceptions.ConnectionError:
                # DNS failure, network unreachable, refused connection etc.
                return "Error fetching URL: unable to reach the website. Check the URL or your network connection."
            except requests.exceptions.HTTPError as e:
                return f"Error fetching URL: server returned {e.response.status_code if e.response else str(e)}"
            except requests.exceptions.RequestException as e:
                return f"Error fetching URL: {str(e)}"

        if response is None:
            return "Error fetching URL: no response received"

        try:
            soup = BeautifulSoup(response.content, 'html.parser')

            for element in soup(["script", "style", "nav", "footer", "aside", "iframe", "noscript"]):
                element.decompose()

            container = soup.find('article') or soup.find('main') or soup.body
            text = container.get_text(separator=' ', strip=True) if container else ""

            if len(text) < MIN_CONTENT_LENGTH:
                return "Error fetching URL: page has insufficient readable content"

            return text[:20000]
        except Exception as e:
            return f"Error fetching URL: failed to parse page content ({str(e)})"
