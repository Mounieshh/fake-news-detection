import re
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlparse, urljoin
import time

class URLHandler:
    """Handles URL input processing for fake news detection"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.timeout = 30
        self.max_content_length = 100000  # Max characters to extract
        
        # Try to import beautifulsoup4 (already installed)
        try:
            from bs4 import BeautifulSoup
            self.bs4_available = True
            self.BeautifulSoup = BeautifulSoup
        except ImportError:
            self.bs4_available = False
            print("Note: beautifulsoup4 not available. Install with: pip install beautifulsoup4")
    
    def process_url_input(self, url: str) -> Dict[str, Any]:
        """
        Process URL input and extract content
        
        Args:
            url (str): URL to process
            
        Returns:
            Dict containing extracted content and metadata
        """
        try:
            # Check BeautifulSoup availability first
            bs4_check = self.check_bs4_availability()
            if not bs4_check['available']:
                return {
                    'status': 'error',
                    'error': bs4_check['error'],
                    'input_type': 'url'
                }

            # Validate URL
            validation = self._validate_url(url)
            if not validation['valid']:
                return {
                    'status': 'error',
                    'error': validation['error'],
                    'input_type': 'url'
                }
            
            # Fetch content from URL
            content_response = self._fetch_url_content(url)
            if content_response['status'] != 'success':
                return content_response
            
            # Extract text content using BeautifulSoup
            if not self.bs4_available:
                return {
                    'status': 'error',
                    'error': 'BeautifulSoup4 is required for URL processing',
                    'input_type': 'url'
                }
            
            soup = self.BeautifulSoup(content_response['html'], 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Extract article content (customize based on common news site structures)
            article_content = ""
            
            # Try to find article content in common containers
            article_selectors = [
                'article',
                '.article-body',
                '.article-content',
                '#article-content',
                '.story-content',
                '[itemprop="articleBody"]',
                '.content-body',
                '.story-body'
            ]
            
            for selector in article_selectors:
                content = soup.select(selector)
                if content:
                    article_content = ' '.join([elem.get_text(strip=True) for elem in content])
                    break
            
            # If no article content found, try to get main content
            if not article_content:
                main_content = soup.find('main')
                if main_content:
                    article_content = main_content.get_text(strip=True)
                else:
                    # Fallback to body text
                    article_content = soup.get_text(strip=True)
            
            # Extract content using BeautifulSoup
            extracted_content = self._extract_content_from_html(
                content_response['html'], 
                content_response['url']
            )
            
            return {
                'status': 'success',
                'input_type': 'url',
                'url': content_response['url'],
                'title': extracted_content.get('title', ''),
                'extracted_content': extracted_content,
                'metadata': {
                    'http_status': content_response.get('http_status'),
                    'content_type': content_response.get('content_type'),
                    'fetch_time': content_response.get('fetch_time')
                }
            }
            
            # Combine all information
            result = {
                'status': 'success',
                'input_type': 'url',
                'original_url': url,
                'final_url': content_response['url'],
                'http_status': content_response['http_status'],
                'content_type': content_response['content_type'],
                'extracted_content': extracted_content,
                'fetch_time': content_response['fetch_time']
            }
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'input_type': 'url'
            }
    
    def check_bs4_availability(self) -> Dict[str, bool]:
        """Check if BeautifulSoup is available"""
        return {
            'available': self.bs4_available,
            'error': None if self.bs4_available else 'BeautifulSoup4 is not installed. Install with: pip install beautifulsoup4'
        }

    def _validate_url(self, url: str) -> Dict[str, Any]:
        """Validate URL format and accessibility"""
        try:
            # Check URL format
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return {
                    'valid': False,
                    'error': 'Invalid URL format'
                }
            
            # Check if it's a supported scheme
            if parsed.scheme not in ['http', 'https']:
                return {
                    'valid': False,
                    'error': f'Unsupported URL scheme: {parsed.scheme}'
                }
            
            # Check for common fake news indicators in domain
            domain = parsed.netloc.lower()
            suspicious_domains = ['fake', 'hoax', 'satire', 'parody', 'clickbait']
            if any(word in domain for word in suspicious_domains):
                return {
                    'valid': True,
                    'warning': f'Domain contains suspicious keywords: {domain}'
                }
            
            return {
                'valid': True,
                'message': 'URL is valid'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'URL validation error: {str(e)}'
            }
    
    def _fetch_url_content(self, url: str) -> Dict[str, Any]:
        """Fetch content from URL"""
        try:
            start_time = time.time()
            
            # Make HTTP request
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            fetch_time = time.time() - start_time
            
            # Check HTTP status
            if response.status_code != 200:
                return {
                    'status': 'error',
                    'error': f'HTTP {response.status_code}: {response.reason}',
                    'input_type': 'url'
                }
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return {
                    'status': 'error',
                    'error': f'Unsupported content type: {content_type}',
                    'input_type': 'url'
                }
            
            # Extract title from meta tags or title tag
            soup = self.BeautifulSoup(response.text, 'html.parser')
            title = ''
            
            # Try to get title from meta tags
            meta_title = soup.find('meta', property='og:title')
            if meta_title:
                title = meta_title.get('content', '')
            
            # Fallback to title tag
            if not title:
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.string
            
            return {
                'status': 'success',
                'html': response.text,
                'url': response.url,
                'title': title,
                'http_status': response.status_code,
                'content_type': content_type,
                'fetch_time': fetch_time
            }
            
        except requests.exceptions.Timeout:
            return {
                'status': 'error',
                'error': 'Request timeout',
                'input_type': 'url'
            }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'error': f'Request failed: {str(e)}',
                'input_type': 'url'
            }
    
    def _extract_content_from_html(self, html: str, url: str) -> Dict[str, Any]:
        """Extract relevant content from HTML"""
        if not self.bs4_available:
            return {
                'status': 'bs4_not_available',
                'message': 'Install beautifulsoup4 for HTML parsing'
            }
        
        try:
            soup = self.BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            # Extract links
            links = self._extract_links(soup, url)
            
            return {
                'status': 'success',
                'title': title,
                'main_content': main_content,
                'metadata': metadata,
                'links': links,
                'content_length': len(main_content),
                'word_count': len(main_content.split())
            }
            
        except Exception as e:
            return {
                'status': 'extraction_error',
                'error': str(e)
            }
    
    def _extract_title(self, soup) -> str:
        """Extract article title"""
        # Try different title selectors
        title_selectors = [
            'h1',
            'title',
            '[property="og:title"]',
            '[name="twitter:title"]',
            '.article-title',
            '.post-title',
            '.entry-title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text().strip()
                if title and len(title) > 10:  # Minimum title length
                    return title
        
        # Fallback to page title
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        return "No title found"
    
    def _extract_main_content(self, soup) -> str:
        """Extract main article content"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Try different content selectors
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '.main-content'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(separator=' ', strip=True)
                if len(content) > 100:  # Minimum content length
                    return content[:self.max_content_length]
        
        # Fallback to body text
        body = soup.find('body')
        if body:
            content = body.get_text(separator=' ', strip=True)
            return content[:self.max_content_length]
        
        return "No content found"
    
    def _extract_metadata(self, soup, url: str) -> Dict[str, Any]:
        """Extract article metadata"""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc,
            'author': None,
            'publish_date': None,
            'keywords': [],
            'description': None
        }
        
        # Extract author
        author_selectors = [
            '[name="author"]',
            '[property="article:author"]',
            '.author',
            '.byline'
        ]
        
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                metadata['author'] = author_elem.get('content') or author_elem.get_text().strip()
                break
        
        # Extract description
        desc_elem = soup.find('meta', attrs={'name': 'description'})
        if desc_elem:
            metadata['description'] = desc_elem.get('content', '').strip()
        
        # Extract keywords
        keywords_elem = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_elem:
            keywords = keywords_elem.get('content', '')
            metadata['keywords'] = [k.strip() for k in keywords.split(',') if k.strip()]
        
        return metadata
    
    def _extract_links(self, soup, base_url: str) -> list:
        """Extract relevant links from the page"""
        links = []
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text().strip()
            
            if href and text:
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)
                
                # Only include external links and significant internal links
                link_domain = urlparse(absolute_url).netloc
                if link_domain != base_domain or len(text) > 20:
                    links.append({
                        'url': absolute_url,
                        'text': text[:100],  # Truncate long link text
                        'external': link_domain != base_domain
                    })
        
        return links[:20]  # Limit to 20 links
    
    def validate_url(self, url: str) -> Dict[str, Any]:
        """Validate URL input"""
        validation = self._validate_url(url)
        if not validation['valid']:
            return validation
        
        # Check if URL is accessible
        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                return {
                    'valid': True,
                    'message': 'URL is valid and accessible',
                    'status_code': response.status_code
                }
            else:
                return {
                    'valid': False,
                    'error': f'URL returned status code: {response.status_code}'
                }
        except Exception as e:
            return {
                'valid': False,
                'error': f'Cannot access URL: {str(e)}'
            }
