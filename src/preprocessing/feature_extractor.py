"""
Feature Extraction Module
Extracts linguistic, statistical, and semantic features from preprocessed text
"""

import re
import math
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
import string

class FeatureExtractor:
    """Extract comprehensive features from preprocessed text for fake news detection"""
    
    def __init__(self):
        """Initialize feature extractor"""
        from utils.logger import get_logger
        self.logger = get_logger(__name__)
        self.nltk_available = False
        self.spacy_available = False
        self.spacy_model = None
        
        # Initialize NLTK and spaCy
        self._init_nltk()
        self._init_spacy()
        
        # Fake news indicators
        self.fake_news_indicators = {
            'emotional_words': [
                'shocking', 'amazing', 'incredible', 'unbelievable', 'outrageous',
                'scandalous', 'explosive', 'devastating', 'terrifying', 'horrifying',
                'stunning', 'mind-blowing', 'jaw-dropping', 'earth-shattering'
            ],
            'urgency_words': [
                'urgent', 'breaking', 'immediate', 'critical', 'emergency',
                'alert', 'warning', 'caution', 'asap', 'now', 'today',
                'this instant', 'right now', 'immediately'
            ],
            'exaggeration_words': [
                'never', 'always', 'all', 'every', 'none', 'completely',
                'totally', 'absolutely', 'definitely', 'certainly', 'surely',
                'obviously', 'clearly', 'undoubtedly', 'without doubt'
            ],
            'conspiracy_words': [
                'conspiracy', 'cover-up', 'secret', 'hidden', 'classified',
                'confidential', 'leaked', 'exposed', 'revealed', 'uncovered',
                'whistleblower', 'insider', 'sources say', 'according to sources'
            ],
            'clickbait_words': [
                'you won\'t believe', 'doctors hate', 'this one trick',
                'what happens next', 'number 7 will shock you', 'click here',
                'find out', 'discover', 'revealed', 'exposed', 'secret'
            ]
        }
        
        # Sentiment indicators
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'brilliant', 'outstanding', 'perfect', 'best', 'love', 'like',
            'happy', 'joy', 'success', 'victory', 'win', 'achieve'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'angry', 'sad', 'disappointed', 'failure', 'lose', 'defeat',
            'problem', 'issue', 'crisis', 'disaster', 'tragedy'
        }
    
    def _init_nltk(self):
        """Initialize NLTK components"""
        try:
            import nltk
            self.nltk_available = True
            
            # Download required NLTK data with better error handling
            
            required_data = [
                ('tokenizers/punkt', 'punkt'),
                ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
                ('corpora/wordnet', 'wordnet'),
                ('tokenizers/punkt_tab', 'punkt_tab')
            ]
            
            for data_path, package_name in required_data:
                try:
                    nltk.data.find(data_path)
                except LookupError:
                    try:
                        self.logger.info(f"Downloading NLTK {package_name}...")
                        nltk.download(package_name, quiet=True)
                    except Exception as download_error:
                        self.logger.warning(f"Failed to download {package_name}: {download_error}")
                        continue
            
            # Import NLTK components with error handling
            try:
                from nltk.tokenize import word_tokenize, sent_tokenize
                from nltk.tag import pos_tag
                from nltk.corpus import wordnet
                
                self.word_tokenize = word_tokenize
                self.sent_tokenize = sent_tokenize
                self.pos_tag = pos_tag
                self.wordnet = wordnet
                
                self.logger.info("NLTK initialized successfully")
                
            except ImportError as import_error:
                self.logger.warning(f"Failed to import NLTK components: {import_error}")
                self._set_nltk_fallbacks()
            
        except ImportError:
            self.logger.warning("NLTK not available for feature extraction")
            self._set_nltk_fallbacks()
        except Exception as e:
            self.logger.warning(f"NLTK initialization error: {e}")
            self._set_nltk_fallbacks()
    
    def _set_nltk_fallbacks(self):
        """Set fallback methods when NLTK is not available"""
        self.word_tokenize = None
        self.sent_tokenize = None
        self.pos_tag = None
        self.wordnet = None
    
    def _init_spacy(self):
        """Initialize spaCy components"""
        try:
            import spacy
            self.spacy_available = True
            
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
            except OSError:
                self.spacy_model = spacy.blank("en")
                
        except ImportError:
            self.logger.warning("spaCy not available for feature extraction")
        except Exception as e:
            self.logger.warning(f"spaCy initialization error: {e}")
    
    def extract_all_features(self, text: str, cleaned_text: str = None) -> Dict[str, Any]:
        """
        Extract all features from text
        
        Args:
            text (str): Original text
            cleaned_text (str): Preprocessed text
            
        Returns:
            Dict containing all extracted features
        """
        if not text:
            return {}
        
        # Use cleaned text if available, otherwise use original
        working_text = cleaned_text if cleaned_text else text
        
        features = {}
        
        # Basic statistical features
        features.update(self._extract_statistical_features(working_text))
        
        # Linguistic features
        features.update(self._extract_linguistic_features(working_text))
        
        # Readability features
        features.update(self._extract_readability_features(working_text))
        
        # Sentiment features
        features.update(self._extract_sentiment_features(working_text))
        
        # Fake news indicator features
        features.update(self._extract_fake_news_indicators(working_text))
        
        # N-gram features
        features.update(self._extract_ngram_features(working_text))
        
        # POS tag features
        if self.nltk_available:
            features.update(self._extract_pos_features(working_text))
        
        # Named entity features
        if self.spacy_available:
            features.update(self._extract_ner_features(working_text))
        
        # URL and link features
        features.update(self._extract_url_features(text))
        
        # Capitalization features
        features.update(self._extract_capitalization_features(text))
        
        # Punctuation features
        features.update(self._extract_punctuation_features(text))
        
        return features
    
    def _extract_statistical_features(self, text: str) -> Dict[str, Any]:
        """Extract basic statistical features"""
        if not text:
            return {}
        
        words = text.split()
        sentences = self._split_sentences(text)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'character_count': len(text),
            'character_count_no_spaces': len(text.replace(' ', '')),
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'unique_word_count': len(set(word.lower() for word in words)),
            'lexical_diversity': len(set(word.lower() for word in words)) / max(len(words), 1),
            'longest_word_length': max((len(word) for word in words), default=0),
            'shortest_word_length': min((len(word) for word in words), default=0)
        }
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features"""
        if not text:
            return {}
        
        words = text.split()
        
        # Word length distribution
        word_lengths = [len(word) for word in words]
        
        # Syllable count estimation
        syllable_counts = [self._count_syllables(word) for word in words]
        
        return {
            'words_1_char': sum(1 for length in word_lengths if length == 1),
            'words_2_chars': sum(1 for length in word_lengths if length == 2),
            'words_3_chars': sum(1 for length in word_lengths if length == 3),
            'words_4_chars': sum(1 for length in word_lengths if length == 4),
            'words_5_chars': sum(1 for length in word_lengths if length == 5),
            'words_6_plus_chars': sum(1 for length in word_lengths if length >= 6),
            'avg_syllables_per_word': sum(syllable_counts) / max(len(words), 1),
            'total_syllables': sum(syllable_counts),
            'words_with_3_plus_syllables': sum(1 for count in syllable_counts if count >= 3)
        }
    
    def _extract_readability_features(self, text: str) -> Dict[str, Any]:
        """Extract readability features"""
        if not text:
            return {}
        
        words = text.split()
        sentences = self._split_sentences(text)
        
        # Flesch Reading Ease Score
        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        if total_sentences > 0 and total_words > 0:
            flesch_score = 206.835 - (1.015 * (total_words / total_sentences)) - (84.6 * (total_syllables / total_words))
        else:
            flesch_score = 0
        
        # Flesch-Kincaid Grade Level
        if total_sentences > 0 and total_words > 0:
            fk_grade = (0.39 * (total_words / total_sentences)) + (11.8 * (total_syllables / total_words)) - 15.59
        else:
            fk_grade = 0
        
        return {
            'flesch_reading_ease': flesch_score,
            'flesch_kincaid_grade': fk_grade,
            'readability_level': self._classify_readability(flesch_score)
        }
    
    def _extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """Extract sentiment features"""
        if not text:
            return {}
        
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_words = len(words)
        
        return {
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'positive_word_ratio': positive_count / max(total_words, 1),
            'negative_word_ratio': negative_count / max(total_words, 1),
            'sentiment_polarity': (positive_count - negative_count) / max(total_words, 1),
            'sentiment_subjectivity': (positive_count + negative_count) / max(total_words, 1)
        }
    
    def _extract_fake_news_indicators(self, text: str) -> Dict[str, Any]:
        """Extract fake news indicator features"""
        if not text:
            return {}
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        features = {}
        
        for category, indicators in self.fake_news_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            features[f'{category}_count'] = count
            features[f'{category}_ratio'] = count / max(total_words, 1)
        
        # Overall fake news score
        total_indicators = sum(features[f'{cat}_count'] for cat in self.fake_news_indicators.keys())
        features['fake_news_indicator_score'] = total_indicators / max(total_words, 1)
        
        return features
    
    def _extract_ngram_features(self, text: str) -> Dict[str, Any]:
        """Extract n-gram features"""
        if not text:
            return {}
        
        words = text.lower().split()
        
        # Bigrams
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)
        
        # Trigrams
        trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
        trigram_counts = Counter(trigrams)
        
        return {
            'unique_bigrams': len(bigram_counts),
            'unique_trigrams': len(trigram_counts),
            'most_common_bigram': bigram_counts.most_common(1)[0] if bigram_counts else None,
            'most_common_trigram': trigram_counts.most_common(1)[0] if trigram_counts else None,
            'bigram_diversity': len(bigram_counts) / max(len(bigrams), 1),
            'trigram_diversity': len(trigram_counts) / max(len(trigrams), 1)
        }
    
    def _extract_pos_features(self, text: str) -> Dict[str, Any]:
        """Extract Part-of-Speech features using NLTK"""
        if not self.nltk_available or not text or self.word_tokenize is None or self.pos_tag is None:
            return {}
        
        try:
            words = self.word_tokenize(text)
            pos_tags = self.pos_tag(words)
            
            # Count POS tags
            pos_counts = Counter(tag for word, tag in pos_tags)
            
            return {
                'noun_count': pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0),
                'verb_count': pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0),
                'adjective_count': pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0),
                'adverb_count': pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0),
                'pronoun_count': pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0),
                'determiner_count': pos_counts.get('DT', 0),
                'preposition_count': pos_counts.get('IN', 0),
                'conjunction_count': pos_counts.get('CC', 0),
                'interjection_count': pos_counts.get('UH', 0),
                'total_pos_tags': len(pos_tags)
            }
        except Exception as e:
            self.logger.warning(f"Error in POS tagging: {e}")
            return {}
    
    def _extract_ner_features(self, text: str) -> Dict[str, Any]:
        """Extract Named Entity Recognition features using spaCy"""
        if not self.spacy_available or not text:
            return {}
        
        try:
            doc = self.spacy_model(text)
            
            # Count named entities
            entity_counts = Counter(ent.label_ for ent in doc.ents)
            
            return {
                'person_count': entity_counts.get('PERSON', 0),
                'organization_count': entity_counts.get('ORG', 0),
                'location_count': entity_counts.get('GPE', 0) + entity_counts.get('LOC', 0),
                'date_count': entity_counts.get('DATE', 0),
                'time_count': entity_counts.get('TIME', 0),
                'money_count': entity_counts.get('MONEY', 0),
                'percent_count': entity_counts.get('PERCENT', 0),
                'total_entities': len(doc.ents),
                'entity_density': len(doc.ents) / max(len(text.split()), 1)
            }
        except Exception as e:
            self.logger.warning(f"Error in NER: {e}")
            return {}
    
    def _extract_url_features(self, text: str) -> Dict[str, Any]:
        """Extract URL and link features"""
        if not text:
            return {}
        
        # URL patterns
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        urls = url_pattern.findall(text)
        
        # Email patterns
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        emails = email_pattern.findall(text)
        
        return {
            'url_count': len(urls),
            'email_count': len(emails),
            'has_urls': len(urls) > 0,
            'has_emails': len(emails) > 0,
            'url_density': len(urls) / max(len(text.split()), 1)
        }
    
    def _extract_capitalization_features(self, text: str) -> Dict[str, Any]:
        """Extract capitalization features"""
        if not text:
            return {}
        
        words = text.split()
        total_chars = len(text)
        
        uppercase_chars = sum(1 for char in text if char.isupper())
        lowercase_chars = sum(1 for char in text if char.islower())
        
        all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        title_case_words = sum(1 for word in words if word.istitle())
        
        return {
            'uppercase_ratio': uppercase_chars / max(total_chars, 1),
            'lowercase_ratio': lowercase_chars / max(total_chars, 1),
            'all_caps_word_count': all_caps_words,
            'title_case_word_count': title_case_words,
            'all_caps_ratio': all_caps_words / max(len(words), 1),
            'title_case_ratio': title_case_words / max(len(words), 1)
        }
    
    def _extract_punctuation_features(self, text: str) -> Dict[str, Any]:
        """Extract punctuation features"""
        if not text:
            return {}
        
        total_chars = len(text)
        
        # Count different punctuation marks
        exclamation_count = text.count('!')
        question_count = text.count('?')
        period_count = text.count('.')
        comma_count = text.count(',')
        semicolon_count = text.count(';')
        colon_count = text.count(':')
        quote_count = text.count('"') + text.count("'")
        
        # Count multiple punctuation
        multiple_exclamation = len(re.findall(r'!{2,}', text))
        multiple_question = len(re.findall(r'\?{2,}', text))
        
        return {
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'period_count': period_count,
            'comma_count': comma_count,
            'semicolon_count': semicolon_count,
            'colon_count': colon_count,
            'quote_count': quote_count,
            'multiple_exclamation_count': multiple_exclamation,
            'multiple_question_count': multiple_question,
            'punctuation_density': (exclamation_count + question_count + period_count + comma_count) / max(total_chars, 1),
            'exclamation_ratio': exclamation_count / max(total_chars, 1),
            'question_ratio': question_count / max(total_chars, 1)
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if self.nltk_available and self.sent_tokenize is not None:
            try:
                return self.sent_tokenize(text)
            except:
                pass
        
        # Fallback: simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(syllable_count, 1)
    
    def _classify_readability(self, flesch_score: float) -> str:
        """Classify readability level based on Flesch score"""
        if flesch_score >= 90:
            return 'very_easy'
        elif flesch_score >= 80:
            return 'easy'
        elif flesch_score >= 70:
            return 'fairly_easy'
        elif flesch_score >= 60:
            return 'standard'
        elif flesch_score >= 50:
            return 'fairly_difficult'
        elif flesch_score >= 30:
            return 'difficult'
        else:
            return 'very_difficult'
    
    def get_feature_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of extracted features"""
        if not features:
            return {}
        
        # Categorize features
        categories = {
            'statistical': [k for k in features.keys() if any(x in k for x in ['count', 'length', 'diversity', 'avg'])],
            'linguistic': [k for k in features.keys() if any(x in k for x in ['syllable', 'char', 'word'])],
            'sentiment': [k for k in features.keys() if any(x in k for x in ['sentiment', 'positive', 'negative'])],
            'fake_news': [k for k in features.keys() if 'fake_news' in k or any(x in k for x in ['emotional', 'urgency', 'exaggeration', 'conspiracy', 'clickbait'])],
            'readability': [k for k in features.keys() if any(x in k for x in ['flesch', 'readability', 'grade'])],
            'pos': [k for k in features.keys() if any(x in k for x in ['noun', 'verb', 'adjective', 'adverb', 'pronoun'])],
            'ner': [k for k in features.keys() if any(x in k for x in ['person', 'organization', 'location', 'entity'])],
            'punctuation': [k for k in features.keys() if any(x in k for x in ['exclamation', 'question', 'period', 'comma', 'punctuation'])],
            'capitalization': [k for k in features.keys() if any(x in k for x in ['uppercase', 'lowercase', 'caps', 'title'])],
            'url': [k for k in features.keys() if any(x in k for x in ['url', 'email', 'link'])]
        }
        
        return {
            'total_features': len(features),
            'feature_categories': {cat: len(feats) for cat, feats in categories.items()},
            'available_categories': list(categories.keys()),
            'feature_names': list(features.keys())
        }
