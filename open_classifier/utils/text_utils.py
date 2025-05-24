import re
import string
from typing import List, Optional, Dict, Any, Tuple
import unicodedata
from collections import Counter
import html

class TextProcessor:
    """Advanced text processing utilities for classification."""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
            'when', 'where', 'how', 'is', 'in', 'of', 'to', 'for', 'with', 'on',
            'at', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves'
        }
    
    def clean_text(self, text: str) -> str:
        """Comprehensive text cleaning."""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        return text.strip()
    
    def preprocess_text(self, text: str, 
                       lowercase: bool = True,
                       remove_punctuation: bool = False,
                       remove_numbers: bool = False) -> str:
        """Preprocess text with various options."""
        text = self.clean_text(text)
        
        if lowercase:
            text = text.lower()
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract various text features."""
        cleaned_text = self.clean_text(text)
        
        # Basic statistics
        char_count = len(cleaned_text)
        word_count = len(cleaned_text.split())
        sentence_count = len(re.findall(r'[.!?]+', cleaned_text))
        
        # Average word length
        words = cleaned_text.split()
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        
        # Punctuation density
        punct_count = sum(1 for char in cleaned_text if char in string.punctuation)
        punct_density = punct_count / max(char_count, 1)
        
        # Uppercase ratio
        upper_count = sum(1 for char in cleaned_text if char.isupper())
        upper_ratio = upper_count / max(char_count, 1)
        
        # Question marks and exclamation marks
        question_marks = cleaned_text.count('?')
        exclamation_marks = cleaned_text.count('!')
        
        # Readability approximation (simplified)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'punctuation_density': punct_density,
            'uppercase_ratio': upper_ratio,
            'question_marks': question_marks,
            'exclamation_marks': exclamation_marks,
            'has_urls': bool(re.search(r'http[s]?://', text)),
            'has_emails': bool(re.search(r'\S+@\S+', text)),
            'has_numbers': bool(re.search(r'\d', cleaned_text))
        }
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords using frequency analysis."""
        # Clean and preprocess
        cleaned_text = self.preprocess_text(text, lowercase=True, remove_punctuation=True)
        
        # Tokenize
        words = [word for word in cleaned_text.split() 
                if len(word) > 2 and word not in self.stop_words]
        
        # Count frequencies
        word_freq = Counter(words)
        
        # Return top keywords
        return [word for word, _ in word_freq.most_common(num_keywords)]
    
    def detect_language_hints(self, text: str) -> Dict[str, Any]:
        """Detect language hints from text patterns."""
        # Simple heuristics for language detection
        hints = {
            'likely_english': True,  # Default assumption
            'has_non_ascii': bool(re.search(r'[^\x00-\x7F]', text)),
            'has_accents': bool(re.search(r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', text.lower())),
            'has_cyrillic': bool(re.search(r'[а-яё]', text.lower())),
            'has_chinese': bool(re.search(r'[\u4e00-\u9fff]', text)),
            'has_arabic': bool(re.search(r'[\u0600-\u06ff]', text)),
        }
        
        # Adjust English likelihood based on patterns
        if hints['has_cyrillic'] or hints['has_chinese'] or hints['has_arabic']:
            hints['likely_english'] = False
        
        return hints
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def truncate_text(self, text: str, max_length: int = 8000, 
                     preserve_words: bool = True) -> str:
        """Truncate text intelligently."""
        if len(text) <= max_length:
            return text
        
        if preserve_words:
            # Find the last complete word within the limit
            truncated = text[:max_length-3]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # Only if we don't lose too much
                truncated = truncated[:last_space]
            return truncated + "..."
        else:
            return text[:max_length-3] + "..."

# Global instance for convenience
text_processor = TextProcessor()

# Convenience functions
def clean_text(text: str) -> str:
    """Clean text using the global processor."""
    return text_processor.clean_text(text)

def preprocess_text(text: str, **kwargs) -> str:
    """Preprocess text using the global processor."""
    return text_processor.preprocess_text(text, **kwargs)

def extract_features(text: str) -> Dict[str, Any]:
    """Extract features using the global processor."""
    return text_processor.extract_features(text)

def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """Extract keywords using the global processor."""
    return text_processor.extract_keywords(text, num_keywords)

def validate_text_input(text: str, 
                       min_length: int = 1,
                       max_length: int = 10000,
                       allow_empty: bool = False) -> Tuple[bool, str]:
    """Validate text input with detailed error messages."""
    if not text and not allow_empty:
        return False, "Text cannot be empty"
    
    if not text:
        text = ""
    
    if len(text) < min_length:
        return False, f"Text must be at least {min_length} characters long"
    
    if len(text) > max_length:
        return False, f"Text must be no more than {max_length} characters long"
    
    # Check for suspicious patterns
    if len(text.strip()) == 0 and not allow_empty:
        return False, "Text cannot be only whitespace"
    
    # Check for extremely repetitive content
    words = text.split()
    if len(words) > 10:
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.1:  # Less than 10% unique words
            return False, "Text appears to be too repetitive"
    
    return True, "Valid"

def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and normalizing unicode."""
    return text_processor.clean_text(text)

def truncate_text(text: str, max_length: int = 8000) -> str:
    """Truncate text to maximum length."""
    return text_processor.truncate_text(text, max_length)