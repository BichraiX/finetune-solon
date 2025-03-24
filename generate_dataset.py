#!/usr/bin/env python3
"""
Generate Synthetic Dataset with Deepseek-v3

This script generates a synthetic dataset of (query, relevant documents) pairs from a corpus
of legal documents without labelers by leveraging Deepseek-v3.
"""

import json
import uuid
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from logging.handlers import RotatingFileHandler
import torch
from tqdm import tqdm
import unicodedata
import urllib.request  # For downloading tesseract language files
import hashlib  # For generating cache file names based on file content
import shutil
from datetime import datetime
from glob import glob

import pypdf
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
# For BM25-based search
from rank_bm25 import BM25Okapi

# For improved sentence-based chunking for French text
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
from nltk.data import load
try:
    # Load the French punkt tokenizer model
    tokenizer_fr = load('tokenizers/punkt/french.pickle')
except:
    # Fall back to downloading it if not available
    nltk.download('punkt', quiet=True)

# For OCR capabilities
try:
    import pytesseract
    import pdf2image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    logging.warning("pytesseract or pdf2image not installed. OCR fallback will not be available.")

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Add file handler for query logging
LOGS_DIR = './logs'
Path(LOGS_DIR).mkdir(exist_ok=True)
query_logger = logging.getLogger('query_logger')
query_logger.setLevel(logging.INFO)
query_file_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, 'generated_queries.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
query_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
query_logger.addHandler(query_file_handler)
query_logger.propagate = False  # Don't propagate to parent logger

# Output paths
DATA_DIR = '/Data/amine.chraibi/rag/data_mistral'
Path(DATA_DIR).mkdir(exist_ok=True)
CACHE_DIR = './cache'
Path(CACHE_DIR).mkdir(exist_ok=True)
OCR_CACHE_DIR = os.path.join(CACHE_DIR, 'ocr')
Path(OCR_CACHE_DIR).mkdir(exist_ok=True)
TESSDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tessdata')
os.makedirs(TESSDATA_DIR, exist_ok=True)

# Models directory
MODELS_DIR = '/Data/amine.chraibi/models'
os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_CORPUS_FPATH = f'{DATA_DIR}/train_corpus.json'
VAL_CORPUS_FPATH = f'{DATA_DIR}/val_corpus.json'
TRAIN_QUERIES_FPATH = f'{DATA_DIR}/train_queries.json'
TRAIN_RELEVANT_DOCS_FPATH = f'{DATA_DIR}/train_relevant_docs.json'
VAL_QUERIES_FPATH = f'{DATA_DIR}/val_queries.json'
VAL_RELEVANT_DOCS_FPATH = f'{DATA_DIR}/val_relevant_docs.json'
TRAIN_DATASET_FPATH = f'{DATA_DIR}/train_dataset.json'
VAL_DATASET_FPATH = f'{DATA_DIR}/val_dataset.json'

# Set environment variable for tesseract language data
os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
# Set environment variable for Transformers cache
os.environ['TRANSFORMERS_CACHE'] = MODELS_DIR
os.environ['HF_HOME'] = MODELS_DIR

# Chunk parameters (in approximate words)
CHUNK_WORD_LIMIT = 1000
OVERLAP_WORDS = 200

# Hard negative settings
NUM_HARD_NEGATIVES = 3  # Number of hard negatives to include per query

# List of refusal phrases for filtering (from generate_reformulation_dataset.py)
REFUSAL_PHRASES = [
    "désolé", 
    "je ne peux pas",
    "je regrette",
    "je suis navré",
    "impossible",
    "en tant qu'assistant",
    "en tant qu'ia",
    "assistant ia",
    "modèle de langage",
    "model disclaimer",
    "je ne suis pas autorisé",
    "je ne suis pas en mesure"
]

# Add a constant for checkpointing
CHECKPOINT_INTERVAL = 500  # Save checkpoint every 500 examples
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class EmbeddingModel:
    """Wrapper for sentence embedding models to be used for quality assessment."""
    
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", model_dir=MODELS_DIR):
        """
        Initialize the embedding model.
        Default is a multilingual SBERT model that works well with French text.
        """
        logging.info(f"Loading embedding model: {model_name} from {model_dir}")
        try:
            # Set cache directory for model and tokenizer
            cache_dir = os.path.join(model_dir, 'embedding_models')
            os.makedirs(cache_dir, exist_ok=True)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            
            # Move model to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            logging.info(f"Embedding model loaded on {self.device}")
            
            self.model_name = model_name
            self.is_initialized = True
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            self.is_initialized = False
    
    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        Returns a matrix of shape (len(texts), embedding_dimension).
        """
        if not self.is_initialized:
            logging.error("Embedding model not initialized")
            return np.array([])
        
        # Mean Pooling function for BERT-like models
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        all_embeddings = []
        
        # Process in batches to avoid OOM issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and prepare inputs
            try:
                encoded_input = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model output
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                
                # Apply mean pooling to get sentence embeddings
                embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Move to CPU and convert to numpy
                all_embeddings.append(embeddings.cpu().numpy())
            
            except Exception as e:
                logging.error(f"Error during embedding batch {i}-{i+batch_size}: {e}")
                # Return empty batch for this part
                shape = (len(batch_texts), self.model.config.hidden_size)
                all_embeddings.append(np.zeros(shape))
        
        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])
    
    def compute_similarities(self, queries: List[str], docs: List[str]) -> List[float]:
        """Compute cosine similarities between queries and corresponding documents."""
        if len(queries) != len(docs):
            logging.error(f"Number of queries ({len(queries)}) != number of docs ({len(docs)})")
            return []
        
        if not queries:
            return []
        
        # Get embeddings
        query_embeddings = self.encode(queries)
        doc_embeddings = self.encode(docs)
        
        if len(query_embeddings) == 0 or len(doc_embeddings) == 0:
            return []
        
        # Compute cosine similarities
        similarities = []
        for i in range(len(queries)):
            if i < len(query_embeddings) and i < len(doc_embeddings):
                sim = np.dot(query_embeddings[i], doc_embeddings[i])
                similarities.append(float(sim))
        
        return similarities

class BM25Scorer:
    """Wrapper for BM25 scoring to evaluate query-document relevance."""
    
    def __init__(self):
        """Initialize the BM25 scorer."""
        self.is_initialized = True
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 processing."""
        # Simple tokenization by splitting on whitespace and lowercasing
        return text.lower().split()
    
    def compute_similarities(self, queries: List[str], docs: List[str]) -> List[float]:
        """Compute BM25 scores between queries and corresponding documents."""
        if len(queries) != len(docs):
            logging.error(f"Number of queries ({len(queries)}) != number of docs ({len(docs)})")
            return []
        
        if not queries:
            return []
        
        similarities = []
        
        for i, (query, doc) in enumerate(zip(queries, docs)):
            # Tokenize the document and query
            tokenized_doc = self.tokenize_text(doc)
            tokenized_query = self.tokenize_text(query)
            
            # Skip empty documents or queries
            if not tokenized_doc or not tokenized_query:
                similarities.append(0.0)
                continue
            
            # Create a BM25 index with just this document
            bm25 = BM25Okapi([tokenized_doc])
            
            # Get BM25 score
            score = bm25.get_scores(tokenized_query)[0]
            
            # Normalize score to be in [0,1] range
            # BM25 scores are unbounded, so we'll use a simple normalization
            # This is a heuristic approach - may need adjustment based on your corpus
            normalized_score = min(1.0, score / 10.0)
            
            similarities.append(float(normalized_score))
        
        return similarities

def init_llm(model_name="mistralai/Mistral-7B-v0.3", model_dir=MODELS_DIR):
    """Initialize Mistral model for text generation using a specific cache directory"""
    cache_subdir = os.path.join(model_dir, 'llm_models')
    os.makedirs(cache_subdir, exist_ok=True)
    
    logging.info(f"Loading {model_name} model and tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_subdir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_subdir
    )
    logging.info(f"Model loaded successfully from {model_dir}")
    return model, tokenizer

def setup_tesseract_language(lang_code='fra'):
    """
    Set up Tesseract OCR with specified language support.
    Creates a local tessdata directory and downloads language files if needed.
    
    Args:
        lang_code: Language code (default: 'fra' for French)
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    if not HAS_OCR:
        return False
        
    try:
        # Create tessdata directory in the current folder
        os.makedirs(TESSDATA_DIR, exist_ok=True)
        
        # Check if language file already exists
        lang_file = os.path.join(TESSDATA_DIR, f'{lang_code}.traineddata')
        if os.path.exists(lang_file) and os.path.getsize(lang_file) > 0:
            logging.info(f"Tesseract language file {lang_code}.traineddata already exists")
            return True
        
        # If not, download the language file
        logging.info(f"Downloading Tesseract language file for {lang_code}...")
        url = f"https://github.com/tesseract-ocr/tessdata/raw/main/{lang_code}.traineddata"
        
        urllib.request.urlretrieve(url, lang_file)
        
        if os.path.exists(lang_file) and os.path.getsize(lang_file) > 0:
            logging.info(f"Successfully downloaded {lang_code}.traineddata")
            return True
        else:
            logging.error(f"Failed to download {lang_code}.traineddata")
            return False
    except Exception as e:
        logging.error(f"Error setting up Tesseract language {lang_code}: {e}")
        return False

def contains_refusal_phrases(text: str) -> bool:
    """
    Check if the text contains phrases that indicate a model refusal or non-compliance.
    
    Args:
        text: The text to check
        
    Returns:
        True if the text contains refusal phrases, False otherwise
    """
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check if any refusal phrase is in the text
    for phrase in REFUSAL_PHRASES:
        if phrase in text_lower:
            return True
    
    return False

def normalize_french_text(text: str) -> str:
    """Normalize French text: replace common OCR errors, fix spacing, etc."""
    # Normalize Unicode characters (NFD to NFC)
    text = unicodedata.normalize('NFC', text)
    
    # Fix common spacing issues
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'(\d)\.(\d)', r'\1,\2', text)  # European decimal format
    
    # Fix common OCR errors with French accents
    text = re.sub(r'e\´', 'é', text)
    text = re.sub(r'e\`', 'è', text)
    text = re.sub(r'a\`', 'à', text)
    text = re.sub(r'e\^', 'ê', text)
    
    # Remove any control characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    return text.strip()

def get_file_hash(file_path: str) -> str:
    """
    Generate a hash for a file based on its path and modification time.
    This creates a unique identifier for the file that changes when the file is modified.
    
    Args:
        file_path: Path to the file
        
    Returns:
        A hash string that uniquely identifies the file version
    """
    try:
        file_stat = os.stat(file_path)
        file_mod_time = file_stat.st_mtime
        file_size = file_stat.st_size
        
        # Create a unique hash based on file path, modification time, and size
        unique_id = f"{file_path}_{file_mod_time}_{file_size}"
        return hashlib.md5(unique_id.encode()).hexdigest()
    except Exception as e:
        logging.error(f"Error generating file hash for {file_path}: {e}")
        # Fallback to just using the filename if error occurs
        return hashlib.md5(file_path.encode()).hexdigest()

def extract_text_with_ocr(file_path: str, lang_code: str = 'fra') -> str:
    """
    Extract text from a PDF file using OCR with caching.
    
    Args:
        file_path: Path to the PDF file
        lang_code: Language code for OCR (default: 'fra' for French)
        
    Returns:
        Extracted text
    """
    if not HAS_OCR:
        logging.error("OCR dependencies not available. Install pytesseract and pdf2image.")
        return ""
    
    # Generate a hash for the file to use as cache key
    file_hash = get_file_hash(file_path)
    cache_file = os.path.join(OCR_CACHE_DIR, f"{file_hash}_{lang_code}_ocr.txt")
    
    # Check if we have a cached version
    if os.path.exists(cache_file):
        logging.info(f"Loading cached OCR text for {file_path}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_text = f.read()
            if cached_text:
                return cached_text
            logging.warning(f"Cached OCR file was empty, reprocessing {file_path}")
        except Exception as e:
            logging.warning(f"Error reading cached OCR file: {e}, reprocessing {file_path}")
    
    try:
        # Setup OCR language
        setup_tesseract_language(lang_code)
        
        logging.info(f"Using OCR to extract text from {file_path}")
        
        # Convert PDF to images
        images = pdf2image.convert_from_path(file_path)
        
        # Extract text from each page using OCR
        text = ""
        for i, img in enumerate(images):
            logging.info(f"Performing OCR on page {i+1}/{len(images)}")
            page_text = pytesseract.image_to_string(img, lang=lang_code)
            if page_text:
                text += page_text + "\n"
                
        # Normalize the OCR text
        normalized_text = normalize_french_text(text)
        
        # Cache the OCR result
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(normalized_text)
            logging.info(f"Cached OCR text to {cache_file}")
        except Exception as e:
            logging.error(f"Error caching OCR text: {e}")
        
        return normalized_text
    except Exception as e:
        logging.error(f"Error performing OCR on {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file with error handling and French text normalization."""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
    
    # If text extraction failed or returned very little text, try OCR
    if not text or len(text.strip()) < 100:
        logging.warning(f"Regular text extraction failed or returned minimal text. Trying OCR for {file_path}")
        text = extract_text_with_ocr(file_path)
    
    # Normalize the extracted text
    return normalize_french_text(text)

def cache_extracted_text(file_path: str) -> str:
    """Use caching to avoid reprocessing PDFs."""
    # Generate a filename for the cache file based on original filename
    base_filename = os.path.basename(file_path)
    cache_file = os.path.join(CACHE_DIR, f"{base_filename}.txt")
    
    # Check if we have a cached version
    if os.path.exists(cache_file):
        # Check if the file has been modified since we cached it
        file_hash = get_file_hash(file_path)
        hash_file = os.path.join(CACHE_DIR, f"{base_filename}.hash")
        
        cached_hash = ""
        if os.path.exists(hash_file):
            try:
                with open(hash_file, 'r') as f:
                    cached_hash = f.read().strip()
            except Exception as e:
                logging.warning(f"Error reading hash file: {e}")
        
        if cached_hash == file_hash:
            logging.info(f"Loading cached text for {file_path}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            logging.info(f"File {file_path} has been modified, reprocessing")
    
    # If not cached or modified, extract text
    logging.info(f"Extracting text from {file_path}")
    text = extract_text_from_pdf(file_path)
    
    # Save extracted text and hash
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Save the hash of the processed file
        file_hash = get_file_hash(file_path)
        hash_file = os.path.join(CACHE_DIR, f"{base_filename}.hash")
        with open(hash_file, 'w') as f:
            f.write(file_hash)
            
        logging.info(f"Cached text and hash for {file_path}")
    except Exception as e:
        logging.error(f"Error caching text: {e}")
    
    return text

def chunk_text(text: str, chunk_word_limit: int = CHUNK_WORD_LIMIT, overlap_words: int = OVERLAP_WORDS) -> List[str]:
    """
    Split text into chunks using advanced context-aware techniques.
    Each chunk contains approximately chunk_word_limit words,
    with an overlap of approximately overlap_words words between chunks.
    
    This improved version tries to split at paragraph boundaries when possible
    and maintains coherent sections.
    """
    # First try to use langchain's text splitter which has more advanced chunking
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # Estimate average word length (including spaces) for this document
        avg_word_length = len(text) / max(1, len(text.split()))
        
        # Convert word limits to character limits (approximate)
        chunk_size = int(chunk_word_limit * avg_word_length)
        chunk_overlap = int(overlap_words * avg_word_length)
        
        # Create text splitter with paragraph-aware splitting
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "! ", "? ", ";", ":", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Split text using langchain's recursive splitter
        chunks = text_splitter.split_text(text)
        
        # Apply sentence boundary detection to avoid cutting in the middle of sentences
        refined_chunks = []
        for chunk in chunks:
            # Try to end chunks at sentence boundaries if they're not already
            if not chunk.endswith('.') and not chunk.endswith('!') and not chunk.endswith('?'):
                # Find the last sentence boundary
                last_period = max(chunk.rfind('. '), chunk.rfind('! '), chunk.rfind('? '))
                if last_period > 0 and last_period > len(chunk) * 0.7:  # If we can find a good boundary
                    # Add 2 to include the period and space
                    refined_chunks.append(chunk[:last_period + 2])
                    # The remainder becomes part of the next chunk via overlap
                else:
                    refined_chunks.append(chunk)
            else:
                refined_chunks.append(chunk)
        
        # Check if we got reasonable chunks
        if refined_chunks and all(len(chunk.split()) >= min(20, len(text.split())) for chunk in refined_chunks):
            return refined_chunks
    except ImportError:
        logging.warning("langchain not available, falling back to basic chunking")
    except Exception as e:
        logging.warning(f"Error in advanced chunking: {e}, falling back to basic chunking")
    
    # Fallback to the original sentence-based method if the above fails
    try:
        # Use French-specific sentence tokenization
        sentences = sent_tokenize(text, language='french')
    except:
        # Fallback to default tokenizer if French tokenizer fails
        sentences = sent_tokenize(text)
    
    # Check for extremely long sentences that might indicate poor tokenization
    avg_sent_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    if avg_sent_length > 50:
        logging.warning(f"Very long average sentence length detected ({avg_sent_length:.1f} words). "
                       f"Text might not be properly tokenized.")
    
    # Compute word counts for each sentence
    word_counts = [len(sentence.split()) for sentence in sentences]
    
    chunks = []
    i = 0
    while i < len(sentences):
        current_words = 0
        j = i
        while j < len(sentences) and current_words < chunk_word_limit:
            current_words += word_counts[j]
            j += 1
        
        # Ensure we always make progress
        if j == i:
            j = i + 1
            
        chunk = " ".join(sentences[i:j])
        chunks.append(chunk)
        
        # Determine overlap: backtrack until approximately overlap_words are reached
        overlap = 0
        k = j - 1
        while k >= i and overlap < overlap_words:
            overlap += word_counts[k]
            k -= 1
            
        # Ensure progress even if overlap_words cannot be met
        i = max(i + 1, k + 1)
    
    return chunks

def load_corpus(files: List[str], verbose: bool = False) -> Dict[str, str]:
    """Load and chunk PDF documents into a corpus using sequential processing and caching."""
    corpus = {}

    def process_file(file_path: str) -> List[Tuple[str, str]]:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            return []
        text = cache_extracted_text(file_path)
        chunks = chunk_text(text)
        return [(str(uuid.uuid4()), chunk) for chunk in chunks]

    # Process each file sequentially
    for fp in files:
        if verbose:
            logging.info(f"Processing file: {fp}")
        file_chunks = process_file(fp)
        for node_id, chunk in file_chunks:
                corpus[node_id] = chunk

    if verbose:
        logging.info(f"Loaded and chunked into {len(corpus)} text chunks")
    return corpus

def generate_text(model, tokenizer, prompt: str, max_tokens: int = 1024) -> str:
    """
    Generate text from the given prompt using the LLM.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        prompt: The prompt to generate text from
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        The generated text
    """
    try:
        # Create inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        return ""

def parse_questions(text: str) -> List[str]:
    """
    Parse questions from the model's response.
    Uses multiple strategies to extract properly formatted questions.
    """
    questions = []
    
    # Clean the text first
    text = text.strip()
    
    # Remove any preamble text before the questions start
    if "1." in text:
        text = text[text.find("1."):]
    
    # Try to find numbered questions (e.g., "1. What is...")
    numbered_pattern = re.compile(r'(?:^|\n)\s*(\d+\.?\s*)(.*?)(?=\n\s*\d+\.|\n\s*$|$)', re.DOTALL)
    matches = numbered_pattern.findall(text)
    
    if matches:
        for number, question in matches:
            # Clean the question text
            clean_question = question.strip()
            
            # Skip if it's just a number with no content
            if not clean_question:
                continue
                
            # Skip placeholders like "First question", "Second question", etc.
            if re.match(r"^(?:premi[èe]re?|deuxi[èe]me|troisi[èe]me|quatri[èe]me|cinqui[èe]me|sixi[èe]me|septi[èe]me|huiti[èe]me|neuvi[èe]me|dixi[èe]me|derni[èe]re)\s+question\s*\??$", clean_question, re.IGNORECASE):
                continue
            
            # Add question mark if missing but question is otherwise valid
            if clean_question and not clean_question.endswith('?'):
                if not any(clean_question.endswith(punct) for punct in ['.', '!', ';', ':']):
                    clean_question += '?'
                
            # Skip instructions, examples, and other non-question content
            skip_markers = ["exemple", "example", "format", "sortie", "output", "voici", "réponds", "comme suit", "suivant"]
            if any(marker in clean_question.lower() for marker in skip_markers) and '?' not in clean_question:
                continue
                
            # Only add if it's actually a question or starts with common question words
            is_question = (
                '?' in clean_question or 
                re.search(r'^(?:pourquoi|comment|qui|quand|où|quel|quelle|quels|quelles|combien|est-ce|que|qu\')', 
                         clean_question.lower())
            )
            
            if is_question and clean_question:
                questions.append(clean_question)
        
        if questions:
            return questions
    
    # If no numbered questions found, try to find bullet points
    bullet_pattern = re.compile(r'(?:^|\n)\s*[-•*]\s*(.*?)(?=\n\s*[-•*]|\n\s*$|$)', re.DOTALL)
    matches = bullet_pattern.findall(text)
    
    if matches:
        for question in matches:
            clean_question = question.strip()
            
            # Skip empty or template questions
            if not clean_question or re.match(r"^(?:premi[èe]re?|deuxi[èe]me|troisi[èe]me|etc\.)\s+question\s*\??$", clean_question, re.IGNORECASE):
                continue
            
            # Add question mark if it seems to be a question but is missing one
            if clean_question and not clean_question.endswith('?'):
                if not any(clean_question.endswith(punct) for punct in ['.', '!', ';', ':']):
                    clean_question += '?'
            
            # Only add if it's actually a question
            if ('?' in clean_question or 
                re.search(r'^(?:pourquoi|comment|qui|quand|où|quel|quelle|quels|quelles|combien|est-ce|que|qu\')', 
                         clean_question.lower())):
                questions.append(clean_question)
        
        if questions:
            return questions
    
    # If no structured format, split by lines and look for question marks or question words
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        
        # Skip empty lines or numbering without content
        if not line or re.match(r'^\d+\.?\s*$', line):
            continue
            
        # Skip placeholder questions
        if re.match(r"^(?:premi[èe]re?|deuxi[èe]me|troisi[èe]me|etc\.)\s+question\s*\??$", line, re.IGNORECASE):
            continue
            
        # Check if it's a proper question
        is_question = (
            '?' in line or 
            any(line.lower().startswith(q) for q in [
                'pourquoi', 'comment', 'qui', 'quand', 'où', 'combien',
                'quel', 'quelle', 'quels', 'quelles', 
                'est-ce', 'peut-on', 'pouvez-vous', 'que', 'qu\'', 'quoi'
            ])
        )
        
        if is_question:
            clean_question = line.strip()
            if clean_question and not clean_question.endswith('?'):
                clean_question += '?'
            questions.append(clean_question)
    
    return questions

def contains_refusal_phrase(text: str) -> bool:
    """
    Check if text contains AI refusal phrases or template/example markers.
    More comprehensive for French text.
    """
    # Standard refusal phrases
    refusal_phrases = [
        # French refusal phrases
        "je suis désolé", "désolé", "je m'excuse", "navrée?", 
        "je ne peux pas", "ne peux pas fournir", "impossible de",
        "incapable de", "je n'ai pas", "n'ai pas accès", "pas d'accès",
        "je suis une ia", "je suis un assistant", "en tant qu'ia", "en tant qu'assistant", 
        "je ne suis pas capable", "au-delà de mes", "contre ma programmation",
        "contre la politique", "interdit de", "pas approprié", "inapproprié",
        "je suis juste un", "je suis seulement un", "non autorisé",
        "mes capacités", "limites", "mes limites", "mes limitations",
        "ne permet pas de", "ne me permet pas", "ne peux pas vous aider",
        
        # English refusal phrases (for completeness)
        "i'm sorry", "i apologize", "i cannot", "cannot provide", 
        "unable to", "i don't have", "don't have access", "no access",
        "i'm an ai", "as an ai", "as an assistant", "i'm not able",
        "beyond my", "against my programming", "against policy",
        "prohibited from", "not appropriate", "inappropriate",
        "i'm just an", "i'm only an",
        
        # Example/template markers
        "exemple", "example", "voici", "format de sortie", "output format",
        "format de réponse", "réponse attendue", "comme suit", "following format",
        "###", "```", "instruction", "modèle de réponse", "pour référence"
    ]
    
    # Template patterns (regex)
    template_patterns = [
        r"^\d+\.\s*(?:Première|Deuxième|Troisième|Quatrième|Cinquième|Sixième|Septième|Huitième|Neuvième|Dixième|Dernière)\s+question",
        r"(?:Première|Deuxième|Troisième|Quatrième|Cinquième|Sixième|Septième|Huitième|Neuvième|Dixième|Dernière)\s+question\s*$",
        r"(?:Première|Deuxième|Troisième|Quatrième|Cinquième|Sixième|Septième|Huitième|Neuvième|Dixième|Dernière)\s+question\s*\?$",
        r"exemple d[e'](?:une)? (?:réponse|question)",
        r"exemples? de (?:format|sortie)",
    ]
    
    text_lower = text.lower()
    
    # Check for phrase matches
    if any(phrase in text_lower for phrase in refusal_phrases):
        return True
    
    # Check for regex pattern matches
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in template_patterns):
        return True
    
    return False

def generate_queries(
    corpus: Dict[str, str],
    model,
    tokenizer,
    max_length: int = 650,
    num_queries_per_doc: int = 15,
    use_filter: bool = True,
    resume_from_checkpoint: bool = True
) -> Dict[str, Tuple[str, str]]:
    """
    Generate synthetic queries for the given corpus using the LLM.
    Returns a dictionary mapping query IDs to tuples of (doc_id, query).
    
    Args:
        corpus: Dictionary mapping document IDs to document texts
        model: Language model to use for generation
        tokenizer: Tokenizer for the language model
        max_length: Maximum length of context to provide to the model
        num_queries_per_doc: Number of queries to generate per document
        use_filter: Whether to filter out queries that may indicate refusal
        resume_from_checkpoint: Whether to resume from the latest checkpoint if available
    
    Returns:
        Dictionary mapping query IDs to tuples of (doc_id, query)
    """
    # Try to load from checkpoint if requested
    queries = {}
    checkpoint_number = 0
    
    if resume_from_checkpoint:
        loaded_queries, checkpoint_number = load_latest_checkpoint()
        if loaded_queries:
            queries = loaded_queries
            query_count = max(int(qid.replace('q', '')) for qid in queries.keys()) + 1 if queries else 0
            
            # Determine which documents have already been processed
            processed_docs = set(doc_id for _, (doc_id, _) in queries.items())
            corpus = {doc_id: text for doc_id, text in corpus.items() if doc_id not in processed_docs}
            
            logging.info(f"Resuming from checkpoint {checkpoint_number} with {len(queries)} existing queries")
            logging.info(f"Remaining documents to process: {len(corpus)}")
    else:
        query_count = 0
    
    logging.info(f"Generating synthetic queries for {len(corpus)} documents...")
    
    # Create a tqdm progress bar for document processing
    pbar = tqdm(corpus.items(), desc="Generating queries", total=len(corpus))
    
    # Keep track of documents that failed to generate good queries
    failed_docs = []
    retry_docs = []
    
    for doc_id, text in pbar:
        # Skip empty or very short documents
        if not text or len(text.split()) < 20:
            continue
        
        chunk_text = text[:max_length] if len(text) > max_length else text
        
        # Updated prompt with more comprehensive instructions and legal expertise emphasis
        prompt = f"""Tu es un assistant juridique expérimenté, chargé de générer des questions précises et spécifiques qu'un professionnel du droit pourrait poser à propos du document fourni. Ces questions serviront dans un système de recherche juridique spécialisé.

Contexte : {chunk_text}

### Instructions précises pour questions juridiques spécifiques :
1. Analyse méticuleusement le contenu, les faits spécifiques, les noms des personnes, des sociétés, les dates et les montants mentionnés dans le document ci-dessus.
2. Génère exactement {num_queries_per_doc} questions EN FRANÇAIS, formulées avec précision juridique.
3. Les questions DOIVENT être extrêmement spécifiques et faire référence aux entités explicitement nommées dans le document (noms propres, entreprises, montants exacts, dates, lieux, etc.).
4. Par exemple, utilise "Quelle est la fonction de Me Jean Dupont dans l'acte de vente du 12 mars 2022 ?" plutôt que "Quelle est la fonction de la personne mentionnée ?".
5. Évite ABSOLUMENT les questions vagues qui supposent une connaissance préalable du document comme "Que stipule le contrat ?", "Que doit faire le cessionnaire ?", ou "Que dit le document à propos de la clause ?"
6. Chaque question doit être autonome et compréhensible SANS avoir lu le document au préalable.
7. Pour les termes juridiques spécifiques (cessionnaire, cédant, etc.), inclus toujours leur identification précise, par exemple : "Quelles sont les obligations de la société XYZ en tant que cessionnaire selon l'acte du [date] ?"
8. Les questions doivent couvrir proportionnellement les différentes parties et aspects du document.
9. Utilise un langage juridique précis et formel, comme un avocat ou un notaire le ferait.
10. Évite absolument de copier des phrases directement du document.

### Exemples de bonnes formulations :
✓ "Quelle est la date précise de signature du contrat entre la SCI MATHO et la société GUICLA ?"
✓ "Quel montant M. Thomas ALBERT a-t-il reçu lors de la cession de ses actions le 15 juin 2023 ?"
✓ "Quelles sont les conditions suspensives mentionnées dans l'article 4 du protocole d'accord ?"

### Exemples de formulations à éviter absolument :
✗ "Que dit le document sur le cessionnaire ?"
✗ "Quelles sont les obligations mentionnées dans le contrat ?"
✗ "Qui est la personne en question ?"

### Format de sortie (RESPECTE STRICTEMENT CE FORMAT) :
1. Première question en français (spécifique et contextuelle)
2. Deuxième question en français (spécifique et contextuelle)
3. Troisième question en français (spécifique et contextuelle)
...
{num_queries_per_doc}. Dernière question en français (spécifique et contextuelle)

N'ajoute rien d'autre avant ou après les questions numérotées.
"""

        try:
            # Get model response and parse questions
            response = generate_text(model, tokenizer, prompt)
            
            if not response:
                failed_docs.append(doc_id)
                continue
                
            questions = parse_questions(response)
            
            # Additional filtering for malformed questions
            filtered_questions = []
            seen_questions = set()  # For duplicate detection
            
            for question in questions:
                # Skip very short questions
                if not question or len(question) < 10:
                    continue
                
                # Filter out questions containing specific templates or markers
                skip_markers = [
                    "exemple", "example", "sortie", "output", "voici", "question",
                    "###", "```", "réponse", "answer"
                ]
                
                should_skip = False
                for marker in skip_markers:
                    if marker.lower() in question.lower():
                        logging.debug(f"Skipping question with template marker: {question[:50]}...")
                        should_skip = True
                        break
                
                # Skip questions with minimal content (e.g. "Quatrième question?")
                if re.match(r"^[a-zA-ZÀ-ÿ]+\s+question\s*\??$", question, re.IGNORECASE):
                    logging.debug(f"Skipping placeholder question: {question}")
                    should_skip = True
                
                # Skip questions with refusal phrases if requested
                if use_filter and contains_refusal_phrase(question):
                    should_skip = True
                
                # Check if the question is in French
                if not is_french_text(question):
                    logging.debug(f"Skipping non-French question: {question[:50]}...")
                    should_skip = True
                
                if not should_skip:
                    # Normalize for duplicate detection
                    normalized_question = normalize_for_deduplication(question)
                    if normalized_question not in seen_questions:
                        filtered_questions.append(question)
                        seen_questions.add(normalized_question)
            
            # If we have enough good questions, add them to the results
            if len(filtered_questions) >= num_queries_per_doc * 0.7:  # At least 70% of requested number
                for question in filtered_questions[:num_queries_per_doc]:
                    query_id = f"q{query_count}"
                    queries[query_id] = (doc_id, question)
                    query_logger.info(f"Generated query {query_id}: '{question}' for doc {doc_id}")
                    query_count += 1
            else:
                # If not enough good questions, add this document to retry list
                retry_docs.append((doc_id, chunk_text))
                logging.debug(f"Not enough good questions for {doc_id}, will retry. Got {len(filtered_questions)}")
                
            # Update progress bar with current count
            pbar.set_postfix({"queries": len(queries), "retries": len(retry_docs), "failed": len(failed_docs)})
            
            # Save checkpoint periodically
            if query_count % CHECKPOINT_INTERVAL == 0 and query_count > 0:
                checkpoint_number += 1
                save_checkpoint(queries, corpus, checkpoint_number)
                
        except Exception as e:
            logging.warning(f"Error generating questions for {doc_id}: {e}")
            failed_docs.append(doc_id)
    
    # Process retry documents if needed
    if retry_docs and len(retry_docs) > 0:
        logging.info(f"Retrying generation for {len(retry_docs)} documents with a simpler prompt...")
        
        for doc_id, chunk_text in retry_docs:
            # Use a simpler, more direct prompt for retry
            retry_prompt = f"""En tant qu'assistant juridique spécialisé, génère {num_queries_per_doc} questions PRÉCISES et SPÉCIFIQUES en français basées sur ce document juridique :

"{chunk_text[:300]}..."

Les questions DOIVENT :
- Faire référence explicite aux noms propres, entités, dates, montants et articles mentionnés dans le texte
- Être formulées de manière autonome et compréhensible pour quelqu'un qui n'a pas lu le document
- Utiliser un langage juridique précis et formel
- Éviter toute ambiguïté comme "Que dit le document sur..." ou "Que stipule le contrat..."
- Spécifier exactement de quelles personnes, entités ou clauses il s'agit, par exemple : "Quelles sont les obligations de la société X selon l'article Y du contrat daté du Z ?"

Format requis : uniquement des questions numérotées de 1 à {num_queries_per_doc}, toutes avec point d'interrogation."""
            
            try:
                retry_response = generate_text(model, tokenizer, retry_prompt)
                if not retry_response:
                    continue
                
                retry_questions = parse_questions(retry_response)
                
                # Apply the same filtering
                valid_questions = []
                seen_questions = set()  # For duplicate detection
                
                for question in retry_questions:
                    # Apply all filters
                    if (len(question) >= 10 and 
                        not contains_refusal_phrase(question) and
                        not re.match(r"^[a-zA-ZÀ-ÿ]+\s+question\s*\??$", question, re.IGNORECASE) and
                        is_french_text(question)):
                        
                        # Check for duplicates
                        normalized_question = normalize_for_deduplication(question)
                        if normalized_question not in seen_questions:
                            valid_questions.append(question)
                            seen_questions.add(normalized_question)
                
                # Add valid questions
                for question in valid_questions[:num_queries_per_doc]:
                    query_id = f"q{query_count}"
                    queries[query_id] = (doc_id, question)
                    query_logger.info(f"Generated query {query_id} (retry): '{question}' for doc {doc_id}")
                    query_count += 1
                
                # Save checkpoint periodically during retries as well
                if query_count % CHECKPOINT_INTERVAL == 0 and query_count > 0:
                    checkpoint_number += 1
                    save_checkpoint(queries, corpus, checkpoint_number)
            
            except Exception as e:
                logging.warning(f"Retry failed for document {doc_id}: {e}")
    
    # If we have a high failure rate, generate some general questions as fallbacks
    if len(failed_docs) > (0.3 * len(corpus)) and query_count < (0.5 * len(corpus) * num_queries_per_doc):
        logging.warning(f"High failure rate detected ({len(failed_docs)} failed documents). Generating fallback questions...")
        
        # Generate some general legal/document questions that might be relevant
        fallback_prompt = """En tant qu'expert juridique, génère 30 questions précises qu'un professionnel du droit pourrait poser à propos d'un document juridique français. 

Les questions doivent :
1. Être spécifiques et faire référence à des éléments juridiques concrets (ex: "Quelles sont les mentions obligatoires d'un acte de cession de parts sociales en droit français ?")
2. Utiliser un langage juridique approprié et formel
3. Couvrir différents aspects comme les définitions légales, les parties impliquées, les obligations contractuelles, les dates d'effet, les montants financiers, etc.
4. Être formulées de manière précise et sans ambiguïté
5. Éviter les questions vagues comme "Que dit le document ?"

Numérote chaque question et assure-toi qu'elles soient toutes différentes et substantielles."""
        
        try:
            fallback_response = generate_text(model, tokenizer, fallback_prompt)
            if fallback_response:
                fallback_questions = parse_questions(fallback_response)
                
                # Add fallback questions to failed documents
                for failed_doc_id in failed_docs:
                    # Get the document text
                    failed_doc_text = corpus.get(failed_doc_id, "")
                    if not failed_doc_text:
                        continue
                    
                    # Use up to 5 fallback questions per failed document
                    for i in range(min(5, len(fallback_questions))):
                        if i < len(fallback_questions):
                            query_id = f"q{query_count}"
                            queries[query_id] = (failed_doc_id, fallback_questions[i])
                            query_logger.info(f"Generated fallback query {query_id}: '{fallback_questions[i]}' for doc {failed_doc_id}")
                            query_count += 1
                            
                            # Remove used question to avoid duplicates
                            fallback_questions.pop(i)
                
                # Save checkpoint after adding fallback questions
                checkpoint_number += 1
                save_checkpoint(queries, corpus, checkpoint_number)
        except Exception as e:
            logging.error(f"Error generating fallback questions: {e}")
    
    # Final checkpoint at the end
    checkpoint_number += 1
    save_checkpoint(queries, corpus, checkpoint_number)
    
    logging.info(f"Generated {len(queries)} synthetic queries from {len(corpus)} documents")
    logging.info(f"Failed documents: {len(failed_docs)}, Retry documents: {len(retry_docs)}")
    logging.info(f"Final checkpoint saved: {checkpoint_number}")
    
    return queries

def normalize_for_deduplication(text: str) -> str:
    """
    Normalize text for deduplication purposes by removing punctuation,
    extra whitespace, accents, and converting to lowercase.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def is_french_text(text: str) -> bool:
    """
    Check if the text is likely to be in French.
    Uses heuristics based on French-specific characters and common words.
    """
    # Check if text is too short
    if len(text) < 10:
        return False
    
    # Check for French-specific characters
    french_chars = ['é', 'è', 'ê', 'à', 'ù', 'û', 'ç', 'ô', 'î', 'ï', 'ë', 'ü']
    has_french_chars = any(char in text.lower() for char in french_chars)
    
    # Check for common French words
    french_words = [
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'ce', 'cette', 'ces',
        'est', 'sont', 'et', 'ou', 'pour', 'dans', 'par', 'sur', 'avec', 'qui', 'que',
        'quoi', 'comment', 'pourquoi', 'quand', 'où', 'quel', 'quelle', 'quels', 'quelles'
    ]
    
    words = re.findall(r'\b\w+\b', text.lower())
    french_word_count = sum(1 for word in words if word in french_words)
    french_word_ratio = french_word_count / max(1, len(words))
    
    # Text is likely French if it has French characters or a substantial ratio of French words
    return has_french_chars or french_word_ratio > 0.15

def normalized_discounted_cumulative_gain(relevance_scores: List[float], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) for a list of relevance scores.
    
    Args:
        relevance_scores: List of relevance scores, where higher scores indicate more relevance
        k: Number of top items to consider
        
    Returns:
        NDCG score between 0 and 1
    """
    if not relevance_scores:
        return 0.0
    
    # Take top k scores
    scores = relevance_scores[:k]
    
    # Calculate DCG
    dcg = scores[0]
    for i in range(1, len(scores)):
        dcg += scores[i] / np.log2(i + 2)
    
    # Calculate ideal DCG (IDCG)
    ideal_scores = sorted(scores, reverse=True)
    idcg = ideal_scores[0]
    for i in range(1, len(ideal_scores)):
        idcg += ideal_scores[i] / np.log2(i + 2)
    
    # Calculate NDCG
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(relevance_scores: List[float], k: int = 10, threshold: float = 0.5) -> float:
    """
    Compute Precision@k for a list of relevance scores.
    
    Args:
        relevance_scores: List of relevance scores, where higher scores indicate more relevance
        k: Number of top items to consider
        threshold: Threshold above which an item is considered relevant
        
    Returns:
        Precision@k score between 0 and 1
    """
    if not relevance_scores or k <= 0:
        return 0.0
    
    # Take top k scores
    scores = relevance_scores[:k]
    
    # Count items above threshold
    relevant = sum(1 for score in scores if score >= threshold)
    
    return relevant / len(scores) if scores else 0.0

def mean_average_precision(relevance_lists: List[List[float]], threshold: float = 0.5) -> float:
    """
    Compute Mean Average Precision (MAP) across multiple relevance lists.
    
    Args:
        relevance_lists: List of lists of relevance scores
        threshold: Threshold above which an item is considered relevant
        
    Returns:
        MAP score between 0 and 1
    """
    if not relevance_lists:
        return 0.0
    
    avg_precisions = []
    
    for relevance in relevance_lists:
        # Calculate precision at each relevant position
        precisions = []
        for k in range(1, len(relevance) + 1):
            if relevance[k-1] >= threshold:
                precisions.append(precision_at_k(relevance, k, threshold))
        
        # Calculate average precision
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_precisions.append(avg_precision)
    
    # Calculate mean average precision
    return sum(avg_precisions) / len(avg_precisions) if avg_precisions else 0.0

def evaluate_dataset_quality(
    queries: Dict[str, str], 
    corpus: Dict[str, str], 
    relevant_docs: Dict[str, List[str]],
    embedding_model: Optional[EmbeddingModel] = None,
    quality_threshold: float = 0.5
):
    """
    Compute comprehensive quality metrics for the generated dataset.
    Uses multiple relevance assessment methods including BM25 and neural embeddings.
    Logs warnings if any metrics indicate potential quality issues.
    """
    if not queries or not corpus or not relevant_docs:
        logging.warning("Empty dataset provided for evaluation.")
        return "poor"
    
    logging.info(f"Dataset size: {len(queries)} queries, {len(corpus)} document chunks")
    
    # 1. Basic statistics
    query_lengths = [len(q.split()) for q in queries.values()]
    avg_query_length = np.mean(query_lengths)
    doc_lengths = [len(d.split()) for d in corpus.values()]
    avg_doc_length = np.mean(doc_lengths)
    
    logging.info(f"Average query length: {avg_query_length:.1f} words")
    logging.info(f"Average document chunk length: {avg_doc_length:.1f} words")
    
    if avg_query_length < 5:
        logging.warning("Average query length is very short. Queries might not be sufficiently detailed.")
    
    # 2. Query-document relevance using advanced embedding model and BM25
    query_list, doc_list, query_ids = [], [], []
    for qid, query in queries.items():
        doc_ids = relevant_docs.get(qid, [])
        if doc_ids:
            doc_text = corpus.get(doc_ids[0], "")
            if doc_text:
                query_list.append(query)
                doc_list.append(doc_text)
                query_ids.append(qid)
    
    if not query_list:
        logging.warning("No valid query-document pairs found for evaluation.")
        return "poor"
    
    # Initialize BM25 scorer
    bm25_scorer = BM25Scorer()
    
    # Use advanced embedding model if available
    embedding_similarities = []
    bm25_similarities = []
    
    # Get BM25 similarities
    logging.info("Computing query-document similarities with BM25...")
    bm25_similarities = bm25_scorer.compute_similarities(query_list, doc_list)
    
    if embedding_model and embedding_model.is_initialized:
        logging.info("Computing query-document similarities with advanced embedding model...")
        embedding_similarities = embedding_model.compute_similarities(query_list, doc_list)
    else:
        # Fall back to TF-IDF if advanced model isn't available
        logging.warning("Advanced embedding model not available. Using only BM25 scores.")
        embedding_similarities = bm25_similarities.copy()
    
    # Combine similarities (with weights if you want to emphasize one method)
    similarities = []
    for i in range(len(bm25_similarities)):
        if i < len(embedding_similarities):
            # Equal weight to both methods
            combined_score = 0.5 * bm25_similarities[i] + 0.5 * embedding_similarities[i]
            similarities.append(combined_score)
        else:
            similarities.append(bm25_similarities[i])
    
    if not similarities:
        logging.warning("Failed to compute similarities.")
        return "poor"
    
    # Calculate IR metrics
    ndcg_score = normalized_discounted_cumulative_gain(similarities)
    precision_score = precision_at_k(similarities, k=len(similarities), threshold=quality_threshold)
    
    # For MAP, we need to generate additional similarity lists between queries and multiple documents
    # This is a simplified approach, just using the existing similarities
    map_score = mean_average_precision([similarities], threshold=quality_threshold)
    
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    
    logging.info(f"Query-document similarity: avg={avg_sim:.3f}, min={min_sim:.3f}, max={max_sim:.3f}")
    logging.info(f"IR metrics - NDCG: {ndcg_score:.3f}, Precision@k: {precision_score:.3f}, MAP: {map_score:.3f}")
    
    # Create quality results for each query-document pair
    quality_results = {
        qid: {
            "query": query_list[i], 
            "similarity": similarities[i], 
            "bm25_score": bm25_similarities[i] if i < len(bm25_similarities) else 0,
            "embedding_score": embedding_similarities[i] if i < len(embedding_similarities) else 0,
            "quality": "good" if similarities[i] >= quality_threshold else "poor"
        }
        for i, qid in enumerate(query_ids)
    }
    
    # Identify low-similarity pairs
    low_sim_threshold = 0.4
    low_sim_pairs = [(qid, query_list[i], doc_list[i][:100]+"...", similarities[i]) 
                     for i, qid in enumerate(query_ids) if similarities[i] < low_sim_threshold]
    
    if low_sim_pairs:
        logging.warning(f"Found {len(low_sim_pairs)} query-document pairs with very low similarity (<{low_sim_threshold})")
        for qid, q, d, sim in low_sim_pairs[:3]:  # Show up to 3 examples
            logging.warning(f"Low similarity example (id={qid[:8]}, sim={sim:.3f}) - Query: '{q}', Document: '{d}'")
    
    # 3. Query diversity analysis
    try:
        # Assess n-gram diversity
        all_queries = list(queries.values())
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
        query_vectors = vectorizer.fit_transform(all_queries)
        
        # Count unique n-grams per query
        unique_ngrams_per_query = query_vectors.sum(axis=1).mean()
        logging.info(f"Average unique n-grams per query: {unique_ngrams_per_query:.1f}")
        
        # Feature overlap between queries
        total_features = len(vectorizer.get_feature_names_out())
        features_used = (query_vectors.sum(axis=0) > 0).sum()
        feature_usage_ratio = features_used / total_features if total_features > 0 else 0
        logging.info(f"Query vocabulary diversity: {feature_usage_ratio:.3f} " +
                    f"({features_used} features used out of {total_features})")
    except Exception as e:
        logging.error(f"Error in query diversity analysis: {e}")
    
    # 4. Generate dataset quality histogram
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=20, alpha=0.7)
        plt.axvline(x=avg_sim, color='r', linestyle='--', label=f'Mean: {avg_sim:.3f}')
        plt.axvline(x=quality_threshold, color='green', linestyle=':', label=f'Quality Threshold: {quality_threshold}')
        plt.xlabel('Query-Document Similarity (Combined BM25 + Embedding)')
        plt.ylabel('Count')
        plt.title('Distribution of Query-Document Similarity Scores')
        plt.legend()
        plt.tight_layout()
        
        quality_plot_path = os.path.join(DATA_DIR, 'quality_histogram.png')
        plt.savefig(quality_plot_path)
        logging.info(f"Quality histogram saved to {quality_plot_path}")
        
        # Also save BM25 histogram
        plt.figure(figsize=(10, 6))
        plt.hist(bm25_similarities, bins=20, alpha=0.7)
        plt.axvline(x=np.mean(bm25_similarities), color='r', linestyle='--', label=f'Mean: {np.mean(bm25_similarities):.3f}')
        plt.axvline(x=quality_threshold, color='green', linestyle=':', label=f'Quality Threshold: {quality_threshold}')
        plt.xlabel('BM25 Similarity Score')
        plt.ylabel('Count')
        plt.title('Distribution of BM25 Similarity Scores')
        plt.legend()
        plt.tight_layout()
        
        bm25_plot_path = os.path.join(DATA_DIR, 'bm25_histogram.png')
        plt.savefig(bm25_plot_path)
        logging.info(f"BM25 histogram saved to {bm25_plot_path}")
    except Exception as e:
        logging.error(f"Error generating quality histogram: {e}")
    
    # 5. Export quality report with per-query results
    try:
        report = {
            'dataset_size': {
                'queries': len(queries),
                'documents': len(corpus),
            },
            'query_stats': {
                'avg_length': float(avg_query_length),
                'min_length': min(query_lengths),
                'max_length': max(query_lengths),
            },
            'document_stats': {
                'avg_length': float(avg_doc_length),
                'min_length': min(doc_lengths),
                'max_length': max(doc_lengths),
            },
            'similarity_stats': {
                'avg_similarity': float(avg_sim),
                'min_similarity': float(min_sim),
                'max_similarity': float(max_sim),
                'low_similarity_pairs': len(low_sim_pairs),
            },
            'ir_metrics': {
                'ndcg': float(ndcg_score),
                'precision_at_k': float(precision_score),
                'map': float(map_score),
            },
            'query_quality': quality_results,
            'overall_quality': 'good' if avg_sim >= quality_threshold else 'poor'
        }
        
        quality_report_path = os.path.join(DATA_DIR, 'quality_report.json')
        with open(quality_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        logging.info(f"Quality report saved to {quality_report_path}")
    except Exception as e:
        logging.error(f"Error generating quality report: {e}")

    # Return an overall quality assessment based on multiple metrics
    overall_score = (avg_sim + ndcg_score + precision_score + map_score) / 4
    
    if overall_score < 0.4:
        return "poor"
    elif overall_score < quality_threshold:
        return "mediocre"
    else:
        return "good"

def load_hard_negatives(negatives_dir: str, positive_corpus: Dict[str, str], verbose: bool = False) -> Dict[str, str]:
    """
    Load and chunk negative examples from the specified directory.
    These will be used as hard negatives in the dataset.
    Uses BM25 to filter out negatives that are too similar to positives.
    """
    if not os.path.exists(negatives_dir):
        logging.warning(f"Hard negatives directory {negatives_dir} does not exist. No hard negatives will be used.")
        return {}
    
    logging.info(f"Loading hard negative documents from {negatives_dir}")
    
    negative_files = []
    for root, _, files in os.walk(negatives_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                negative_files.append(os.path.join(root, file))
    
    if not negative_files:
        logging.warning(f"No PDF files found in negatives directory {negatives_dir}")
        return {}
    
    logging.info(f"Found {len(negative_files)} potential hard negative documents")
    
    # Load negative documents
    negative_corpus = load_corpus(negative_files, verbose=verbose)
    
    # Ensure we're not using any chunks that are too similar to positive examples
    if positive_corpus:
        logging.info("Filtering out negative examples that are too similar to positive examples...")
        filtered_negative_corpus = {}
        
        # Use BM25 to compute similarities
        positive_texts = list(positive_corpus.values())
        negative_texts = list(negative_corpus.values())
        negative_ids = list(negative_corpus.keys())
        
        if positive_texts and negative_texts:
            try:
                # Initialize BM25 scorer
                bm25_scorer = BM25Scorer()
                
                # For each negative document, compute its similarity to all positive documents
                for i, neg_text in enumerate(negative_texts):
                    # Prepare for maximum similarity calculation
                    max_sim = 0.0
                    
                    # Find maximum similarity with any positive document
                    for pos_text in positive_texts:
                        sim = bm25_scorer.compute_similarities([neg_text], [pos_text])[0]
                        max_sim = max(max_sim, sim)
                    
                    # Filter based on similarity threshold
                    similarity_threshold = 0.5
                    if max_sim < similarity_threshold:
                        filtered_negative_corpus[negative_ids[i]] = negative_corpus[negative_ids[i]]
                
                logging.info(f"Filtered negative corpus from {len(negative_corpus)} to {len(filtered_negative_corpus)} chunks " 
                             f"using BM25 similarity threshold of {similarity_threshold}")
                return filtered_negative_corpus
            except Exception as e:
                logging.warning(f"Error filtering negative examples with BM25: {e}")
                # Fall back to original method using TF-IDF
                logging.warning("Falling back to TF-IDF for filtering negatives")
                
                vectorizer = TfidfVectorizer(ngram_range=(1, 2))
                try:
                    combined = positive_texts + negative_texts
                    tfidf = vectorizer.fit_transform(combined)
                    
                    positive_vectors = tfidf[:len(positive_texts)]
                    negative_vectors = tfidf[len(positive_texts):]
                    
                    # Calculate similarities between each negative and all positives
                    # Keep only those with similarity below threshold
                    similarity_threshold = 0.5
                    
                    for i, neg_vector in enumerate(negative_vectors):
                        max_sim = max([cosine_similarity(neg_vector, pos_vector)[0][0] for pos_vector in positive_vectors])
                        if max_sim < similarity_threshold:
                            filtered_negative_corpus[negative_ids[i]] = negative_corpus[negative_ids[i]]
                    
                    logging.info(f"Filtered negative corpus from {len(negative_corpus)} to {len(filtered_negative_corpus)} chunks " 
                                f"using TF-IDF similarity threshold of {similarity_threshold}")
                    return filtered_negative_corpus
                except Exception as e:
                    logging.warning(f"Error filtering negative examples with TF-IDF: {e}")
                    return negative_corpus
    
    return negative_corpus

def filter_low_quality_pairs(
    queries: Dict[str, str], 
    corpus: Dict[str, str], 
    relevant_docs: Dict[str, List[str]],
    embedding_model: Optional[EmbeddingModel] = None,
    sim_threshold: float = 0.4,
    filter_refusals: bool = True,
    precision_threshold: float = 0.6,
    ndcg_threshold: float = 0.7
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Filter out low-quality query-document pairs based on multiple quality metrics.
    
    This function uses a combination of:
    - Semantic similarity (via embedding model)
    - Lexical similarity (via BM25)
    - IR metrics (Precision@k and NDCG)
    - Refusal phrase detection
    
    Args:
        queries: Dictionary of query IDs to query texts
        corpus: Dictionary of document IDs to document texts
        relevant_docs: Dictionary of query IDs to lists of relevant document IDs
        embedding_model: Optional embedding model for semantic similarity
        sim_threshold: Threshold for similarity scores
        filter_refusals: Whether to filter out queries with refusal phrases
        precision_threshold: Threshold for precision@k scores
        ndcg_threshold: Threshold for NDCG scores
        
    Returns:
        Filtered queries dictionary and relevant_docs dictionary
    """
    if not queries or not corpus or not relevant_docs:
        logging.warning("Empty dataset provided for filtering.")
        return queries, relevant_docs
    
    logging.info(f"Filtering low-quality pairs from {len(queries)} queries...")
    
    filtered_queries = {}
    filtered_relevant_docs = {}
    
    # First, filter out queries with refusal phrases if requested
    if filter_refusals:
        refusal_count = 0
        filtered_query_ids = []
        for qid, query in queries.items():
            if contains_refusal_phrase(query):
                logging.warning(f"Removing query containing refusal phrase: {query[:100]}...")
                refusal_count += 1
                filtered_query_ids.append(qid)
        
        # Create new dictionaries without the refusal phrases
        if refusal_count > 0:
            logging.info(f"Removed {refusal_count} queries containing refusal phrases")
            queries = {qid: query for qid, query in queries.items() if qid not in filtered_query_ids}
            relevant_docs = {qid: docs for qid, docs in relevant_docs.items() if qid not in filtered_query_ids}
    
    # Initialize BM25 scorer
    bm25_scorer = BM25Scorer()
    
    # Group queries by document for batch processing
    doc_to_queries = {}
    for qid, query in queries.items():
        doc_ids = relevant_docs.get(qid, [])
        if not doc_ids:
            continue
        doc_id = doc_ids[0]  # Take the first relevant doc
        if doc_id not in doc_to_queries:
            doc_to_queries[doc_id] = []
        doc_to_queries[doc_id].append((qid, query))
    
    # Process each document and its queries
    for doc_id, query_pairs in doc_to_queries.items():
        doc_text = corpus.get(doc_id, "")
        if not doc_text:
            continue
        
        doc_queries = [q for _, q in query_pairs]
        doc_query_ids = [qid for qid, _ in query_pairs]
        
        # 1. Compute BM25 similarities
        bm25_similarities = []
        for query in doc_queries:
            bm25_score = bm25_scorer.compute_similarities([query], [doc_text])
            if bm25_score:
                bm25_similarities.append(bm25_score[0])
            else:
                bm25_similarities.append(0.0)
                
        # 2. Compute embedding similarities if model is available
        embedding_similarities = []
        if embedding_model and embedding_model.is_initialized:
            doc_texts = [doc_text] * len(doc_queries)
            embedding_similarities = embedding_model.compute_similarities(doc_queries, doc_texts)
            
            # Handle the case where embedding model returns fewer similarities
            if len(embedding_similarities) < len(doc_queries):
                embedding_similarities.extend([0.0] * (len(doc_queries) - len(embedding_similarities)))
        else:
            # If no embedding model, use BM25 scores as fallback
            embedding_similarities = bm25_similarities.copy()
        
        # 3. Combine BM25 and embedding similarities
        combined_similarities = []
        for i, (bm25_sim, embed_sim) in enumerate(zip(bm25_similarities, embedding_similarities)):
            # Equal weighting to both lexical and semantic similarity
            combined_sim = 0.5 * bm25_sim + 0.5 * embed_sim
            combined_similarities.append(combined_sim)
                
        # 4. Compute NDCG and Precision@k
        ndcg_score = normalized_discounted_cumulative_gain(combined_similarities)
        precision_score = precision_at_k(combined_similarities, k=len(combined_similarities), threshold=sim_threshold)
            
        # 5. Apply filtering based on combined criteria
        for i, (qid, query) in enumerate(query_pairs):
            if i >= len(combined_similarities):
                continue
                
            sim_score = combined_similarities[i]
            
            # Filter based on individual similarity score
            if sim_score >= sim_threshold:
                # Additional quality check based on IR metrics for documents with multiple queries
                if len(doc_queries) > 1:
                    if ndcg_score >= ndcg_threshold and precision_score >= precision_threshold:
                        filtered_queries[qid] = query
                        filtered_relevant_docs[qid] = [doc_id]
                else:
                    # For documents with a single query, rely on similarity score
                    filtered_queries[qid] = query
                    filtered_relevant_docs[qid] = [doc_id]
            else:
                logging.debug(f"Filtering out query (ID: {qid}) due to low similarity score: {sim_score:.3f}")
    
    # Log filtering statistics
    initial_count = len(queries)
    final_count = len(filtered_queries)
    removed_count = initial_count - final_count
    retained_pct = (final_count / max(1, initial_count)) * 100
    
    logging.info(f"Filtering complete: kept {final_count} queries ({retained_pct:.1f}%), removed {removed_count} low-quality pairs")
    
    # Provide some example queries that were kept/removed
    if final_count > 0 and removed_count > 0:
        kept_examples = list(filtered_queries.items())[:3]
        
        removed_qids = set(queries.keys()) - set(filtered_queries.keys())
        removed_examples = [(qid, queries[qid]) for qid in list(removed_qids)[:3]]
        
        logging.info("Examples of KEPT queries:")
        for qid, query in kept_examples:
            logging.info(f"  - {qid}: {query[:100]}...")
            
        logging.info("Examples of REMOVED queries:")
        for qid, query in removed_examples:
            logging.info(f"  - {qid}: {query[:100]}...")
    
    return filtered_queries, filtered_relevant_docs

def assign_hard_negatives(
    queries: Dict[str, str],
    corpus: Dict[str, str],
    relevant_docs: Dict[str, List[str]],
    negative_corpus: Dict[str, str],
    embedding_model: Optional[EmbeddingModel] = None,
    num_hard_negatives: int = NUM_HARD_NEGATIVES,
    quality_threshold: float = 0.3,  # Threshold for what constitutes a "good" hard negative
    num_queries_per_doc: int = 15    # Added to match the number of hard negatives per query
) -> Dict[str, List[str]]:
    """
    Assign hard negatives to each query based on BM25 and semantic similarity.
    Returns an updated relevant_docs dictionary containing hard negatives.
    
    The number of hard negatives per query will match num_queries_per_doc if possible,
    ensuring a balanced dataset with equal numbers of queries and negatives.
    """
    if not queries or not negative_corpus:
        return relevant_docs
    
    logging.info(f"Assigning {num_hard_negatives} hard negatives to {len(queries)} queries using BM25...")
    
    # Use num_queries_per_doc as the number of hard negatives if specified
    if num_queries_per_doc > 0:
        num_hard_negatives = num_queries_per_doc
        logging.info(f"Setting number of hard negatives to match queries per document: {num_hard_negatives}")
    
    # Create a copy of the relevant_docs to modify
    updated_relevant_docs = {qid: docs.copy() for qid, docs in relevant_docs.items()}
    
    # Group queries by their positive documents to process in batches
    positive_doc_to_queries = {}
    for qid, query in queries.items():
        if qid not in relevant_docs or not relevant_docs[qid]:
            continue
        
        positive_doc_id = relevant_docs[qid][0]  # Take the first positive doc
        if positive_doc_id not in positive_doc_to_queries:
            positive_doc_to_queries[positive_doc_id] = []
        positive_doc_to_queries[positive_doc_id].append((qid, query))
    
    # Initialize BM25 scorer
    bm25_scorer = BM25Scorer()
    
    # Process each positive document and its queries
    for pos_doc_id, query_pairs in positive_doc_to_queries.items():
        queries_for_doc = [q for _, q in query_pairs]
        query_ids = [qid for qid, _ in query_pairs]
        
        # Get potential negative docs
        negative_docs = list(negative_corpus.values())
        negative_doc_ids = list(negative_corpus.keys())
        
        if not negative_docs:
            continue
        
        # Calculate similarities between queries and all negative docs
        all_similarities = []
        
        # Calculate similarity scores using BM25
        for query in queries_for_doc:
            query_similarities = []
            for neg_doc in negative_docs:
                # Calculate BM25 similarity between query and negative document
                sim = bm25_scorer.compute_similarities([query], [neg_doc])[0]
                query_similarities.append(sim)
            all_similarities.append(query_similarities)
        
        # If embedding model is available, combine with BM25 scores
        if embedding_model and embedding_model.is_initialized:
            logging.info("Using both BM25 and embedding model for hard negative selection")
            
            # Process in smaller batches to avoid OOM
            embedding_similarities = []
            
            # If we have too many negative docs, sample a subset to avoid memory issues
            max_negatives = 100
            if len(negative_docs) > max_negatives:
                logging.info(f"Sampling {max_negatives} out of {len(negative_docs)} negative docs for hard negative selection")
                indices = np.random.choice(len(negative_docs), max_negatives, replace=False)
                sampled_negative_docs = [negative_docs[i] for i in indices]
                sampled_negative_doc_ids = [negative_doc_ids[i] for i in indices]
            else:
                sampled_negative_docs = negative_docs
                sampled_negative_doc_ids = negative_doc_ids
            
            # Process each query separately
            for query in queries_for_doc:
                query_batch = [query] * len(sampled_negative_docs)
                sims = embedding_model.compute_similarities(query_batch, sampled_negative_docs)
                embedding_similarities.append(sims)
            
            # Combine BM25 and embedding scores if we sampled
            if len(negative_docs) > max_negatives:
                # Create new similarity lists with combined scores
                combined_similarities = []
                for i, query_sims in enumerate(all_similarities):
                    # Start with BM25 scores
                    combined_query_sims = query_sims.copy()
                    
                    # Replace sampled items with combined scores
                    for j, idx in enumerate(indices):
                        if j < len(embedding_similarities[i]):
                            # Average of BM25 and embedding score
                            combined_score = 0.5 * query_sims[idx] + 0.5 * embedding_similarities[i][j]
                            combined_query_sims[idx] = combined_score
                    
                    combined_similarities.append(combined_query_sims)
                
                all_similarities = combined_similarities
            else:
                # Directly combine scores
                combined_similarities = []
                for i, (bm25_sims, emb_sims) in enumerate(zip(all_similarities, embedding_similarities)):
                    if len(bm25_sims) == len(emb_sims):
                        combined_query_sims = [0.5 * b + 0.5 * e for b, e in zip(bm25_sims, emb_sims)]
                        combined_similarities.append(combined_query_sims)
                    else:
                        # Handle size mismatch
                        combined_similarities.append(bm25_sims)
                
                if combined_similarities:
                    all_similarities = combined_similarities
        
        # Assign hard negatives to each query
        for i, qid in enumerate(query_ids):
            if i >= len(all_similarities):
                continue
                
            # Get similarities for this query
            sims = all_similarities[i]
            if len(sims) == 0:
                continue
            
            # Sort negative docs by similarity (descending)
            neg_with_sim = [(negative_doc_ids[j], sims[j]) for j in range(min(len(sims), len(negative_doc_ids)))]
            
            # Sort by similarity in descending order (higher similarity first)
            neg_with_sim.sort(key=lambda x: x[1], reverse=True)
            
            # Select appropriate hard negatives within suitable similarity range
            # Hard negatives should be somewhat similar but not too similar
            hard_negs = []
            
            # First, try to find negatives in the optimal similarity range
            for neg_id, sim in neg_with_sim:
                # Skip if similarity is too low (not relevant enough to be challenging)
                # or too high (might actually be relevant)
                if 0.15 <= sim <= quality_threshold:
                    hard_negs.append(neg_id)
                    if len(hard_negs) >= num_hard_negatives:
                        break
            
            # If we don't have enough, relax the constraints
            if len(hard_negs) < num_hard_negatives and neg_with_sim:
                logging.info(f"Couldn't find enough hard negatives in optimal range for query {qid}, relaxing constraints")
                
                # Sort all negatives by how close their similarity is to the ideal mid-range value
                ideal_sim = (0.15 + quality_threshold) / 2
                sorted_by_ideal = sorted(neg_with_sim, key=lambda x: abs(x[1] - ideal_sim))
                
                # Add more negatives until we reach the desired count
                for neg_id, _ in sorted_by_ideal:
                    if neg_id not in hard_negs:
                        hard_negs.append(neg_id)
                        if len(hard_negs) >= num_hard_negatives:
                            break
            
            # Add hard negatives to this query's relevant docs (replacing any existing ones)
            # Keep only the first document (positive example) and add hard negatives
            updated_relevant_docs[qid] = [updated_relevant_docs[qid][0]] + hard_negs
    
    # Count how many queries got hard negatives
    queries_with_negs = sum(1 for qid in updated_relevant_docs if len(updated_relevant_docs[qid]) > 1)
    avg_negs = sum(len(docs) - 1 for docs in updated_relevant_docs.values()) / max(1, len(updated_relevant_docs))
    
    logging.info(f"Hard negative assignment complete: {queries_with_negs} queries got hard negatives (avg {avg_negs:.1f} per query)")
    
    return updated_relevant_docs

def save_dataset(queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, List[str]], output_dir: str):
    """
    Save the dataset files (queries, corpus, and relevance judgments) to the specified directory.
    Also saves a combined dataset format suitable for training.
    """
    logging.info(f"Saving dataset to {output_dir}")
    
    # Save corpus
    corpus_path = os.path.join(output_dir, "corpus.json")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved {len(corpus)} documents to {corpus_path}")
    
    # Save queries
    queries_path = os.path.join(output_dir, "queries.json")
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved {len(queries)} queries to {queries_path}")
    
    # Save relevance judgments
    qrels_path = os.path.join(output_dir, "qrels.json")
    with open(qrels_path, 'w', encoding='utf-8') as f:
        json.dump(relevant_docs, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved relevance judgments to {qrels_path}")
    
    # Save in BEIR format
    beir_path = os.path.join(output_dir, "beir_format")
    os.makedirs(beir_path, exist_ok=True)
    
    # BEIR corpus format: {'_id': doc_id, 'title': '', 'text': text}
    beir_corpus = {}
    for doc_id, text in corpus.items():
        beir_corpus[doc_id] = {'_id': doc_id, 'title': '', 'text': text}
    
    with open(os.path.join(beir_path, "corpus.jsonl"), 'w', encoding='utf-8') as f:
        for doc_id, doc in beir_corpus.items():
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # BEIR queries format: {'_id': query_id, 'text': query}
    beir_queries = {}
    for query_id, query in queries.items():
        beir_queries[query_id] = {'_id': query_id, 'text': query}
    
    with open(os.path.join(beir_path, "queries.jsonl"), 'w', encoding='utf-8') as f:
        for query_id, query in beir_queries.items():
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    # BEIR qrels format: {query_id: {doc_id: score}}
    beir_qrels = {}
    for query_id, doc_ids in relevant_docs.items():
        beir_qrels[query_id] = {}
        for i, doc_id in enumerate(doc_ids):
            # First document is the positive example (score=1), others are hard negatives (score=0)
            score = 1 if i == 0 else 0
            beir_qrels[query_id][doc_id] = score
    
    with open(os.path.join(beir_path, "qrels.tsv"), 'w', encoding='utf-8') as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for query_id, docs in beir_qrels.items():
            for doc_id, score in docs.items():
                f.write(f"{query_id}\t{doc_id}\t{score}\n")
    
    logging.info(f"Saved dataset in BEIR format to {beir_path}")
    
    # Save in training-ready format for sentence transformers
    train_data = []
    for query_id, query_text in queries.items():
        if query_id in relevant_docs and relevant_docs[query_id]:
            pos_id = relevant_docs[query_id][0]  # First doc is positive
            if pos_id in corpus:
                pos_text = corpus[pos_id]
                
                # Add hard negatives if available
                hard_negs = []
                if len(relevant_docs[query_id]) > 1:
                    for neg_id in relevant_docs[query_id][1:]:
                        if neg_id in corpus:
                            hard_negs.append(corpus[neg_id])
                
                train_data.append({
                    "query": query_text,
                    "positive": pos_text,
                    "hard_negatives": hard_negs
                })
    
    train_data_path = os.path.join(output_dir, "train_data.json")
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Saved {len(train_data)} training examples to {train_data_path}")

def save_checkpoint(queries: Dict[str, Tuple[str, str]], corpus: Dict[str, str], checkpoint_number: int):
    """
    Save the current progress to a checkpoint file.
    
    Args:
        queries: The generated queries dictionary
        corpus: The corpus dictionary
        checkpoint_number: Current checkpoint number
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_number}.json")
    
    # Convert the queries dict (which has tuples as values) to a format that can be JSON-serialized
    serializable_queries = {}
    for query_id, (doc_id, question) in queries.items():
        serializable_queries[query_id] = {
            "doc_id": doc_id,
            "question": question
        }
    
    # Create a checkpoint object
    checkpoint_data = {
        "queries": serializable_queries,
        "checkpoint_number": checkpoint_number,
        "timestamp": str(datetime.datetime.now()),
        "num_queries": len(queries),
        "num_docs": len(set(doc_id for _, (doc_id, _) in queries.items()))  # Count unique docs
    }
    
    # Save as JSON
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Saved checkpoint {checkpoint_number} with {len(queries)} queries")

def load_latest_checkpoint():
    """
    Load the most recent checkpoint if one exists.
    
    Returns:
        Tuple of (queries dict, checkpoint number) or (None, 0) if no checkpoint found
    """
    checkpoint_files = glob(os.path.join(CHECKPOINT_DIR, "checkpoint_*.json"))
    if not checkpoint_files:
        return None, 0
    
    # Get the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda f: int(re.search(r'checkpoint_(\d+)\.json', f).group(1)))
    checkpoint_number = int(re.search(r'checkpoint_(\d+)\.json', latest_checkpoint).group(1))
    
    try:
        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        # Convert the serialized queries back to the original format
        queries = {}
        for query_id, query_info in checkpoint_data["queries"].items():
            queries[query_id] = (query_info["doc_id"], query_info["question"])
        
        logging.info(f"Loaded checkpoint {checkpoint_number} with {len(queries)} queries")
        return queries, checkpoint_number
    except Exception as e:
        logging.error(f"Error loading checkpoint {latest_checkpoint}: {e}")
        return None, 0

def main():
    """Main function to generate the dataset."""
    import datetime
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate a question answering dataset from PDF documents")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output directory for the dataset")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.3", help="HuggingFace model to use")
    parser.add_argument("--num_queries_per_doc", type=int, default=15, help="Number of queries to generate per document")
    parser.add_argument("--num_hard_negatives", type=int, default=5, help="Number of hard negative examples per query")
    parser.add_argument("--filter_refusals", action="store_true", help="Filter out refusal responses")
    parser.add_argument("--hard_negatives_dir", type=str, default=None, help="Directory containing hard negative documents")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint")
    parser.add_argument("--clear_checkpoints", action="store_true", help="Clear existing checkpoints before starting")
    args = parser.parse_args()
    
    # Set up the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clear checkpoints if requested
    if args.clear_checkpoints:
        import shutil
        if os.path.exists(CHECKPOINT_DIR):
            logging.info(f"Clearing existing checkpoints in {CHECKPOINT_DIR}")
            shutil.rmtree(CHECKPOINT_DIR)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all PDF files in the data directory
    pdf_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    logging.info(f"Found {len(pdf_files)} PDF files to process")

    # Load PDF corpus
    corpus = load_corpus(pdf_files, verbose=not args.quiet)
    
    # Load hard negatives if specified
    hard_negative_corpus = {}
    if args.hard_negatives_dir:
        logging.info(f"Loading hard negatives from {args.hard_negatives_dir}")
        hard_negative_corpus = load_hard_negatives(args.hard_negatives_dir, corpus, verbose=not args.quiet)
        logging.info(f"Loaded {len(hard_negative_corpus)} hard negative documents")

    # Initialize embedding model
    logging.info(f"Initializing embedding model: {args.embedding_model}")
    embedding_model = EmbeddingModel(model_name=args.embedding_model, model_dir=args.model_dir)
    
    # Initialize LLM - now we unpack the tuple here
    model, tokenizer = init_llm(model_name=args.model_name, model_dir=args.model_dir)

    # Generate queries with checkpoint support
    queries = generate_queries(
        corpus=corpus,
        model=model,
        tokenizer=tokenizer,
        num_queries_per_doc=args.num_queries_per_doc,
        use_filter=args.filter_refusals,
        resume_from_checkpoint=args.resume
    )
    
    # Create relevance judgments (qid -> [did])
    relevant_docs = {qid: [did] for qid, (did, _) in queries.items()}
    
    # Extract just the queries for evaluation and filtering
    query_texts = {qid: q for qid, (_, q) in queries.items()}
    
    # Filter low-quality examples if requested
    if args.filter_low_quality and embedding_model.is_initialized:
        logging.info(f"Filtering low-quality examples using similarity threshold {args.quality_threshold}...")
        initial_count = len(query_texts)
        query_texts, relevant_docs = filter_low_quality_pairs(
            query_texts, 
            corpus, 
            relevant_docs,
            embedding_model=embedding_model,
            sim_threshold=args.quality_threshold,
            filter_refusals=args.filter_refusals,
            precision_threshold=args.precision_threshold,
            ndcg_threshold=args.ndcg_threshold
        )
        filtered_count = initial_count - len(query_texts)
        logging.info(f"Filtered out {filtered_count} low-quality examples, kept {len(query_texts)}")
    
    # Assign hard negatives if available
    if hard_negative_corpus and query_texts:
        logging.info("Assigning hard negatives to queries...")
        relevant_docs = assign_hard_negatives(
            queries=query_texts,
            corpus=corpus,
            relevant_docs=relevant_docs,
            negative_corpus=hard_negative_corpus,
            embedding_model=embedding_model,
            num_hard_negatives=args.num_hard_negatives,
            num_queries_per_doc=args.num_queries_per_doc  # Pass the number of queries per chunk
        )
    
    # Calculate quality metrics on the final dataset
    if embedding_model.is_initialized and query_texts:
        quality_report = evaluate_dataset_quality(
            queries=query_texts,
            corpus=corpus,
            relevant_docs=relevant_docs,
            embedding_model=embedding_model,
            quality_threshold=args.quality_threshold
        )
        
        # Save quality report
        quality_report_path = os.path.join(args.output_dir, "quality_report.json")
        with open(quality_report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        logging.info(f"Quality report saved to {quality_report_path}")
    
    # Save dataset files
    if query_texts:
        save_dataset(
            queries=query_texts,
            corpus=corpus,
            relevant_docs=relevant_docs,
            output_dir=args.output_dir
        )
    else:
        logging.error("No valid queries remain after filtering. Dataset not saved.")
        return 1
    
    # Save metadata
    metadata = {
        "num_queries": len(query_texts),
        "num_documents": len(corpus),
        "num_queries_per_chunk": args.num_queries_per_doc,
        "embedding_model": args.embedding_model,
        "llm_model": args.model_name,
        "llm_max_length": args.llm_max_length,
        "use_filter": args.filter_refusals,
        "filter_low_quality": args.filter_low_quality,
        "quality_threshold": args.quality_threshold if args.filter_low_quality else None,
        "precision_threshold": args.precision_threshold if args.filter_low_quality else None,
        "ndcg_threshold": args.ndcg_threshold if args.filter_low_quality else None,
        "use_ocr": args.use_ocr,
        "ocr_lang": args.ocr_lang if args.use_ocr else None,
        "hard_negatives": bool(hard_negative_corpus),
        "num_hard_negatives": args.num_hard_negatives if hard_negative_corpus else 0,
        "timestamp": datetime.now().isoformat(),
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Metadata saved to {metadata_path}")
    logging.info(f"Dataset generation complete: {len(query_texts)} queries, {len(corpus)} documents")
    
    return 0

if __name__ == "__main__":
    import argparse
    import random
    main()
