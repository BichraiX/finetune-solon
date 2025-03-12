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
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import unicodedata

import pypdf
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

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

# For quality evaluation
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Input document paths
DOCUMENTS_DIR = '/Data/amine.chraibi/rag/docs'
NEGATIVES_DIR = '/Data/amine.chraibi/rag/negatives'
TRAIN_FILES = [
    f'{DOCUMENTS_DIR}/142.2020 07 02 - Contrat de prêt aux entreprises - CRAVEDI - BRED BANQUE POPULAIRE (signé).pdf',
    f'{DOCUMENTS_DIR}/Compare - PROTO CRAVEDI - 21 12 23 - Version signature.pdf'
]
VAL_FILES = [f'{DOCUMENTS_DIR}/2019 05 16 - MATHO - Bail commercial GUICLA vdef.pdf']

# Output paths
DATA_DIR = '/Data/amine.chraibi/rag/data_deepseek'
Path(DATA_DIR).mkdir(exist_ok=True)
CACHE_DIR = './cache'
Path(CACHE_DIR).mkdir(exist_ok=True)

TRAIN_CORPUS_FPATH = f'{DATA_DIR}/train_corpus.json'
VAL_CORPUS_FPATH = f'{DATA_DIR}/val_corpus.json'
TRAIN_QUERIES_FPATH = f'{DATA_DIR}/train_queries.json'
TRAIN_RELEVANT_DOCS_FPATH = f'{DATA_DIR}/train_relevant_docs.json'
VAL_QUERIES_FPATH = f'{DATA_DIR}/val_queries.json'
VAL_RELEVANT_DOCS_FPATH = f'{DATA_DIR}/val_relevant_docs.json'
TRAIN_DATASET_FPATH = f'{DATA_DIR}/train_dataset.json'
VAL_DATASET_FPATH = f'{DATA_DIR}/val_dataset.json'

# Chunk parameters (in approximate words)
CHUNK_WORD_LIMIT = 1000
OVERLAP_WORDS = 200

# Hard negative settings
NUM_HARD_NEGATIVES = 3  # Number of hard negatives to include per query

class EmbeddingModel:
    """Wrapper for sentence embedding models to be used for quality assessment."""
    
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the embedding model.
        Default is a multilingual SBERT model that works well with French text.
        """
        logging.info(f"Loading embedding model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
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

def init_llm():
    """Initialize Deepseek-v3 6B model for text generation"""
    logging.info("Loading Deepseek-v3 6B model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-instruct", 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

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
    
    # Normalize the extracted text
    return normalize_french_text(text)

def cache_extracted_text(file_path: str) -> str:
    """Use caching to avoid reprocessing PDFs."""
    cache_file = os.path.join(CACHE_DIR, os.path.basename(file_path) + ".txt")
    if os.path.exists(cache_file):
        logging.info(f"Loading cached text for {file_path}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        logging.info(f"Extracting text from {file_path}")
        text = extract_text_from_pdf(file_path)
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return text

def chunk_text(text: str, chunk_word_limit: int = CHUNK_WORD_LIMIT, overlap_words: int = OVERLAP_WORDS) -> List[str]:
    """
    Split text into chunks using French sentence boundaries.
    Each chunk contains approximately chunk_word_limit words,
    with an overlap of approximately overlap_words words between chunks.
    """
    # Use French-specific sentence tokenization
    try:
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
    """Load and chunk PDF documents into a corpus using parallel processing and caching."""
    corpus = {}

    def process_file(file_path: str) -> List[Tuple[str, str]]:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            return []
        text = cache_extracted_text(file_path)
        chunks = chunk_text(text)
        return [(str(uuid.uuid4()), chunk) for chunk in chunks]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, fp) for fp in files]
        for future in futures:
            for node_id, chunk in future.result():
                corpus[node_id] = chunk

    if verbose:
        logging.info(f"Loaded and chunked into {len(corpus)} text chunks")
    return corpus

def generate_query_with_llm(model, tokenizer, prompt: str) -> str:
    """Generate a response using the Deepseek model with error handling."""
    formatted_prompt = f"### Instruction: {prompt}\n\n### Response:"
    try:
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=512,  # Increased from 256 to allow for more detailed responses
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response
    except Exception as e:
        logging.error(f"Error generating query: {e}")
        return ""

def generate_queries(
    corpus: Dict[str, str],
    model,
    tokenizer,
    num_questions_per_chunk: int = 15,
    prompt_template: str = None,
    verbose: bool = False,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Generate synthetic questions for each document chunk using Deepseek-v3.
    The prompt template is cleaned and formatted robustly.
    """
    # Use a cleaned-up prompt (no extra indentation)
    prompt_template = prompt_template or (
        "Tu es une intelligence artificielle chargée de générer des questions réalistes qu'un utilisateur pourrait poser "
        "et pour lesquelles le contexte fourni serait une réponse pertinente.\n\n"
        
        "Contexte : {context_str}\n\n"
        
        "### Instructions :\n"
        "1. Analyse le contenu, les faits et les concepts du document ci-dessus.\n"
        "2. Génère exactement {num_questions_per_chunk} questions différentes mais thématiquement liées.\n"
        "3. Les questions doivent être formulées de façon naturelle, comme un utilisateur réel les poserait.\n"
        "4. Inclus occasionnellement des fautes de frappe mineures, des accents manquants ou des expressions familières pour refléter des requêtes authentiques.\n"
        "5. Les questions doivent être diverses dans leur formulation tout en ciblant la même information principale.\n"
        "6. Évite absolument de copier des phrases directement du document.\n\n"
        
        "### Format de sortie :\n"
        "Retourne UNIQUEMENT un objet JSON valide avec la structure suivante :\n"
        
        "```json\n"
        "{{\n"
        "  \"questions\": [\n"
        "    \"Première question générée\",\n"
        "    \"Deuxième question générée\"\n"
        "  ]\n"
        "}}\n"
        "```\n\n"
        
        "Ne réponds rien d'autre que cet objet JSON valide."
    )

    queries = {}
    relevant_docs = {}
    
    for node_id, text in tqdm(corpus.items(), desc="Generating queries"):
        prompt = prompt_template.format(
            context_str=text,
            num_questions_per_chunk=num_questions_per_chunk
        )
        response = generate_query_with_llm(model, tokenizer, prompt)
        
        # Extract JSON from the response
        json_match = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(1)
                parsed_json = json.loads(json_str)
                
                # Handle the expected format with "questions" key
                if "questions" in parsed_json and isinstance(parsed_json["questions"], list):
                    questions = parsed_json["questions"]
                # Fall back to handling old format with question_1, question_2 keys
                else:
                    questions = [v for k, v in parsed_json.items() if isinstance(v, str)]
                
                # Filter empty questions
                questions = [q.strip() for q in questions if q and len(q.strip()) > 10]
                
                if verbose and questions:
                    logging.info(f"Generated {len(questions)} questions for chunk {node_id[:8]}...")
                
                for question in questions:
                    question_id = str(uuid.uuid4())
                    queries[question_id] = question
                    relevant_docs[question_id] = [node_id]
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON for chunk {node_id[:8]}: {e}")
        else:
            # Fallback to line-by-line parsing if JSON parsing fails
            logging.warning(f"No valid JSON found in response for chunk {node_id[:8]}. Attempting fallback parsing.")
            lines = response.strip().split('\n')
            questions = []
            for line in lines:
                # Remove leading numbers, bullet points, etc.
                cleaned = re.sub(r'^\s*(\d+[\.\)]|\*|\-)\s*', '', line).strip()
                if cleaned and len(cleaned) > 10 and not cleaned.startswith(('```', '{', '}')):
                    questions.append(cleaned)
            
            if questions:
                for question in questions[:num_questions_per_chunk]:
                    question_id = str(uuid.uuid4())
                    queries[question_id] = question
                    relevant_docs[question_id] = [node_id]
    
    return queries, relevant_docs

def evaluate_dataset_quality(
    queries: Dict[str, str], 
    corpus: Dict[str, str], 
    relevant_docs: Dict[str, List[str]],
    embedding_model: Optional[EmbeddingModel] = None,
    quality_threshold: float = 0.5
):
    """
    Compute comprehensive quality metrics for the generated dataset.
    Uses a semantic embedding model for more accurate relevance assessment.
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
    
    # 2. Query-document relevance using advanced embedding model
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
    
    # Use advanced embedding model if available
    similarities = []
    
    if embedding_model and embedding_model.is_initialized:
        logging.info("Computing query-document similarities with advanced embedding model...")
        similarities = embedding_model.compute_similarities(query_list, doc_list)
    else:
        # Fall back to TF-IDF if advanced model isn't available
        logging.warning("Advanced embedding model not available. Falling back to TF-IDF.")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        try:
            combined = query_list + doc_list
            tfidf = vectorizer.fit_transform(combined)
            query_tfidf = tfidf[:len(query_list)]
            doc_tfidf = tfidf[len(query_list):]
            
            for i in range(len(query_list)):
                sim = cosine_similarity(query_tfidf[i], doc_tfidf[i]).item()
                similarities.append(sim)
        except Exception as e:
            logging.error(f"Error in similarity calculation: {e}")
            return "poor"
    
    if not similarities:
        logging.warning("Failed to compute similarities.")
        return "poor"
    
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    logging.info(f"Query-document similarity: avg={avg_sim:.3f}, min={min_sim:.3f}")
    
    # Create quality results for each query-document pair
    quality_results = {
        qid: {"query": query, "similarity": sim, "quality": "good" if sim >= quality_threshold else "poor"}
        for qid, query, sim in zip(query_ids, query_list, similarities)
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
        plt.xlabel('Query-Document Similarity')
        plt.ylabel('Count')
        plt.title('Distribution of Query-Document Similarity Scores')
        plt.legend()
        plt.tight_layout()
        
        quality_plot_path = os.path.join(DATA_DIR, 'quality_histogram.png')
        plt.savefig(quality_plot_path)
        logging.info(f"Quality histogram saved to {quality_plot_path}")
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
                'max_similarity': float(max(similarities)),
                'low_similarity_pairs': len(low_sim_pairs),
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

    # Return an overall quality assessment
    if avg_sim < 0.4:
        return "poor"
    elif avg_sim < quality_threshold:
        return "mediocre"
    else:
        return "good"

def load_hard_negatives(negatives_dir: str, positive_corpus: Dict[str, str], verbose: bool = False) -> Dict[str, str]:
    """
    Load and chunk negative examples from the specified directory.
    These will be used as hard negatives in the dataset.
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
        
        # Use TF-IDF to compute similarities (this is just for filtering, not final quality assessment)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        positive_texts = list(positive_corpus.values())
        negative_texts = list(negative_corpus.values())
        
        if positive_texts and negative_texts:
            try:
                combined = positive_texts + negative_texts
                tfidf = vectorizer.fit_transform(combined)
                
                positive_vectors = tfidf[:len(positive_texts)]
                negative_vectors = tfidf[len(positive_texts):]
                
                # Calculate similarities between each negative and all positives
                # Keep only those with similarity below threshold
                similarity_threshold = 0.5
                negative_ids = list(negative_corpus.keys())
                
                for i, neg_vector in enumerate(negative_vectors):
                    max_sim = max([cosine_similarity(neg_vector, pos_vector)[0][0] for pos_vector in positive_vectors])
                    if max_sim < similarity_threshold:
                        filtered_negative_corpus[negative_ids[i]] = negative_corpus[negative_ids[i]]
                
                logging.info(f"Filtered negative corpus from {len(negative_corpus)} to {len(filtered_negative_corpus)} chunks")
                return filtered_negative_corpus
            except Exception as e:
                logging.warning(f"Error filtering negative examples: {e}")
                return negative_corpus
    
    return negative_corpus

def filter_low_quality_pairs(
    queries: Dict[str, str], 
    corpus: Dict[str, str], 
    relevant_docs: Dict[str, List[str]],
    embedding_model: Optional[EmbeddingModel] = None,
    sim_threshold: float = 0.4
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Filter out low-quality query-document pairs based on semantic similarity."""
    if not queries or not corpus or not relevant_docs:
        logging.warning("Empty dataset provided for filtering.")
        return queries, relevant_docs
    
    logging.info(f"Filtering low-quality pairs from {len(queries)} queries using similarity threshold {sim_threshold}...")
    
    filtered_queries = {}
    filtered_relevant_docs = {}
    
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
        
        # Compute similarities
        similarities = []
        if embedding_model and embedding_model.is_initialized:
            # Use neural embeddings
            doc_texts = [doc_text] * len(doc_queries)
            similarities = embedding_model.compute_similarities(doc_queries, doc_texts)
        else:
            # Fall back to TF-IDF
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            try:
                combined = doc_queries + [doc_text]
                tfidf = vectorizer.fit_transform(combined)
                query_vectors = tfidf[:len(doc_queries)]
                doc_vector = tfidf[len(doc_queries):]
                
                for i in range(len(doc_queries)):
                    sim = cosine_similarity(query_vectors[i], doc_vector)[0][0]
                    similarities.append(sim)
            except Exception as e:
                logging.error(f"Error in similarity calculation during filtering: {e}")
                similarities = [0.0] * len(doc_queries)  # Assume low similarity on error
        
        # Keep only high-quality pairs
        for i, (qid, query) in enumerate(query_pairs):
            if i < len(similarities) and similarities[i] >= sim_threshold:
                filtered_queries[qid] = query
                filtered_relevant_docs[qid] = [doc_id]
    
    removed_count = len(queries) - len(filtered_queries)
    retained_pct = (len(filtered_queries) / max(1, len(queries))) * 100
    
    logging.info(f"Filtering complete: kept {len(filtered_queries)} queries ({retained_pct:.1f}%), removed {removed_count} low-quality pairs.")
    
    return filtered_queries, filtered_relevant_docs

def assign_hard_negatives(
    queries: Dict[str, str],
    corpus: Dict[str, str],
    relevant_docs: Dict[str, List[str]],
    negative_corpus: Dict[str, str],
    embedding_model: Optional[EmbeddingModel] = None,
    num_hard_negatives: int = NUM_HARD_NEGATIVES,
    quality_threshold: float = 0.3  # Threshold for what constitutes a "good" hard negative
) -> Dict[str, List[str]]:
    """
    Assign hard negatives to each query based on semantic similarity.
    Returns an updated relevant_docs dictionary containing hard negatives.
    """
    if not queries or not negative_corpus:
        return relevant_docs
    
    logging.info(f"Assigning hard negatives to {len(queries)} queries...")
    
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
        
        if embedding_model and embedding_model.is_initialized:
            # Calculate all-to-all similarities with neural embeddings
            # This approach is more computationally expensive but gives better hard negatives
            
            # If we have too many negative docs, sample a subset to avoid memory issues
            max_negatives = 100
            if len(negative_docs) > max_negatives:
                logging.info(f"Sampling {max_negatives} out of {len(negative_docs)} negative docs for hard negative selection")
                indices = np.random.choice(len(negative_docs), max_negatives, replace=False)
                negative_docs = [negative_docs[i] for i in indices]
                negative_doc_ids = [negative_doc_ids[i] for i in indices]
            
            # Process each query separately to keep memory usage manageable
            for query in queries_for_doc:
                # Repeat the query for each negative doc
                query_batch = [query] * len(negative_docs)
                sims = embedding_model.compute_similarities(query_batch, negative_docs)
                all_similarities.append(sims)
        else:
            # Fall back to TF-IDF for similarity calculation
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            try:
                # Combine all texts for vectorization
                all_texts = queries_for_doc + negative_docs
                tfidf = vectorizer.fit_transform(all_texts)
                
                # Split vectorized texts
                query_vectors = tfidf[:len(queries_for_doc)]
                negative_vectors = tfidf[len(queries_for_doc):]
                
                # Calculate cosine similarity between each query and each negative doc
                for i in range(len(queries_for_doc)):
                    sims = []
                    for j in range(len(negative_docs)):
                        sim = cosine_similarity(query_vectors[i], negative_vectors[j])[0][0]
                        sims.append(sim)
                    all_similarities.append(sims)
            except Exception as e:
                logging.error(f"Error calculating similarities for hard negatives: {e}")
                all_similarities = [[0.0] * len(negative_docs) for _ in range(len(queries_for_doc))]
        
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
            neg_with_sim.sort(key=lambda x: x[1], reverse=True)
            
            # Select top N hard negatives within suitable similarity range (not too similar, not too dissimilar)
            hard_negs = []
            for neg_id, sim in neg_with_sim:
                # Skip if similarity is too low (not relevant enough to be challenging)
                # or too high (might actually be relevant)
                if 0.15 <= sim <= quality_threshold:
                    hard_negs.append(neg_id)
                    if len(hard_negs) >= num_hard_negatives:
                        break
            
            # Add hard negatives to this query's relevant docs
            updated_relevant_docs[qid].extend(hard_negs)
    
    # Count how many queries got hard negatives
    queries_with_negs = sum(1 for qid in updated_relevant_docs if len(updated_relevant_docs[qid]) > 1)
    avg_negs = sum(len(docs) - 1 for docs in updated_relevant_docs.values()) / max(1, len(updated_relevant_docs))
    
    logging.info(f"Hard negative assignment complete: {queries_with_negs} queries got hard negatives (avg {avg_negs:.1f} per query)")
    
    return updated_relevant_docs

def main():
    """Main function to generate the dataset and evaluate its quality."""
    # Set up parameters
    parser = argparse.ArgumentParser(description="Generate synthetic RAG dataset for embedding model fine-tuning")
    parser.add_argument("--docs_dir", type=str, default=DOCUMENTS_DIR, 
                        help="Directory containing PDF documents")
    parser.add_argument("--negatives_dir", type=str, default=NEGATIVES_DIR, 
                        help="Directory containing negative PDF documents")
    parser.add_argument("--output_dir", type=str, default=DATA_DIR, 
                        help="Directory to save the generated dataset")
    parser.add_argument("--questions_per_chunk", type=int, default=15, 
                        help="Number of questions to generate per document chunk")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_WORD_LIMIT, 
                        help="Approximate word limit for each document chunk")
    parser.add_argument("--overlap", type=int, default=OVERLAP_WORDS, 
                        help="Number of words to overlap between chunks")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                        help="Ratio of documents to use for training vs validation")
    parser.add_argument("--quality_threshold", type=str, default="mediocre", 
                        choices=["poor", "mediocre", "good"],
                        help="Minimum quality threshold for accepting the dataset")
    parser.add_argument("--similarity_threshold", type=float, default=0.4,
                        help="Minimum similarity threshold for filtering queries")
    parser.add_argument("--use_hard_negatives", action="store_true",
                        help="Include hard negatives in the dataset")
    parser.add_argument("--num_hard_negatives", type=int, default=NUM_HARD_NEGATIVES,
                        help="Number of hard negatives to include per query")
    parser.add_argument("--embedding_model", type=str, 
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Embedding model to use for quality assessment")
    args = parser.parse_args()
    
    # Create output directories
    Path(args.output_dir).mkdir(exist_ok=True)
    Path(CACHE_DIR).mkdir(exist_ok=True)
    
    # Find all PDF documents
    logging.info(f"Searching for PDF documents in {args.docs_dir}")
    all_files = []
    for root, _, files in os.walk(args.docs_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        logging.error(f"No PDF files found in {args.docs_dir}")
        return
    
    logging.info(f"Found {len(all_files)} PDF documents")
    
    # Split into train and validation sets
    random.shuffle(all_files)
    split_idx = int(len(all_files) * args.train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    if not train_files:
        logging.error("No training files available after splitting")
        return
    
    if not val_files:
        logging.warning("No validation files available. Using a training file for validation.")
        val_files = [train_files[0]]
    
    # Initialize model for text generation
    model, tokenizer = init_llm()
    
    # Initialize embedding model for quality assessment
    embedding_model = None
    try:
        embedding_model = EmbeddingModel(args.embedding_model)
        if not embedding_model.is_initialized:
            logging.warning(f"Failed to initialize embedding model {args.embedding_model}. Falling back to TF-IDF.")
            embedding_model = None
    except Exception as e:
        logging.error(f"Error initializing embedding model: {e}")
    
    # Process training corpus
    logging.info(f"Generating training corpus from {len(train_files)} documents...")
    train_corpus = load_corpus(train_files, verbose=True)
    train_corpus_path = os.path.join(args.output_dir, "train_corpus.json")
    with open(train_corpus_path, 'w', encoding='utf-8') as f:
        json.dump(train_corpus, f, ensure_ascii=False, indent=2)
    logging.info(f"Training corpus saved with {len(train_corpus)} chunks")
    
    # Process validation corpus
    logging.info(f"Generating validation corpus from {len(val_files)} documents...")
    val_corpus = load_corpus(val_files, verbose=True)
    val_corpus_path = os.path.join(args.output_dir, "val_corpus.json")
    with open(val_corpus_path, 'w', encoding='utf-8') as f:
        json.dump(val_corpus, f, ensure_ascii=False, indent=2)
    logging.info(f"Validation corpus saved with {len(val_corpus)} chunks")
    
    # Load hard negatives if enabled
    negative_corpus = {}
    if args.use_hard_negatives:
        logging.info(f"Loading hard negatives from {args.negatives_dir}...")
        # Combine train and val corpus to ensure negatives are dissimilar from both
        combined_corpus = {**train_corpus, **val_corpus}
        negative_corpus = load_hard_negatives(args.negatives_dir, combined_corpus, verbose=True)
        
        if not negative_corpus:
            logging.warning("No hard negatives could be loaded. Proceeding without hard negatives.")
    
    # Generate training queries
    logging.info(f"Generating {args.questions_per_chunk} questions per chunk for training data...")
    train_queries, train_relevant_docs = generate_queries(
        train_corpus, model, tokenizer, 
        num_questions_per_chunk=args.questions_per_chunk, 
        verbose=True
    )
    
    # Generate validation queries
    logging.info(f"Generating {args.questions_per_chunk} questions per chunk for validation data...")
    val_queries, val_relevant_docs = generate_queries(
        val_corpus, model, tokenizer, 
        num_questions_per_chunk=args.questions_per_chunk, 
        verbose=True
    )
    
    # Filter out low-quality pairs
    logging.info(f"Filtering low-quality pairs using similarity threshold {args.similarity_threshold}...")
    train_queries, train_relevant_docs = filter_low_quality_pairs(
        train_queries, train_corpus, train_relevant_docs,
        embedding_model, args.similarity_threshold
    )
    
    val_queries, val_relevant_docs = filter_low_quality_pairs(
        val_queries, val_corpus, val_relevant_docs,
        embedding_model, args.similarity_threshold
    )
    
    # Assign hard negatives if available
    if args.use_hard_negatives and negative_corpus:
        logging.info(f"Assigning hard negatives ({args.num_hard_negatives} per query)...")
        
        train_relevant_docs = assign_hard_negatives(
            train_queries, train_corpus, train_relevant_docs, negative_corpus,
            embedding_model, args.num_hard_negatives
        )
        
        val_relevant_docs = assign_hard_negatives(
            val_queries, val_corpus, val_relevant_docs, negative_corpus,
            embedding_model, args.num_hard_negatives
        )
    
    # Save the generated data
    train_queries_path = os.path.join(args.output_dir, "train_queries.json")
    with open(train_queries_path, 'w', encoding='utf-8') as f:
        json.dump(train_queries, f, ensure_ascii=False, indent=2)
    
    train_relevant_docs_path = os.path.join(args.output_dir, "train_relevant_docs.json")
    with open(train_relevant_docs_path, 'w', encoding='utf-8') as f:
        json.dump(train_relevant_docs, f, ensure_ascii=False, indent=2)
    
    val_queries_path = os.path.join(args.output_dir, "val_queries.json")
    with open(val_queries_path, 'w', encoding='utf-8') as f:
        json.dump(val_queries, f, ensure_ascii=False, indent=2)
    
    val_relevant_docs_path = os.path.join(args.output_dir, "val_relevant_docs.json")
    with open(val_relevant_docs_path, 'w', encoding='utf-8') as f:
        json.dump(val_relevant_docs, f, ensure_ascii=False, indent=2)
    
    # Create final datasets
    logging.info("Creating final datasets...")
    
    # Include negative corpus in the final dataset for training with hard negatives
    final_train_corpus = train_corpus.copy()
    final_val_corpus = val_corpus.copy()
    
    if args.use_hard_negatives and negative_corpus:
        # Add negative documents to corpus
        for neg_id, neg_text in negative_corpus.items():
            if any(neg_id in docs for docs in train_relevant_docs.values()):
                final_train_corpus[neg_id] = neg_text
            if any(neg_id in docs for docs in val_relevant_docs.values()):
                final_val_corpus[neg_id] = neg_text
    
    train_dataset = {
        'queries': train_queries,
        'corpus': final_train_corpus,
        'relevant_docs': train_relevant_docs,
    }
    train_dataset_path = os.path.join(args.output_dir, "train_dataset.json")
    with open(train_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)
    
    val_dataset = {
        'queries': val_queries,
        'corpus': final_val_corpus,
        'relevant_docs': val_relevant_docs,
    }
    val_dataset_path = os.path.join(args.output_dir, "val_dataset.json")
    with open(val_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, ensure_ascii=False, indent=2)
    
    # Run quality evaluation
    logging.info("Evaluating training dataset quality...")
    train_quality = evaluate_dataset_quality(
        train_queries, final_train_corpus, train_relevant_docs,
        embedding_model, args.similarity_threshold
    )
    logging.info(f"Training dataset quality assessment: {train_quality.upper()}")
    
    logging.info("Evaluating validation dataset quality...")
    val_quality = evaluate_dataset_quality(
        val_queries, final_val_corpus, val_relevant_docs,
        embedding_model, args.similarity_threshold
    )
    logging.info(f"Validation dataset quality assessment: {val_quality.upper()}")
    
    # Check against quality threshold
    quality_levels = {"poor": 0, "mediocre": 1, "good": 2}
    if quality_levels[train_quality] < quality_levels[args.quality_threshold]:
        logging.error(f"QUALITY CHECK FAILED: Training dataset quality ({train_quality}) is below the required threshold ({args.quality_threshold})")
        logging.error("Consider adjusting parameters or the prompt template and regenerating the dataset.")
    else:
        logging.info(f"Dataset generation complete! Files saved to {args.output_dir}")
        logging.info(f"Training dataset: {len(train_queries)} queries, {len(final_train_corpus)} document chunks")
        logging.info(f"Validation dataset: {len(val_queries)} queries, {len(final_val_corpus)} document chunks")
        
        # Create summary file
        summary = {
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": vars(args),
            "train_dataset": {
                "queries": len(train_queries),
                "chunks": len(final_train_corpus),
                "quality": train_quality,
                "hard_negatives": sum(len(docs) - 1 for docs in train_relevant_docs.values())
            },
            "val_dataset": {
                "queries": len(val_queries),
                "chunks": len(final_val_corpus),
                "quality": val_quality,
                "hard_negatives": sum(len(docs) - 1 for docs in val_relevant_docs.values())
            }
        }
        
        summary_path = os.path.join(args.output_dir, "dataset_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logging.info(f"Dataset summary saved to {summary_path}")

if __name__ == "__main__":
    import argparse
    import random
    from datetime import datetime
    main()
