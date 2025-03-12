#!/usr/bin/env python3

"""
Analyze Dataset Quality

This script analyzes the quality of the dataset by checking various metrics and properties
that could affect the training quality. It computes statistics on query and document lengths,
relevance, query diversity (via pairwise similarity), query-document similarity, and corpus coverage.
Warnings are issued when any metric falls outside recommended thresholds.
"""

import json
import numpy as np
from collections import Counter
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

# Create output directory for plots if it doesn't exist
os.makedirs('./Data', exist_ok=True)

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def load_data(data_dir: str = './data_deepseek'):
    """Load dataset files"""
    with open(f"{data_dir}/train_queries.json", 'r') as f:
        train_queries = json.load(f)
    with open(f"{data_dir}/train_corpus.json", 'r') as f:
        train_corpus = json.load(f)
    with open(f"{data_dir}/train_relevant_docs.json", 'r') as f:
        train_relevant_docs = json.load(f)
    return train_queries, train_corpus, train_relevant_docs

def analyze_query_stats(queries: Dict[str, str]):
    """Analyze query statistics"""
    lengths = [len(q.split()) for q in queries.values()]
    avg_length = np.mean(lengths) if lengths else 0
    logging.info(f"\nQuery Statistics:")
    logging.info(f"Total queries: {len(queries)}")
    logging.info(f"Average query length: {avg_length:.2f} words")
    logging.info(f"Min query length: {min(lengths) if lengths else 0} words")
    logging.info(f"Max query length: {max(lengths) if lengths else 0} words")
    
    # Warning if average query length is very short
    if avg_length < 5:
        logging.warning("Average query length is very short. Queries may lack sufficient detail.")
    
    # Plot query length distribution
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title('Query Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig('./Data/query_length_distribution.png')
    plt.close()

def analyze_document_stats(corpus: Dict[str, str]):
    """Analyze document statistics"""
    lengths = [len(doc.split()) for doc in corpus.values()]
    avg_length = np.mean(lengths) if lengths else 0
    logging.info(f"\nDocument Statistics:")
    logging.info(f"Total documents: {len(corpus)}")
    logging.info(f"Average document length: {avg_length:.2f} words")
    logging.info(f"Min document length: {min(lengths) if lengths else 0} words")
    logging.info(f"Max document length: {max(lengths) if lengths else 0} words")
    
    # Warning if average document length is too short
    if avg_length < 50:
        logging.warning("Average document length is very short. Chunks may not contain sufficient context.")
    
    # Plot document length distribution
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=30, color='salmon', edgecolor='black')
    plt.title('Document Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig('./Data/document_length_distribution.png')
    plt.close()

def analyze_relevance_stats(queries: Dict[str, str], relevant_docs: Dict[str, List[str]]):
    """Analyze relevance statistics"""
    rel_counts = [len(docs) for docs in relevant_docs.values()]
    avg_rel = np.mean(rel_counts) if rel_counts else 0
    logging.info(f"\nRelevance Statistics:")
    logging.info(f"Queries with relevant docs: {len(relevant_docs)}")
    logging.info(f"Average relevant docs per query: {avg_rel:.2f}")
    logging.info(f"Min relevant docs per query: {min(rel_counts) if rel_counts else 0}")
    logging.info(f"Max relevant docs per query: {max(rel_counts) if rel_counts else 0}")
    
    # Warning if average relevant docs per query is very low
    if avg_rel < 1:
        logging.warning("Average relevant docs per query is low. Some queries may lack associated documents.")
    
    # Plot relevant documents distribution
    plt.figure(figsize=(10, 5))
    plt.hist(rel_counts, bins=30, color='lightgreen', edgecolor='black')
    plt.title('Relevant Documents per Query Distribution')
    plt.xlabel('Number of Relevant Documents')
    plt.ylabel('Frequency')
    plt.savefig('./Data/relevance_distribution.png')
    plt.close()

def analyze_query_similarity(queries: Dict[str, str]):
    """Analyze query similarity to detect potential duplicates or near-duplicates"""
    vectorizer = TfidfVectorizer(stop_words='english')
    query_texts = list(queries.values())
    tfidf_matrix = vectorizer.fit_transform(query_texts)
    
    # Calculate pairwise similarities
    similarities = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarities, 0)  # Ignore self-similarity
    
    # Find highly similar pairs
    threshold = 0.8
    similar_pairs = []
    for i in range(len(query_texts)):
        for j in range(i + 1, len(query_texts)):
            if similarities[i, j] > threshold:
                similar_pairs.append((query_texts[i], query_texts[j], similarities[i, j]))
    
    # Calculate average pairwise similarity (excluding self-similarity)
    if similarities.size:
        avg_pairwise_similarity = np.sum(similarities) / (len(query_texts)*(len(query_texts)-1))
    else:
        avg_pairwise_similarity = 0
    logging.info(f"\nQuery Similarity Analysis:")
    logging.info(f"Average pairwise query similarity (TF-IDF): {avg_pairwise_similarity:.3f}")
    if avg_pairwise_similarity > 0.8:
        logging.warning("High average pairwise query similarity. Queries may be too homogenous, reducing dataset diversity.")
    
    logging.info(f"Number of highly similar query pairs (similarity > {threshold}): {len(similar_pairs)}")
    if similar_pairs:
        logging.info("Example similar pairs:")
        for q1, q2, sim in similar_pairs[:5]:
            logging.info(f"Similarity: {sim:.3f}")
            logging.info(f"Query 1: {q1}")
            logging.info(f"Query 2: {q2}\n")

def analyze_query_doc_similarity(queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, List[str]]):
    """
    Compute average cosine similarity between each query and its corresponding document using TF-IDF.
    Issues a warning if the average similarity is below a threshold (e.g., 0.2).
    """
    query_list, doc_list = [], []
    for qid, query in queries.items():
        doc_ids = relevant_docs.get(qid, [])
        if doc_ids:
            doc_text = corpus.get(doc_ids[0], "")
            if doc_text:
                query_list.append(query)
                doc_list.append(doc_text)
    if not query_list or not doc_list:
        logging.warning("No valid query-document pairs found for similarity evaluation.")
        return

    # Combine texts for a joint vectorization
    vectorizer = TfidfVectorizer()
    combined_texts = query_list + doc_list
    tfidf = vectorizer.fit_transform(combined_texts)
    query_tfidf = tfidf[:len(query_list)]
    doc_tfidf = tfidf[len(query_list):]
    
    similarities = []
    for i in range(len(query_list)):
        sim = cosine_similarity(query_tfidf[i], doc_tfidf[i])
        similarities.append(sim.item())
    
    avg_similarity = np.mean(similarities)
    logging.info(f"\nQuery-Document Similarity Analysis:")
    logging.info(f"Average cosine similarity between queries and corresponding documents: {avg_similarity:.3f}")
    if avg_similarity < 0.2:
        logging.warning("Low average query-document similarity. Queries might not be well aligned with document content.")

def analyze_coverage(queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, List[str]]):
    """Analyze dataset coverage"""
    # Calculate document usage statistics
    doc_usage = Counter()
    for docs in relevant_docs.values():
        doc_usage.update(docs)
    
    unused_docs = set(corpus.keys()) - set(doc_usage.keys())
    
    logging.info(f"\nCoverage Analysis:")
    logging.info(f"Total unique documents used as relevant: {len(doc_usage)}")
    logging.info(f"Unused documents: {len(unused_docs)} out of {len(corpus)}")
    
    usage_counts = list(doc_usage.values())
    if usage_counts:
        mean_usage = np.mean(usage_counts)
        max_usage = max(usage_counts)
    else:
        mean_usage, max_usage = 0, 0
    logging.info(f"Mean usage per document: {mean_usage:.2f}")
    logging.info(f"Max usage of a single document: {max_usage}")
    
    # Plot document usage distribution
    plt.figure(figsize=(10, 5))
    plt.hist(usage_counts, bins=30, color='orchid', edgecolor='black')
    plt.title('Document Usage Distribution')
    plt.xlabel('Number of Times Used as Relevant')
    plt.ylabel('Frequency')
    plt.savefig('./Data/document_usage_distribution.png')
    plt.close()
    
    # Compute coverage: percentage of corpus referenced by at least one query
    coverage = len(doc_usage) / len(corpus) if corpus else 0
    logging.info(f"Corpus coverage: {coverage * 100:.2f}% of document chunks are referenced by queries.")
    if coverage < 0.5:
        logging.warning("Less than 50% of document chunks are referenced by queries. Consider generating more queries per chunk.")

def main():
    """Main function to analyze dataset quality"""
    logging.info("Starting dataset analysis...")
    
    # Load data
    train_queries, train_corpus, train_relevant_docs = load_data()
    
    # Run analyses
    analyze_query_stats(train_queries)
    analyze_document_stats(train_corpus)
    analyze_relevance_stats(train_queries, train_relevant_docs)
    analyze_query_similarity(train_queries)
    analyze_query_doc_similarity(train_queries, train_corpus, train_relevant_docs)
    analyze_coverage(train_queries, train_corpus, train_relevant_docs)
    
    logging.info("\nAnalysis complete. Check the Data directory for visualizations.")

if __name__ == "__main__":
    main()
