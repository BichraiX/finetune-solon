#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune the solon_embedding model using the methodology from FT_Embedding_Models_on_Domain_Specific_Data
"""

import os
import json
import torch
import logging
import argparse
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerModelCardData,
    SentenceTransformerTrainingArguments, 
    SentenceTransformerTrainer
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finetune_solon.log')
    ]
)
logger = logging.getLogger(__name__)


def load_json_file(file_path):
    """Load a JSON file and return its contents"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune Solon Embedding model")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data_deepseek",
        help="Directory containing the dataset files"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="OrdalieTech/Solon-embeddings-base-0.1",
        help="Identifier for the model to fine-tune"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="Data/amine.chraibi/solon-embedding-finetuned-15",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=4,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Per device training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=16,
        help="Per device evaluation batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-5,
        help="Learning rate"
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Ensure output directory is an absolute path
    if not os.path.isabs(args.output_dir):
        # If the path doesn't start with /, make it absolute with respect to the root directory
        args.output_dir = f"/{args.output_dir}"
        logger.info(f"Modified output directory to absolute path: {args.output_dir}")
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Using output directory: {args.output_dir}")

    # Load dataset
    data_dir = args.data_dir
    logger.info(f"Loading datasets from {data_dir}")
    
    # Load train and validation data
    train_queries_file = os.path.join(data_dir, "train_queries.json")
    train_corpus_file = os.path.join(data_dir, "train_corpus.json")
    train_relevant_docs_file = os.path.join(data_dir, "train_relevant_docs.json")
    
    val_queries_file = os.path.join(data_dir, "val_queries.json")
    val_corpus_file = os.path.join(data_dir, "val_corpus.json")
    val_relevant_docs_file = os.path.join(data_dir, "val_relevant_docs.json")
    
    # Check if all required files exist
    required_files = [
        train_queries_file, train_corpus_file, train_relevant_docs_file,
        val_queries_file, val_corpus_file, val_relevant_docs_file
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    logger.info("Loading training data...")
    train_queries = load_json_file(train_queries_file)
    train_corpus = load_json_file(train_corpus_file)
    train_relevant_docs = load_json_file(train_relevant_docs_file)
    
    logger.info("Loading validation data...")
    val_queries = load_json_file(val_queries_file)
    val_corpus = load_json_file(val_corpus_file)
    val_relevant_docs = load_json_file(val_relevant_docs_file)
    
    # Print some stats
    logger.info(f"Train queries: {len(train_queries)}")
    logger.info(f"Train corpus entries: {len(train_corpus)}")
    logger.info(f"Train query-document pairs: {sum(len(docs) for docs in train_relevant_docs.values())}")
    
    logger.info(f"Validation queries: {len(val_queries)}")
    logger.info(f"Validation corpus entries: {len(val_corpus)}")
    logger.info(f"Validation query-document pairs: {sum(len(docs) for docs in val_relevant_docs.values())}")
    
    # Create a dataset for training in sentence-transformers format
    # For training, we need pairs of (anchor, positive) texts
    train_examples = []
    skipped_examples = 0
    
    for query_id, relevant_doc_ids in train_relevant_docs.items():
        if query_id in train_queries:
            query_text = train_queries[query_id]
            for doc_id in relevant_doc_ids:
                if doc_id in train_corpus:
                    doc_text = train_corpus[doc_id]
                    # Create a pair (anchor=query, positive=document)
                    train_examples.append({
                        "anchor": query_text,
                        "positive": doc_text,
                        "query_id": query_id,
                        "doc_id": doc_id
                    })
                else:
                    skipped_examples += 1
                    logger.warning(f"Document ID {doc_id} not found in corpus, skipping example")
        else:
            skipped_examples += len(relevant_doc_ids)
            logger.warning(f"Query ID {query_id} not found in queries, skipping {len(relevant_doc_ids)} examples")
    
    if skipped_examples > 0:
        logger.warning(f"Skipped {skipped_examples} examples due to missing query or document IDs")
    
    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_list(train_examples)
    logger.info(f"Created training dataset with {len(train_dataset)} examples")
    if len(train_dataset) > 0:
        logger.info(f"Sample example: {train_dataset[0]}")
    else:
        logger.error("Training dataset is empty. Please check your data files.")
        raise ValueError("Training dataset is empty. Please check your data files.")
    
    # Create a validation dataset similar to the training dataset
    val_examples = []
    val_skipped_examples = 0
    
    for query_id, relevant_doc_ids in val_relevant_docs.items():
        if query_id in val_queries:
            query_text = val_queries[query_id]
            for doc_id in relevant_doc_ids:
                if doc_id in val_corpus:
                    doc_text = val_corpus[doc_id]
                    # Create a pair (anchor=query, positive=document)
                    val_examples.append({
                        "anchor": query_text,
                        "positive": doc_text,
                        "query_id": query_id,
                        "doc_id": doc_id
                    })
                else:
                    val_skipped_examples += 1
                    logger.warning(f"Validation document ID {doc_id} not found in corpus, skipping example")
        else:
            val_skipped_examples += len(relevant_doc_ids)
            logger.warning(f"Validation query ID {query_id} not found in queries, skipping {len(relevant_doc_ids)} examples")
    
    if val_skipped_examples > 0:
        logger.warning(f"Skipped {val_skipped_examples} validation examples due to missing query or document IDs")
    
    # Convert to HuggingFace Dataset format
    val_dataset = Dataset.from_list(val_examples)
    logger.info(f"Created validation dataset with {len(val_dataset)} examples")
    
    # Determine the evaluation strategy based on validation data availability
    eval_strategy = "epoch"
    min_eval_examples = 10  # Minimum number of examples needed for evaluation
    
    if len(val_dataset) < min_eval_examples:
        logger.warning(f"Validation dataset contains fewer than {min_eval_examples} examples. Setting eval_strategy to 'no'.")
        eval_strategy = "no"
    else:
        logger.info(f"Sample validation example: {val_dataset[0]}")
    
    # Prepare training and validation datasets
    train_dataset = train_dataset.select_columns(["anchor", "positive"])
    
    # Only prepare validation dataset if we're using it
    if eval_strategy != "no":
        val_dataset = val_dataset.select_columns(["anchor", "positive"])
    
    # Dimensions for matryoshka embedding - ordered from largest to smallest
    matryoshka_dimensions = [768, 512, 256, 128, 64]
    logger.info(f"Using matryoshka dimensions: {matryoshka_dimensions}")
    
    # Create evaluators for each dimension
    logger.info("Creating evaluators")
    matryoshka_evaluators = []
    
    for dim in matryoshka_dimensions:
        # Define the evaluator for this dimension
        ir_evaluator = InformationRetrievalEvaluator(
            queries=val_queries,
            corpus=val_corpus,
            relevant_docs=val_relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to the respective dimension
            score_functions={"cosine": cos_sim},
        )
        # Add to list
        matryoshka_evaluators.append(ir_evaluator)
    
    # Create a sequential evaluator to run all dimension-specific evaluators
    evaluator = SequentialEvaluator(matryoshka_evaluators)
    
    # Initialize the solon embedding model
    model_id = args.model_id
    logger.info(f"Loading model: {model_id}")
    
    try:
        # Load model with eager attention implementation instead of sdpa
        # XLMRoberta-based models don't support sdpa yet
        model = SentenceTransformer(
            model_id,
            model_kwargs={"attn_implementation": "eager"},
            model_card_data=SentenceTransformerModelCardData(
                language="en",
                license="apache-2.0",
                model_name="Solon Embedding Model Fine-tuned",
            ),
        )
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        raise
    
    # Define loss functions
    logger.info("Setting up loss functions")
    # Initial Loss - MultipleNegativesRankingLoss
    base_loss = MultipleNegativesRankingLoss(model)
    
    # Wrap with MatryoshkaLoss
    train_loss = MatryoshkaLoss(
        model, base_loss, matryoshka_dims=matryoshka_dimensions
    )
    
    # Training Arguments
    logger.info(f"Setting up training arguments with eval_strategy={eval_strategy}")
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,                 # output directory
        num_train_epochs=args.epochs,               # number of epochs
        per_device_train_batch_size=args.batch_size,  # train batch size
        gradient_accumulation_steps=16,             # for a global batch size of batch_size*16
        per_device_eval_batch_size=args.eval_batch_size,  # evaluation batch size
        warmup_ratio=0.1,                           # warmup ratio
        learning_rate=args.learning_rate,           # learning rate
        lr_scheduler_type="cosine",                 # use cosine learning rate scheduler
        optim="adamw_torch_fused",                  # use fused adamw optimizer
        tf32=True,                                  # use tf32 precision
        bf16=True,                                  # use bf16 precision
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Benefits from no duplicate samples
        eval_strategy=eval_strategy,                # evaluate after each epoch or "no"
        save_strategy="epoch",                      # save after each epoch
        logging_steps=10,                           # log every 10 steps
        save_total_limit=3,                         # save only the last 3 models
        load_best_model_at_end=eval_strategy != "no",  # Only load best model if evaluating
        metric_for_best_model="eval_dim_128_cosine_ndcg@10" if eval_strategy != "no" else None,  # Optimizing metric
        report_to="none",                            # Turning off training logging for now
    )
    
    # Create the trainer
    logger.info("Creating the trainer")
    
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "loss": train_loss,
        "evaluator": evaluator,
    }
    
    # Only add eval_dataset if we're using evaluation
    if eval_strategy != "no" and len(val_dataset) > 0:
        logger.info("Adding validation dataset to trainer")
        trainer_kwargs["eval_dataset"] = val_dataset
    
    trainer = SentenceTransformerTrainer(**trainer_kwargs)
    
    # Start training
    logger.info("Starting training")
    try:
        trainer.train()
        
        # Save the best model
        logger.info(f"Saving the model to {args.output_dir}")
        trainer.save_model()
        
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise 