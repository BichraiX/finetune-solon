# Solon Embedding Model Fine-tuning

This repository contains code to fine-tune the Solon Embedding model using the matryoshka representation learning technique.

## Overview

The `finetune_solon.py` script fine-tunes the Solon Embedding model on a custom dataset using the methodology from `FT_Embedding_Models_on_Domain_Specific_Data`. It implements:

- Matryoshka Representation Learning (MRL) for creating embeddings that can be truncated to different sizes
- Multiple Negatives Ranking Loss for retrieval-oriented training
- Evaluation across multiple embedding dimensions (768, 512, 256, 128, 64)
- Information Retrieval metrics for evaluation (NDCG, Precision, Recall, etc.)

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset in the expected format (see Dataset Format section below)

## Usage

Run the fine-tuning script with default parameters:

```bash
python finetune_solon.py
```

### Command Line Arguments

The script accepts several command-line arguments:

- `--data_dir`: Directory containing the dataset files (default: "data_deepseek")
- `--model_id`: Identifier for the model to fine-tune (default: "OrdalieTech/Solon-embeddings-base-0.1")
- `--output_dir`: Directory to save the fine-tuned model (default: "solon-embedding-finetuned")
- `--epochs`: Number of training epochs (default: 4)
- `--batch_size`: Per device training batch size (default: 32)
- `--eval_batch_size`: Per device evaluation batch size (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)

Example with custom parameters:

```bash
python finetune_solon.py --output_dir="my-finetuned-model" --epochs=6 --batch_size=64 --learning_rate=1e-5
```

## Dataset Format

The script expects the following JSON files in the data directory:

1. `train_queries.json`: Dictionary mapping query IDs to query texts for training
2. `train_corpus.json`: Dictionary mapping document IDs to document texts for training
3. `train_relevant_docs.json`: Dictionary mapping query IDs to lists of relevant document IDs for training
4. `val_queries.json`: Dictionary mapping query IDs to query texts for validation
5. `val_corpus.json`: Dictionary mapping document IDs to document texts for validation
6. `val_relevant_docs.json`: Dictionary mapping query IDs to lists of relevant document IDs for validation

Example dataset structure:
```
data_deepseek/
├── train_queries.json
├── train_corpus.json
├── train_relevant_docs.json
├── val_queries.json
├── val_corpus.json
└── val_relevant_docs.json
```

### JSON Format Examples

#### queries.json
```json
{
  "query_id1": "What is the legal significance of a signature?",
  "query_id2": "Explain the concept of electronic signatures"
}
```

#### corpus.json
```json
{
  "doc_id1": "A signature is a legally binding...",
  "doc_id2": "Electronic signatures are recognized as valid..."
}
```

#### relevant_docs.json
```json
{
  "query_id1": ["doc_id1", "doc_id3"],
  "query_id2": ["doc_id2"]
}
```

## Output

The script produces the following outputs:

1. Fine-tuned model saved in the specified output directory
2. Training logs in `finetune_solon.log`
3. Evaluation metrics for each dimension at each epoch during training

The best model is automatically saved based on the NDCG@10 performance on the 128-dimension embeddings.

## Using the Fine-tuned Model

After fine-tuning, you can load and use the model like this:

```python
from sentence_transformers import SentenceTransformer

# Load the fine-tuned model
model = SentenceTransformer('solon-embedding-finetuned')

# Create embeddings
sentences = ['This is a query', 'This is a document']
embeddings = model.encode(sentences)

# For reduced dimension (e.g., 128)
embeddings_dim128 = model.encode(sentences, normalize_embeddings=True, truncate_dim=128)
```

## Troubleshooting

If you encounter issues, check the log file `finetune_solon.log` for detailed error messages.

Common issues:
- CUDA out of memory: Reduce batch size or use gradient accumulation
- Missing files: Ensure your dataset directory has all required files
- Model not found: Verify the model_id is correct and accessible (default is "OrdalieTech/Solon-embeddings-base-0.1") 