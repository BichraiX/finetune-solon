# Quality Filtering for Reformulation Dataset

This document describes the quality filtering functionality added to the reformulation dataset generator.

## Overview

The quality filtering system ensures that the generated reformulation dataset contains only high-quality examples. It implements several quality checks:

1. **Basic checks during generation:**
   - Reformulations must end with a question mark
   - Reformulations must be different from the original input
   - Semantic similarity between input and reformulation must be above a threshold

2. **Comprehensive quality evaluation:**
   - Statistical analysis of input and reformulation lengths
   - Evaluation of question mark usage
   - Semantic similarity assessment between inputs and reformulations

3. **Post-generation filtering:**
   - Remove examples not ending with a question mark
   - Filter examples with low semantic similarity to their inputs

## NEW: Refusal Phrase Detection

The system now automatically detects and filters out examples where the model has generated refusal language in the input, such as:

- "désolé" (sorry)
- "je ne peux pas" (I cannot)
- "je regrette" (I regret)
- "en tant qu'assistant" (as an assistant)
- "modèle de langage" (language model)
- And many other similar phrases

This ensures the dataset contains only genuine user queries and not model disclaimers or refusals.

## Usage

### During Dataset Generation

Quality filtering is enabled by default during generation:

```bash
python generate_reformulation_dataset.py --num-examples 200 --output-file data/reformulation_dataset.json
```

You can adjust the similarity threshold:

```bash
python generate_reformulation_dataset.py --sim-threshold 0.5
```

To disable refusal phrase filtering (not recommended):

```bash
python generate_reformulation_dataset.py --no-filter-refusals
```

### Evaluating an Existing Dataset

To evaluate and filter an existing dataset:

```bash
python generate_reformulation_dataset.py --evaluate-only --input-file data/reformulation_dataset.json
```

This will:
1. Check for and report refusal phrases in the dataset
2. Generate a quality report at `data/reformulation_quality_report.json`
3. Create a filtered version at `data/reformulation_dataset_filtered.json`

## Quality Metrics

The quality evaluation produces a comprehensive report with the following metrics:

- **Dataset size:** Number of examples
- **Input statistics:** Average/min/max length of user inputs
- **Reformulation statistics:** Average/min/max length, percentage with question marks
- **Similarity statistics:** Average/min/max semantic similarity between inputs and reformulations
- **Quality summary:** Count and percentage of good and poor examples
- **Overall quality:** "good", "mediocre", or "poor" based on percentage of high-quality examples

## Requirements

The quality filtering system requires these additional dependencies:

- numpy>=1.24.0
- scikit-learn>=1.3.0
- sentence-transformers>=2.2.0

Install them with:

```bash
pip install numpy scikit-learn sentence-transformers
``` 