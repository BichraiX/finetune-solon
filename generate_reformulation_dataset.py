#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a synthetic dataset of reformulated queries for legal documents in French.
This script uses the DeepSeek model to generate examples of original user prompts and
their reformulated versions that improve RAG retrieval performance.
"""

# Requirements:
# transformers>=4.37.0
# accelerate>=0.25.0
# torch>=2.0.0
# numpy>=1.24.0 
# scikit-learn>=1.3.0
# sentence-transformers>=2.2.0

import os
import json
import argparse
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
import traceback
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re

# Import optional libraries with error handling
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_DEEPSEEK = True
except ImportError:
    HAS_DEEPSEEK = False
    print("Warning: transformers or torch not found. You'll need to install them to use DeepSeek model.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generate_reformulation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DOCUMENT_TYPES = [
    "pua",
    "promesse",
    "cession",
    "protocole",
    "bails",
    "garantie",
    "acte réitératif",
    "pacte",
    "transaction"
]

# Example scenarios for each document type
SCENARIOS = {
    "pua": [
        "acquisition d'une entreprise",
        "fusion de sociétés",
        "investissement dans une startup",
        "accord préalable à une vente",
        "partenariat commercial",
        "création d'une joint-venture",
        "achat de brevets technologiques",
        "collaboration de recherche et développement",
        "négociation de droits de propriété intellectuelle",
        "préparation d'une offre publique d'achat"
    ],
    "promesse": [
        "achat immobilier",
        "vente de parts sociales", 
        "acquisition de brevets",
        "cession de fonds de commerce",
        "embauche d'un dirigeant",
        "investissement dans des actions cotées",
        "vente d'un bien industriel",
        "bail commercial",
        "transmission d'une entreprise familiale",
        "acquisition d'une marque commerciale"
    ],
    "cession": [
        "parts sociales d'une SAS",
        "fonds de commerce",
        "portefeuille clients",
        "droits d'auteur",
        "créances commerciales",
        "brevets industriels",
        "marques déposées",
        "logiciels professionnels",
        "base de données clients",
        "actifs immobiliers d'entreprise"
    ],
    "protocole": [
        "résolution de litige commercial",
        "accord entre actionnaires",
        "fusion-acquisition",
        "restructuration d'entreprise",
        "développement commun d'un produit"
    ],
    "bails": [
        "local commercial",
        "bureaux professionnels",
        "entrepôt logistique",
        "espace de coworking",
        "boutique en centre commercial"
    ],
    "garantie": [
        "prêt bancaire",
        "investissement",
        "opération immobilière",
        "cession d'entreprise",
        "contrat de franchise"
    ],
    "acte réitératif": [
        "vente immobilière",
        "fusion d'entreprises",
        "transfert de propriété",
        "cession de parts sociales",
        "confirmation d'une promesse"
    ],
    "pacte": [
        "actionnaires",
        "associés d'une SARL",
        "investisseurs",
        "fondateurs d'une startup",
        "partenaires commerciaux"
    ],
    "transaction": [
        "règlement d'un litige",
        "résolution d'un conflit commercial",
        "accord sur des dommages et intérêts",
        "compensation d'un préjudice",
        "arrangement amiable entre parties"
    ]
}

# Define templates for user prompts (used as fallbacks or for templates-only mode)
USER_PROMPT_TEMPLATES = [
    "J'aimerais des informations sur {doc_type} concernant {scenario}.",
    "Pouvez-vous m'aider avec un {doc_type} pour {scenario}?",
    "Je cherche un {doc_type} lié à {scenario}, comment procéder?",
    "Besoin d'aide pour un {doc_type} dans le cadre de {scenario}.",
    "Comment obtenir un {doc_type} pour {scenario}?",
    "Questions sur le {doc_type} dans la situation de {scenario}.",
    "Conseils juridiques pour {doc_type} dans le contexte {scenario}?",
    "Démarches à suivre pour {doc_type} suite à {scenario}.",
    "Quelles sont les règles du {doc_type} qui s'appliquent à {scenario}?",
    "Je dois préparer un {doc_type} pour {scenario}, que faire?",
    "Existe-t-il un modèle de {doc_type} adapté à {scenario}?",
    "Droits et obligations dans un {doc_type} pour {scenario}?",
    "Aspects légaux du {doc_type} en cas de {scenario}?"
]

class EmbeddingModel:
    """Embedding model for assessing semantic similarity between texts."""
    
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_initialized = False
        
        try:
            # Try to import sentence-transformers
            logger.info(f"Loading embedding model {model_name}...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            self.is_initialized = True
            logger.info(f"Initialized embedding model {model_name} on {self.device}")
        except ImportError:
            logger.warning("sentence-transformers package not installed. Install with: pip install sentence-transformers")
            self.is_initialized = False
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {str(e)}")
            self.is_initialized = False
    
    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode texts into embeddings."""
        if not self.is_initialized:
            raise ValueError("Embedding model not initialized")
        
        # Ensure all inputs are strings
        texts = [str(t) for t in texts]
        
        # Get embeddings using sentence-transformers
        embeddings = self.model.encode(texts, batch_size=batch_size, 
                                        show_progress_bar=False, convert_to_numpy=True)
        return embeddings
    
    def compute_similarities(self, source_texts: List[str], target_texts: List[str]) -> List[float]:
        """Compute semantic similarities between pairs of texts."""
        if not self.is_initialized:
            raise ValueError("Embedding model not initialized")
        
        if len(source_texts) != len(target_texts):
            raise ValueError("Source and target text lists must have the same length")
        
        # Get embeddings
        source_embeddings = self.encode(source_texts)
        target_embeddings = self.encode(target_texts)
        
        # Compute cosine similarities
        similarities = []
        for i in range(len(source_texts)):
            source_emb = source_embeddings[i]
            target_emb = target_embeddings[i]
            
            # Normalize and compute dot product (cosine similarity)
            source_norm = np.linalg.norm(source_emb)
            target_norm = np.linalg.norm(target_emb)
            
            if source_norm > 0 and target_norm > 0:
                sim = np.dot(source_emb, target_emb) / (source_norm * target_norm)
                similarities.append(float(sim))
            else:
                similarities.append(0.0)
        
        return similarities

def load_deepseek_model(model_id: str = "deepseek-ai/deepseek-coder-6.7b-instruct"):
    """Load the DeepSeek model and tokenizer."""
    if not HAS_DEEPSEEK:
        raise ImportError("You need to install transformers and torch to use this function.")
    
    logger.info(f"Loading DeepSeek model: {model_id}")
    
    try:
        # Use 4-bit quantization for efficiency if supported
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True
            )
            
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading DeepSeek model: {str(e)}")
        raise

def generate_user_prompt_with_deepseek(
    model,
    tokenizer,
    doc_type: str,
    scenario: str
) -> str:
    """Generate a realistic user prompt using the DeepSeek model."""
    
    system_prompt = """Tu es une intelligence artificielle chargée de générer des requêtes juridiques authentiques et réalistes qu'un utilisateur pourrait poser.

### Instructions:
1. Génère des demandes d'utilisateurs concernant des documents juridiques.
2. Les demandes doivent être en français et sembler 100% authentiques, comme si elles provenaient d'un vrai utilisateur.
3. Elles peuvent être vagues, précises, formelles ou informelles, mais toujours réalistes.
4. N'inclus JAMAIS de méta-commentaires comme "à la va-vite", "pas très clair", "c'est urgent" ou des indications similaires.
5. N'annonce jamais que tu es un assistant, un modèle ou une IA. Tu dois simuler un utilisateur réel cherchant des informations juridiques.
6. Les demandes doivent être concises (30-80 mots maximum) et directes.
7. Les erreurs de frappe occasionnelles ou formulations imparfaites sont acceptables pour plus de réalisme.

La requête doit être exactement ce qu'un utilisateur écrirait, pas ce qu'un assistant répondrait."""

    user_message = f"""Génère une demande d'utilisateur authentique concernant un document juridique de type "{doc_type}" dans le contexte de "{scenario}".

Contraintes importantes:
- N'inclus AUCUN méta-commentaire ("pas très clair", "à la va-vite", etc.)
- Ne commence pas par "Utilisateur:" ou similaire
- N'indique jamais que c'est un exemple ou une simulation
- Écris exactement comme un vrai utilisateur, avec parfois des formulations imparfaites
- Sois concis et direct (moins de 60 mots)

Donne uniquement le texte brut que l'utilisateur écrirait."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        # Format messages for DeepSeek
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.92,
        )
        user_prompt = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up the prompt
        user_prompt = user_prompt.strip().strip('"\'')
        
        # Remove potential meta-commentary patterns
        user_prompt = re.sub(r"(?i)à la va-vite|pas très clair|pas clair|en urgence", "", user_prompt)
        user_prompt = re.sub(r"(?i)je suis un utilisateur|en tant qu'utilisateur", "", user_prompt)
        
        logger.info(f"Generated user prompt for {doc_type} - {scenario}")
        return user_prompt
    
    except Exception as e:
        logger.error(f"Error generating user prompt: {str(e)}")
        # Fallback to templates if the model generation fails
        prompt_template = random.choice(USER_PROMPT_TEMPLATES)
        return prompt_template.format(doc_type=doc_type, scenario=scenario)

def generate_example_with_deepseek(
    model,
    tokenizer,
    doc_type: str,
    scenario: str
) -> Dict[str, Any]:
    """Generate a synthetic example using the DeepSeek model."""
    # Decide whether to use a template or generate a user prompt with DeepSeek
    # Use DeepSeek to generate user prompt 70% of the time for greater variety
    if random.random() < 0.7:
        user_prompt = generate_user_prompt_with_deepseek(model, tokenizer, doc_type, scenario)
    else:
        # Fallback to templates for the remaining 30%
        prompt_template = random.choice(USER_PROMPT_TEMPLATES)
        user_prompt = prompt_template.format(doc_type=doc_type, scenario=scenario)
    
    # Construct prompt for DeepSeek for reformulation
    system_prompt = """Tu es un expert spécialisé dans la reformulation de requêtes pour un système de recherche juridique en français.

### Contexte:
Dans un système de Retrieval Augmented Generation (RAG) juridique, la qualité de la requête initiale affecte considérablement la pertinence des résultats. Ta mission est de convertir des demandes d'utilisateurs en questions précises et structurées qui optimiseront la recherche documentaire.

### Instructions:
1. Analyse la demande de l'utilisateur concernant un document juridique
2. Reformule-la sous forme d'une question claire, précise et bien structurée
3. Conserve tous les éléments juridiques importants présents dans la requête originale
4. La reformulation DOIT être sous forme de question et se terminer par un point d'interrogation
5. Utilise un langage juridique approprié mais accessible
6. La question doit être autonome et compréhensible sans contexte supplémentaire
7. Garde une longueur raisonnable (10-25 mots)

### Exemples:
User: "Je veux savoir comment contester mon bail commercial."
Reformulated query: "Quelles sont les procédures légales pour contester un bail commercial en France?"

User: "Besoin d'info sur le droit de préemption pour mon voisin"
Reformulated query: "Comment s'applique le droit de préemption entre voisins dans le cadre d'une vente immobilière?"

Format de ta réponse: "Reformulated query: [ta question reformulée]" sans aucune autre explication."""

    user_message = f"""Voici la demande authentique d'un utilisateur concernant un document juridique:
"{user_prompt}"

Reformule cette demande en une question claire, précise et bien structurée qui optimisera la recherche dans un système RAG juridique.
La reformulation doit:
- Être sous forme de question et se terminer par ?
- Conserver tous les éléments juridiques importants de la demande originale
- Être autonome et compréhensible sans contexte supplémentaire
- Utiliser un langage juridique approprié mais accessible

Commence ta réponse par "Reformulated query: " suivi directement de la question reformulée. Ne donne aucune explication avant ou après."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        # Format messages for DeepSeek
        logger.info(f"Generating example for {doc_type} - {scenario}")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.65,  # Slightly lower temperature for more focused outputs
            top_p=0.9,
        )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract the reformulated query from the response
        reformulated_query = response.strip()
        
        # Extract only the reformulated query using the prefix
        if "Reformulated query : " in reformulated_query:
            reformulated_query = reformulated_query.split("Reformulated query : ", 1)[1].strip()
        # Try alternative formats the model might use
        elif "Reformulated query: " in reformulated_query:
            reformulated_query = reformulated_query.split("Reformulated query: ", 1)[1].strip()
        elif "reformulated query : " in reformulated_query.lower():
            reformulated_query = reformulated_query.split("reformulated query : ", 1)[1].strip()
        elif "reformulated query: " in reformulated_query.lower():
            reformulated_query = reformulated_query.split("reformulated query: ", 1)[1].strip()
        
        # In case the model still gives a longer explanation or has other text, try to extract just the question
        elif len(reformulated_query.split('\n')) > 1:
            for line in reformulated_query.split('\n'):
                if '?' in line:
                    reformulated_query = line.strip()
                    break
        
        # Remove quotes if present
        reformulated_query = reformulated_query.strip('"\'')
        
        # If there's text after the question mark, trim it to just the question
        if '?' in reformulated_query:
            question_parts = reformulated_query.split('?', 1)
            reformulated_query = question_parts[0] + '?'
        # Ensure the query ends with a question mark
        elif not reformulated_query.endswith('?'):
            reformulated_query = reformulated_query + '?'
        
        # Create the example
        example = {
            "input": user_prompt,
            "output": {
                "type": doc_type,
                "reformulated_query": reformulated_query
            }
        }
        
        logger.info(f"Generated example: {example}")
        return example
    
    except Exception as e:
        logger.error(f"Error generating example: {str(e)}")
        raise

def generate_scenario_with_deepseek(
    model,
    tokenizer,
    doc_type: str
) -> str:
    """Generate a realistic scenario using the DeepSeek model."""
    
    system_prompt = """Tu es spécialisé dans la création de scénarios juridiques réalistes pour la France.

Ta tâche est de générer des scénarios courts (3-6 mots) et authentiques concernant des situations juridiques spécifiques.
Ces scénarios seront utilisés pour générer des requêtes juridiques réalistes.

Exemples de bons scénarios:
- Pour "bail": "location appartement meublé", "rupture anticipée bail commercial", "caution non rendue"
- Pour "contrat": "vente véhicule occasion", "litige prestation service", "clause de non-concurrence"
- Pour "procuration": "vente immobilière", "gestion compte bancaire", "représentation assemblée générale"

Les scénarios doivent être:
1. Concis (3-6 mots maximum)
2. Spécifiques au type de document juridique indiqué
3. Réalistes et ancrés dans le contexte juridique français
4. Formulés avec des termes clairs et précis"""

    user_message = f"""Génère un scénario court et réaliste (3-6 mots maximum) pour un document juridique de type "{doc_type}".

Le scénario doit être concis, pertinent juridiquement et utilisable dans une requête d'utilisateur concernant ce type de document.

Donne uniquement le texte du scénario, sans phrase d'introduction ou commentaire."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        # Format messages for DeepSeek
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,  # Short output
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        scenario = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up and limit length
        scenario = scenario.strip().strip('"\'')
        # Limit to 6 words maximum
        words = scenario.split()
        if len(words) > 6:
            scenario = " ".join(words[:6])
        
        logger.info(f"Generated scenario for {doc_type}: {scenario}")
        return scenario
    
    except Exception as e:
        logger.error(f"Error generating scenario: {str(e)}")
        # Fallback to a generic scenario if generation fails
        return random.choice(SCENARIOS.get(doc_type, SCENARIOS["protocole"]))

def contains_refusal_phrases(text: str) -> bool:
    """
    Check if the text contains phrases that indicate a model refusal or non-compliance.
    
    Args:
        text: The text to check
        
    Returns:
        True if the text contains refusal phrases, False otherwise
    """
    # List of refusal phrases to check for (lowercase)
    refusal_phrases = [
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
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check if any refusal phrase is in the text
    for phrase in refusal_phrases:
        if phrase in text_lower:
            return True
    
    return False

def evaluate_reformulation_quality(
    examples: List[Dict[str, Any]],
    embedding_model: Optional[EmbeddingModel] = None,
    quality_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate the quality of reformulated queries using various metrics.
    
    Args:
        examples: List of reformulation examples
        embedding_model: Optional embedding model for semantic similarity
        quality_threshold: Threshold for determining good quality reformulations
        
    Returns:
        A report with quality metrics and assessments
    """
    if not examples:
        logger.warning("Empty dataset provided for evaluation.")
        return {"overall_quality": "poor", "example_count": 0}
    
    logger.info(f"Evaluating quality of {len(examples)} reformulation examples")
    
    # 1. Basic statistics
    input_lengths = [len(ex["input"].split()) for ex in examples]
    avg_input_length = np.mean(input_lengths)
    
    output_lengths = [len(ex["output"]["reformulated_query"].split()) for ex in examples]
    avg_output_length = np.mean(output_lengths)
    
    logger.info(f"Average input length: {avg_input_length:.1f} words")
    logger.info(f"Average reformulated query length: {avg_output_length:.1f} words")
    
    if avg_output_length < 5:
        logger.warning("Average reformulated query length is very short. Queries might not be sufficiently detailed.")
    
    # 2. Check if reformulations end with a question mark
    has_question_mark = [ex["output"]["reformulated_query"].strip().endswith('?') for ex in examples]
    pct_with_question = sum(has_question_mark) / len(examples) * 100
    logger.info(f"{pct_with_question:.1f}% of reformulations end with a question mark")
    
    if pct_with_question < 95:
        logger.warning(f"Only {pct_with_question:.1f}% of reformulations end with a question mark. Expected at least 95%.")
    
    # 3. Calculate input-output similarity using embedding model
    input_list, output_list = [], []
    for ex in examples:
        input_list.append(ex["input"])
        output_list.append(ex["output"]["reformulated_query"])
    
    similarities = []
    
    if embedding_model and embedding_model.is_initialized:
        logger.info("Computing input-reformulation similarities with embedding model...")
        similarities = embedding_model.compute_similarities(input_list, output_list)
    else:
        # Fall back to TF-IDF if embedding model isn't available
        logger.warning("Advanced embedding model not available. Falling back to TF-IDF.")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        try:
            combined = input_list + output_list
            tfidf = vectorizer.fit_transform(combined)
            input_tfidf = tfidf[:len(input_list)]
            output_tfidf = tfidf[len(input_list):]
            
            for i in range(len(input_list)):
                sim = cosine_similarity(input_tfidf[i], output_tfidf[i]).item()
                similarities.append(sim)
        except Exception as e:
            logger.error(f"Error in similarity calculation: {e}")
            similarities = [0.0] * len(input_list)
    
    if not similarities:
        logger.warning("Failed to compute similarities.")
        return {"overall_quality": "poor", "example_count": len(examples)}
    
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    logger.info(f"Input-reformulation similarity: avg={avg_sim:.3f}, min={min_sim:.3f}")
    
    # 4. Identify potentially poor quality reformulations
    quality_results = []
    for i, ex in enumerate(examples):
        sim = similarities[i] if i < len(similarities) else 0.0
        has_question = has_question_mark[i] if i < len(has_question_mark) else False
        
        quality = "good" if sim >= quality_threshold and has_question else "poor"
        quality_results.append({
            "id": i,
            "input": ex["input"],
            "reformulated_query": ex["output"]["reformulated_query"],
            "similarity": sim,
            "has_question_mark": has_question,
            "quality": quality
        })
    
    # Count good quality examples
    good_examples = sum(1 for r in quality_results if r["quality"] == "good")
    good_pct = good_examples / len(examples) * 100
    
    # 5. Generate quality report
    try:
        report = {
            "dataset_size": len(examples),
            "input_stats": {
                "avg_length": float(avg_input_length),
                "min_length": min(input_lengths),
                "max_length": max(input_lengths),
            },
            "reformulation_stats": {
                "avg_length": float(avg_output_length),
                "min_length": min(output_lengths),
                "max_length": max(output_lengths),
                "pct_with_question_mark": float(pct_with_question)
            },
            "similarity_stats": {
                "avg_similarity": float(avg_sim),
                "min_similarity": float(min_sim),
                "max_similarity": float(max(similarities)) if similarities else 0.0,
            },
            "quality_summary": {
                "good_examples": good_examples,
                "poor_examples": len(examples) - good_examples,
                "good_percentage": float(good_pct)
            },
            "overall_quality": "good" if good_pct >= 80 else "mediocre" if good_pct >= 60 else "poor"
        }
        
        quality_report_path = os.path.join("data", "reformulation_quality_report.json")
        os.makedirs(os.path.dirname(quality_report_path), exist_ok=True)
        with open(quality_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        logger.info(f"Quality report saved to {quality_report_path}")
        
        return report
    except Exception as e:
        logger.error(f"Error generating quality report: {e}")
        return {"overall_quality": "unknown", "example_count": len(examples), "error": str(e)}

def filter_low_quality_reformulations(
    examples: List[Dict[str, Any]],
    embedding_model: Optional[EmbeddingModel] = None,
    sim_threshold: float = 0.4,
    require_question_mark: bool = True,
    filter_refusals: bool = True
) -> List[Dict[str, Any]]:
    """
    Filter out low-quality reformulations based on quality criteria.
    
    Args:
        examples: List of reformulation examples
        embedding_model: Optional embedding model for semantic similarity
        sim_threshold: Minimum semantic similarity between input and reformulation
        require_question_mark: Whether to require reformulations to end with a question mark
        filter_refusals: Whether to filter out examples containing refusal phrases
        
    Returns:
        Filtered list of high-quality examples
    """
    if not examples:
        logger.warning("Empty dataset provided for filtering.")
        return []
    
    logger.info(f"Filtering low-quality reformulations from {len(examples)} examples...")
    
    # Filter examples with refusal phrases if requested
    if filter_refusals:
        refusal_count = 0
        filtered_without_refusals = []
        for ex in examples:
            if contains_refusal_phrases(ex["input"]):
                refusal_count += 1
                continue
            filtered_without_refusals.append(ex)
        
        if refusal_count > 0:
            logger.info(f"Removed {refusal_count} examples containing refusal phrases")
        examples = filtered_without_refusals
    
    # Filter examples that don't end with a question mark if required
    if require_question_mark:
        question_filtered = [ex for ex in examples if ex["output"]["reformulated_query"].strip().endswith('?')]
        logger.info(f"Kept {len(question_filtered)}/{len(examples)} examples that end with a question mark")
    else:
        question_filtered = examples
    
    # If no embedding model available or no examples left, return current filtered set
    if not embedding_model or not embedding_model.is_initialized or not question_filtered:
        return question_filtered
    
    # Calculate input-output similarity
    input_list, output_list = [], []
    for ex in question_filtered:
        input_list.append(ex["input"])
        output_list.append(ex["output"]["reformulated_query"])
    
    # Compute similarities with embedding model
    similarities = embedding_model.compute_similarities(input_list, output_list)
    
    # Keep examples with high enough similarity
    filtered_examples = []
    for i, ex in enumerate(question_filtered):
        if i < len(similarities) and similarities[i] >= sim_threshold:
            filtered_examples.append(ex)
    
    removed_count = len(question_filtered) - len(filtered_examples)
    retained_pct = (len(filtered_examples) / max(1, len(examples))) * 100
    
    logger.info(f"Filtering complete: kept {len(filtered_examples)} examples ({retained_pct:.1f}%), removed {removed_count} low-quality examples.")
    
    return filtered_examples

def generate_dataset(
    model, 
    tokenizer, 
    num_examples: int = 15000, 
    output_file: str = "reformulation_dataset.json",
    resume_from: int = 0,
    quality_filter: bool = True,
    sim_threshold: float = 0.4
) -> List[Dict[str, Any]]:
    """Generate a synthetic dataset of reformulated queries with quality filtering."""
    # Check if we're resuming from a previous run
    dataset = []
    if resume_from > 0 and os.path.exists(output_file.replace('.json', f'_intermediate_{resume_from}.json')):
        # Load existing dataset
        with open(output_file.replace('.json', f'_intermediate_{resume_from}.json'), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Resuming from example {resume_from}. Loaded {len(dataset)} existing examples.")
    
    # Initialize embedding model for quality assessment if needed
    embedding_model = None
    if quality_filter:
        try:
            embedding_model = EmbeddingModel()
            if not embedding_model.is_initialized:
                logger.warning("Failed to initialize embedding model. Quality filtering will be limited.")
        except Exception as e:
            logger.warning(f"Error initializing embedding model: {str(e)}")
    
    # Generate examples
    examples_generated = 0
    target_examples = num_examples
    max_attempts = num_examples * 2  # Allow up to 2x attempts to reach target number
    refusal_count = 0
    
    for i in range(resume_from, resume_from + max_attempts):
        if examples_generated >= target_examples:
            break
            
        try:
            # Randomly select a document type
            doc_type = random.choice(DOCUMENT_TYPES)
            
            # Decide whether to use a predefined scenario or generate one with DeepSeek
            if random.random() < 0.6:  # 60% chance to generate a new scenario
                scenario = generate_scenario_with_deepseek(model, tokenizer, doc_type)
            else:
                # Fallback to predefined scenarios
                scenario = random.choice(SCENARIOS.get(doc_type, SCENARIOS["protocole"]))
            
            # Generate an example
            example = generate_example_with_deepseek(model, tokenizer, doc_type, scenario)
            
            # Check for refusal phrases in the input
            if contains_refusal_phrases(example["input"]):
                logger.warning(f"Discarding example with refusal phrase in input: {example['input'][:100]}...")
                refusal_count += 1
                continue
            
            # Apply basic quality checks
            reformulated_query = example["output"]["reformulated_query"]
            
            # Basic quality check: Must end with a question mark
            if not reformulated_query.strip().endswith('?'):
                logger.warning(f"Discarding example that doesn't end with a question mark: {reformulated_query}")
                continue
                
            # Basic quality check: Reformulation must be different from input
            if example["input"].lower() == reformulated_query.lower():
                logger.warning(f"Discarding example where reformulation is identical to input: {reformulated_query}")
                continue
                
            # Semantic similarity check if embedding model is available
            if embedding_model and embedding_model.is_initialized and quality_filter:
                similarity = embedding_model.compute_similarities([example["input"]], [reformulated_query])[0]
                if similarity < sim_threshold:
                    logger.warning(f"Discarding example with low input-output similarity ({similarity:.3f}): {reformulated_query}")
                    continue
            
            # If passes all checks, add to dataset
            dataset.append(example)
            examples_generated += 1
            
            # Log progress
            if examples_generated % 10 == 0:
                logger.info(f"Progress: {examples_generated}/{target_examples} examples generated ({examples_generated / target_examples * 100:.1f}%)")
                logger.info(f"Refusal phrases detected and filtered: {refusal_count}")
                
                # Save intermediate results every 50 examples
                if examples_generated % 50 == 0:
                    intermediate_file = output_file.replace('.json', f'_intermediate_{examples_generated}.json')
                    with open(intermediate_file, 'w', encoding='utf-8') as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved intermediate results to {intermediate_file}")
            
            # Add a short delay to avoid overloading the model
            time.sleep(0.2)
            
        except Exception as e:
            # Log the error but continue with the next example
            logger.error(f"Error generating example {i}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Save current progress
            intermediate_file = output_file.replace('.json', f'_error_at_{i}.json')
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved progress before error to {intermediate_file}")
            
            # Longer cooldown if we encounter an error (could be rate limiting)
            logger.info("Cooling down for 30 seconds before continuing...")
            time.sleep(30)
    
    # Report on refusal phrases filtered
    if refusal_count > 0:
        logger.info(f"Total refusal phrases detected and filtered: {refusal_count}")
    
    # Final quality evaluation
    if quality_filter and embedding_model and embedding_model.is_initialized:
        logger.info("Performing final quality evaluation on the complete dataset...")
        quality_report = evaluate_reformulation_quality(dataset, embedding_model)
        logger.info(f"Overall dataset quality: {quality_report['overall_quality']}")
        
        # If quality is poor, apply final filtering
        if quality_report['overall_quality'] == 'poor':
            logger.warning("Dataset quality is poor. Applying final quality filtering...")
            dataset = filter_low_quality_reformulations(dataset, embedding_model, sim_threshold)
    
    # Save the final dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Generated {len(dataset)} examples, saved to {output_file}")
    logger.info(f"Total refusal examples filtered: {refusal_count}")
    return dataset

def generate_examples_without_model(
    num_examples: int = 20, 
    output_file: str = "reformulation_dataset_template.json"
) -> List[Dict[str, Any]]:
    """
    Generate a set of examples using templates without requiring a language model.
    Useful for testing or when a model is not available.
    """
    logger.info(f"Generating {num_examples} examples using templates (no model)")
    
    # Sample input pairs
    example_pairs = [
        # Bail (Lease)
        {
            "input": "Je veux savoir comment résilier mon bail d'appartement avant la fin du contrat.",
            "reformulated_query": "Quelles sont les conditions légales pour résilier un bail d'habitation de manière anticipée?"
        },
        {
            "input": "Mon propriétaire veut augmenter mon loyer de 10%, il a le droit?",
            "reformulated_query": "Dans quelles conditions un propriétaire peut-il légalement augmenter le montant d'un loyer en cours de bail?"
        },
        {
            "input": "Problème avec ma caution non rendue après 3 mois",
            "reformulated_query": "Quels sont les délais légaux de restitution d'un dépôt de garantie après la fin d'un bail?"
        },
        
        # Contrat (Contract)
        {
            "input": "Je veux annuler un contrat signé hier avec un commercial",
            "reformulated_query": "Quel est le délai légal de rétractation pour un contrat commercial signé à domicile?"
        },
        {
            "input": "Clause de non concurrence dans mon contrat de travail",
            "reformulated_query": "Quelles sont les conditions de validité d'une clause de non-concurrence dans un contrat de travail en France?"
        },
        {
            "input": "Mon employeur peut-il modifier mon contrat sans mon accord?",
            "reformulated_query": "Dans quelles conditions un employeur peut-il modifier unilatéralement un contrat de travail en France?"
        },
        
        # Testament (Will)
        {
            "input": "Comment faire un testament sans notaire?",
            "reformulated_query": "Quelles sont les formes légales de testament olographe reconnues en droit français?"
        },
        {
            "input": "Mon père est décédé sans testament, comment ça se passe?",
            "reformulated_query": "Comment est répartie la succession en l'absence de testament selon le droit français?"
        },
        
        # Procuration (Power of Attorney)
        {
            "input": "Je dois donner procuration à mon frère pour vendre ma maison",
            "reformulated_query": "Quelles sont les formalités juridiques pour établir une procuration de vente immobilière en France?"
        },
        {
            "input": "Procuration bancaire pour ma mère âgée",
            "reformulated_query": "Comment établir une procuration bancaire pour une personne âgée et quelles sont ses limites juridiques?"
        }
    ]
    
    dataset = []
    doc_types = DOCUMENT_TYPES
    
    # Create examples by combining templates and document types
    for i in range(num_examples):
        # Choose a document type randomly
        doc_type = random.choice(doc_types)
        
        # Choose a random example pair that will serve as a template
        template = random.choice(example_pairs)
        
        # Create the example
        example = {
            "input": template["input"],
            "output": {
                "type": doc_type,
                "reformulated_query": template["reformulated_query"]
            }
        }
        
        dataset.append(example)
    
    # Save the dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Generated {len(dataset)} template examples, saved to {output_file}")
    return dataset

def main():
    """Entry point for generating the reformulation dataset."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset of reformulated queries")
    parser.add_argument('--num-examples', type=int, default=15000, help='Number of examples to generate')
    parser.add_argument('--output-file', type=str, default='/Data/amine.chraibi/rag/reformulation_dataset.json', help='Output file path')
    parser.add_argument('--resume-from', type=int, default=0, help='Resume generation from a specific count')
    parser.add_argument('--template-only', action='store_true', help='Generate template examples without using a model')
    parser.add_argument('--quality-filter', action='store_true', default=True, help='Apply quality filtering to examples')
    parser.add_argument('--no-quality-filter', action='store_false', dest='quality_filter', help='Disable quality filtering')
    parser.add_argument('--sim-threshold', type=float, default=0.4, help='Similarity threshold for quality filtering')
    parser.add_argument('--filter-refusals', action='store_true', default=True, help='Filter out examples with refusal phrases')
    parser.add_argument('--no-filter-refusals', action='store_false', dest='filter_refusals', help='Disable refusal phrase filtering')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate an existing dataset without generating new examples')
    parser.add_argument('--input-file', type=str, help='Input dataset file for evaluation only mode')
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Evaluate-only mode
    if args.evaluate_only:
        if not args.input_file or not os.path.exists(args.input_file):
            logger.error("Input file is required for evaluate-only mode")
            return
            
        logger.info(f"Evaluating quality of existing dataset: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        # First check for refusal phrases if requested
        if args.filter_refusals:
            logger.info("Checking for refusal phrases in the dataset...")
            refusal_count = sum(1 for ex in dataset if contains_refusal_phrases(ex["input"]))
            if refusal_count > 0:
                logger.warning(f"Found {refusal_count} examples containing refusal phrases")
        
        embedding_model = EmbeddingModel()
        if embedding_model.is_initialized:
            quality_report = evaluate_reformulation_quality(dataset, embedding_model)
            logger.info(f"Overall dataset quality: {quality_report['overall_quality']}")
            
            if args.quality_filter or quality_report['overall_quality'] == 'poor':
                logger.warning("Applying quality filtering...")
                filtered_dataset = filter_low_quality_reformulations(
                    dataset, 
                    embedding_model, 
                    sim_threshold=args.sim_threshold,
                    filter_refusals=args.filter_refusals
                )
                
                # Save filtered dataset
                filtered_output = args.input_file.replace('.json', '_filtered.json')
                with open(filtered_output, 'w', encoding='utf-8') as f:
                    json.dump(filtered_dataset, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved filtered dataset ({len(filtered_dataset)} examples) to {filtered_output}")
        else:
            logger.error("Could not initialize embedding model for quality evaluation")
        return
    
    # Generate dataset with or without a model
    if args.template_only:
        logger.info(f"Generating {args.num_examples} template examples without using a model")
        dataset = generate_examples_without_model(args.num_examples, args.output_file)
    else:
        # Load the model
        logger.info("Loading DeepSeek model...")
        model, tokenizer = load_deepseek_model()
        
        # Generate the dataset
        logger.info(f"Generating {args.num_examples} examples using DeepSeek")
        dataset = generate_dataset(
            model, 
            tokenizer, 
            num_examples=args.num_examples, 
            output_file=args.output_file,
            resume_from=args.resume_from,
            quality_filter=args.quality_filter,
            sim_threshold=args.sim_threshold
        )
    
    logger.info(f"Dataset generation complete. Generated {len(dataset)} examples.")

if __name__ == "__main__":
    main() 