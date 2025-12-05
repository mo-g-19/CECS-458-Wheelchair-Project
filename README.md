# Wheelchair Friendly Restaurant Assistant
## A Deep Learning System for Ranking Restaurants by Accessibility and Experience

This repository contains the codebase for the Wheelchair Accessibility Recommendation System, a CECS-458 project integrating Transformer-based NLP, Graph Neural Networks (GraphSAGE, GCN, GAT), and late-fusion ranking to generate personalized restaurant recommendations optimized for both accessibility and user experience.

The system focuses on restaurants in the Los Angeles area and supports dynamic updates as new users, reviews, and venues are added.

## Core Features
### 1) Transformer-Based NLP Embeddings
- Cleans and tokenizes user reviews
- Generates semantic embeddings using SentenceTransformer (MiniLM-L6-v2)
- Extracts accessibility-related information from free-text

### 2) Multi-Stage GNN Pipeline
- GraphSAGE: Maintains the full citywide graph (restaurants, users, reviews). Inductive updates allow new nodes to be added daily.
- GCN: Builds a per-query subgraph based on user location and smooths local features for fast, context-aware inference.
- GAT: Applies attention weighting with reviewer credibility (PageRank-style), refining accessibility and experience scores.

### 3) Modular Late Fusion
Final restauraunt ranking is computed by fusing:
- NLP embeddings
- GAT-refined accessibility and experience scores

Supports two ranking modes:
- Accessibility-first
- Enviroment / Enjoyment

### 4) Data Services (Future implementation)
Integrating multiple storage layers
- Supabase – spatial restaurant data
- Firestore – unstructured review text
- FAISS – vector database for embeddings
- Railway – backend hosting (API + inference)

## Repository Structure
.
├── src/

│   ├── gnn/

│   │   ├── graphsage.py

│   │   ├── gcn.py

│   │   ├── gat.py

│   │   └── build_features.py

│   ├── nlp/

│   │   ├── cleaning.py

│   │   ├── embedder.py

│   └── api/

│       ├── yelp_api.py

│       └── inference.py

├── tests/

│   ├── test_features.py

│   └── test_models.py

├── data/

│   └── sample_inputs/

├── requirements.txt

└── README.md  ← you are here

## How the System Works
1) User Query → NLP Encoder
    - Extracts key phrases
    - Produces semantic embedding

2) Location → GraphSAGE
    - Filters restaurants to a 5-mile radius
    - Provides inductive generalization

3) Local Subgraph to GCN
    - Builds per-query graph
    - Smooths features

4) Credibility Adjustment → GAT
    - Weighs users by historical review frequency
    - Refines accessibility and experience scores

5) Late Fusion Ranking
    - Combines NLP + GCN signals
    - Produces final sorted restaurant list

## Running Tests (Still refining it)
pytest -q

For now, people can run it locally
1) Install dependencies
pip install -r requirements.txt

2) Try it out: python demo.py

## System Architecture Diagram
User Query → Transformer ┐
                         ├── Late Fusion → Ranking → Output
GraphSAGE → GCN → GAT ───┘

- GraphSAGE = large-scale graph backbone (daily/weekly updates)
- GCN = per-query local context
- GAT = credibility-based weighting

## Project Status
X - NLP embedding pipeline
X - Basic GNN models (GraphSAGE / GCN / GAT)
X - Subgraph selection for real-time queries
  - Full backend integration
  - Production performance tuning
  - UI/mobile interface