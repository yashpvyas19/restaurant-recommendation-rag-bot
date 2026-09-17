# 🎯 Discussion Session & Interview Preparation Guide

This guide is designed to help you confidently explain your approach, architecture, design trade-offs, and key learnings during your presentation and discussion session.

---

## 1. ⚡ 60-Second Elevator Pitch

> *"For this project, I built **ConciergeAI**, a domain-specific restaurant recommendation chatbot powered by Retrieval-Augmented Generation (RAG). Instead of relying on a raw LLM that might hallucinate non-existent restaurants or outdated menus, ConciergeAI grounds its answers in a verified, structured knowledge base.
> 
> The system uses local dense vector embeddings (`all-MiniLM-L6-v2`) and **ChromaDB** to perform semantic search across cuisines, dietary restrictions, price brackets, and ambiance. The retrieved context is augmented into a strict grounding prompt evaluated by **Google Gemini Flash**. I also engineered resilience mechanisms—including multi-model failover across quotas and deterministic fallback—so the system guarantees reliable performance even under strict cloud rate limits."*

---

## 2. 🏛️ Architectural Walkthrough

Be prepared to draw or explain the 5 stages of the pipeline:

```
[Curated Catalog (JSON/CSV)] 
          │
          ▼
 [Document Ingestion & Metadata Tagging]
          │
          ▼
 [Dense Vector Embeddings (all-MiniLM-L6-v2)] ──► [ChromaDB Vector Store]
                                                              │
   [User Query: "Vegan + Gluten-free under $30"]              │ Semantic
          │                                                   │ Search
          ▼                                                   ▼
 [Query Embedding] ──────────────────────────────► [Top-K Retrieved Context]
                                                              │
                                                              ▼
                                              [Augmented Grounding Prompt]
                                                              │
                                                              ▼
                                              [Google Gemini LLM Generation]
                                                              │
                                                              ▼
                                              [Transparent Verified Output]
```

### Key Technical Decisions:
1. **Local Embeddings vs Cloud Embeddings**:
   * *Decision*: Used HuggingFace `sentence-transformers/all-MiniLM-L6-v2` locally instead of remote cloud embedding APIs.
   * *Rationale*: Runs in-memory, eliminates external API latency, avoids cloud rate limits, and protects the system from endpoint deprecation (e.g. 404s on retired embedding versions).
2. **Vector Store**:
   * *Decision*: ChromaDB.
   * *Rationale*: Lightweight, zero-setup in-memory vector indexing with cosine similarity, perfectly suited for rapid prototyping and low-latency retrieval.
3. **Generative LLM**:
   * *Decision*: Google Gemini Flash (`gemini-3.8-flash` / `gemini-2.5-flash` / `gemini-2.0-flash`).
   * *Rationale*: Low latency, high reasoning fidelity, structured output following, and free-tier accessibility for Colab demonstrations.

---

## 3. ❓ Anticipated Discussion Questions & Model Answers

### Q1: *"Why did you choose RAG over fine-tuning a model on restaurant data?"*
* **Answer**:
  1. **Zero Hallucination & Traceability**: RAG provides citations. Users and business stakeholders can verify *why* a restaurant was recommended against real catalog entries.
  2. **Data Dynamism**: Menus, hours, and prices change weekly. Updating a RAG vector store takes seconds (`vector_db.add_documents`), whereas fine-tuning is expensive, slow, and risks catastrophic forgetting.
  3. **Cost Efficiency**: RAG utilizes generalist foundation models with smaller prompt windows rather than retraining expensive model weights.

### Q2: *"How do you handle conflicting constraints (e.g., cheap price vs. upscale ambiance)?"*
* **Answer**:
  * Semantic vector search creates a high-dimensional representation that naturally balances conceptual similarity.
  * In the prompt stage, we instruct the LLM to prioritize hard constraints (e.g., dietary allergies and price caps) while acknowledging when an ambiance trade-off is made.

### Q3: *"What happened when you encountered API rate limits (429) or high demand (503), and how did you solve it?"*
* **Answer**:
  * *"During testing, free-tier cloud APIs often hit strict requests-per-minute (RPM) or daily caps (like 20 requests/day on preview models). I implemented two layers of resilience:*
    1. *`Model Failover Pool`: A fallback queue rotating between `gemini-3.8-flash`, `gemini-2.5-flash`, and `gemini-2.0-flash`. Because quotas are tracked per model, rotation provides immediate recovery.*
    2. *`Deterministic Grounded Fallback`: If all cloud API quotas are exhausted, the engine formats the retrieved ChromaDB chunks into a clean, structured summary directly from the vector store so the user experience never crashes."*

### Q4: *"How would you scale this prototype to a production system with 1,000,000 restaurants?"*
* **Answer**:
  1. **Vector Database**: Migrate from in-memory ChromaDB to a managed distributed vector database like **Qdrant**, **Pinecone**, or **pgvector** with HNSW indexing.
  2. **Hybrid Search**: Combine dense semantic embeddings with sparse lexical search (**BM25**) using Reciprocal Rank Fusion (RRF) to ensure exact keyword matching on restaurant names or dish codes.
  3. **Two-Stage Re-ranking**: Use a fast bi-encoder for initial retrieval (top 50) and a cross-encoder (**Cohere Rerank** or `bge-reranker`) to re-score the top 3.
  4. **Evaluation Framework**: Implement **RAGAS** or **TruLens** to continuously monitor *Faithfulness*, *Answer Relevance*, and *Context Recall*.

---

## 4. 💡 Key Learnings

1. **Prompt Grounding is as Crucial as Retrieval**: Without negative constraints ("If no match is found, do not fabricate"), LLMs tend to invent plausible-sounding restaurants. Strict negative guardrails eliminated hallucinations.
2. **Metadata Filtering Powers Hybrid Accuracy**: Storing categorical fields (price, cuisine, neighborhood) as searchable vector metadata enables pre-filtering or post-filtering for guaranteed constraint satisfaction.
3. **Graceful Degradation Matters**: Production AI apps must fail safely. Having fallback mechanisms between models ensures business continuity.
