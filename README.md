# MedReg Intelligence

**AI-powered Regulatory Intelligence for Medical Devices**

A production-grade RAG system that answers complex regulatory questions across multiple jurisdictions — with exact legal citations. Built for Regulatory Affairs professionals, Quality Managers, and MedTech consultants.

**[Live Demo](https://medreg-intelligence.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## The Problem

Regulatory professionals in MedTech spend **up to 30% of their time** searching through legislation. With regulations spanning the EU, UK, Switzerland, Canada, and beyond, cross-jurisdictional comparison is manual, slow, and error-prone.

## The Solution

MedReg Intelligence provides instant, source-grounded answers across multiple regulatory frameworks. Every response includes exact article references — no hallucinations, no guesswork.

**Example:**
> *"Compare the classification rules for medical device software (SaMD) under EU MDR and UK MDR 2002."*

The system retrieves relevant passages from both regulations, reranks them for relevance, and generates a structured comparison table with precise legal citations.

---

## Key Features

### Multi-Jurisdiction RAG Pipeline
- **7+ regulatory documents** pre-indexed and ready to query
- Covers **EU MDR**, **MPDG** (Germany), **MepV** (Switzerland), **UK MDR 2002**, **CMDR** (Canada), and more
- Users can upload additional PDFs to expand the knowledge base on the fly

### Advanced Retrieval Architecture
- **Structure-aware chunking** that respects article boundaries, paragraph numbering, and annex structure
- **Semantic search** via Azure OpenAI Embeddings (text-embedding-3-small)
- **Cohere Rerank v3.5** for precision — retrieves 20 candidates, reranks to top 10
- **Azure GPT-4.1** for response generation with streaming output

### Professional Export
- **Word export (.docx)** — formatted analysis documents with question, answer, and branding
- **Audit Trail (HTML)** — full conversation history for compliance documentation
- **Copy-to-clipboard** for quick sharing

### User Experience
- Instant access — pre-built vector database loads automatically, no setup required
- Clickable example questions for quick onboarding
- Additive document upload with automatic processing and real-time progress feedback
- Clean, professional interface built with Streamlit

---

## Architecture

```
User Query
    |
    v
[ChromaDB Semantic Search] --> 20 candidates
    |
    v
[Cohere Rerank v3.5] -------> Top 10 relevant passages
    |
    v
[Azure GPT-4.1] ------------> Structured answer with citations
    |
    v
[Streamlit UI] -------------> Formatted response + export options
```

### RAG Pipeline Detail

| Stage | Technology | Purpose |
|-------|-----------|---------|
| Chunking | Custom (chunking.py) | Structure-aware splitting respecting legal article boundaries |
| Embedding | Azure text-embedding-3-small | Semantic vector representation |
| Vector Store | ChromaDB (persistent) | Fast similarity search |
| Reranking | Cohere Rerank v3.5 | Precision filtering: 20 candidates to top 10 |
| Generation | Azure GPT-4.1 | Streaming response with citation grounding |
| Orchestration | LangChain | Prompt templates, chat history, chain composition |

---

## Pre-indexed Regulations

| Document | Jurisdiction | Language |
|----------|-------------|----------|
| EU MDR (Regulation 2017/745) | EU | DE |
| MPDG (Medizinprodukterecht-Durchfuhrungsgesetz) | Germany | DE |
| MepV (Medizinprodukteverordnung) | Switzerland | DE |
| UK MDR 2002 (Medical Devices Regulations) | UK | EN |
| UK MDR 2002 - Conformity Assessment | UK | EN |
| MDCG 2021-24 Classification Guidance | EU | EN |
| UK IVD Guidance (MHRA) | UK | EN |

Users can upload additional regulatory PDFs (e.g., FDA 21 CFR 820, IVDR, CMDR) directly through the interface.

---

## Quick Start

### Live Demo
Visit **[medreg-intelligence.streamlit.app](https://medreg-intelligence.streamlit.app)** — no installation required.

### Local Development

```bash
git clone https://github.com/leelesemann-sys/medreg-intelligence.git
cd medreg-intelligence
pip install -r requirements.txt
```

Create a `.env` file:
```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
COHERE_API_KEY=your_key
```

Run:
```bash
streamlit run app.py
```

The pre-built vector database (14 MB) downloads automatically on first start.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Azure OpenAI GPT-4.1 |
| **Embeddings** | Azure text-embedding-3-small |
| **Reranking** | Cohere Rerank v3.5 |
| **Vector Database** | ChromaDB (persistent) |
| **Orchestration** | LangChain |
| **Frontend** | Streamlit |
| **Export** | python-docx, HTML |
| **Hosting** | Streamlit Cloud |
| **DB Hosting** | GitHub Releases |

---

## Project Structure

```
medreg-intelligence/
  app.py              # Main application (~400 lines)
  chunking.py         # Structure-aware document chunking
  requirements.txt    # Dependencies
  .gitignore
  docs/               # Documentation & status reports
```

---

## Why This Matters

Traditional keyword search fails for regulatory questions because:
- Legal language is dense and cross-referential
- The same concept uses different terminology across jurisdictions
- Answers often require synthesizing information from multiple articles

MedReg Intelligence solves this through **semantic understanding** — it finds relevant passages based on meaning, not just keywords, and synthesizes cross-jurisdictional comparisons with exact legal citations.

---

## Roadmap

- [ ] Source transparency: distinguish between database-grounded and general knowledge answers
- [ ] Confidence indicators for response quality
- [ ] Additional jurisdictions (FDA, PMDA Japan, TGA Australia)
- [ ] Persistent chat history across sessions

---

*Built by [Lesemann AI Solutions & Consulting](https://github.com/leelesemann-sys)*
