# 🏥 MDR Navigator - Regulatory AI Assistant

Ein RAG-basiertes (Retrieval Augmented Generation) Assistenzsystem für die Medizintechnik-Branche. 
Es beantwortet komplexe Fragen zur **EU-Verordnung 2017/745 (MDR)** basierend auf offiziellen Dokumenten und liefert präzise Quellenangaben.

![Status](https://img.shields.io/badge/Status-Prototype-blue) ![Stack](https://img.shields.io/badge/Tech-Azure_OpenAI_|_LangChain_|_Streamlit-green)

## 🎯 Business Value
Ingenieure und Quality Manager verbringen bis zu 30% ihrer Zeit mit der Recherche in regulatorischen Dokumenten. 
Der MDR Navigator reduziert diesen Aufwand durch semantische Suche und generative KI, die:
- **Zitationen liefert:** Jede Antwort ist auf den genauen Artikel im Gesetz referenziert (Grounding).
- **Halluzinationen minimiert:** Durch strikte Prompt-Engineering-Vorgaben und Temperature 0.
- **Sicher ist:** Daten verlassen den Azure Tenant nicht (Sovereign AI Approach).

## 🛠 Tech Stack
- **Cloud:** Microsoft Azure (OpenAI Service, AI Search)
- **Framework:** LangChain
- **Frontend:** Streamlit
- **Sprache:** Python 3.10+

## 🚀 Installation & Setup

1. Repository klonen:
   ```bash
   git clone [https://github.com/DEIN-USER/mdr-navigator.git](https://github.com/DEIN-USER/mdr-navigator.git)