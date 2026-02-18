# MedReg Intelligence

> **Sprache:** [English](README.md) | Deutsch

**KI-gestützte regulatorische Intelligenz für Medizinprodukte**

Ein produktionsreifes RAG-System, das komplexe regulatorische Fragen über mehrere Jurisdiktionen hinweg beantwortet — mit exakten Gesetzeszitaten. Entwickelt für Regulatory Affairs Professionals, Qualitätsmanager und MedTech-Berater.

**[Live-Demo](https://medreg-intelligence.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Das Problem

Regulatory-Affairs-Fachleute in der MedTech-Branche verbringen **bis zu 30% ihrer Arbeitszeit** mit der Suche in Gesetzestexten. Bei Regulierungen, die EU, UK, Schweiz, Kanada und weitere Märkte umfassen, ist der jurisdiktionsübergreifende Vergleich manuell, langsam und fehleranfällig.

## Die Lösung

MedReg Intelligence liefert sofortige, quellenbasierte Antworten über mehrere regulatorische Rahmenwerke hinweg. Jede Antwort enthält exakte Artikelverweise — keine Halluzinationen, kein Raten.

**Beispiel:**
> *"Vergleiche die Klassifizierungsregeln für Medizinprodukte-Software (SaMD) unter EU MDR und UK MDR 2002."*

Das System ruft relevante Passagen aus beiden Verordnungen ab, ordnet sie nach Relevanz und erstellt eine strukturierte Vergleichstabelle mit präzisen Gesetzeszitaten.

---

## Hauptfunktionen

### Multi-Jurisdiktions RAG-Pipeline
- **7+ regulatorische Dokumente** vorindiziert und abfragebereit
- Abdeckung von **EU MDR**, **MPDG** (Deutschland), **MepV** (Schweiz), **UK MDR 2002**, **CMDR** (Kanada) und mehr
- Nutzer können zusätzliche PDFs hochladen, um die Wissensbasis spontan zu erweitern

### Fortschrittliche Retrieval-Architektur
- **Strukturbewusstes Chunking**, das Artikelgrenzen, Absatznummerierung und Annexstruktur respektiert
- **Semantische Suche** via Azure OpenAI Embeddings (text-embedding-3-small)
- **Cohere Rerank v3.5** für Präzision — ruft 20 Kandidaten ab, ordnet auf Top 10 um
- **Azure GPT-4.1** für Antwortgenerierung mit Streaming-Ausgabe

### Professioneller Export
- **Word-Export (.docx)** — formatierte Analysedokumente mit Frage, Antwort und Branding
- **Audit Trail (HTML)** — vollständiger Gesprächsverlauf für Compliance-Dokumentation
- **In-die-Zwischenablage-kopieren** für schnelles Teilen

### Nutzererlebnis
- Sofortiger Zugang — vorgebaute Vektordatenbank lädt automatisch, kein Setup erforderlich
- Klickbare Beispielfragen für schnellen Einstieg
- Additiver Dokument-Upload mit automatischer Verarbeitung und Echtzeit-Fortschritt
- Saubere, professionelle Oberfläche mit Streamlit

---

## Architektur

```
Nutzeranfrage
    |
    v
[ChromaDB Semantische Suche] --> 20 Kandidaten
    |
    v
[Cohere Rerank v3.5] ----------> Top 10 relevante Passagen
    |
    v
[Azure GPT-4.1] ---------------> Strukturierte Antwort mit Zitaten
    |
    v
[Streamlit UI] ----------------> Formatierte Antwort + Exportoptionen
```

### RAG-Pipeline im Detail

| Stufe | Technologie | Zweck |
|-------|-----------|---------|
| Chunking | Custom (chunking.py) | Strukturbewusste Aufteilung unter Berücksichtigung von Artikelgrenzen |
| Embedding | Azure text-embedding-3-small | Semantische Vektordarstellung |
| Vector Store | ChromaDB (persistent) | Schnelle Ähnlichkeitssuche |
| Reranking | Cohere Rerank v3.5 | Präzisionsfilterung: 20 Kandidaten auf Top 10 |
| Generierung | Azure GPT-4.1 | Streaming-Antwort mit Zitatverankerung |
| Orchestrierung | LangChain | Prompt-Templates, Chat-Historie, Chain-Komposition |

---

## Vorindizierte Regulierungen

| Dokument | Jurisdiktion | Sprache |
|----------|-------------|----------|
| EU MDR (Verordnung 2017/745) | EU | DE |
| MPDG (Medizinprodukterecht-Durchführungsgesetz) | Deutschland | DE |
| MepV (Medizinprodukteverordnung) | Schweiz | DE |
| UK MDR 2002 (Medical Devices Regulations) | UK | EN |
| UK MDR 2002 - Conformity Assessment | UK | EN |
| MDCG 2021-24 Classification Guidance | EU | EN |
| UK IVD Guidance (MHRA) | UK | EN |

Nutzer können zusätzliche regulatorische PDFs (z.B. FDA 21 CFR 820, IVDR, CMDR) direkt über die Oberfläche hochladen.

---

## Schnellstart

### Live-Demo
Besuche **[medreg-intelligence.streamlit.app](https://medreg-intelligence.streamlit.app)** — keine Installation erforderlich.

### Lokale Entwicklung

```bash
git clone https://github.com/leelesemann-sys/medreg-intelligence.git
cd medreg-intelligence
pip install -r requirements.txt
```

`.env`-Datei erstellen:
```
AZURE_OPENAI_API_KEY=dein_key
AZURE_OPENAI_ENDPOINT=dein_endpoint
COHERE_API_KEY=dein_key
```

Starten:
```bash
streamlit run app.py
```

Die vorgebaute Vektordatenbank (14 MB) wird beim ersten Start automatisch heruntergeladen.

---

## Tech Stack

| Komponente | Technologie |
|-----------|-----------|
| **LLM** | Azure OpenAI GPT-4.1 |
| **Embeddings** | Azure text-embedding-3-small |
| **Reranking** | Cohere Rerank v3.5 |
| **Vektordatenbank** | ChromaDB (persistent) |
| **Orchestrierung** | LangChain |
| **Frontend** | Streamlit |
| **Export** | python-docx, HTML |
| **Hosting** | Streamlit Cloud |
| **DB-Hosting** | GitHub Releases |

---

## Projektstruktur

```
medreg-intelligence/
  app.py              # Hauptanwendung (~400 Zeilen)
  chunking.py         # Strukturbewusstes Dokument-Chunking
  requirements.txt    # Abhängigkeiten
  .gitignore
  docs/               # Dokumentation & Statusberichte
```

---

## Warum das wichtig ist

Traditionelle Stichwortsuche versagt bei regulatorischen Fragen, weil:
- Rechtssprache dicht und querverweisend ist
- Dasselbe Konzept in verschiedenen Jurisdiktionen unterschiedliche Terminologie verwendet
- Antworten oft die Synthese von Informationen aus mehreren Artikeln erfordern

MedReg Intelligence löst dies durch **semantisches Verständnis** — es findet relevante Passagen basierend auf Bedeutung, nicht nur Stichwörtern, und synthetisiert jurisdiktionsübergreifende Vergleiche mit exakten Gesetzeszitaten.

---

## Roadmap

- [ ] Quelltransparenz: Unterscheidung zwischen datenbankgestützten und allgemeinen Wissensantworten
- [ ] Konfidenzindikatoren für Antwortqualität
- [ ] Zusätzliche Jurisdiktionen (FDA, PMDA Japan, TGA Australien)
- [ ] Persistente Chat-Historie über Sitzungen hinweg

---

*Entwickelt von [Lesemann AI Solutions & Consulting](https://github.com/leelesemann-sys)*
