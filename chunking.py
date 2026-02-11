"""
chunking.py - Strukturbewusstes Chunking für regulatorische Dokumente

Erkennt den Dokumenttyp automatisch und splittet an Artikelgrenzen
statt blind alle N Zeichen. Erzeugt Chunks mit reichhaltigen Metadaten
(Jurisdiktion, Artikel-ID, Kapitel, Sprache etc.).

Unterstützte Dokumenttypen:
- EU MDR 2017/745 (DE) - mit OCR-Artefakt-Bereinigung
- MPDG (DE)
- MepV Schweiz (DE)
- UK Medical Devices Regulations 2002 (EN)
- Guidance-Dokumente (Fallback)
"""

import re
import tempfile
import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
MAX_CHUNK_SIZE = 2000   # Artikel über 2000 Zeichen werden an Absätzen gesplittet
MIN_CHUNK_SIZE = 200    # Sehr kurze Artikel werden zusammengefasst
FALLBACK_CHUNK_SIZE = 800
FALLBACK_CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# OCR-Bereinigung (speziell für EU MDR PDF)
# ---------------------------------------------------------------------------
# Das EU-MDR PDF hat systematische OCR-Artefakte: Großbuchstabe + Leerzeichen + Rest
# z.B. "V erordnung", "K ommission", "Ar tikel", "K onf or mit ät"

def clean_ocr_artifacts(text: str) -> str:
    """Repariert OCR-Artefakte im EU MDR PDF."""

    # Pass 1: Häufige bekannte Fehler direkt korrigieren
    known_fixes = {
        "V erordnung": "Verordnung",
        "K ommission": "Kommission",
        "P arlament": "Parlament",
        "P arlaments": "Parlaments",
        "P atienten": "Patienten",
        "P atient": "Patient",
        "K onf or mit ät": "Konformität",
        "K onf or mit äts": "Konformitäts",
        "K onf or mit": "Konformit",
        "T ransparenz": "Transparenz",
        "A ufbereitung": "Aufbereitung",
        "A ufbereiter": "Aufbereiter",
        "Ü bereinstimmung": "Übereinstimmung",
        "Ü ber wachung": "Überwachung",
        "Ü ber": "Über",
        "R ates": "Rates",
        "R ahmen": "Rahmen",
        "U nion": "Union",
        "U nionsebene": "Unionsebene",
        "K onsultationen": "Konsultationen",
        "K oordinier": "Koordinier",
        "A ußerdem": "Außerdem",
        "W irtschaftsakteur": "Wirtschaftsakteur",
    }
    for wrong, right in known_fixes.items():
        text = text.replace(wrong, right)

    # Pass 2: Generisches Pattern - Einzelner Großbuchstabe + Leerzeichen + Kleinbuchstaben
    # z.B. "V erfahren" -> "Verfahren", "K ör per" -> "Körper"
    text = re.sub(r'\b([A-ZÄÖÜ])\s([a-zäöüß])', r'\1\2', text)

    # Pass 3: Mittlere Fragmente - Kleinbuchstaben + Leerzeichen + 1-3 Kleinbuchstaben + Leerzeichen
    # z.B. "Konf or mit" -> "Konformit", "sch wer wiegend" -> "schwerwiegend"
    # Vorsichtig: nur wenn Fragment < 4 Zeichen (vermeidet echte Wörter)
    for _ in range(3):  # Mehrfach durchlaufen für verschachtelte Artefakte
        text = re.sub(r'([a-zäöüß])\s([a-zäöüß]{1,3})\s([a-zäöüß])', r'\1\2\3', text)

    # Pass 4: Mehrfach-Leerzeichen zu einem zusammenfassen
    text = re.sub(r'  +', ' ', text)

    return text


# ---------------------------------------------------------------------------
# Header/Footer-Bereinigung
# ---------------------------------------------------------------------------

def remove_mpdg_headers(text: str) -> str:
    """Entfernt wiederkehrende Header/Footer aus MPDG-Seiten."""
    text = re.sub(
        r'Ein Service des Bundesministerium der Justiz.*?www\.gesetze-im-internet\.de\s*',
        '', text
    )
    text = re.sub(r'- Seite \d+ von \d+ -', '', text)
    return text


def remove_mepv_headers(text: str) -> str:
    """Entfernt Header/Footer aus MepV-Seiten."""
    # Nur alleinstehende Header-Zeilen entfernen (nicht wenn Teil eines Satzes)
    text = re.sub(r'^Medizinprodukteverordnung\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Heilmittel\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\s*/\s*64\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^812\.213\s*$', '', text, flags=re.MULTILINE)
    return text


def remove_uk_headers(text: str) -> str:
    """Entfernt Header/Footer aus UK MDR-Seiten."""
    text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
    return text


# ---------------------------------------------------------------------------
# Text-Extraktion aus PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_or_path) -> tuple[str, int]:
    """Extrahiert den gesamten Text und Seitenzahl aus einem PDF.

    Args:
        file_or_path: Dateipfad (str) oder Streamlit UploadedFile-Objekt

    Returns:
        (full_text, total_pages)
    """
    if isinstance(file_or_path, str):
        reader = PdfReader(file_or_path)
    else:
        # Streamlit UploadedFile -> temporäre Datei
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_or_path.getvalue())
            tmp_path = tmp.name
        reader = PdfReader(tmp_path)
        os.remove(tmp_path)

    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    return "\n\n".join(pages_text), len(reader.pages)


def extract_text_with_pages(file_or_path) -> list[tuple[str, int]]:
    """Extrahiert Text seitenweise mit Seitennummer.

    Returns:
        Liste von (text, page_number) Tupeln (1-indexed)
    """
    if isinstance(file_or_path, str):
        reader = PdfReader(file_or_path)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_or_path.getvalue())
            tmp_path = tmp.name
        reader = PdfReader(tmp_path)
        os.remove(tmp_path)

    result = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        result.append((text, i + 1))

    return result


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def estimate_page(article_start: int, full_text: str, pages: list[tuple[str, int]]) -> int:
    """Schätzt die Seitennummer basierend auf der Textposition."""
    char_count = 0
    for text, page_num in pages:
        char_count += len(text) + 2  # +2 für \n\n Separator
        if char_count >= article_start:
            return page_num
    return pages[-1][1] if pages else 1


def split_large_article(text: str, article_header: str, metadata: dict) -> list[Document]:
    """Splittet einen zu großen Artikel an Absatzgrenzen.

    Versucht verschiedene Split-Strategien in absteigender Priorität:
    1. Nummerierte Absätze: (1), (2) oder 1, 2 (am Zeilenanfang)
    2. Buchstaben-Items: a), b) oder a., b.
    3. Doppelte Zeilenumbrüche (Absatzgrenzen)
    4. RecursiveCharacterTextSplitter als Fallback

    Jeder Sub-Chunk bekommt den Artikel-Header vorangestellt.
    """
    paragraphs = None

    # Strategie 1: (1), (2), ... am Zeilenanfang
    parts = re.split(r'(?=(?:^|\n)\(\d+\)\s)', text)
    if len(parts) > 1:
        paragraphs = parts

    # Strategie 2: Nummern am Zeilenanfang ohne Klammern (Schweizer Stil: "1 Text")
    if not paragraphs or len(paragraphs) <= 1:
        parts = re.split(r'(?=\n\d+\s+[A-ZÄÖÜ])', text)
        if len(parts) > 1:
            paragraphs = parts

    # Strategie 3: Buchstaben-Items
    if not paragraphs or len(paragraphs) <= 1:
        parts = re.split(r'(?=\n[a-z][\.\)]\s)', text)
        if len(parts) > 1:
            paragraphs = parts

    # Strategie 4: Doppelte Zeilenumbrüche
    if not paragraphs or len(paragraphs) <= 1:
        parts = text.split('\n\n')
        if len(parts) > 1:
            paragraphs = parts

    # Strategie 5: Fallback RecursiveCharacterTextSplitter
    if not paragraphs or len(paragraphs) <= 1:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=200
        )
        sub_docs = splitter.create_documents(
            [text],
            metadatas=[metadata]
        )
        for doc in sub_docs:
            doc.page_content = f"{article_header}\n{doc.page_content}"
        return sub_docs

    # Absätze zu Chunks zusammenfassen (max MAX_CHUNK_SIZE)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > MAX_CHUNK_SIZE and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += para
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Falls ein einzelner Chunk immer noch zu groß ist -> nochmal mit Splitter
    final_docs = []
    for i, chunk in enumerate(chunks):
        meta = metadata.copy()
        meta["chunk_part"] = i + 1
        meta["chunk_total"] = len(chunks)
        content = f"{article_header}\n{chunk}" if i > 0 else chunk

        if len(content) > MAX_CHUNK_SIZE * 1.5:
            # Immer noch zu groß -> Splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_CHUNK_SIZE,
                chunk_overlap=200
            )
            sub_docs = splitter.create_documents([content], metadatas=[meta])
            final_docs.extend(sub_docs)
        else:
            final_docs.append(Document(page_content=content, metadata=meta))

    return final_docs


# ---------------------------------------------------------------------------
# Dokument-Typ-Erkennung
# ---------------------------------------------------------------------------

def detect_document_type(text: str, filename: str) -> dict:
    """Erkennt den Dokumenttyp anhand des Texts und Dateinamens.

    WICHTIG: Spezifischere Checks (MPDG, MepV) kommen VOR dem EU-Check,
    weil diese Dokumente die EU-Verordnung 2017/745 referenzieren.

    Returns:
        Dict mit document_type, jurisdiction, language, document_title, parser
    """
    text_lower = text[:5000].lower()
    filename_lower = filename.lower()

    # 1. MPDG (DE) - vor EU, weil MPDG die EU-Verordnung referenziert
    if ("medizinprodukterecht-durchführungsgesetz" in text_lower
            or "mpdg" in text_lower
            or "durchführungsgesetz" in text_lower
            or "bundesministerium der justiz" in text_lower):
        return {
            "document_type": "law",
            "jurisdiction": "DE",
            "language": "de",
            "document_title": "MPDG (Medizinprodukterecht-Durchführungsgesetz)",
            "parser": "de_mpdg"
        }

    # 2. MepV (CH) - vor EU, weil MepV die EU-MDR referenziert
    if ("812.213" in text
            or "schweizerische bundesrat" in text_lower
            or ("medizinprodukteverordnung" in text_lower and "mepv" in text_lower)
            or "fedlex" in filename_lower):
        return {
            "document_type": "regulation",
            "jurisdiction": "CH",
            "language": "de",
            "document_title": "MepV (Medizinprodukteverordnung Schweiz)",
            "parser": "ch_mepv"
        }

    # 3. UK MDR
    if "medical devices regulations 2002" in text_lower:
        return {
            "document_type": "regulation",
            "jurisdiction": "UK",
            "language": "en",
            "document_title": "UK Medical Devices Regulations 2002",
            "parser": "uk_mdr"
        }

    # 4. EU MDR - zuletzt, da am generischsten
    if "2017/745" in text and ("verordnung" in text_lower or "ver ordnung" in text_lower):
        return {
            "document_type": "regulation",
            "jurisdiction": "EU",
            "language": "de",
            "document_title": "EU MDR 2017/745",
            "parser": "eu_mdr"
        }

    # Guidance-Dokumente erkennen
    if "guidance" in text_lower or "mdcg" in text_lower or "conformity" in text_lower:
        lang = "en" if re.search(r'\b(the|and|of|for|with)\b', text[:1000]) else "de"
        if "united kingdom" in text_lower or " gb " in text_lower or "uk mdr" in text_lower:
            jurisdiction = "UK"
        else:
            jurisdiction = "EU"
        return {
            "document_type": "guidance",
            "jurisdiction": jurisdiction,
            "language": lang,
            "document_title": filename.replace(".pdf", ""),
            "parser": "guidance"
        }

    # Unbekannt -> Fallback
    lang = "en" if re.search(r'\b(the|and|of|for|with)\b', text[:1000]) else "de"
    return {
        "document_type": "other",
        "jurisdiction": "unknown",
        "language": lang,
        "document_title": filename.replace(".pdf", ""),
        "parser": "guidance"
    }


# ---------------------------------------------------------------------------
# Parser: EU MDR
# ---------------------------------------------------------------------------

def parse_eu_mdr(file_or_path, filename: str, doc_info: dict) -> list[Document]:
    """Parser für EU MDR 2017/745 (Deutsch, mit OCR-Artefakt-Bereinigung)."""
    full_text, total_pages = extract_text_from_pdf(file_or_path)
    pages = extract_text_with_pages(file_or_path)

    # OCR-Artefakte bereinigen
    full_text = clean_ocr_artifacts(full_text)

    documents = []
    base_meta = {
        "source_document": filename,
        "document_title": doc_info["document_title"],
        "document_type": doc_info["document_type"],
        "jurisdiction": doc_info["jurisdiction"],
        "language": doc_info["language"],
    }

    # --- Erwägungsgründe (Recitals) ---
    recitals_match = re.search(
        r'in\s+Erwägung\s+nachstehender\s+Gründe:(.+?)(?=Artikel\s+1\b|KAPITEL\s+I\b)',
        full_text, re.DOTALL | re.IGNORECASE
    )
    if recitals_match:
        recitals_text = recitals_match.group(1)
        recital_parts = re.split(r'(?=\(\d+\)\s)', recitals_text)
        for part in recital_parts:
            part = part.strip()
            if len(part) < 50:
                continue
            num_match = re.match(r'\((\d+)\)', part)
            num = num_match.group(1) if num_match else "?"
            meta = base_meta.copy()
            meta.update({
                "article_id": f"Erwägungsgrund ({num})",
                "article_title": "",
                "chapter": "Erwägungsgründe",
                "section": "",
                "page": estimate_page(recitals_match.start(), full_text, pages),
                "chunk_type": "recital",
            })
            if len(part) > MAX_CHUNK_SIZE:
                documents.extend(split_large_article(part, f"Erwägungsgrund ({num})", meta))
            else:
                documents.append(Document(page_content=part, metadata=meta))

    # --- Artikel ---
    current_chapter = ""
    current_chapter_title = ""

    # Finde alle Kapitel-Überschriften
    chapters = {}
    for m in re.finditer(r'KAPITEL\s+([IVXLC]+)\s*\n(.+?)(?=\n)', full_text):
        chapters[m.start()] = {
            "chapter": f"KAPITEL {m.group(1)}",
            "chapter_title": m.group(2).strip()
        }

    # Finde alle Artikel-Überschriften (am Zeilenanfang oder nach Leerzeile)
    # Muss "Artikel X\n" sein (eigene Zeile), nicht "gemäß Artikel 52 Absatz 3"
    article_pattern = re.compile(
        r'(?:^|\n)Artikel\s+(\d+)\s*\n(.+?)(?=\nArtikel\s+\d+\s*\n|ANHANG|$)',
        re.DOTALL
    )

    for match in article_pattern.finditer(full_text):
        article_num = match.group(1)
        article_body = match.group(2).strip()

        # Kapitel-Kontext bestimmen
        for pos in sorted(chapters.keys()):
            if pos < match.start():
                current_chapter = chapters[pos]["chapter"]
                current_chapter_title = chapters[pos]["chapter_title"]

        # Titel = erste Zeile des Artikelkörpers
        lines = article_body.split('\n')
        article_title = lines[0].strip() if lines else ""
        # Titel oft nur eine Zeile, danach kommt (1) oder Fließtext
        if article_title and not article_title.startswith('('):
            article_body_text = '\n'.join(lines[1:]).strip()
        else:
            article_title = ""
            article_body_text = article_body

        full_article = f"Artikel {article_num} – {article_title}\n{article_body_text}" if article_title else f"Artikel {article_num}\n{article_body_text}"

        meta = base_meta.copy()
        meta.update({
            "article_id": f"Artikel {article_num}",
            "article_title": article_title,
            "chapter": current_chapter,
            "section": current_chapter_title,
            "page": estimate_page(match.start(), full_text, pages),
            "chunk_type": "article",
        })

        if len(full_article) > MAX_CHUNK_SIZE:
            documents.extend(split_large_article(full_article, f"Artikel {article_num} – {article_title}", meta))
        else:
            documents.append(Document(page_content=full_article, metadata=meta))

    return documents


# ---------------------------------------------------------------------------
# Parser: DE MPDG
# ---------------------------------------------------------------------------

def parse_de_mpdg(file_or_path, filename: str, doc_info: dict) -> list[Document]:
    """Parser für deutsches MPDG."""
    full_text, _ = extract_text_from_pdf(file_or_path)
    pages = extract_text_with_pages(file_or_path)

    # Header entfernen
    full_text = remove_mpdg_headers(full_text)

    documents = []
    base_meta = {
        "source_document": filename,
        "document_title": doc_info["document_title"],
        "document_type": doc_info["document_type"],
        "jurisdiction": doc_info["jurisdiction"],
        "language": doc_info["language"],
    }

    # Kapitel-Tracking
    current_chapter = ""
    current_chapter_title = ""
    chapters = {}
    for m in re.finditer(r'Kapitel\s+(\d+)\s*\n(.+?)(?=\n)', full_text):
        chapters[m.start()] = {
            "chapter": f"Kapitel {m.group(1)}",
            "chapter_title": m.group(2).strip()
        }

    # Paragraphen finden - "§ X" am Zeilenanfang (nicht mitten in Verweisen)
    para_pattern = re.compile(
        r'(?:^|\n)§\s*(\d+\w?)\s+(.+?)(?=\n§\s*\d|$)',
        re.DOTALL
    )

    for match in para_pattern.finditer(full_text):
        para_num = match.group(1)
        para_body = match.group(2).strip()

        # Kapitel-Kontext
        for pos in sorted(chapters.keys()):
            if pos < match.start():
                current_chapter = chapters[pos]["chapter"]
                current_chapter_title = chapters[pos]["chapter_title"]

        # Titel = erste Zeile
        lines = para_body.split('\n')
        para_title = lines[0].strip() if lines else ""
        if para_title and not re.match(r'^[\(\d]', para_title):
            body_text = '\n'.join(lines[1:]).strip()
        else:
            para_title = ""
            body_text = para_body

        full_para = f"§ {para_num} {para_title}\n{body_text}" if para_title else f"§ {para_num}\n{body_text}"

        meta = base_meta.copy()
        meta.update({
            "article_id": f"§ {para_num}",
            "article_title": para_title,
            "chapter": current_chapter,
            "section": current_chapter_title,
            "page": estimate_page(match.start(), full_text, pages),
            "chunk_type": "article",
        })

        if len(full_para) > MAX_CHUNK_SIZE:
            documents.extend(split_large_article(full_para, f"§ {para_num} {para_title}", meta))
        else:
            documents.append(Document(page_content=full_para, metadata=meta))

    return documents


# ---------------------------------------------------------------------------
# Parser: CH MepV
# ---------------------------------------------------------------------------

def parse_ch_mepv(file_or_path, filename: str, doc_info: dict) -> list[Document]:
    """Parser für Schweizer MepV."""
    full_text, _ = extract_text_from_pdf(file_or_path)
    pages = extract_text_with_pages(file_or_path)

    # Header entfernen
    full_text = remove_mepv_headers(full_text)

    documents = []
    base_meta = {
        "source_document": filename,
        "document_title": doc_info["document_title"],
        "document_type": doc_info["document_type"],
        "jurisdiction": doc_info["jurisdiction"],
        "language": doc_info["language"],
    }

    # Kapitel-Tracking
    current_chapter = ""
    current_chapter_title = ""
    chapters = {}
    for m in re.finditer(r'(\d+)\.\s*Kapitel[:\s]+(.+?)(?=\n)', full_text):
        chapters[m.start()] = {
            "chapter": f"Kapitel {m.group(1)}",
            "chapter_title": m.group(2).strip()
        }

    # Artikel finden - "Art." gefolgt von Nummer, dann Titel
    art_pattern = re.compile(
        r'Art\.\s*(\d+\w?)\s+(.+?)(?=\nArt\.\s*\d|$)',
        re.DOTALL
    )

    for match in art_pattern.finditer(full_text):
        art_num = match.group(1)
        art_body = match.group(2).strip()

        # Kapitel-Kontext
        for pos in sorted(chapters.keys()):
            if pos < match.start():
                current_chapter = chapters[pos]["chapter"]
                current_chapter_title = chapters[pos]["chapter_title"]

        # Titel = erste Zeile
        lines = art_body.split('\n')
        art_title = lines[0].strip() if lines else ""
        if art_title and not re.match(r'^[\d]', art_title):
            body_text = '\n'.join(lines[1:]).strip()
        else:
            art_title = ""
            body_text = art_body

        full_art = f"Art. {art_num} {art_title}\n{body_text}" if art_title else f"Art. {art_num}\n{body_text}"

        meta = base_meta.copy()
        meta.update({
            "article_id": f"Art. {art_num}",
            "article_title": art_title,
            "chapter": current_chapter,
            "section": current_chapter_title,
            "page": estimate_page(match.start(), full_text, pages),
            "chunk_type": "article",
        })

        if len(full_art) > MAX_CHUNK_SIZE:
            documents.extend(split_large_article(full_art, f"Art. {art_num} {art_title}", meta))
        else:
            documents.append(Document(page_content=full_art, metadata=meta))

    return documents


# ---------------------------------------------------------------------------
# Parser: UK MDR
# ---------------------------------------------------------------------------

def parse_uk_mdr(file_or_path, filename: str, doc_info: dict) -> list[Document]:
    """Parser für UK Medical Devices Regulations 2002."""
    full_text, _ = extract_text_from_pdf(file_or_path)
    pages = extract_text_with_pages(file_or_path)

    full_text = remove_uk_headers(full_text)

    documents = []
    base_meta = {
        "source_document": filename,
        "document_title": doc_info["document_title"],
        "document_type": doc_info["document_type"],
        "jurisdiction": doc_info["jurisdiction"],
        "language": doc_info["language"],
    }

    # Part-Tracking
    current_part = ""
    current_part_title = ""
    parts = {}
    for m in re.finditer(r'PART\s+([IVXLC]+)\s*\n(.+?)(?=\n)', full_text):
        parts[m.start()] = {
            "part": f"Part {m.group(1)}",
            "part_title": m.group(2).strip()
        }

    # Regulations finden (nummeriert: "1.", "2.", etc. am Zeilenanfang)
    reg_pattern = re.compile(
        r'(?:^|\n)(\d+)\.\s+([A-Z].+?)(?=\n\d+\.\s+[A-Z]|SCHEDULE|$)',
        re.DOTALL
    )

    for match in reg_pattern.finditer(full_text):
        reg_num = match.group(1)
        reg_body = match.group(2).strip()

        # Part-Kontext
        for pos in sorted(parts.keys()):
            if pos < match.start():
                current_part = parts[pos]["part"]
                current_part_title = parts[pos]["part_title"]

        # Titel = erste Zeile
        lines = reg_body.split('\n')
        reg_title = lines[0].strip() if lines else ""
        if reg_title and not reg_title.startswith('('):
            body_text = '\n'.join(lines[1:]).strip()
        else:
            reg_title = ""
            body_text = reg_body

        full_reg = f"Regulation {reg_num}. {reg_title}\n{body_text}" if reg_title else f"Regulation {reg_num}\n{body_text}"

        meta = base_meta.copy()
        meta.update({
            "article_id": f"Regulation {reg_num}",
            "article_title": reg_title,
            "chapter": current_part,
            "section": current_part_title,
            "page": estimate_page(match.start(), full_text, pages),
            "chunk_type": "article",
        })

        if len(full_reg) > MAX_CHUNK_SIZE:
            documents.extend(split_large_article(full_reg, f"Regulation {reg_num}. {reg_title}", meta))
        else:
            documents.append(Document(page_content=full_reg, metadata=meta))

    return documents


# ---------------------------------------------------------------------------
# Parser: Guidance Fallback
# ---------------------------------------------------------------------------

def parse_guidance(file_or_path, filename: str, doc_info: dict) -> list[Document]:
    """Fallback-Parser für Guidance-Dokumente und unbekannte Formate."""
    full_text, _ = extract_text_from_pdf(file_or_path)

    base_meta = {
        "source_document": filename,
        "document_title": doc_info["document_title"],
        "document_type": doc_info["document_type"],
        "jurisdiction": doc_info["jurisdiction"],
        "language": doc_info["language"],
        "article_id": "",
        "article_title": "",
        "chapter": "",
        "section": "",
        "page": 1,
        "chunk_type": "guidance",
    }

    # Versuche Überschriften zu erkennen (nummeriert oder GROSSBUCHSTABEN)
    heading_pattern = re.compile(r'^(\d+\.?\d*\.?\s+[A-Z].{5,80})$', re.MULTILINE)
    headings = list(heading_pattern.finditer(full_text))

    if headings:
        # Nutze erkannte Überschriften als Split-Punkte
        documents = []
        for i, heading_match in enumerate(headings):
            start = heading_match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(full_text)
            section_text = full_text[start:end].strip()
            section_title = heading_match.group(1).strip()

            meta = base_meta.copy()
            meta["section"] = section_title
            meta["article_id"] = section_title.split()[0] if section_title else ""

            if len(section_text) > MAX_CHUNK_SIZE:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=FALLBACK_CHUNK_SIZE,
                    chunk_overlap=FALLBACK_CHUNK_OVERLAP
                )
                sub_docs = splitter.create_documents([section_text], metadatas=[meta])
                documents.extend(sub_docs)
            else:
                documents.append(Document(page_content=section_text, metadata=meta))

        return documents

    # Kein Muster erkannt -> einfacher RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP
    )
    docs = splitter.create_documents([full_text], metadatas=[base_meta])
    return docs


# ---------------------------------------------------------------------------
# Haupt-Funktion
# ---------------------------------------------------------------------------

PARSERS = {
    "eu_mdr": parse_eu_mdr,
    "de_mpdg": parse_de_mpdg,
    "ch_mepv": parse_ch_mepv,
    "uk_mdr": parse_uk_mdr,
    "guidance": parse_guidance,
}


def chunk_document(file_or_path, filename: str) -> list[Document]:
    """Chunked ein einzelnes PDF-Dokument mit automatischer Typ-Erkennung.

    Args:
        file_or_path: Dateipfad (str) oder Streamlit UploadedFile
        filename: Name der Datei (für Metadaten)

    Returns:
        Liste von Document-Objekten mit reichhaltigen Metadaten
    """
    # Text extrahieren für Typ-Erkennung
    full_text, _ = extract_text_from_pdf(file_or_path)
    doc_info = detect_document_type(full_text, filename)

    parser = PARSERS.get(doc_info["parser"], parse_guidance)
    documents = parser(file_or_path, filename, doc_info)

    # Validierung: Falls Parser keine Dokumente gefunden hat -> Fallback
    if not documents:
        print(f"  Warnung: Parser '{doc_info['parser']}' fand keine Chunks für {filename}. Nutze Fallback.")
        documents = parse_guidance(file_or_path, filename, doc_info)

    return documents


def chunk_documents(files, filenames: list[str] = None) -> list[Document]:
    """Chunked mehrere PDF-Dokumente.

    Args:
        files: Liste von Dateipfaden (str) oder Streamlit UploadedFile-Objekten
        filenames: Optionale Dateinamen (falls files Pfade sind)

    Returns:
        Liste aller Document-Objekte
    """
    all_documents = []

    for i, f in enumerate(files):
        if isinstance(f, str):
            name = filenames[i] if filenames and i < len(filenames) else os.path.basename(f)
        else:
            name = f.name

        print(f"Chunking: {name}...")
        docs = chunk_document(f, name)
        print(f"  -> {len(docs)} Chunks erstellt")
        all_documents.extend(docs)

    print(f"\nGesamt: {len(all_documents)} Chunks aus {len(files)} Dokumenten")
    return all_documents


# ---------------------------------------------------------------------------
# Standalone-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data_dir = "./data"

    if not os.path.exists(data_dir):
        print(f"Ordner {data_dir} nicht gefunden.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    print(f"Gefunden: {len(pdf_files)} PDFs\n")

    for pdf_file in sorted(pdf_files):
        path = os.path.join(data_dir, pdf_file)
        docs = chunk_document(path, pdf_file)

        print(f"\n{'='*60}")
        print(f"  {pdf_file}")
        print(f"  Chunks: {len(docs)}")

        if docs:
            # Zeige Metadaten des ersten Chunks
            meta = docs[0].metadata
            print(f"  Typ: {meta.get('document_type')} | Jurisdiktion: {meta.get('jurisdiction')} | Sprache: {meta.get('language')}")
            print(f"  Erster Chunk: {docs[0].page_content[:100]}...")

            # Statistiken
            sizes = [len(d.page_content) for d in docs]
            print(f"  Chunk-Größen: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")

            # Zeige ein paar Artikel-IDs
            article_ids = [d.metadata.get("article_id", "?") for d in docs[:5]]
            print(f"  Erste Artikel: {', '.join(article_ids)}")
