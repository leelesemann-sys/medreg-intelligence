import streamlit as st
import os
import shutil
import zipfile
import requests
from datetime import datetime
from dotenv import load_dotenv
import cohere

# --- IMPORTS ---
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from chunking import chunk_documents

# --- KONFIGURATION ---
load_dotenv()
DB_DIR = "./db_storage"
DB_ZIP_URL = "https://github.com/leelesemann-sys/medreg-intelligence/releases/download/v1.0.0/db_storage.zip"
API_VERSION = "2024-08-01-preview"

# Keys: .env (lokal) oder Streamlit Secrets (Cloud)
def get_secret(key):
    """Liest einen Secret-Wert: zuerst aus Streamlit Secrets, dann aus .env/Umgebung."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, "")

AZURE_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
COHERE_API_KEY = get_secret("COHERE_API_KEY")

os.environ["AZURE_OPENAI_API_KEY"] = AZURE_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_ENDPOINT
os.environ["AZURE_OPENAI_API_VERSION"] = API_VERSION

# --- COHERE RERANKER ---
reranker = cohere.ClientV2(api_key=COHERE_API_KEY) if COHERE_API_KEY else None

st.set_page_config(page_title="MedReg Intelligence", layout="wide", page_icon="⚖️")

# --- Minimales CSS: nur File-Uploader Fix ---
st.markdown("""
<style>
    [data-testid="stFileUploaderFileList"] {
        max-height: 400px;
        overflow-y: auto;
    }
    [data-testid="stFileUploaderFileList"] + div[data-testid] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ MedReg Intelligence")

# --- ROBUSTER HTML EXPORT ---
def generate_audit_html(history, files):
    now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; }}
            .header {{ background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 100%); color: white; padding: 30px; border-radius: 5px; margin-bottom: 30px; }}
            .header h1 {{ margin: 0; font-size: 24px; }}
            .header .brand {{ font-size: 0.7em; font-weight: 300; opacity: 0.85; margin-top: 4px; }}
            .meta {{ color: #eee; font-size: 0.9em; margin-top: 10px; }}
            .section-title {{ border-bottom: 2px solid #2980b9; color: #2980b9; padding-bottom: 5px; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .chat-entry {{ margin-bottom: 25px; padding: 15px; border-radius: 5px; }}
            .user {{ background-color: #f8f9fa; border-left: 5px solid #bdc3c7; }}
            .ai {{ background-color: #e3f2fd; border-left: 5px solid #2980b9; }}
            .role {{ font-weight: bold; margin-bottom: 10px; font-size: 0.8em; text-transform: uppercase; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚖️ MedReg Intelligence</h1>
            <div class="brand">Lesemann AI Solutions & Consulting</div>
            <div class="meta">Regulatory Audit Trail | Erstellt: {now}</div>
        </div>

        <h3 class="section-title">Datenbasis (Analysierte Gesetze)</h3>
        <ul>{"".join([f"<li>{f}</li>" for f in files])}</ul>

        <h3 class="section-title">Protokollierter Beratungsverlauf</h3>
    """
    for msg in history:
        role_class = "user" if msg.type == "human" else "ai"
        role_label = "Nutzerfrage" if msg.type == "human" else "KI-Analyse (MedReg Intelligence)"
        content = msg.content.replace("\n", "<br>")
        html += f"""
        <div class="chat-entry {role_class}">
            <div class="role">{role_label}</div>
            <div class="text">{content}</div>
        </div>"""

    html += "</body></html>"
    return html

# --- INITIALISIERUNG ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "indexing_done" not in st.session_state:
    st.session_state.indexing_done = False

# --- UI ---

with st.sidebar:
    st.header("MedReg Intelligence")
    if AZURE_API_KEY and AZURE_ENDPOINT:
        st.success("🔑 Azure API verbunden")
    else:
        st.error("⚠️ API Keys fehlen! Bitte .env oder Streamlit Secrets konfigurieren.")
        st.stop()
    if reranker:
        st.success("🎯 Cohere Reranker aktiv")
    else:
        st.warning("⚠️ Kein Cohere Key – Reranking deaktiviert")

    if st.session_state.chat_history:
        st.divider()
        st.subheader("📄 Export & Audit")
        html_report = generate_audit_html(st.session_state.chat_history, st.session_state.processed_files)
        st.download_button(
            label="Audit Trail herunterladen (HTML)",
            data=html_report,
            file_name=f"Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html"
        )

    if st.button("Speicher komplett leeren"):
        if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
        st.session_state.chat_history = []
        st.session_state.processed_files = []
        st.session_state.indexing_done = False
        st.rerun()

# --- VORBERECHNETE WISSENSBASIS (Auto-Download) ---
def ensure_default_db():
    """Lädt die vorberechnete ChromaDB von GitHub Releases, falls lokal nicht vorhanden."""
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        return True
    try:
        zip_path = "db_storage.zip"
        with st.spinner("📥 Lade vorberechnete Wissensbasis herunter..."):
            resp = requests.get(DB_ZIP_URL, stream=True, timeout=60)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DB_DIR)
            os.remove(zip_path)
        return True
    except Exception as e:
        st.warning(f"⚠️ Auto-Download fehlgeschlagen: {e}")
        return False

# --- VEKTORDATENBANK ---
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    openai_api_version=API_VERSION
)

# 1. Versuche vorberechnete DB zu laden (oder herunterladen)
db_ready = ensure_default_db()

if db_ready:
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    st.session_state.indexing_done = True

    try:
        content_data = vectorstore.get(include=['metadatas'])
        if content_data['metadatas']:
            unique_files = sorted(list(set([m['source_document'] for m in content_data['metadatas'] if 'source_document' in m])))
            st.sidebar.success(f"✅ Wissensbasis geladen ({len(unique_files)} Dokumente)")
            with st.sidebar.expander("Gelesene Dateien anzeigen"):
                for f in unique_files:
                    st.write(f"📄 {f}")
            st.session_state.processed_files = unique_files
    except Exception as e:
        st.sidebar.warning("Wissensbasis aktiv, Inhaltsliste konnte nicht geladen werden.")
else:
    vectorstore = None

# 2. Eigene Dokumente hochladen (zusätzlich oder als Fallback)
with st.sidebar.expander("📂 Eigene Dokumente hochladen"):
    uploaded_files = st.file_uploader("PDFs auswählen:", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Eigene Dokumente indexieren"):
        # Bestehende DB löschen und neu aufbauen
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        with st.spinner("Strukturbewusstes Chunking läuft..."):
            all_splits = chunk_documents(uploaded_files)
            st.info(f"✂️ {len(all_splits)} Chunks aus {len(uploaded_files)} Dokumenten erstellt")
        prog = st.progress(0, text="Indexiere Chunks...")
        batch_size = 50
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size]
            if i == 0:
                vectorstore = Chroma.from_documents(documents=batch, embedding=embeddings, persist_directory=DB_DIR)
            else:
                vectorstore.add_documents(batch)
            prog.progress(min((i + batch_size) / len(all_splits), 1.0))
        st.session_state.processed_files = [f.name for f in uploaded_files]
        st.session_state.indexing_done = True
        st.rerun()

# --- COPY-BUTTON HELPER ---
def copy_button(text, key):
    """Erzeugt einen Copy-to-Clipboard Button via HTML/JS."""
    import html as html_lib
    escaped = html_lib.escape(text).replace("\n", "\\n").replace("'", "\\'")
    st.markdown(f"""
    <button onclick="navigator.clipboard.writeText('{escaped}');this.textContent='✅ Kopiert!';setTimeout(()=>this.textContent='📋 Kopieren',1500)"
        style="background:none;border:1px solid #ddd;border-radius:5px;padding:4px 12px;cursor:pointer;font-size:0.8em;color:#666;margin-top:4px;">
        📋 Kopieren
    </button>
    """, unsafe_allow_html=True)

# --- CHAT MIT RERANKING-LOGIK ---
if vectorstore:
    for message in st.session_state.chat_history:
        with st.chat_message(message.type):
            st.markdown(message.content)
            if message.type == "ai":
                copy_button(message.content, f"copy_{id(message)}")

    if prompt := st.chat_input("Ihre regulatorische Frage..."):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # 1. Retrieval: Breite Suche (20 Kandidaten)
            candidate_count = 20 if reranker else 10
            docs = vectorstore.as_retriever(search_kwargs={"k": candidate_count}).invoke(prompt)

            # 2. Reranking: Cohere wählt die besten 10 aus
            if reranker and docs:
                try:
                    rerank_response = reranker.rerank(
                        model="rerank-v3.5",
                        query=prompt,
                        documents=[d.page_content for d in docs],
                        top_n=10
                    )
                    reranked_indices = [r.index for r in rerank_response.results]
                    docs = [docs[i] for i in reranked_indices]
                except Exception as e:
                    st.warning(f"⚠️ Reranking fehlgeschlagen, nutze Standard-Ranking: {e}")

            context = "\n\n".join([
                f"QUELLE: {d.metadata.get('document_title', d.metadata.get('source_document', '?'))} "
                f"- {d.metadata.get('article_id', '')} "
                f"({d.metadata.get('jurisdiction', '?')}, S.{d.metadata.get('page', '?')})\n"
                f"{d.page_content}"
                for d in docs
            ])

            # 2. Modell & Prompt
            llm = AzureChatOpenAI(azure_deployment="gpt-4.1", api_version=API_VERSION, streaming=True, temperature=0)
            sys_msg = """Du bist ein Regulatory Intelligence Experte für Medizinprodukte.

Regeln:
1. Trenne Jurisdiktionen strikt: EU (MDR), Deutschland (MPDG), Schweiz (MepV), UK (UK MDR 2002).
2. Nutze Tabellen für Vergleiche zwischen Jurisdiktionen.
3. Belege JEDE Aussage mit der exakten Quelle: Gesetz, Artikel/§ und Jurisdiktion.
4. Wenn eine Frage mehrere Jurisdiktionen betrifft, zeige die Unterschiede klar auf.
5. Antworte in der Sprache der Frage (Deutsch oder Englisch)."""

            chain = ChatPromptTemplate.from_messages([
                ("system", sys_msg),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Kontext:\n{context}\n\nFrage: {input}")
            ]) | llm | StrOutputParser()

            # 3. Stream & Store
            full_res = st.write_stream(chain.stream({"input": prompt, "context": context, "chat_history": st.session_state.chat_history[:-1]}))
            copy_button(full_res, "copy_latest")
            st.session_state.chat_history.append(AIMessage(content=full_res))
            st.rerun()
