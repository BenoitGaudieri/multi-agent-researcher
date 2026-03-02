# multi-agent-researcher

Sistema di ricerca multi-agente basato su **LangGraph** che orchestra automaticamente tre fonti di conoscenza: documenti locali (RAG), ricerca web, o entrambe — decidendo in autonomia quale strategia usare in base alla domanda.

---

## Architettura

```
user question
      │
      ▼
 ┌────────────┐
 │ Orchestrate│  ← LLM classifica la domanda: RAG / WEB / BOTH / NONE
 └─────┬──────┘
       │
       │ conditional fan-out
       ├─────────────────────────┐
       ▼                         ▼
 ┌───────────┐           ┌──────────────┐
 │  RAG Agent│           │  Web Agent   │
 │  (FAISS)  │           │(Tavily/DDG)  │
 └─────┬─────┘           └──────┬───────┘
       │                         │
       └────────────┬────────────┘
                    ▼
             ┌────────────┐
             │ Synthesize │  ← assembla tutto e genera la risposta finale
             └─────┬──────┘
                   ▼
             final answer
```

### Nodi del grafo

| Nodo | Ruolo |
|------|-------|
| **orchestrate** | Usa l'LLM per classificare la domanda e decidere il routing |
| **rag_agent** | Recupera chunks rilevanti dal FAISS index locale |
| **web_agent** | Esegue una ricerca web (Tavily o DuckDuckGo) |
| **synthesize** | Combina tutti i contesti e genera la risposta finale |

### Logica di routing

L'orchestratore chiede all'LLM di scegliere esattamente una tra quattro opzioni:

- **RAG** — la domanda riguarda documenti privati / dominio specifico → solo FAISS
- **WEB** — serve informazione recente o generale → solo ricerca web
- **BOTH** → entrambi gli agenti in parallelo (fan-out), poi fan-in su `synthesize`
- **NONE** — risposta nota / conversazionale → vai diretto a `synthesize`

---

## Stack

| Componente | Tecnologia |
|------------|------------|
| Orchestrazione agenti | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Framework LLM | [LangChain](https://python.langchain.com/) |
| LLM & Embeddings | [Ollama](https://ollama.com/) (locale, no API key) |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) |
| Web search (primario) | [Tavily](https://tavily.com/) (gratuito, API key richiesta) |
| Web search (fallback) | [ddgs](https://pypi.org/project/ddgs/) — DuckDuckGo (no API key) |
| CLI | [Typer](https://typer.tiangolo.com/) + [Rich](https://github.com/Textualize/rich) |

---

## Struttura del progetto

```
multi-agent-researcher/
├── main.py               # Entry point CLI
├── requirements.txt
├── .env.example          # Template variabili d'ambiente
│
├── graph/
│   ├── __init__.py
│   ├── state.py          # ResearchState — stato condiviso nel grafo
│   ├── nodes.py          # Implementazione dei 4 nodi
│   └── builder.py        # Costruzione e compilazione del grafo LangGraph
│
├── rag/
│   ├── __init__.py
│   ├── config.py         # Configurazione via env vars
│   └── indexer.py        # Indicizzatore documenti → FAISS
│
├── docs/                 # Documenti da indicizzare
└── web_results/          # Risultati web salvati automaticamente (Markdown)
```

---

## Prerequisiti

### 1. Python 3.11+

```bash
python --version
```

### 2. Ollama installato e in esecuzione

```bash
# Installa da https://ollama.com/
ollama serve

# Scarica i modelli necessari
ollama pull llama3.2          # LLM per ragionamento e risposta
ollama pull nomic-embed-text  # Modello di embedding per FAISS
```

### 3. (Opzionale) API key Tavily

Registrati gratuitamente su [tavily.com](https://tavily.com/) per ottenere una API key.
Senza di essa il web agent usa automaticamente DuckDuckGo.

---

## Installazione

```bash
# Clona o naviga nella cartella del progetto
cd multi-agent-researcher

# Crea e attiva un virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Installa le dipendenze
pip install -r requirements.txt

# Configura le variabili d'ambiente
cp .env.example .env
# Modifica .env con il tuo editor
```

---

## Configurazione (`.env`)

```dotenv
# Modelli Ollama
RAG_LLM_MODEL=llama3.2          # LLM per orchestrazione e sintesi
RAG_EMBED_MODEL=nomic-embed-text # Modello embedding per FAISS

# FAISS
RAG_INDEX_DIR=./faiss_db        # Cartella dove salvare gli indici
RAG_COLLECTION=default          # Nome della collection di default

# Chunking dei documenti
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200

# Numero di chunks da recuperare per query RAG
RAG_TOP_K=5

# Web search (lascia vuoto per usare DuckDuckGo)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx

# Numero massimo di risultati web
WEB_MAX_RESULTS=5

# Cartella dove salvare i risultati web (Markdown)
WEB_RESULTS_DIR=./web_results
```

---

## Utilizzo

### Indicizzare documenti

```bash
# Singolo file
python main.py index ./docs/manuale.pdf

# Intera cartella (ricorsivo — supporta .pdf, .txt, .md, .docx)
python main.py index ./docs/

# Collection custom
python main.py index ./docs/ --collection aziendale
```

### Fare una domanda

```bash
# Domanda singola
python main.py research "Quali sono le clausole di rescissione del contratto?"

# Con output verboso (mostra route + contesti raw)
python main.py research "Ultime notizie su LangGraph?" --verbose

# Usando una collection specifica
python main.py research "Chi è il CEO?" --collection aziendale
```

### Modalità interattiva (REPL)

```bash
python main.py ask

# Con verbose e collection custom
python main.py ask --verbose --collection aziendale
```

```
> Come funziona il FAISS index?
> Quali sono le news di oggi su AI?
> exit
```

### Listare le collections indicizzate

```bash
python main.py list
```

```
Indexed collections:
  default   — 142 chunks  (manuale.pdf, policy.md) · 2025-03-01
  aziendale — 87 chunks   (organigramma.pdf)        · 2025-02-28
```

### Risultati web salvati

Ogni volta che il web agent viene invocato, i risultati vengono salvati automaticamente in `web_results/` come file Markdown:

```
web_results/
├── 20260302-143021-ultime-notizie-su-langchain.md
├── 20260302-150812-chi-ha-vinto-le-elezioni.md
└── ...
```

Formato del file:

```markdown
# Web search results

**Query:** ultime notizie su LangGraph
**Date:** 2026-03-02 14:30 UTC
**Results:** 5

---

## 1. LangGraph 0.3 released — major improvements
**URL:** https://example.com/article

Content of the result...
```

I file `.md` possono essere re-indicizzati nel RAG per costruire una knowledge base dai risultati web:

```bash
python main.py index ./web_results/ --collection web-archive
```

---

## Esempi di routing automatico

| Domanda | Route scelto | Motivazione |
|---------|-------------|-------------|
| *"Cosa dice l'articolo 3 del contratto?"* | **RAG** | Documento privato, sicuramente indicizzato |
| *"Chi ha vinto le elezioni ieri?"* | **WEB** | Informazione recente non in documenti locali |
| *"Cosa dice la policy aziendale sul remote working e quali sono le ultime leggi in materia?"* | **BOTH** | Serve sia doc locale sia normativa aggiornata |
| *"Quanto fa 2+2?"* | **NONE** | Risposta banale, nessuna ricerca necessaria |

---

## Come funziona internamente

### 1. Stato condiviso (`graph/state.py`)

```python
class ResearchState(TypedDict):
    question: str
    needs_rag: bool
    needs_web: bool
    rag_results: Annotated[list[str], add]   # reducer: concatena risultati paralleli
    web_results: Annotated[list[str], add]
    final_answer: Optional[str]
```

Il reducer `add` su `rag_results` e `web_results` è fondamentale: quando i due agenti girano in parallelo e scrivono entrambi sullo stato, i risultati vengono concatenati invece di sovrascriversi.

### 2. Nodo `orchestrate`

Invia all'LLM un prompt di classificazione e legge una singola parola: `RAG`, `WEB`, `BOTH`, o `NONE`. Aggiorna `needs_rag` e `needs_web` di conseguenza.

### 3. Edge condizionale (routing)

```python
def _route(state) -> list[str]:
    destinations = []
    if state["needs_rag"]: destinations.append("rag_agent")
    if state["needs_web"]: destinations.append("web_agent")
    return destinations or ["synthesize"]
```

Restituire una lista fa scattare il **fan-out parallelo** di LangGraph.

### 4. Fan-in automatico

LangGraph aspetta che **tutti** i branch attivi completino prima di eseguire `synthesize`. Non serve alcuna logica di sincronizzazione manuale.

### 5. Nodo `synthesize`

Riceve lo stato con tutti i risultati aggregati e genera la risposta finale chiedendo all'LLM di combinare i contesti disponibili.

---

## Aggiungere un nuovo agente

1. Scrivi la funzione nodo in `graph/nodes.py`:

```python
def database_agent(state: ResearchState) -> dict:
    results = query_my_database(state["question"])
    return {"rag_results": [results]}   # riutilizza un reducer esistente
```

2. Registra il nodo e collegalo in `graph/builder.py`:

```python
g.add_node("database_agent", database_agent)
g.add_edge("database_agent", "synthesize")
```

3. Aggiungi `needs_database: bool` a `ResearchState` e aggiorna la logica di routing in `_route()` e nel prompt dell'orchestratore.

---

## Troubleshooting

**Ollama non risponde**
```bash
ollama serve   # assicurati che il server sia avviato
ollama list    # verifica che llama3.2 e nomic-embed-text siano installati
```

**"No index found. Run `index` first."**
Il RAG agent non trova un FAISS index. Esegui prima `python main.py index <path>`.

**DuckDuckGo ritorna errore**
`ddgs` ha rate limiting aggressivo. Imposta `TAVILY_API_KEY` nel `.env` per una ricerca web più affidabile. In alternativa, riduci `WEB_MAX_RESULTS` a 3.

**L'orchestratore sceglie sempre RAG (o sempre WEB)**
Prova un modello LLM più capace: `RAG_LLM_MODEL=llama3.1:8b` o `mistral`. Il prompt di classificazione funziona meglio con modelli instruction-tuned.
