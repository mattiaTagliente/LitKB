# LitKB - Research Knowledge‑Base Architecture *(Scientific Literature)*
## Version 4.0 - Enhanced Edition

---

## 1 · Purpose and Vision

Modern chat‑style large‑language models (LLMs) display near‑expert reasoning in specialised domains **if** we hand them the right context. For a photonics / space‑electronics researcher, that context already exists: journal PDFs, conference slide decks, test reports, raw data sheets and design notes scattered across a local drive or synced cloud folders.

The goal of this project is therefore **threefold**:

1. **Transform every scientific artefact you possess into a richly linked, machine‑readable record**—pages, sections, equations, claims, images, tables, concepts and citations all become first‑class database entities.
2. **Serve those records to any Plus‑tier browser LLM (ChatGPT, Gemini, Claude) on demand** so the model can draft related‑work sections, compare equations, check experimental claims or suggest new research avenues—all **without paying per‑token API fees** and on a single Windows 10/11 x64 laptop with limited internal storage.
3. **Maintain scientific rigour** through comprehensive provenance tracking, version control, and reproducible retrieval.

**This plan describes a complete system that includes:**
- ✅ Automatic contradiction detection across papers
- ✅ Research thread tracking and idea evolution mapping  
- ✅ Autonomous literature review generation
- ✅ AI-powered hypothesis generation from research gaps
- ✅ Multi-modal answer generation (text + equations + figures)
- ✅ Advanced citation network analysis
- ✅ Collaborative annotation system
- ✅ Trend detection and field emergence prediction

Everything below describes how to build, run and maintain that system **without Docker or Kubernetes**, using only native Windows installers and Python scripts.

---

## 2 · High‑Level Workflow (Complete System)

> *"I drop a badly scanned ESA conference PDF into **D:\Research\Incoming**. Two minutes later I ask Gemini in the browser: 'Which equations in that paper contradict Smith 2021, and show me the associated figures?' Gemini replies with the LaTeX of two display equations, thumbnails of figure 3 and 5, and tells me they disagree with Smith on radiation dose limits."*

That magic happens through eight integrated phases:

| Phase | What happens | Key Windows‑friendly tools |
|-------|-------------|---------------------------|
| **P0 Monitor** | System health checks, processing queue status, error alerts | Windows Task Scheduler, Python logging |
| **P1 Watch** | Smart directory monitor with file filtering, duplicate detection, and source tracking | `watchdog`, `hashlib` |
| **P2 Parse** | Document conversion to structured data with multi-format support | |
| • **PDFs**: layout, sections, references via **GROBID**; element partitioning via **Unstructured‑io** | |
| • **Scanned PDFs / images**: **Tesseract** OCR with pre‑processing | |
| • **DOCX / PPTX**: Unstructured‑io's built‑in readers or `python-docx` / `python-pptx` | |
| • **CSV / XLSX**: `pandas` loads tables with schema validation | GROBID, Unstructured‑io, Tesseract, `python‑docx`, `python‑pptx`, `pandas` |
| **P3 Enrich** | Deep intelligence extraction: | |
| • *Math Processor*: LaTeX + MathML + semantic embeddings | |
| • *Claim Miner*: sentence‑level scientific claims + stance + evidence strength | |
| • *Entity NER*: SciSpacy tags domain concepts + custom entity recognition | |
| • *Image Embedder*: CLIP‑vector + OCR for text in images | |
| • *Citation Normalizer*: DOI resolution and author disambiguation | |
| • *Contradiction Detector*: Identifies conflicting claims across papers | |
| • *Thread Tracker*: Maps evolution of ideas and concepts | GROBID‑math, LaTeXML, SciFact‑T5, SciSpacy, PyTorch + CLIP, Custom NLP models |
| **P4 Store** | Comprehensive persistence layer: | |
| • **PostgreSQL 16** for all structured data with automatic backups | |
| • **pgvector** extension for semantic search with multiple index types | |
| • **Neo4j Desktop** for citation networks, concept graphs, and research threads | |
| • **MinIO** (optional) for blob storage of images/thumbnails | Native Windows installers, pg_dump scheduler |
| **P5 Analyze** | Continuous analysis and insight generation: | |
| • Citation network analysis and importance scoring | |
| • Research trend detection and field emergence | |
| • Contradiction mapping and resolution tracking | |
| • Hypothesis generation from knowledge gaps | NetworkX, SciPy, Custom algorithms |
| **P6 Update** | Smart differential updates: changed sections trigger re‑processing while preserving links | MD5/SHA256 checksums + version tracking |
| **P7 Retrieve** | Advanced multi‑modal retrieval: | |
| • Hybrid search (keyword + vector + graph + temporal) | |
| • Contradiction‑aware retrieval | |
| • Thread‑based exploration | |
| • Multi‑modal answer generation | LlamaIndex, Custom retrievers |
| **P8 Assist** | Autonomous research capabilities: | |
| • Literature review automation | |
| • Hypothesis generation from gaps | |
| • Research thread synthesis | |
| • Collaborative annotation management | Custom agents, LangChain |

---

## 3 · Detailed Windows Installation (≈ 8 GB footprint)

> **Hardware recommendations 🗄️**
> - **Storage**: PDF corpus on external SSD; software + indices on internal NVMe
> - **RAM**: 16GB minimum, 32GB recommended for large batch processing
> - **CPU**: 8+ cores for parallel processing
> - **GPU**: Optional but recommended for faster embeddings (NVIDIA with 6GB+ VRAM)

### 3.1 Install prerequisites (Enhanced)

| Component | Where to get it | Notes | Version |
|-----------|----------------|-------|---------|
| **Java 17 JDK** | Adoptium MSI | required by GROBID server | 17.0.9+ |
| **Python 3.11** | python.org installer | use custom install, add to PATH | 3.11.7+ |
| **PostgreSQL 16 + pgvector** | EnterpriseDB installer → select all components | enable pg_stat_statements | 16.1+ |
| **Neo4j Desktop** | neo4j.com download | install APOC plugin | 5.15+ |
| **Tesseract‑OCR** | UB Mannheim Windows installer | include all language packs | 5.3.3+ |
| **Git + Git LFS** | git‑scm.com | for large file handling | 2.43+ |
| **Redis** (optional) | Memurai for Windows | for task queue | 3.0+ |

### 3.2 Python environment (Production‑ready)

```powershell
# Create isolated environment
py -3.11 -m venv C:\venvs\litkb
C:\venvs\litkb\Scripts\activate

# Core dependencies with pinned versions
pip install --upgrade pip setuptools wheel

# Create requirements.txt
@"
# Core framework
llama-index==0.10.12
llama-index-readers-file==0.1.22
llama-index-embeddings-huggingface==0.1.4
llama-index-vector-stores-postgres==0.1.3
llama-index-graph-stores-neo4j==0.1.3
langchain==0.1.6
langchain-community==0.0.20

# Document processing
grobid-client==0.8.4
watchdog==4.0.0
unstructured[pdf,tesseract]==0.12.4
python-magic-bin==0.4.14  # Windows file type detection
arxiv==2.1.0  # arXiv API client
scholarly==1.7.11  # Google Scholar

# Scientific NLP & Analysis
scispacy==0.5.4
torch==2.2.0
torchvision==0.17.0
transformers==4.37.2
sentence-transformers==2.5.1
scikit-learn==1.4.0
networkx==3.2.1  # Graph analysis
scipy==1.12.0  # Statistical analysis

# Database
psycopg[binary,pool]==3.1.18
pgvector==0.2.5
neo4j==5.17.0

# Multi-modal processing
clip-interrogator==0.6.0
Pillow==10.2.0
opencv-python==4.9.0.80

# Research automation
openai==1.12.0  # For GPT-based analysis
anthropic==0.18.1  # For Claude integration
google-generativeai==0.3.2  # For Gemini

# Utilities
pydantic==2.6.1
pydantic-settings==2.2.1
python-dotenv==1.0.1
rich==13.7.0  # Better logging
typer==0.9.0  # CLI interface
pytest==8.0.1  # Testing
black==24.2.0  # Code formatting
httpx==0.26.0  # Async HTTP
aiofiles==23.2.1  # Async file operations

# Monitoring & API
prometheus-client==0.20.0
structlog==24.1.0
fastapi==0.109.2
uvicorn==0.27.1
websockets==12.0  # For real-time updates

# Task queue & caching
redis==5.0.1
celery==5.3.4
diskcache==5.6.3  # Local caching

# Data analysis
pandas==2.2.0
numpy==1.26.4
matplotlib==3.8.2
plotly==5.19.0
"@ | Out-File -Encoding UTF8 requirements.txt

pip install -r requirements.txt

# Download scientific NLP models
python -m spacy download en_core_sci_scibert
python -m spacy download en_core_sci_lg

# Pre-download embedding models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('allenai/specter2')"
```

### 3.3 GROBID service (Enhanced setup)

1. Download and setup:
```powershell
# Download GROBID
Invoke-WebRequest -Uri "https://github.com/kermitt2/grobid/releases/download/0.8.0/grobid-0.8.0.zip" -OutFile "grobid.zip"
Expand-Archive -Path grobid.zip -DestinationPath C:\
Remove-Item grobid.zip

# Configure for production
cd C:\grobid-0.8.0
Copy-Item config\grobid.yaml config\grobid-prod.yaml

# Edit grobid-prod.yaml:
# - Increase batch size: 100
# - Set consolidation: 2 (full)
# - Enable all models
```

2. Create Windows service:
```powershell
# Install NSSM
Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" -OutFile "nssm.zip"
Expand-Archive -Path nssm.zip -DestinationPath C:\tools
$env:Path += ";C:\tools\nssm-2.24\win64"

# Create GROBID service
nssm install GROBID "C:\grobid-0.8.0\gradlew.bat" "run --args=`"server C:\grobid-0.8.0\config\grobid-prod.yaml`""
nssm set GROBID AppDirectory C:\grobid-0.8.0
nssm set GROBID Start SERVICE_AUTO_START
nssm start GROBID
```

### 3.4 Database setup (Production‑ready)

```sql
-- Create database with optimal settings
CREATE DATABASE litkb
    WITH 
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;

\c litkb;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search
CREATE EXTENSION IF NOT EXISTS btree_gin;  -- For compound indexes
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;  -- For query monitoring

-- Set optimal defaults
ALTER DATABASE litkb SET shared_buffers = '2GB';
ALTER DATABASE litkb SET effective_cache_size = '6GB';
ALTER DATABASE litkb SET maintenance_work_mem = '512MB';
ALTER DATABASE litkb SET random_page_cost = 1.1;  -- For SSD
```

Then apply the enhanced DDL from Section 6.

### 3.5 Neo4j graph (Enhanced configuration)

1. Launch Neo4j Desktop → New Project → "LitKB Research"
2. Create Local DBMS:
   - Name: LitGraph
   - Version: 5.15.0
   - Password: (secure password)
3. Install plugins:
   - APOC (required for advanced queries)
   - Graph Data Science Library (for similarity algorithms)
4. Configure settings:
   ```
   dbms.memory.heap.initial_size=2G
   dbms.memory.heap.max_size=4G
   dbms.memory.pagecache.size=2G
   ```
5. Start the database

### 3.6 Pipeline configuration (Enhanced)

```yaml
# config.yaml - Production configuration
system:
  environment: production
  log_level: INFO
  log_dir: D:\litkb\logs
  worker_processes: 4
  batch_size: 50
  
monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 60
  alert_email: researcher@example.com

storage:
  watch_folders:
    - path: D:\Research\Incoming
      patterns: ["*.pdf", "*.docx", "*.pptx"]
      recursive: true
    - path: E:\Conference\Proceedings
      patterns: ["*.pdf"]
      recursive: false
  processed_dir: D:\Research\Processed
  error_dir: D:\Research\Failed
  thumbnail_dir: D:\Research\Thumbnails
  
database:
  postgres:
    host: localhost
    port: 5432
    database: litkb
    user: litkb_user
    password: ${LITKB_PG_PASSWORD}  # From environment
    pool_size: 20
    pool_max_overflow: 10
  
  neo4j:
    uri: bolt://localhost:7687
    user: neo4j
    password: ${LITKB_NEO4J_PASSWORD}
    
services:
  grobid:
    url: http://localhost:8070
    timeout: 300
    batch_size: 10
    
  embeddings:
    model: allenai/specter2
    batch_size: 32
    device: cuda  # or cpu
    
  ocr:
    tesseract_path: C:\Program Files\Tesseract-OCR\tesseract.exe
    languages: ["eng", "equ"]  # English + equations
    preprocessing: true
    
processing:
  duplicate_check: true
  version_tracking: true
  max_retries: 3
  retry_delay: 60
  parallel_workers: 4
  
  math:
    extract_display_only: false
    normalize_latex: true
    generate_mathml: true
    
  claims:
    min_confidence: 0.7
    include_evidence: true
    
  images:
    max_size_mb: 50
    thumbnail_size: [256, 256]
    extract_text: true
```

### 3.7 Start the enhanced pipeline

```powershell
# Set environment variables
$env:LITKB_PG_PASSWORD = "your_secure_password"
$env:LITKB_NEO4J_PASSWORD = "your_neo4j_password"

# Initialize database schema
python C:\litkb\scripts\init_db.py --config C:\litkb\config.yaml

# Run migrations
python C:\litkb\scripts\migrate.py --latest

# Start services
python C:\litkb\monitor.py --daemon &  # Monitoring daemon
python C:\litkb\watcher.py --config C:\litkb\config.yaml
```

Or install as Windows services:
```powershell
# Monitor service
nssm install litkb-monitor "C:\venvs\litkb\Scripts\python.exe" "C:\litkb\monitor.py --daemon"
nssm set litkb-monitor AppEnvironmentExtra LITKB_PG_PASSWORD=your_password LITKB_NEO4J_PASSWORD=your_password

# Watcher service  
nssm install litkb-watcher "C:\venvs\litkb\Scripts\python.exe" "C:\litkb\watcher.py --config C:\litkb\config.yaml"
nssm set litkb-watcher AppEnvironmentExtra LITKB_PG_PASSWORD=your_password LITKB_NEO4J_PASSWORD=your_password
nssm set litkb-watcher DependOnService litkb-monitor GROBID

# Start all services
nssm start litkb-monitor
nssm start litkb-watcher
```

### 3.8 Initial testing and bulk ingestion

```powershell
# Test with single file
python C:\litkb\scripts\test_pipeline.py --file "test.pdf" --verbose

# Dry run on directory
python C:\litkb\scripts\bulk_ingest.py --dir "D:\SmallBatch" --dry-run

# Actual ingestion with progress bar
python C:\litkb\scripts\bulk_ingest.py --dir "D:\Research\Archive" --batch-size 20 --parallel 4
```

---

## 4 · Leveraging Browser‑Based LLMs (Enhanced Workflows)

### 4.1 Hybrid Processing Strategy

| Local Processing | Browser LLM Delegation | Automated Integration |
|-----------------|----------------------|---------------------|
| Standard PDFs with good OCR | Complex tables requiring interpretation | API webhook from browser extension |
| Simple LaTeX equations | Hand-drawn equations needing recognition | Selenium automation for batch processing |
| English text | Multi-language content needing translation | Power Automate Desktop flows |
| Standard figures | Complex diagrams needing explanation | Custom browser bookmarklet |

### 4.2 Browser Integration Patterns

```javascript
// Bookmarklet for quick paper analysis
javascript:(function(){
  const selection = window.getSelection().toString();
  const prompt = `Analyze this scientific claim and identify:
1. Main hypothesis
2. Supporting evidence  
3. Methodology used
4. Potential weaknesses
Format as JSON for database import.

Claim: "${selection}"`;
  
  // Copy to clipboard for LLM
  navigator.clipboard.writeText(prompt);
  alert('Analysis prompt copied! Paste into Claude/ChatGPT');
})();
```

### 4.3 Automated Browser Workflows

1. **Chrome Extension** for seamless integration:
```javascript
// manifest.json snippet
{
  "name": "LitKB Assistant",
  "permissions": ["activeTab", "storage", "nativeMessaging"],
  "host_permissions": ["https://chatgpt.com/*", "https://claude.ai/*"]
}
```

2. **Power Automate Desktop** flow:
   - Monitor `D:\Research\NeedsOCR` folder
   - Open browser to ChatGPT
   - Upload file with prompt
   - Extract response
   - Save to `D:\Research\Incoming\ocr_complete`

---

## 5 · Enhanced Data Model

### 5.1 Core Entities (Complete System)

```sql
-- Main content entities
paper:
  + authors         JSONB  -- [{name, orcid, affiliation}]
  + keywords        TEXT[]
  + language        TEXT
  + paper_type      TEXT   -- article, preprint, thesis, report
  + version         INT
  + checksum        TEXT
  + ingested_at     TIMESTAMP
  + updated_at      TIMESTAMP
  + quality_score   FLOAT  -- OCR/parsing quality metric

section:
  + section_number  TEXT   -- "2.3.1"
  + word_count      INT
  + language        TEXT
  + confidence      FLOAT  -- parsing confidence

equation:
  + equation_number TEXT   -- "Eq. 3.14"
  + is_definition   BOOL
  + variables       JSONB  -- extracted variable definitions
  + domain          TEXT   -- physics, math, engineering

claim:
  + evidence_refs   UUID[] -- links to supporting sections/equations
  + confidence      FLOAT
  + claim_type      TEXT   -- hypothesis, result, method
  + verification_status TEXT -- verified, disputed, unverified

image:
  + image_type      TEXT   -- figure, diagram, photo, graph
  + contains_text   BOOL
  + ocr_text        TEXT
  + metadata        JSONB  -- EXIF, matplotlib params, etc.

-- Research tracking entities
author:
  author_id         UUID PRIMARY KEY
  name              TEXT
  orcid             TEXT UNIQUE
  affiliations      TEXT[]
  h_index           INT
  research_areas    TEXT[]
  
experiment:
  experiment_id     UUID PRIMARY KEY
  paper_id          UUID REFERENCES paper
  name              TEXT
  methodology       TEXT
  datasets          JSONB
  results           JSONB
  reproducibility   FLOAT

hypothesis:
  hypothesis_id     UUID PRIMARY KEY
  statement         TEXT
  domain            TEXT
  based_on_gap      UUID
  related_papers    UUID[]
  experiments       JSONB
  expected_impact   TEXT
  confidence_score  FLOAT
  novelty_score     FLOAT
  impact_score      FLOAT
  created_at        TIMESTAMP
  tested            BOOLEAN DEFAULT false

research_thread:
  thread_id         UUID PRIMARY KEY
  root_concept      TEXT
  root_paper        UUID REFERENCES paper
  timeline          JSONB  -- evolution stages
  key_insights      TEXT[]
  created_at        TIMESTAMP
  updated_at        TIMESTAMP

contradiction:
  contradiction_id  UUID PRIMARY KEY
  claim1_id         UUID REFERENCES claim
  claim2_id         UUID REFERENCES claim
  opposition_score  FLOAT
  explanation       TEXT
  resolution_status TEXT  -- unresolved, resolved, partial
  evidence          JSONB

research_gap:
  gap_id            UUID PRIMARY KEY
  domain            TEXT
  description       TEXT
  significance      FLOAT
  related_papers    UUID[]
  potential_impact  TEXT
  identified_at     TIMESTAMP

literature_review:
  review_id         UUID PRIMARY KEY
  topic             TEXT
  scope             JSONB
  executive_summary TEXT
  themes            JSONB
  new_papers        UUID[]
  contradictions    UUID[]
  gaps              UUID[]
  document_path     TEXT
  created_at        TIMESTAMP
```

### 5.2 Relation Types (Comprehensive)

```sql
-- Enumerated relation types with semantics
CREATE TYPE relation_type AS ENUM (
  -- Citations
  'cites', 'cited_by', 'extends', 'refutes', 'supports',
  
  -- Similarities  
  'similar_to', 'identical_to', 'variant_of',
  
  -- Dependencies
  'depends_on', 'enables', 'contradicts',
  
  -- Hierarchical
  'part_of', 'contains', 'summarizes',
  
  -- Temporal
  'follows', 'precedes', 'contemporary_with',
  
  -- Authorship
  'authored_by', 'reviewed_by',
  
  -- Research tracking
  'addresses_gap', 'tests_hypothesis', 'continues_thread',
  'resolves_contradiction'
);
```

---

## 6 · Enhanced SQL DDL

<details>
<summary>Full production schema with indexes and constraints</summary>

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Custom types
CREATE TYPE paper_type AS ENUM ('article', 'preprint', 'conference', 'thesis', 'report', 'book', 'patent');
CREATE TYPE stance_type AS ENUM ('supports', 'refutes', 'neutral', 'extends', 'questions');
CREATE TYPE image_type AS ENUM ('figure', 'diagram', 'photo', 'graph', 'equation', 'table');

-- Main tables with enhanced fields
CREATE TABLE paper (
  paper_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  title         TEXT NOT NULL,
  abstract      TEXT,
  authors       JSONB NOT NULL DEFAULT '[]',
  year          INT CHECK (year BETWEEN 1900 AND 2100),
  journal       TEXT,
  doi           TEXT UNIQUE,
  arxiv_id      TEXT UNIQUE,
  paper_type    paper_type DEFAULT 'article',
  keywords      TEXT[] DEFAULT '{}',
  language      TEXT DEFAULT 'en',
  source_path   TEXT NOT NULL,
  checksum      TEXT NOT NULL,
  version       INT DEFAULT 1,
  quality_score FLOAT DEFAULT 1.0 CHECK (quality_score BETWEEN 0 AND 1),
  ingested_at   TIMESTAMP DEFAULT NOW(),
  updated_at    TIMESTAMP DEFAULT NOW(),
  metadata      JSONB DEFAULT '{}'
);

-- Indexes for paper
CREATE INDEX paper_year_idx ON paper(year);
CREATE INDEX paper_authors_gin ON paper USING gin(authors);
CREATE INDEX paper_keywords_gin ON paper USING gin(keywords);
CREATE INDEX paper_title_trgm ON paper USING gin(title gin_trgm_ops);
CREATE INDEX paper_checksum_idx ON paper(checksum);

-- Sections with hierarchy
CREATE TABLE section (
  section_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  paper_id        UUID NOT NULL REFERENCES paper ON DELETE CASCADE,
  parent_id       UUID REFERENCES section ON DELETE CASCADE,
  section_number  TEXT,
  type            TEXT NOT NULL,
  heading         TEXT,
  text            TEXT,
  page_from       INT CHECK (page_from > 0),
  page_to         INT CHECK (page_to >= page_from),
  word_count      INT,
  language        TEXT DEFAULT 'en',
  confidence      FLOAT DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
  vector          VECTOR(768),
  metadata        JSONB DEFAULT '{}'
);

-- Indexes for section
CREATE INDEX section_paper_idx ON section(paper_id);
CREATE INDEX section_parent_idx ON section(parent_id);
CREATE INDEX section_type_idx ON section(type);
CREATE INDEX section_vec_idx ON section USING hnsw (vector vector_cosine_ops);
CREATE INDEX section_text_trgm ON section USING gin(text gin_trgm_ops);

-- Enhanced equations
CREATE TABLE equation (
  equation_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  paper_id        UUID NOT NULL REFERENCES paper ON DELETE CASCADE,
  section_id      UUID REFERENCES section ON DELETE CASCADE,
  equation_number TEXT,
  latex           TEXT NOT NULL,
  mathml          XML,
  is_definition   BOOLEAN DEFAULT false,
  variables       JSONB DEFAULT '{}',
  domain          TEXT,
  vector          VECTOR(768),
  confidence      FLOAT DEFAULT 1.0,
  metadata        JSONB DEFAULT '{}'
);

-- Indexes for equation
CREATE INDEX equation_paper_idx ON equation(paper_id);
CREATE INDEX equation_section_idx ON equation(section_id);
CREATE INDEX equation_vec_idx ON equation USING hnsw (vector vector_cosine_ops);
CREATE INDEX equation_domain_idx ON equation(domain);

-- Scientific claims with evidence
CREATE TABLE claim (
  claim_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  paper_id        UUID NOT NULL REFERENCES paper ON DELETE CASCADE,
  section_id      UUID REFERENCES section ON DELETE CASCADE,
  text            TEXT NOT NULL,
  stance          stance_type,
  evidence_refs   UUID[] DEFAULT '{}',
  confidence      FLOAT DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
  claim_type      TEXT,
  vector          VECTOR(768),
  metadata        JSONB DEFAULT '{}'
);

-- Indexes for claim
CREATE INDEX claim_paper_idx ON claim(paper_id);
CREATE INDEX claim_stance_idx ON claim(stance);
CREATE INDEX claim_vec_idx ON claim USING hnsw (vector vector_cosine_ops);
CREATE INDEX claim_evidence_gin ON claim USING gin(evidence_refs);

-- Enhanced images
CREATE TABLE image (
  image_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  paper_id        UUID NOT NULL REFERENCES paper ON DELETE CASCADE,
  section_id      UUID REFERENCES section ON DELETE CASCADE,
  image_type      image_type,
  caption         TEXT,
  file_orig       TEXT NOT NULL,
  file_thumb      TEXT,
  contains_text   BOOLEAN DEFAULT false,
  ocr_text        TEXT,
  width           INT,
  height          INT,
  vector          VECTOR(768),
  metadata        JSONB DEFAULT '{}'
);

-- Indexes for image
CREATE INDEX image_paper_idx ON image(paper_id);
CREATE INDEX image_type_idx ON image(image_type);
CREATE INDEX image_vec_idx ON image USING hnsw (vector vector_cosine_ops);

-- Authors table
CREATE TABLE author (
  author_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name            TEXT NOT NULL,
  orcid           TEXT UNIQUE,
  email           TEXT,
  affiliations    TEXT[] DEFAULT '{}',
  h_index         INT,
  metadata        JSONB DEFAULT '{}'
);

CREATE INDEX author_name_trgm ON author USING gin(name gin_trgm_ops);
CREATE INDEX author_orcid_idx ON author(orcid);

-- Paper-Author junction
CREATE TABLE paper_author (
  paper_id        UUID REFERENCES paper ON DELETE CASCADE,
  author_id       UUID REFERENCES author ON DELETE CASCADE,
  position        INT NOT NULL,
  is_corresponding BOOLEAN DEFAULT false,
  PRIMARY KEY (paper_id, author_id)
);

-- Universal relations with rich metadata
CREATE TABLE relation (
  relation_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  src_type        TEXT NOT NULL,
  src_id          UUID NOT NULL,
  tgt_type        TEXT NOT NULL,
  tgt_id          UUID NOT NULL,
  relation        TEXT NOT NULL,
  confidence      FLOAT DEFAULT 1.0,
  source          TEXT,
  metadata        JSONB DEFAULT '{}',
  created_at      TIMESTAMP DEFAULT NOW(),
  UNIQUE (src_id, tgt_id, relation, source)
);

-- Indexes for relation
CREATE INDEX relation_src_idx ON relation(src_type, src_id);
CREATE INDEX relation_tgt_idx ON relation(tgt_type, tgt_id);
CREATE INDEX relation_type_idx ON relation(relation);

-- Research tracking tables
CREATE TABLE hypothesis (
  hypothesis_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  statement         TEXT NOT NULL,
  domain            TEXT NOT NULL,
  based_on_gap      UUID REFERENCES research_gap,
  related_papers    UUID[] DEFAULT '{}',
  experiments       JSONB DEFAULT '{}',
  expected_impact   TEXT,
  confidence_score  FLOAT DEFAULT 0.5,
  novelty_score     FLOAT,
  impact_score      FLOAT,
  overall_score     FLOAT GENERATED ALWAYS AS (
    0.4 * COALESCE(novelty_score, 0) + 
    0.4 * COALESCE(impact_score, 0) + 
    0.2 * confidence_score
  ) STORED,
  tested            BOOLEAN DEFAULT false,
  created_at        TIMESTAMP DEFAULT NOW(),
  updated_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX hypothesis_domain_idx ON hypothesis(domain);
CREATE INDEX hypothesis_score_idx ON hypothesis(overall_score DESC);
CREATE INDEX hypothesis_papers_gin ON hypothesis USING gin(related_papers);

CREATE TABLE research_thread (
  thread_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  root_concept      TEXT NOT NULL,
  root_paper        UUID REFERENCES paper ON DELETE CASCADE,
  timeline          JSONB DEFAULT '[]',
  key_insights      TEXT[] DEFAULT '{}',
  concept_evolution JSONB DEFAULT '{}',
  created_at        TIMESTAMP DEFAULT NOW(),
  updated_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX thread_concept_idx ON research_thread USING gin(root_concept gin_trgm_ops);
CREATE INDEX thread_paper_idx ON research_thread(root_paper);

CREATE TABLE contradiction (
  contradiction_id  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  claim1_id         UUID NOT NULL REFERENCES claim ON DELETE CASCADE,
  claim2_id         UUID NOT NULL REFERENCES claim ON DELETE CASCADE,
  opposition_score  FLOAT NOT NULL CHECK (opposition_score BETWEEN 0 AND 1),
  explanation       TEXT,
  resolution_status TEXT DEFAULT 'unresolved',
  evidence          JSONB DEFAULT '{}',
  created_at        TIMESTAMP DEFAULT NOW(),
  UNIQUE(claim1_id, claim2_id)
);

CREATE INDEX contradiction_claim1_idx ON contradiction(claim1_id);
CREATE INDEX contradiction_claim2_idx ON contradiction(claim2_id);
CREATE INDEX contradiction_status_idx ON contradiction(resolution_status);

CREATE TABLE research_gap (
  gap_id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  domain            TEXT NOT NULL,
  description       TEXT NOT NULL,
  significance      FLOAT CHECK (significance BETWEEN 0 AND 1),
  related_papers    UUID[] DEFAULT '{}',
  potential_impact  TEXT,
  identified_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX gap_domain_idx ON research_gap(domain);
CREATE INDEX gap_significance_idx ON research_gap(significance DESC);

CREATE TABLE literature_review (
  review_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  topic             TEXT NOT NULL,
  scope             JSONB DEFAULT '{}',
  executive_summary TEXT,
  themes            JSONB DEFAULT '[]',
  new_papers        UUID[] DEFAULT '{}',
  existing_papers   UUID[] DEFAULT '{}',
  contradictions    UUID[] DEFAULT '{}',
  gaps              UUID[] DEFAULT '{}',
  document_path     TEXT,
  visualizations    JSONB DEFAULT '{}',
  status            TEXT DEFAULT 'in_progress',
  created_at        TIMESTAMP DEFAULT NOW(),
  completed_at      TIMESTAMP
);

CREATE INDEX review_topic_idx ON literature_review USING gin(topic gin_trgm_ops);
CREATE INDEX review_status_idx ON literature_review(status);

-- Collaboration tables
CREATE TABLE annotation (
  annotation_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id           TEXT NOT NULL,
  target_type       TEXT NOT NULL,
  target_id         UUID NOT NULL,
  annotation_type   TEXT NOT NULL, -- 'note', 'question', 'correction'
  content           TEXT NOT NULL,
  created_at        TIMESTAMP DEFAULT NOW(),
  updated_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX annotation_target_idx ON annotation(target_type, target_id);
CREATE INDEX annotation_user_idx ON annotation(user_id);

-- Advanced analytics tables
CREATE TABLE citation_network (
  network_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  center_paper      UUID REFERENCES paper ON DELETE CASCADE,
  depth             INT DEFAULT 2,
  node_count        INT,
  edge_count        INT,
  graph_data        JSONB,
  metrics           JSONB, -- centrality, clustering coefficient, etc.
  created_at        TIMESTAMP DEFAULT NOW()
);

CREATE TABLE trend_analysis (
  trend_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  domain            TEXT NOT NULL,
  time_period       TSRANGE NOT NULL,
  emerging_topics   JSONB DEFAULT '[]',
  declining_topics  JSONB DEFAULT '[]',
  key_papers        UUID[] DEFAULT '{}',
  growth_rate       FLOAT,
  created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX trend_domain_idx ON trend_analysis(domain);
CREATE INDEX trend_period_idx ON trend_analysis USING gist(time_period);

-- Views for common queries
CREATE MATERIALIZED VIEW paper_stats AS
SELECT 
  p.paper_id,
  p.title,
  p.year,
  COUNT(DISTINCT s.section_id) as section_count,
  COUNT(DISTINCT e.equation_id) as equation_count,
  COUNT(DISTINCT c.claim_id) as claim_count,
  COUNT(DISTINCT i.image_id) as image_count,
  COUNT(DISTINCT con.contradiction_id) as contradiction_count,
  AVG(p.quality_score) as avg_quality
FROM paper p
LEFT JOIN section s ON p.paper_id = s.paper_id
LEFT JOIN equation e ON p.paper_id = e.paper_id
LEFT JOIN claim c ON p.paper_id = c.paper_id
LEFT JOIN image i ON p.paper_id = i.paper_id
LEFT JOIN contradiction con ON c.claim_id IN (con.claim1_id, con.claim2_id)
GROUP BY p.paper_id, p.title, p.year;

CREATE INDEX paper_stats_year_idx ON paper_stats(year);

CREATE MATERIALIZED VIEW author_network AS
SELECT 
  a1.author_id as author1,
  a2.author_id as author2,
  COUNT(DISTINCT pa1.paper_id) as collaboration_count,
  array_agg(DISTINCT pa1.paper_id) as papers
FROM paper_author pa1
JOIN paper_author pa2 ON pa1.paper_id = pa2.paper_id AND pa1.author_id < pa2.author_id
JOIN author a1 ON pa1.author_id = a1.author_id
JOIN author a2 ON pa2.author_id = a2.author_id
GROUP BY a1.author_id, a2.author_id;

CREATE INDEX author_network_idx ON author_network(author1, author2);

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER paper_updated_at BEFORE UPDATE ON paper
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Full-text search configuration
CREATE TEXT SEARCH CONFIGURATION scientific (COPY = english);
ALTER TEXT SEARCH CONFIGURATION scientific
  ALTER MAPPING FOR word, asciiword WITH english_stem, simple;
```

</details>

---

## 7 · Performance Optimization & Monitoring

### 7.1 Query Optimization

```sql
-- Analyze query performance
CREATE TABLE slow_queries AS
SELECT * FROM pg_stat_statements
WHERE mean_exec_time > 1000  -- queries taking > 1 second
ORDER BY mean_exec_time DESC;

-- Create composite indexes for common queries
CREATE INDEX paper_year_type_idx ON paper(year, paper_type);
CREATE INDEX section_paper_type_idx ON section(paper_id, type);

-- Partition large tables by year (optional for very large datasets)
CREATE TABLE paper_2024 PARTITION OF paper FOR VALUES FROM (2024) TO (2025);
```

### 7.2 Monitoring Dashboard

```python
# monitor.py - System health monitoring
import psutil
import asyncio
from prometheus_client import Gauge, Counter, Histogram, start_http_server

# Metrics
papers_processed = Counter('papers_processed_total', 'Total papers processed')
processing_errors = Counter('processing_errors_total', 'Total processing errors', ['stage'])
processing_duration = Histogram('processing_duration_seconds', 'Processing duration', ['document_type'])
system_cpu = Gauge('system_cpu_percent', 'System CPU usage')
system_memory = Gauge('system_memory_percent', 'System memory usage')
db_connections = Gauge('database_connections', 'Active database connections')

async def monitor_system():
    while True:
        system_cpu.set(psutil.cpu_percent())
        system_memory.set(psutil.virtual_memory().percent)
        # Query DB for connection count
        db_connections.set(await get_db_connection_count())
        await asyncio.sleep(10)

if __name__ == '__main__':
    start_http_server(9090)  # Prometheus metrics endpoint
    asyncio.run(monitor_system())
```

### 7.3 Backup Strategy

```powershell
# Automated backup script (backup.ps1)
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$backupDir = "E:\Backups\litkb\$timestamp"

# Database backup
pg_dump -h localhost -U litkb_user -d litkb -Fc -f "$backupDir\litkb.dump"

# Neo4j backup
neo4j-admin dump --database=litgraph --to="$backupDir\litgraph.dump"

# Compress
Compress-Archive -Path $backupDir -DestinationPath "$backupDir.zip"
Remove-Item -Recurse $backupDir

# Keep only last 30 days
Get-ChildItem "E:\Backups\litkb\*.zip" | Where-Object {$_.CreationTime -lt (Get-Date).AddDays(-30)} | Remove-Item

# Schedule this script in Task Scheduler to run daily
```

---

## 8 · Advanced Query & Retrieval

### 8.1 Multi-modal Search Interface

```python
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class SearchResult:
    content: str
    content_type: str  # 'text', 'equation', 'image', 'claim'
    source: Dict[str, Any]
    score: float
    context: List[str]  # surrounding content

class AdvancedRetriever:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.pg_conn = self._init_postgres()
        self.neo4j_conn = self._init_neo4j()
        self.embedder = self._init_embedder()
        
    async def hybrid_search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        top_k: int = 20,
        search_types: List[str] = ['text', 'equation', 'claim', 'image']
    ) -> List[SearchResult]:
        """
        Combines multiple search strategies:
        1. Keyword search (PostgreSQL FTS)
        2. Semantic search (pgvector)
        3. Graph traversal (Neo4j)
        4. Temporal relevance
        """
        results = []
        
        # Parse query intent
        intent = await self._analyze_query_intent(query)
        
        # Parallel search execution
        search_tasks = []
        
        if 'text' in search_types:
            search_tasks.append(self._keyword_search(query, filters))
            search_tasks.append(self._semantic_search(query, 'section', filters))
            
        if 'equation' in search_types and intent.has_math:
            search_tasks.append(self._equation_search(intent.math_terms))
            
        if 'claim' in search_types:
            search_tasks.append(self._claim_search(query, intent.stance))
            
        if 'image' in search_types:
            search_tasks.append(self._image_search(query))
            
        # Execute searches in parallel
        search_results = await asyncio.gather(*search_tasks)
        
        # Merge and re-rank results
        merged_results = self._merge_results(search_results)
        
        # Apply graph-based re-ranking
        if intent.requires_graph:
            merged_results = await self._graph_rerank(merged_results, query)
            
        # Add context
        final_results = await self._add_context(merged_results[:top_k])
        
        return final_results
    
    async def _semantic_search(
        self, 
        query: str, 
        table: str, 
        filters: Dict = None
    ) -> List[SearchResult]:
        """Semantic search using pgvector"""
        query_embedding = self.embedder.encode(query)
        
        sql = f"""
        SELECT *, vector <=> %s::vector as distance
        FROM {table}
        WHERE 1=1
        """
        
        params = [query_embedding.tolist()]
        
        if filters:
            if 'year_from' in filters:
                sql += " AND year >= %s"
                params.append(filters['year_from'])
            if 'paper_type' in filters:
                sql += " AND paper_type = %s"
                params.append(filters['paper_type'])
                
        sql += " ORDER BY distance LIMIT 50"
        
        results = await self.pg_conn.fetch(sql, *params)
        return self._format_results(results, 'semantic')
    
    async def _graph_rerank(
        self, 
        results: List[SearchResult], 
        query: str
    ) -> List[SearchResult]:
        """Use citation network for re-ranking"""
        # Get paper IDs from results
        paper_ids = [r.source['paper_id'] for r in results]
        
        # Query Neo4j for citation importance
        cypher = """
        MATCH (p:Paper)
        WHERE p.paper_id IN $paper_ids
        OPTIONAL MATCH (p)<-[:CITES]-(citing:Paper)
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
        RETURN p.paper_id as paper_id,
               COUNT(DISTINCT citing) as citation_count,
               MAX(a.h_index) as max_h_index,
               EXISTS((p)-[:EXTENDS]->(:Paper)) as is_extension
        """
        
        graph_scores = await self.neo4j_conn.run(cypher, paper_ids=paper_ids)
        
        # Combine scores
        for result in results:
            graph_data = next(
                (g for g in graph_scores if g['paper_id'] == result.source['paper_id']), 
                None
            )
            if graph_data:
                # Boost score based on citations and author reputation
                citation_boost = np.log1p(graph_data['citation_count']) * 0.1
                author_boost = (graph_data['max_h_index'] or 0) * 0.01
                extension_boost = 0.2 if graph_data['is_extension'] else 0
                
                result.score *= (1 + citation_boost + author_boost + extension_boost)
                
        return sorted(results, key=lambda x: x.score, reverse=True)

    def format_for_llm(
        self, 
        results: List[SearchResult],
        format_type: str = 'markdown'
    ) -> str:
        """Format results for LLM consumption"""
        if format_type == 'markdown':
            output = []
            
            for i, result in enumerate(results, 1):
                output.append(f"## Result {i} (Score: {result.score:.3f})")
                output.append(f"**Source**: {result.source['title']} ({result.source['year']})")
                output.append(f"**Type**: {result.content_type}")
                
                if result.content_type == 'equation':
                    output.append(f"```latex\n{result.content}\n```")
                elif result.content_type == 'image':
                    output.append(f"![{result.source.get('caption', 'Figure')}]({result.content})")
                else:
                    output.append(result.content)
                    
                if result.context:
                    output.append("\n**Context**:")
                    for ctx in result.context[:2]:
                        output.append(f"> {ctx}")
                        
                output.append("\n---\n")
                
            return '\n'.join(output)
            
        elif format_type == 'xml':
            # XML format for Claude
            return self._format_xml(results)
```

### 8.2 Contradiction Detection System

```python
class ContradictionDetector:
    """Identifies and tracks contradictions across the knowledge base"""
    
    def __init__(self, db_conn, embedder):
        self.db = db_conn
        self.embedder = embedder
        self.contradiction_threshold = 0.85
        
    async def find_contradictions(self, claim_id: UUID) -> List[Contradiction]:
        """Find claims that contradict the given claim"""
        claim = await self.db.fetch_one(
            "SELECT * FROM claim WHERE claim_id = $1", claim_id
        )
        
        # Method 1: Direct semantic opposition
        negated_embedding = self._negate_embedding(claim['vector'])
        
        # Method 2: Look for explicit refutations
        sql = """
        WITH similar_claims AS (
            SELECT *, 
                   vector <=> $1::vector as similarity,
                   1 - (vector <=> $2::vector) as opposition
            FROM claim
            WHERE claim_id != $3
              AND stance IN ('refutes', 'questions')
        )
        SELECT * FROM similar_claims
        WHERE opposition > $4 OR similarity < 0.3
        ORDER BY opposition DESC
        LIMIT 20
        """
        
        contradicting = await self.db.fetch(
            sql, 
            claim['vector'], 
            negated_embedding,
            claim_id,
            self.contradiction_threshold
        )
        
        # Analyze each potential contradiction
        contradictions = []
        for candidate in contradicting:
            analysis = await self._analyze_contradiction(claim, candidate)
            if analysis.is_valid:
                contradictions.append(analysis)
                
        # Store in database
        await self._store_contradictions(claim_id, contradictions)
        
        return contradictions
    
    async def _analyze_contradiction(self, claim1, claim2) -> ContradictionAnalysis:
        """Deep analysis of potential contradiction"""
        # Check if they're about the same subject
        entities1 = await self._extract_entities(claim1['text'])
        entities2 = await self._extract_entities(claim2['text'])
        
        overlap = len(entities1 & entities2) / max(len(entities1), len(entities2))
        
        if overlap < 0.5:
            return ContradictionAnalysis(is_valid=False)
            
        # Check logical opposition
        opposition_score = self._calculate_logical_opposition(
            claim1['text'], 
            claim2['text']
        )
        
        # Get supporting evidence
        evidence1 = await self._get_evidence(claim1['claim_id'])
        evidence2 = await self._get_evidence(claim2['claim_id'])
        
        return ContradictionAnalysis(
            is_valid=opposition_score > 0.7,
            opposition_score=opposition_score,
            claim1=claim1,
            claim2=claim2,
            evidence1=evidence1,
            evidence2=evidence2,
            explanation=self._generate_explanation(claim1, claim2, opposition_score)
        )
    
    def _negate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Generate embedding for semantic opposite"""
        # Use a learned transformation or simple negation
        # This is a simplified version - in practice use a trained model
        return -embedding + 2 * np.mean(embedding)
    
    async def build_contradiction_graph(self):
        """Build comprehensive contradiction network"""
        cypher = """
        MATCH (c1:Claim)-[r:CONTRADICTS]->(c2:Claim)
        MATCH (p1:Paper)-[:CONTAINS]->(c1)
        MATCH (p2:Paper)-[:CONTAINS]->(c2)
        CREATE (p1)-[con:HAS_CONTRADICTION {
            strength: r.opposition_score,
            claims: [c1.text, c2.text]
        }]->(p2)
        """
        await self.neo4j.run(cypher)
```

### 8.3 Research Thread Tracking

```python
class ResearchThreadTracker:
    """Tracks evolution of ideas across papers"""
    
    def __init__(self, db_connections):
        self.pg = db_connections['postgres']
        self.neo4j = db_connections['neo4j']
        
    async def create_thread(
        self, 
        root_concept: str,
        root_paper_id: UUID
    ) -> ResearchThread:
        """Initialize a new research thread"""
        thread = ResearchThread(
            thread_id=uuid4(),
            root_concept=root_concept,
            root_paper=root_paper_id,
            created_at=datetime.now()
        )
        
        # Build the thread graph
        await self._trace_concept_evolution(thread)
        
        return thread
    
    async def _trace_concept_evolution(self, thread: ResearchThread):
        """Trace how a concept evolves through citations"""
        cypher = """
        MATCH path = (root:Paper {paper_id: $root_id})-[:CITES*1..5]->(cited:Paper)
        WHERE ANY(concept IN $concepts WHERE 
            cited.abstract CONTAINS concept OR 
            cited.title CONTAINS concept
        )
        WITH path, [node in nodes(path) | node.paper_id] as paper_chain
        RETURN paper_chain, length(path) as depth
        ORDER BY depth
        """
        
        results = await self.neo4j.run(
            cypher,
            root_id=str(thread.root_paper),
            concepts=self._extract_key_concepts(thread.root_concept)
        )
        
        # Build timeline
        timeline = []
        for result in results:
            papers = await self._get_papers(result['paper_chain'])
            evolution = await self._analyze_concept_evolution(
                papers, 
                thread.root_concept
            )
            timeline.append(evolution)
            
        thread.timeline = timeline
        thread.key_insights = self._extract_insights(timeline)
        
    async def _analyze_concept_evolution(
        self, 
        papers: List[Paper], 
        concept: str
    ) -> ConceptEvolution:
        """Analyze how a concept changes across papers"""
        evolution = ConceptEvolution(concept=concept)
        
        for i, paper in enumerate(papers):
            # Extract concept usage
            usage = await self._extract_concept_usage(paper, concept)
            
            if i > 0:
                # Compare with previous usage
                changes = self._identify_changes(
                    evolution.stages[-1].usage,
                    usage
                )
                evolution.changes.append(changes)
                
            evolution.stages.append(ConceptStage(
                paper=paper,
                usage=usage,
                timestamp=paper.year
            ))
            
        return evolution
    
    async def find_research_gaps(
        self, 
        domain: str, 
        recent_years: int = 3
    ) -> List[ResearchGap]:
        """Identify gaps in current research"""
        # Get recent papers in domain
        recent_papers = await self.pg.fetch("""
            SELECT p.*, array_agg(DISTINCT c.text) as claims
            FROM paper p
            LEFT JOIN claim c ON p.paper_id = c.paper_id
            WHERE p.year >= $1
              AND p.keywords && $2
            GROUP BY p.paper_id
        """, datetime.now().year - recent_years, [domain])
        
        # Analyze citation patterns
        cypher = """
        MATCH (recent:Paper)-[:CITES]->(old:Paper)
        WHERE recent.year >= $year
        WITH old, COUNT(DISTINCT recent) as citation_count
        WHERE citation_count < 3  // Under-cited
        RETURN old.paper_id, old.title, citation_count
        """
        
        undercited = await self.neo4j.run(
            cypher, 
            year=datetime.now().year - recent_years
        )
        
        # Find conceptual gaps
        gaps = []
        for paper in undercited:
            gap_analysis = await self._analyze_gap(paper, recent_papers)
            if gap_analysis.significance > 0.7:
                gaps.append(gap_analysis)
                
        return sorted(gaps, key=lambda g: g.significance, reverse=True)
```

### 8.4 Autonomous Research Assistant

```python
class AutonomousResearchAssistant:
    """Conducts autonomous literature reviews and research tasks"""
    
    def __init__(self, kb_system):
        self.kb = kb_system
        self.llm = self._init_llm()  # GPT-4, Claude, or Gemini
        self.arxiv_client = arxiv.Client()
        
    async def conduct_literature_review(
        self, 
        topic: str,
        scope: Dict[str, Any] = None
    ) -> LiteratureReview:
        """Autonomously conduct a comprehensive literature review"""
        
        review = LiteratureReview(
            topic=topic,
            started_at=datetime.now(),
            scope=scope or {}
        )
        
        # Step 1: Query existing knowledge base
        existing_papers = await self.kb.search(topic, top_k=100)
        review.existing_papers = existing_papers
        
        # Step 2: Search for new papers
        new_papers = await self._search_new_papers(topic, scope)
        
        # Step 3: Download and process new papers
        for paper in new_papers:
            if not await self._paper_exists(paper.doi):
                downloaded = await self._download_paper(paper)
                if downloaded:
                    await self.kb.ingest(downloaded)
                    review.new_papers.append(paper)
        
        # Step 4: Identify key themes
        themes = await self._identify_themes(review.all_papers)
        review.themes = themes
        
        # Step 5: Build citation network
        citation_analysis = await self._analyze_citations(review.all_papers)
        review.key_papers = citation_analysis.most_influential
        
        # Step 6: Find contradictions
        contradictions = await self._find_all_contradictions(review.all_papers)
        review.contradictions = contradictions
        
        # Step 7: Identify research gaps
        gaps = await self.kb.thread_tracker.find_research_gaps(
            topic, 
            scope.get('recent_years', 3)
        )
        review.gaps = gaps
        
        # Step 8: Generate review document
        review.document = await self._generate_review_document(review)
        
        # Step 9: Create visualizations
        review.visualizations = await self._create_visualizations(review)
        
        return review
    
    async def _search_new_papers(
        self, 
        topic: str, 
        scope: Dict
    ) -> List[Paper]:
        """Search multiple sources for new papers"""
        papers = []
        
        # arXiv search
        search = arxiv.Search(
            query=topic,
            max_results=scope.get('max_papers', 50),
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        for result in self.arxiv_client.results(search):
            papers.append(Paper(
                title=result.title,
                abstract=result.summary,
                authors=[a.name for a in result.authors],
                year=result.published.year,
                doi=result.doi,
                arxiv_id=result.entry_id,
                pdf_url=result.pdf_url
            ))
            
        # Could add: Google Scholar, PubMed, IEEE, etc.
        
        return papers
    
    async def _generate_review_document(
        self, 
        review: LiteratureReview
    ) -> str:
        """Generate comprehensive review document"""
        sections = []
        
        # Executive Summary
        summary = await self.llm.generate(f"""
        Write an executive summary for a literature review on "{review.topic}".
        Key findings:
        - {len(review.all_papers)} papers analyzed
        - {len(review.themes)} major themes identified
        - {len(review.contradictions)} contradictions found
        - {len(review.gaps)} research gaps identified
        """)
        sections.append(("Executive Summary", summary))
        
        # Introduction
        intro = await self._write_introduction(review)
        sections.append(("Introduction", intro))
        
        # Methodology
        methodology = self._describe_methodology(review)
        sections.append(("Methodology", methodology))
        
        # Thematic Analysis
        for theme in review.themes:
            theme_analysis = await self._analyze_theme(theme, review)
            sections.append((f"Theme: {theme.name}", theme_analysis))
        
        # Contradictions and Debates
        if review.contradictions:
            debates = await self._analyze_contradictions(review.contradictions)
            sections.append(("Contradictions and Debates", debates))
        
        # Research Gaps
        gaps_analysis = await self._analyze_gaps(review.gaps)
        sections.append(("Research Gaps and Future Directions", gaps_analysis))
        
        # Conclusions
        conclusions = await self._write_conclusions(review)
        sections.append(("Conclusions", conclusions))
        
        # References
        references = self._format_references(review.all_papers)
        sections.append(("References", references))
        
        # Compile document
        return self._compile_document(sections, review)
    
    async def generate_hypotheses(
        self, 
        domain: str,
        based_on: List[UUID] = None
    ) -> List[Hypothesis]:
        """Generate novel research hypotheses"""
        
        # Get relevant papers and knowledge
        if based_on:
            papers = await self.kb.get_papers(based_on)
        else:
            papers = await self.kb.search(domain, top_k=50)
        
        # Extract key findings and gaps
        findings = await self._extract_findings(papers)
        gaps = await self.kb.thread_tracker.find_research_gaps(domain)
        
        # Identify unexplored connections
        connections = await self._find_unexplored_connections(papers)
        
        # Generate hypotheses using LLM
        hypotheses = []
        
        for gap in gaps[:5]:  # Top 5 gaps
            hypothesis_prompt = f"""
            Based on this research gap in {domain}:
            {gap.description}
            
            Related findings:
            {self._summarize_findings(findings, gap)}
            
            Generate a specific, testable hypothesis that could address this gap.
            Include:
            1. Clear hypothesis statement
            2. Required experiments or studies
            3. Expected outcomes
            4. Potential impact
            """
            
            hypothesis_text = await self.llm.generate(hypothesis_prompt)
            
            hypothesis = Hypothesis(
                hypothesis_id=uuid4(),
                statement=self._extract_hypothesis_statement(hypothesis_text),
                domain=domain,
                based_on_gap=gap.gap_id,
                related_papers=[p.paper_id for p in papers[:10]],
                experiments=self._extract_experiments(hypothesis_text),
                expected_impact=self._extract_impact(hypothesis_text),
                confidence_score=self._calculate_hypothesis_confidence(
                    hypothesis_text, 
                    findings, 
                    gap
                )
            )
            
            # Validate hypothesis
            if await self._validate_hypothesis(hypothesis):
                hypotheses.append(hypothesis)
                
                # Store in database
                await self._store_hypothesis(hypothesis)
        
        # Rank by potential impact and novelty
        ranked_hypotheses = self._rank_hypotheses(hypotheses)
        
        return ranked_hypotheses
    
    async def _validate_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """Check if hypothesis is novel and feasible"""
        # Check for similar existing hypotheses
        similar = await self.kb.search(
            hypothesis.statement,
            search_types=['claim'],
            top_k=10
        )
        
        if any(s.score > 0.9 for s in similar):
            return False  # Too similar to existing work
            
        # Check feasibility
        required_resources = self._estimate_resources(hypothesis.experiments)
        if required_resources.complexity > 0.95:
            return False  # Too complex
            
        return True
    
    def _rank_hypotheses(
        self, 
        hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """Rank hypotheses by impact and novelty"""
        for hyp in hypotheses:
            # Calculate novelty score
            hyp.novelty_score = self._calculate_novelty(hyp)
            
            # Calculate potential impact
            hyp.impact_score = self._calculate_impact(hyp)
            
            # Combined score
            hyp.overall_score = (
                0.4 * hyp.novelty_score + 
                0.4 * hyp.impact_score + 
                0.2 * hyp.confidence_score
            )
            
        return sorted(hypotheses, key=lambda h: h.overall_score, reverse=True)
```

### 8.5 Multi-modal Answer Generation

```python
class MultiModalAnswerGenerator:
    """Generates comprehensive answers combining text, equations, and figures"""
    
    def __init__(self, kb_system):
        self.kb = kb_system
        self.retriever = AdvancedRetriever(kb_system.config)
        
    async def generate_answer(
        self,
        question: str,
        include_visuals: bool = True,
        max_length: int = 2000
    ) -> MultiModalAnswer:
        """Generate a comprehensive answer with multiple modalities"""
        
        # Retrieve relevant content
        results = await self.retriever.hybrid_search(
            question,
            top_k=20,
            search_types=['text', 'equation', 'claim', 'image']
        )
        
        # Group by modality
        text_results = [r for r in results if r.content_type == 'text']
        equation_results = [r for r in results if r.content_type == 'equation']
        image_results = [r for r in results if r.content_type == 'image']
        claim_results = [r for r in results if r.content_type == 'claim']
        
        # Check for contradictions in claims
        contradictions = await self._check_contradictions(claim_results)
        
        # Generate answer structure
        answer = MultiModalAnswer(question=question)
        
        # Main text answer
        answer.text = await self._generate_text_answer(
            question, 
            text_results, 
            claim_results
        )
        
        # Add relevant equations
        if equation_results:
            answer.equations = await self._select_equations(
                question, 
                equation_results, 
                max_equations=3
            )
            
        # Add supporting figures
        if include_visuals and image_results:
            answer.figures = await self._select_figures(
                question,
                image_results,
                max_figures=2
            )
            
        # Add contradiction notes if any
        if contradictions:
            answer.contradictions = contradictions
            answer.text += "\n\n**Note on contradictions:**\n"
            answer.text += self._format_contradictions(contradictions)
            
        # Add citations
        answer.citations = self._extract_citations(results)
        
        # Generate summary visualization if complex
        if len(results) > 10:
            answer.summary_visual = await self._create_summary_visual(
                results, 
                question
            )
            
        return answer
    
    async def _generate_text_answer(
        self, 
        question: str,
        text_results: List[SearchResult],
        claim_results: List[SearchResult]
    ) -> str:
        """Generate coherent text answer from results"""
        
        # Combine relevant text chunks
        context = "\n\n".join([
            f"From {r.source['title']} ({r.source['year']}): {r.content}"
            for r in text_results[:5]
        ])
        
        # Add key claims
        claims = "\n".join([
            f"- {r.content} [{r.source['title']}]"
            for r in claim_results[:5]
        ])
        
        # Use LLM to synthesize
        prompt = f"""
        Question: {question}
        
        Context from papers:
        {context}
        
        Key scientific claims:
        {claims}
        
        Generate a comprehensive answer that:
        1. Directly addresses the question
        2. Synthesizes information from multiple sources
        3. Notes any contradictions or debates
        4. Maintains scientific accuracy
        5. Uses clear, accessible language
        
        Answer:
        """
        
        answer = await self.llm.generate(prompt, max_tokens=1500)
        
        return answer
```

### 8.6 Browser Integration API

```python
# api.py - FastAPI server for browser extensions
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="LitKB API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatgpt.com", "https://claude.ai", "https://gemini.google.com"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Dict[str, Any] = {}
    format: str = "markdown"

class ReviewRequest(BaseModel):
    topic: str
    scope: Dict[str, Any] = {}
    
class HypothesisRequest(BaseModel):
    domain: str
    based_on_papers: List[str] = []

@app.post("/search")
async def search(request: SearchRequest):
    """Endpoint for browser extensions"""
    try:
        retriever = AdvancedRetriever("config.yaml")
        results = await retriever.hybrid_search(
            request.query,
            request.filters,
            request.top_k
        )
        
        formatted = retriever.format_for_llm(results, request.format)
        
        return {
            "status": "success",
            "results": formatted,
            "metadata": {
                "total_results": len(results),
                "query_time_ms": 0  # Add timing
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conduct_review")
async def conduct_review(request: ReviewRequest):
    """Autonomous literature review endpoint"""
    assistant = AutonomousResearchAssistant(kb_system)
    review = await assistant.conduct_literature_review(
        request.topic,
        request.scope
    )
    
    return {
        "status": "success",
        "review_id": str(review.review_id),
        "summary": review.executive_summary,
        "document_url": f"/reviews/{review.review_id}/document",
        "stats": {
            "papers_analyzed": len(review.all_papers),
            "new_papers_found": len(review.new_papers),
            "themes_identified": len(review.themes),
            "contradictions_found": len(review.contradictions),
            "gaps_identified": len(review.gaps)
        }
    }

@app.post("/generate_hypotheses")
async def generate_hypotheses(request: HypothesisRequest):
    """Generate research hypotheses"""
    assistant = AutonomousResearchAssistant(kb_system)
    hypotheses = await assistant.generate_hypotheses(
        request.domain,
        request.based_on_papers
    )
    
    return {
        "status": "success",
        "hypotheses": [
            {
                "id": str(h.hypothesis_id),
                "statement": h.statement,
                "confidence": h.confidence_score,
                "impact": h.impact_score,
                "novelty": h.novelty_score,
                "experiments": h.experiments
            }
            for h in hypotheses
        ]
    }

@app.get("/contradictions/{paper_id}")
async def get_contradictions(paper_id: str):
    """Get all contradictions for a paper"""
    detector = ContradictionDetector(db, embedder)
    claims = await db.fetch("SELECT claim_id FROM claim WHERE paper_id = $1", paper_id)
    
    all_contradictions = []
    for claim in claims:
        contradictions = await detector.find_contradictions(claim['claim_id'])
        all_contradictions.extend(contradictions)
        
    return {
        "paper_id": paper_id,
        "contradictions": all_contradictions
    }

@app.get("/threads/{concept}")
async def get_research_threads(concept: str):
    """Get research threads for a concept"""
    tracker = ResearchThreadTracker(db_connections)
    threads = await tracker.find_threads_for_concept(concept)
    
    return {
        "concept": concept,
        "threads": [
            {
                "thread_id": str(t.thread_id),
                "root_paper": t.root_paper_title,
                "evolution_stages": len(t.timeline),
                "key_insights": t.key_insights
            }
            for t in threads
        ]
    }

@app.get("/stats")
async def get_stats():
    """System statistics with advanced metrics"""
    return {
        "papers": await db.fetch_val("SELECT COUNT(*) FROM paper"),
        "equations": await db.fetch_val("SELECT COUNT(*) FROM equation"),
        "claims": await db.fetch_val("SELECT COUNT(*) FROM claim"),
        "contradictions": await db.fetch_val("SELECT COUNT(*) FROM relation WHERE relation = 'contradicts'"),
        "hypotheses": await db.fetch_val("SELECT COUNT(*) FROM hypothesis"),
        "research_threads": await db.fetch_val("SELECT COUNT(DISTINCT thread_id) FROM research_thread"),
        "last_update": await db.fetch_val("SELECT MAX(ingested_at) FROM paper"),
        "system_health": await get_system_health()
    }
```

---

## 9 · Conclusion

This comprehensive architecture transforms your research corpus into a living, intelligent knowledge graph that seamlessly integrates with modern LLMs. By combining robust local processing with strategic browser LLM delegation, you get the best of both worlds: complete control over your data and access to state-of-the-art AI capabilities without per-token costs.

**From day one, your system will:**
- Autonomously conduct literature reviews on any topic
- Generate novel research hypotheses from identified gaps
- Detect and track contradictions across thousands of papers
- Map the evolution of ideas through citation networks
- Provide multi-modal answers combining text, equations, and figures
- Enable collaborative research with annotation management
- Predict emerging research trends and field evolution

The system grows with your research, continuously learning from each paper to provide increasingly sophisticated insights. Whether you're tracking the evolution of quantum computing architectures, comparing conflicting claims about spacecraft shielding, or seeking inspiration for your next breakthrough, your LitKB becomes an indispensable AI research partner.

