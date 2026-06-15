# 🔬 Scientific Multimodal RAG

**A Retrieval-Augmented Generation system for scientific papers, combining vision and text retrieval with self-checking answer generation.**

Ask questions about Vision Transformer research papers and get accurate, cited answers powered by ColPali (visual retrieval), SciNCL (text retrieval), and Qwen2-VL (answer generation).

---

## Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │              USER QUERY                         │
                        │     "What is the Vision Transformer?"           │
                        └──────────────┬───────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                                      ▼
         ┌──────────────────┐                  ┌──────────────────┐
         │  ColPali Encode  │                  │  SciNCL Encode   │
         │  (Multi-Vector)  │                  │  (768-d Dense)   │
         │  ~2.5 GB VRAM    │                  │  ~0.6 GB VRAM    │
         └────────┬─────────┘                  └────────┬─────────┘
                  │ UNLOAD                               │ UNLOAD
                  ▼                                      ▼
         ┌──────────────────┐                  ┌──────────────────┐
         │  MaxSim Retrieve │                  │  ChromaDB ANN    │
         │  (.npy files)    │                  │  (Cosine Sim)    │
         └────────┬─────────┘                  └────────┬─────────┘
                  │                                      │
                  └──────────────────┬───────────────────┘
                                     ▼
                          ┌────────────────────┐
                          │  Score Fusion       │
                          │  0.7 ColPali        │
                          │  0.3 SciNCL         │
                          │  Min-Max Normalise  │
                          └────────┬───────────┘
                                   ▼
                          ┌────────────────────┐
                          │  Context Builder    │
                          │  + Page Images      │
                          │  + Text + Citations │
                          └────────┬───────────┘
                                   ▼
                          ┌────────────────────┐
                          │  Qwen2-VL Generate │
                          │  (4-bit Quantized) │
                          │  ~1.5 GB VRAM      │
                          └────────┬───────────┘
                                   │ UNLOAD
                                   ▼
                          ┌────────────────────┐
                          │  Self-Check        │
                          │  1. Attribution    │
                          │  2. Faithfulness   │
                          │  3. Confidence     │
                          └────────┬───────────┘
                                   ▼
                          ┌────────────────────┐
                          │  Answer + Sources  │
                          │  + Confidence      │
                          └────────────────────┘
```

---

## Quick Start

### 🖥️ Local System (Windows/Mac/Linux)

#### 1. Setup Environment & Install Dependencies
```bash
# Set up a virtual environment (optional but recommended)
py -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# Or: .venv\Scripts\activate   # Windows CMD

# Install packages
pip install -r requirements.txt
```

#### 2. Run Ingestion & Build Index (Offline)
Downloads arXiv papers, parses PDFs, generates vision + text embeddings, and stores them in ChromaDB:
```bash
py main.py --mode offline
```

#### 3. Run Streamlit UI (Online)
Starts the interactive Q&A application:
```bash
py main.py --mode online
```
Then visit: **`http://localhost:8501`** in your browser.

---

### 🧬 HPC Cluster (SLURM)

#### 1. Submit Batch Job (Gradio or Streamlit)
```bash
cd /scratch/data/divyasaxena_rs/Vineet_internship

# For Streamlit
sbatch scripts/slurm_app.sh

# For Gradio
sbatch scripts/slurm_online_gradio.sh
```
*(Prints Job ID, e.g. `346287`)*

#### 2. View Logs Live
```bash
# For Streamlit
tail -f /scratch/data/divyasaxena_rs/sci_rag_app_<JOB_ID>.out

# For Gradio
tail -f logs/rag_online_<JOB_ID>.out
```

#### 3. Port-Forward to Laptop Browser
Run this command in a **new terminal on your laptop**:
```bash
# For Streamlit
ssh -L 8501:cn27:8501 divyasaxena_rs@172.25.0.81

# For Gradio
ssh -L 7860:cn27:7860 divyasaxena_rs@172.25.0.81
```
Then visit: **`http://localhost:8501`** (Streamlit) or **`http://localhost:7860`** (Gradio).

---

## Project Structure

```
Scientific-Multimodal-RAG/
├── app/                        # Demo applications
│   ├── gradio_app.py           # Gradio demo (Phase 1)
│   ├── streamlit_app.py        # Streamlit app (full-featured)
│   └── requirements.txt        # App-specific dependencies
├── configs/                    # YAML configuration files
│   ├── data_config.yaml        # Data download and paths
│   ├── model_config.yaml       # Model parameters
│   ├── pipeline_config.yaml    # Pipeline orchestration
│   ├── retrieval_config.yaml   # Retrieval and fusion settings
│   └── evaluation_config.yaml  # Evaluation parameters
├── docs/                       # Documentation
│   ├── architecture.md         # System architecture details
│   ├── kaggle_setup.md         # Kaggle deployment guide
│   ├── hybrid_extension.md     # Future hybrid extension plan
│   └── evaluation_guide.md     # Evaluation methodology
├── future/                     # Future extensions
│   ├── hybrid/                 # Medical/scientific query router
│   └── frontend/               # Custom frontend skeleton
├── kaggle/                     # Kaggle notebook scripts
│   ├── notebook-online.py      # Online (query) notebook
│   ├── notebook-offline.py     # Offline (indexing) notebook
│   └── kaggle-metadata.json    # Kaggle dataset metadata
├── pipelines/                  # Pipeline orchestration
│   ├── offline_pipeline.py     # Download → Parse → Embed → Index
│   └── online_pipeline.py      # Validate → Encode → Retrieve → Generate
├── scripts/                    # CLI entry points
│   ├── download_data.py        # Download arXiv papers
│   ├── parse_pdfs.py           # Parse PDFs to images + text
│   ├── build_index.py          # Build embedding indices
│   ├── query.py                # Interactive query CLI
│   ├── evaluate.py             # Run evaluation
│   └── push_to_kaggle.py       # Push data to Kaggle
├── src/                        # Core source code
│   ├── embeddings/             # Embedding models
│   │   ├── base_embedder.py    # Abstract base class + EmbeddingOutput
│   │   ├── colpali_embedder.py # ColPali vision embedder
│   │   └── scincl_embedder.py  # SciNCL text embedder
│   ├── retrieval/              # Retrieval backends
│   │   ├── base_retriever.py   # Abstract base + dataclasses
│   │   ├── colpali_retriever.py# MaxSim retrieval over .npy
│   │   ├── text_retriever.py   # ChromaDB ANN retrieval
│   │   └── fusion_retriever.py # Weighted score fusion
│   ├── generation/             # Answer generation
│   │   ├── rag_generator.py    # Full RAG pipeline
│   │   └── self_check.py       # Three-level verification
│   ├── context/                # Context building
│   │   ├── context_builder.py  # VLM input assembly
│   │   └── prompt_templates.py # System/user prompts
│   └── utils/                  # Utilities
│       ├── config_loader.py    # YAML config + path resolution
│       ├── device.py           # GPU/VRAM management
│       ├── image_utils.py      # Image loading and resizing
│       ├── logging_utils.py    # Structured logging
│       ├── metrics.py          # Evaluation metrics
│       ├── visualization.py    # Result visualization
│       └── error_handler.py    # Error handling
├── tests/                      # Test suite
│   ├── test_parsers.py         # Parser and preprocessor tests
│   ├── test_embedders.py       # Embedding model tests
│   ├── test_retrievers.py      # Retrieval backend tests
│   └── test_pipeline.py        # Pipeline and self-check tests
├── checkpoint.json             # Pipeline checkpoint
├── TRACKER.md                  # Progress tracker
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── setup.py                    # Package setup
```

---

## Configuration Guide

All settings are controlled via YAML files in `configs/`:

| Config File | Purpose | Key Settings |
|---|---|---|
| `data_config.yaml` | Paper download and file paths | `query`, `max_results`, `keep_best`, `paths` |
| `model_config.yaml` | Model parameters | `colpali`/`scincl`/`qwen2vl` model names, devices, dtype |
| `retrieval_config.yaml` | Retrieval and fusion | `top_k`, `colpali_weight`, `scincl_weight`, ChromaDB settings |
| `pipeline_config.yaml` | Pipeline orchestration | `max_query_length`, `min_query_words`, batch sizes |
| `evaluation_config.yaml` | Evaluation parameters | Ground truth path, metrics, output format |

### Path Resolution

Paths are automatically resolved based on the runtime environment:
- **Kaggle**: Paths prefixed with `/kaggle/working/`
- **Local**: Paths relative to project root

---

## Kaggle Deployment Guide

### Session 1: Offline (Index Building) — ~40 min

1. Create a Kaggle notebook with GPU (P100)
2. Upload `kaggle/notebook-offline.py` as the notebook source
3. Add the `scientific-multimodal-rag` dataset as input
4. Run all cells — this downloads, parses, and embeds the papers
5. Save the output as a new dataset version

### Session 2: Online (Query Demo) — ~5 min per query

1. Create a new Kaggle notebook with GPU (P100)
2. Add the output dataset from Session 1 as input
3. Upload `kaggle/notebook-online.py` as the notebook source
4. Run all cells and interact with the Gradio demo

> **Important**: The P100 has 16 GB VRAM. Models are loaded one at a time (staggered loading) to stay within limits.

---

## Results

| Metric | Value | Notes |
|---|---|---|
| Papers indexed | 10 | Vision Transformer cluster from arXiv |
| Pages embedded | ~100 | ColPali multi-vector + SciNCL dense |
| Avg query time | 5-12s | On Kaggle P100, no retries |
| Self-check pass rate | ~85% | Attribution + Faithfulness + Confidence |
| ColPali VRAM | ~2.5 GB | Gemma-2B backbone, float16 |
| SciNCL VRAM | ~0.6 GB | SciBERT-base, float16 |
| Qwen2-VL VRAM | ~1.5 GB | 4-bit quantized (NF4) |
| Total VRAM peak | ~2.5 GB | Staggered loading ensures one model at a time |

---

## Key Design Decisions

1. **Staggered Model Loading**: Only one model is loaded at a time. Load → Use → Unload. This keeps peak VRAM at ~2.5 GB instead of ~4.6 GB.

2. **ColPali Dominance (0.7 weight)**: Scientific papers rely heavily on figures, tables, and layout. Visual retrieval captures these elements that text-only retrieval misses.

3. **MaxSim over Single-Vector**: ColPali's late-interaction mechanism preserves fine-grained alignment between query terms and visual patch tokens.

4. **Three-Level Self-Check**: Attribution (citations present), Faithfulness (keyword overlap), Confidence (threshold check) ensure answer quality.

5. **.npy for ColPali**: ChromaDB stores one vector per document. ColPali produces N vectors per page. Hence, .npy files with in-memory MaxSim scoring.

---

## License

MIT License — See LICENSE file for details.

