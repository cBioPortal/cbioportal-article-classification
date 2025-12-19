# cBioPortal Article Classification

> **⚠️ WORK IN PROGRESS**
> This tool is currently under active development and was vibe-coded. Use at your own risk!
> Expect bugs, breaking changes, and incomplete features.

## 📊 See the Latest Results

**[→ View the latest usage analysis report](output/reports/usage_report.md)**

This report includes embedded visualizations, analysis types, research areas, cancer types, temporal trends, and detailed summaries of how researchers use cBioPortal.

---

A comprehensive tool for fetching, classifying, and analyzing scientific papers that cite cBioPortal using LLM-based classification.

## Features

- **Automated Citation Fetching**: Retrieves citations from PubMed with incremental updates
- **PDF Download**: Attempts to download full-text PDFs from multiple sources
- **LLM-Based Classification**: Uses Claude (via AWS Bedrock) with structured outputs
- **Rich Analysis**: Generates visualizations and markdown reports

## Quick Start

### Prerequisites
- Python 3.11+
- AWS account with Bedrock access
- Email address (required for NCBI Entrez API)

### Installation

```bash
# Clone and install
cd cbioportal-article-classification
uv sync

# Activate virtual environment
source .venv/bin/activate  # or activate.fish for fish

# Configure environment
cp .env.example .env
# Edit .env with your NCBI_EMAIL and AWS settings
```

Required in `.env`:
```bash
NCBI_EMAIL=your.email@example.com
AWS_REGION=us-east-1
AWS_PROFILE=default
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

## Usage

```bash
# Check status
cbioportal-classify status

# Fetch citations and download PDFs
cbioportal-classify fetch --download-pdfs --max-downloads 50

# Classify papers using LLM
cbioportal-classify classify --max-papers 20

# Generate analysis report
cbioportal-classify analyze

# Or run everything at once
cbioportal-classify run-all
```

## Key Commands

| Command | Description |
|---------|-------------|
| `status` | Show database state (citations, PDFs, classifications) |
| `fetch` | Fetch new citations from PubMed |
| `fetch --download-pdfs` | Also download PDFs for full-text analysis |
| `classify` | Classify papers using Claude via AWS Bedrock |
| `classify --reclassify` | Force re-classification of existing papers |
| `analyze` | Generate visualizations and markdown report |
| `run-all` | Execute complete pipeline |

## Output Files

- `output/reports/usage_report.md` - Analysis report with embedded visualizations
- `output/plots/` - PNG visualizations (usage analysis, research areas)
- `data/metadata/classifications.json` - Raw classification data
- `data/metadata/classifications.csv` - Classifications in CSV format

## How It Works

1. **Fetcher**: Uses NCBI's Entrez API to find papers citing cBioPortal papers, downloads PDFs from PMC/bioRxiv/Unpaywall
2. **Classifier**: Uses Claude via AWS Bedrock with instructor + Pydantic for structured outputs, classifies by analysis type, cancer type, research area, etc.
3. **Analyzer**: Generates temporal trends, breakdowns by category, and comprehensive markdown reports with embedded plots

## Configuration

Edit `src/cbioportal_article_classification/config.py` to:
- Add/modify classification categories
- Change cBioPortal PMIDs being tracked
- Adjust AWS Bedrock model

## Troubleshooting

**NCBI rate limits**: Get a free [NCBI API key](https://www.ncbi.nlm.nih.gov/account/settings/) for 10 req/sec instead of 3 req/sec

**PDF downloads fail**: Tool will still classify using abstracts. PDFs are often behind paywalls.

**AWS Bedrock errors**: Ensure credentials have Bedrock access and model ID is correct for your region

## License

MIT

## Citation

If you use this tool, please cite the cBioPortal papers:
- de Bruijn et al. (2023) Cancer Res. PMID: 37668528
- Gao et al. (2013) Sci. Signal. PMID: 23550210
- Cerami et al. (2012) Cancer Discov. PMID: 22588877
