# cBioPortal Article Classification

> **⚠️ WORK IN PROGRESS**
> This tool is currently under active development and was vibe-coded. Use at your own risk!
> Expect bugs, breaking changes, and incomplete features.

## 📊 See the Latest Results

**[→ View the latest usage analysis report](output/reports/latest.md)**

This report includes:
- Classification of 133+ papers citing cBioPortal
- Embedded visualizations and plots
- Analysis types, research areas, and cancer types studied
- Temporal trends showing citation patterns over time
- Detailed summaries of how researchers use cBioPortal

---

A comprehensive tool for fetching, classifying, and analyzing scientific papers that cite cBioPortal. This tool helps track how researchers are using cBioPortal across different cancer types, analysis methods, and research areas.

## Features

- **Automated Citation Fetching**: Retrieves citations from PubMed for specified cBioPortal papers
- **Incremental Updates**: Only fetches new articles since the last run
- **PDF Download**: Attempts to download PDFs for full-text analysis
- **LLM-Based Classification**: Uses Claude (via AWS Bedrock) with structured outputs (instructor + Pydantic) to classify papers
- **Rich Analysis**: Generates visualizations and reports showing usage patterns over time
- **CLI Interface**: Clean command-line interface built with Click

## Installation

### Prerequisites

- Python 3.11+
- AWS account with Bedrock access configured
- AWS CLI configured with appropriate credentials
- Email address (required for NCBI Entrez API)

### Setup

1. Clone the repository:
```bash
cd cbioportal-article-classification
```

2. Install using uv:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
# For bash/zsh:
source .venv/bin/activate

# For fish:
source .venv/bin/activate.fish
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

Required environment variables:
```
# REQUIRED: Your email for NCBI (they ask for this to contact you if there are issues)
NCBI_EMAIL=your.email@example.com

# AWS Bedrock settings
AWS_REGION=us-east-1
AWS_PROFILE=default
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

Optional: Get an [NCBI API key](https://www.ncbi.nlm.nih.gov/account/settings/) for higher rate limits (10 req/sec instead of 3 req/sec):
```
NCBI_API_KEY=your_api_key_here
```

## Usage

**Important**: Always activate the virtual environment first before running commands:
```bash
source .venv/bin/activate  # or activate.fish for fish shell
```

The tool provides a CLI with several commands:

### Check Status

```bash
cbioportal-classify status
```

Shows current state of your database including:
- Total citations fetched
- PDFs downloaded
- Papers classified
- Last run information

### Fetch Citations

Fetch new citations from PubMed:

```bash
# Fetch citations only
cbioportal-classify fetch

# Fetch citations and download PDFs (limit to 50)
cbioportal-classify fetch --download-pdfs --max-downloads 50

# Force re-fetch all citations
cbioportal-classify fetch --force
```

### Classify Papers

Classify papers using Claude via AWS Bedrock:

```bash
# Classify up to 20 papers
cbioportal-classify classify --max-papers 20

# Re-classify existing papers
cbioportal-classify classify --reclassify

# Classify all unclassified papers
cbioportal-classify classify
```

### Analyze Results

Generate visualizations and reports:

```bash
# Full analysis with plots and report
cbioportal-classify analyze

# Report only (skip plots)
cbioportal-classify analyze --skip-plots
```

### Run Complete Pipeline

Run all steps in sequence:

```bash
# Fetch, classify, and analyze
cbioportal-classify run-all

# With PDF downloads and limits
cbioportal-classify run-all --download-pdfs --max-downloads 50 --max-papers 20
```

## Project Structure

```
cbioportal-article-classification/
├── src/cbioportal_article_classification/
│   ├── __init__.py
│   ├── config.py          # Configuration and settings
│   ├── fetcher.py         # Citation fetching and PDF download
│   ├── classifier.py      # LLM-based classification
│   ├── analyzer.py        # Analysis and visualization
│   └── main.py            # CLI entry point
├── data/
│   ├── pdfs/              # Downloaded PDFs
│   └── metadata/          # JSON files with citations and classifications
├── output/
│   ├── reports/           # Generated markdown reports
│   └── plots/             # Visualization plots
├── pyproject.toml
└── README.md
```

## Classification Categories

Papers are classified into the following categories:

### Analysis Type
- Mutation analysis
- Copy number analysis
- Gene expression analysis
- Survival analysis
- Pathway analysis
- Multi-omics integration
- Other

### Cancer Type
- Pan-cancer, Breast cancer, Lung cancer, etc.
- Specific tumor types
- Not specified

### Research Area
- Biomarker discovery
- Drug response/resistance
- Pathway analysis
- Tumor evolution
- Immunotherapy
- Precision medicine
- Methods/Tools development
- Database/Resource
- Review/Commentary

### Study Type
- Original research
- Review
- Methods/Software
- Clinical study
- Meta-analysis

### Data Source
- TCGA, ICGC, MSK-IMPACT, GENIE, METABRIC
- Custom/Private data
- Multiple sources

## Output Files

### Metadata Files

- `data/metadata/citations.json`: All fetched citations with metadata
- `data/metadata/classifications.json`: Classification results
- `data/metadata/classifications.csv`: Classifications in CSV format
- `data/metadata/last_run.json`: Information about last fetch

### Reports and Visualizations

- `output/reports/usage_report_YYYYMMDD.md`: Comprehensive markdown report
- `output/plots/usage_analysis_YYYYMMDD.png`: Multi-panel visualization
- `output/plots/research_areas_YYYYMMDD.png`: Research area breakdown

## Architecture

### Fetcher Module

Uses NCBI's Entrez API (via BioPython) to:
- Find papers citing cBioPortal papers using `elink`
- Retrieve comprehensive metadata using `efetch`
- Extract DOIs, abstracts, and author information
- Track which papers have been fetched
- Download PDFs when available (via DOI)

### Classifier Module

Uses **instructor** + **Pydantic** for structured outputs:
- Defines `PaperClassification` Pydantic model
- Uses instructor to patch Anthropic Bedrock client
- Automatically validates and parses LLM responses
- Extracts structured data from paper text

### Analyzer Module

Uses pandas, matplotlib, and seaborn to:
- Aggregate classification results
- Generate temporal trends
- Create visualizations
- Produce comprehensive reports

## Example Workflow

```bash
# Activate virtual environment
source .venv/bin/activate

# 1. Check current status
cbioportal-classify status

# 2. Fetch new citations (limit downloads for testing)
cbioportal-classify fetch --download-pdfs --max-downloads 10

# 3. Classify papers (limit for testing)
cbioportal-classify classify --max-papers 5

# 4. Generate analysis
cbioportal-classify analyze

# 5. Check the report
cat output/reports/usage_report_*.md
```

## Periodic Updates

For regular monitoring, you can run the tool periodically (e.g., monthly):

```bash
# Cron job example (monthly on the 1st at 2am)
0 2 1 * * cd /path/to/cbioportal-article-classification && cbioportal-classify run-all --download-pdfs --max-downloads 100 --max-papers 50
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black src/
uv run ruff check src/
```

## Configuration

Edit `src/cbioportal_article_classification/config.py` to:
- Add/modify classification categories
- Change cBioPortal PMIDs being tracked
- Adjust rate limits and delays
- Modify AWS Bedrock settings

## Troubleshooting

### NCBI API Errors

If you see errors about NCBI email:
- Make sure `NCBI_EMAIL` is set in your `.env` file
- Use a valid email address (NCBI requires this)

If you hit rate limits:
- Get an NCBI API key (free) for 10 req/sec instead of 3 req/sec
- Add `NCBI_API_KEY` to your `.env` file
- Reduce batch sizes if still having issues

### AWS Bedrock Errors

- Ensure your AWS credentials have Bedrock access
- Check that the model ID is correct for your region
- Verify quota limits in AWS console

### PDF Download Failures

- PDFs are often behind paywalls
- The tool will still classify based on abstracts
- Consider using Unpaywall API for open access papers

## Future Enhancements

- [ ] S3 storage integration for PDFs
- [ ] Support for additional LLM providers
- [ ] Web dashboard for results
- [ ] Email notifications for periodic reports
- [ ] Export to other formats (Excel, JSON)
- [ ] Integration with additional citation sources (Scopus, Web of Science)

## License

MIT

## Citation

If you use this tool in your research, please cite the cBioPortal papers:
- Gao et al. (2013) Sci. Signal. PMID: 23550210
- Cerami et al. (2012) Cancer Discov. PMID: 22588877
- de Bruijn et al. (2023) Cancer Res. PMID: 37668528
