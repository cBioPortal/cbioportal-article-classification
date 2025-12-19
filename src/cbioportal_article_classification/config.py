"""Configuration for cBioPortal article classification tool."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# cBioPortal reference papers
CBIOPORTAL_PMIDS = [
    "37668528",  # 2023 paper
    "23550210",  # 2013 paper
    "22588877",  # 2012 paper
]

# AWS Bedrock configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

# S3 configuration (optional, for future use)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")

# Classification Schema Version
# Increment this when the classification schema changes (new fields, categories, major prompt changes)
# Version History:
#   v1: Initial schema (no text_source tracking)
#   v2: Added text_source field (pdf/abstract/none)
#   v3: Improved data_source classification (not released)
#   v4: Added detailed usage tracking (usage_mode, genes_queried, features_used, analysis_location)
CLASSIFICATION_SCHEMA_VERSION = 4

# Classification categories
CLASSIFICATION_CATEGORIES = {
    "analysis_type": [
        "Mutation analysis",
        "Copy number analysis",
        "Gene expression analysis",
        "Survival analysis",
        "Pathway analysis",
        "Multi-omics integration",
        "Other"
    ],
    "cancer_type": [
        "Pan-cancer",
        "Breast cancer",
        "Lung cancer",
        "Colorectal cancer",
        "Prostate cancer",
        "Melanoma",
        "Glioma/Brain cancer",
        "Leukemia/Lymphoma",
        "Other solid tumor",
        "Other hematologic",
        "Not specified"
    ],
    "research_area": [
        "Biomarker discovery",
        "Drug response/resistance",
        "Pathway analysis",
        "Tumor evolution",
        "Immunotherapy",
        "Precision medicine",
        "Methods/Tools development",
        "Database/Resource",
        "Review/Commentary",
        "Other"
    ],
    "study_type": [
        "Original research",
        "Review",
        "Methods/Software",
        "Clinical study",
        "Meta-analysis",
        "Other"
    ],
    "data_source": [
        "TCGA",
        "ICGC",
        "MSK-IMPACT",
        "GENIE",
        "METABRIC",
        "Custom/Private data",
        "Multiple sources",
        "Not specified"
    ],
    "cbioportal_usage_mode": [
        "Data download/export",
        "Web-based analysis",
        "Web-based visualization",
        "API access",
        "Citation only"
    ],
    "cbioportal_features_used": [
        "OncoPrint",
        "Mutation Mapper",
        "Survival analysis",
        "Expression analysis",
        "Enrichment analysis",
        "Group comparison",
        "Download data",
        "Query interface",
        "Not specified"
    ],
    "analysis_location": [
        "cBioPortal platform",
        "External (downloaded data)",
        "Mixed",
        "Unclear"
    ]
}

# NCBI/PubMed settings
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")  # Required by NCBI Entrez API
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")  # Optional, increases rate limit

# Unpaywall settings (for finding open access PDFs)
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", NCBI_EMAIL)  # Required by Unpaywall API

# Fetcher settings
FETCH_DELAY_SECONDS = 0.34  # ~3 requests/second (NCBI rate limit without API key)

# PDF download settings
PDF_SOURCE_PRIORITY = ["pmc", "biorxiv", "unpaywall", "doi"]  # Order to try PDF sources
PDF_DOWNLOAD_TIMEOUT = 30  # Timeout for PDF downloads in seconds
PDF_MAX_WORKERS = 10  # Number of parallel PDF download workers
PDF_PMC_DELAY = 0.34  # Delay only for PMC downloads (NCBI rate limit)

# Classification settings
CLASSIFICATION_MAX_WORKERS = 10  # Number of parallel classification workers
CLASSIFICATION_RATE_LIMIT_DELAY = 0.1  # Delay between classifications (seconds)
