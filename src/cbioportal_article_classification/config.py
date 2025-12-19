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
    ]
}

# NCBI/PubMed settings
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")  # Required by NCBI Entrez API
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")  # Optional, increases rate limit

# Unpaywall settings (for finding open access PDFs)
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", NCBI_EMAIL)  # Required by Unpaywall API

# Fetcher settings
MAX_CITATIONS_PER_PAPER = 1000  # Limit per PMID to avoid overwhelming requests
FETCH_DELAY_SECONDS = 0.34  # ~3 requests/second (NCBI rate limit without API key)

# PDF download settings
PDF_SOURCE_PRIORITY = ["pmc", "biorxiv", "unpaywall", "doi"]  # Order to try PDF sources
PDF_DOWNLOAD_TIMEOUT = 30  # Timeout for PDF downloads in seconds
