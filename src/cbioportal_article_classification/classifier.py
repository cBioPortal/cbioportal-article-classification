"""Module for classifying papers using AWS Bedrock (Claude) with instructor."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import time
from datetime import datetime
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from anthropic import AnthropicBedrock
import instructor
from pydantic import BaseModel, Field
import pandas as pd
import pdfplumber
from tqdm import tqdm

from .config import (
    AWS_REGION,
    AWS_PROFILE,
    BEDROCK_MODEL_ID,
    CLASSIFICATION_CATEGORIES,
    CLASSIFICATION_SCHEMA_VERSION,
    CLASSIFICATION_MAX_WORKERS,
    CLASSIFICATION_RATE_LIMIT_DELAY,
    METADATA_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pdfminer warnings for malformed PDFs
logging.getLogger('pdfminer').setLevel(logging.ERROR)


class PaperClassification(BaseModel):
    """Structured output for paper classification."""

    analysis_type: List[str] = Field(
        description="Types of analysis performed (e.g., mutation analysis, survival analysis)"
    )
    oncotree_code: Optional[str] = Field(
        default=None,
        description="OncoTree tumor type code (e.g., STAD, BRCA, LUAD). Use 'PAN' for pan-cancer studies, None if not specified"
    )
    oncotree_name: Optional[str] = Field(
        default=None,
        description="OncoTree tumor type name (e.g., Stomach Adenocarcinoma, Breast Invasive Carcinoma). None if not specified"
    )
    cancer_type_category: Optional[str] = Field(
        default=None,
        description="Broader cancer category (e.g., Gastrointestinal, Breast, Lung, Hematologic, Pan-Cancer). None if not specified"
    )
    research_area: List[str] = Field(
        description="Research areas (e.g., biomarker discovery, drug response)"
    )
    study_type: List[str] = Field(
        description="Type of study (e.g., original research, review)"
    )
    data_source: List[str] = Field(
        description="Data sources used from cBioPortal (e.g., TCGA, MSK-IMPACT)"
    )
    cbioportal_usage_summary: str = Field(
        description="2-3 sentence summary of how cBioPortal was used in this study"
    )
    specific_datasets: List[str] = Field(
        description="Specific dataset names mentioned (e.g., TCGA-BRCA, GENIE-MSK)"
    )
    cbioportal_usage_mode: List[str] = Field(
        description="How cBioPortal was used: data download, web analysis, visualization, API, or citation only"
    )
    specific_genes_queried: List[str] = Field(
        description="Specific genes mentioned as queried or analyzed (e.g., TP53, KRAS, EGFR). Empty list if none specified."
    )
    cbioportal_features_used: List[str] = Field(
        description="Specific cBioPortal features/tools used (OncoPrint, Mutation Mapper, etc.)"
    )
    analysis_location: str = Field(
        description="Where analysis was performed: cBioPortal platform, External, Mixed, or Unclear"
    )
    cites_study_data_source_paper: bool = Field(
        description="Whether the paper cites any underlying data source papers from cBioPortal studies (e.g., TCGA consortium papers, METABRIC papers)"
    )
    study_data_source_pmids_cited: List[str] = Field(
        description="List of PMIDs of study data source papers cited (empty list if none). Only include PMIDs that are reference publications for cBioPortal studies."
    )
    study_data_source_papers_cited: List[str] = Field(
        description="Names of study data sources cited (e.g., 'TCGA Pan-Cancer Atlas', 'METABRIC'). Empty list if none cited."
    )
    cites_cbioportal_platform_paper: bool = Field(
        description="Whether the paper cites any of the 3 cBioPortal platform reference papers (Cerami 2012, Gao 2013, or de Bruijn 2023)"
    )
    cbioportal_platform_pmids_cited: List[str] = Field(
        description="List of PMIDs of cBioPortal platform papers cited: 22588877 (Cerami 2012), 23550210 (Gao 2013), 37668528 (de Bruijn 2023). Empty list if none."
    )
    cbioportal_platform_papers_cited: List[str] = Field(
        description="Names of cBioPortal platform papers cited (e.g., 'Cerami et al. 2012', 'Gao et al. 2013', 'de Bruijn et al. 2023'). Empty list if none cited."
    )
    confidence: str = Field(
        description="Confidence level in the classification: high, medium, or low"
    )


class PaperClassifier:
    """Classifies papers using AWS Bedrock with Claude and instructor."""

    def __init__(self):
        """Initialize the classifier with instructor-patched Bedrock client."""
        # Store configuration for creating thread-local clients
        self.aws_region = AWS_REGION
        self.aws_profile = AWS_PROFILE
        # Thread-local storage for clients
        self._thread_local = threading.local()

        self.classifications_file = METADATA_DIR / "classifications.json"
        self.classifications = self._load_classifications()
        self.citation_sentences_file = METADATA_DIR / "citation_sentences.json"
        self.citation_sentences = self._load_citation_sentences()
        self.data_source_papers_file = METADATA_DIR / "data_source_papers.json"
        self.data_source_papers = self._load_data_source_papers()

    def _get_client(self):
        """Get or create a thread-local instructor-patched Bedrock client."""
        if not hasattr(self._thread_local, 'client'):
            base_client = AnthropicBedrock(
                aws_region=self.aws_region,
                aws_profile=self.aws_profile,
            )
            # Patch the client with instructor for structured outputs
            self._thread_local.client = instructor.from_anthropic(base_client)
        return self._thread_local.client

    def _load_citation_sentences(self) -> Dict:
        """Load citation sentences from disk."""
        if self.citation_sentences_file.exists():
            with open(self.citation_sentences_file, "r") as f:
                return json.load(f)
        return {}

    def _load_data_source_papers(self) -> Dict:
        """Load data source papers mapping from disk."""
        if self.data_source_papers_file.exists():
            with open(self.data_source_papers_file, "r") as f:
                return json.load(f)
        logger.warning(f"Data source papers file not found: {self.data_source_papers_file}")
        logger.warning("Run 'fetch-reference-data' to download data source paper metadata")
        return {}

    def _load_classifications(self) -> Dict:
        """Load existing classifications from disk."""
        if self.classifications_file.exists():
            with open(self.classifications_file, "r") as f:
                return json.load(f)
        return {}

    def _save_classifications(self):
        """Save classifications to disk."""
        with open(self.classifications_file, "w") as f:
            json.dump(self.classifications, indent=2, fp=f)

    def _save_csv(self):
        """Save classifications to CSV format."""
        if self.classifications:
            df = pd.DataFrame(list(self.classifications.values()))
            csv_path = METADATA_DIR / "classifications.csv"
            df.to_csv(csv_path, index=False)
            logger.debug(f"Saved {len(df)} classifications to CSV")

    def extract_text_from_pdf(self, pdf_path: Path, max_pages: int = 10) -> str:
        """Extract text from PDF file (first + last pages for better coverage).

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to extract (to avoid huge texts)

        Returns:
            Extracted text content
        """
        try:
            text_parts = []

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                # Extract first half + last half to capture both intro/methods
                # and discussion/conclusion sections
                first_n = max_pages // 2
                last_n = max_pages // 2

                # Build list of page indices to extract
                pages_to_extract = []

                # Add first N pages
                pages_to_extract.extend(range(min(first_n, total_pages)))

                # Add last N pages (avoid duplicates if paper is short)
                if total_pages > first_n:
                    last_start = max(first_n, total_pages - last_n)
                    pages_to_extract.extend(range(last_start, total_pages))

                # Extract text from selected pages
                for page_num in pages_to_extract:
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def build_classification_prompt(self, paper_data: Dict, paper_text: str, text_source: str = "abstract", reference_pmids: List[str] = None) -> str:
        """Build the prompt for Claude to classify the paper.

        Args:
            paper_data: Paper metadata
            paper_text: Extracted text from paper
            text_source: Source of the text ("pdf", "abstract", "sentences", "pdf+sentences")
            reference_pmids: List of PMIDs this paper cites (bibliography)

        Returns:
            Formatted prompt for the LLM
        """
        if reference_pmids is None:
            reference_pmids = []

        # Build data source information for the prompt
        data_source_info = "None available (run 'fetch-reference-data' to load data source papers)"
        if self.data_source_papers:
            # Show a sample of important data source papers (limit to top 20 to avoid overwhelming the prompt)
            data_source_lines = []
            for pmid, info in list(self.data_source_papers.items())[:20]:
                title = info.get('title', 'Unknown')[:80]
                studies = info.get('studies', [])
                study_names = ', '.join(studies[:3])  # Show first 3 studies
                if len(studies) > 3:
                    study_names += f' (+{len(studies)-3} more)'
                data_source_lines.append(f"- PMID {pmid}: {title}... (Studies: {study_names})")
            data_source_info = "\n".join(data_source_lines)
            if len(self.data_source_papers) > 20:
                data_source_info += f"\n... and {len(self.data_source_papers) - 20} more data source papers"

        # Build category descriptions, excluding cancer_type (replaced by OncoTree)
        categories_desc = "\n\n".join([
            f"**{cat_name.replace('_', ' ').title()}**:\n" +
            ", ".join(f'"{opt}"' for opt in options)
            for cat_name, options in CLASSIFICATION_CATEGORIES.items()
            if cat_name != "cancer_type"  # Skip cancer_type, we'll use OncoTree instead
        ])

        # Adapt the text section label based on source
        if text_source == "sentences":
            text_label = "Citation Contexts (how cBioPortal was mentioned/used in this paper)"
        elif text_source == "pdf+sentences":
            text_label = "Paper Text and Citation Contexts"
        elif text_source == "pdf":
            text_label = "Paper Text (Abstract and Introduction)"
        else:  # abstract
            text_label = "Paper Abstract"

        prompt = f"""You are analyzing a scientific paper that cites cBioPortal, a cancer genomics data portal and analysis platform.

Paper Title: {paper_data.get('title', 'N/A')}
Authors: {paper_data.get('authors', 'N/A')}
Year: {paper_data.get('year', 'N/A')}
Venue: {paper_data.get('venue', 'N/A')}

{text_label}:
{paper_text[:8000]}

Please classify this paper according to the following categories. For each category, select the most appropriate option(s).

{categories_desc}

**Cancer Type** (OncoTree classification):
Identify the specific cancer type studied using OncoTree codes. Common codes include:
- Breast: BRCA (Breast Invasive Carcinoma), IDC (Breast Invasive Ductal Carcinoma)
- Lung: LUAD (Lung Adenocarcinoma), LUSC (Lung Squamous Cell Carcinoma), NSCLC (Non-Small Cell Lung Cancer)
- Colorectal: COADREAD (Colorectal Adenocarcinoma), COAD (Colon Adenocarcinoma), READ (Rectum Adenocarcinoma)
- Prostate: PRAD (Prostate Adenocarcinoma)
- Gastric: STAD (Stomach Adenocarcinoma), EGC (Esophagogastric Adenocarcinoma)
- Brain: GBM (Glioblastoma), LGG (Lower Grade Glioma)
- Ovarian: OV (Ovarian Serous Cystadenocarcinoma), HGSOC (High-Grade Serous Ovarian Cancer)
- Hematologic: AML (Acute Myeloid Leukemia), ALL (Acute Lymphoblastic Leukemia), CLL (Chronic Lymphocytic Leukemia)
- Melanoma: SKCM (Cutaneous Melanoma), MEL (Melanoma)
- Pan-cancer: PAN (for pan-cancer studies across multiple cancer types)

If the exact OncoTree code is unclear, provide the closest match or None if cancer type is not specified.
Also provide a broader cancer category (e.g., "Gastrointestinal", "Breast", "Lung", "Hematologic", "Brain", "Pan-Cancer").

Instructions:
1. Read the paper content carefully
2. For each category, select the option(s) that best describe the paper
3. Provide a brief summary (2-3 sentences) of how cBioPortal was used in this study
4. Extract specific information about cBioPortal usage:

**cBioPortal Usage Mode** (select all that apply):
- "Data download/export" if they downloaded data for external analysis
- "Web-based analysis" if they used cBioPortal's analysis features on the website
- "Web-based visualization" if they used visualization tools (OncoPrint, plots)
- "API access" if they accessed data programmatically via API
- "Citation only" if usage is unclear or only cited in references

**Specific Genes Queried**: Extract gene symbols explicitly mentioned as being queried or analyzed through cBioPortal (e.g., TP53, KRAS, EGFR, BRCA1). Use standard gene symbols in uppercase. Leave empty if no specific genes mentioned.

**cBioPortal Features Used** (select all that apply):
- "OncoPrint" if they mention using OncoPrint for mutation/CNA visualization
- "Mutation Mapper" if they mention lollipop plots or mutation mapper
- "Survival analysis" if they used Kaplan-Meier plots or survival tools
- "Expression analysis" if they queried mRNA/protein expression
- "Enrichment analysis" if they used pathway/GO enrichment features
- "Group comparison" if they compared patient cohorts
- "Download data" if they downloaded bulk data
- "Query interface" if they used the web query interface
- "Not specified" if unclear

**Analysis Location**:
- "cBioPortal platform" if analysis was performed on cBioPortal
- "External (downloaded data)" if they downloaded data and analyzed elsewhere
- "Mixed" if they did both
- "Unclear" if not specified

**Specific Datasets**: Extract exact dataset names (e.g., TCGA-BRCA, MSK-IMPACT, GENIE-MSK)

**Citation Tracking** (two separate types):

Paper's reference list (PMIDs cited by this paper): {reference_pmids}

1. **Study Data Source Papers** (TCGA, METABRIC, etc.):
Check if this paper cites the underlying data source papers for datasets they used from cBioPortal.

Known data source papers from cBioPortal studies:
{data_source_info}

- Set `cites_study_data_source_paper` to true if ANY references match known data source PMIDs
- List matching PMIDs in `study_data_source_pmids_cited`
- List study names in `study_data_source_papers_cited` (e.g., "TCGA Pan-Cancer Atlas", "METABRIC")
- If no matches, set to false and leave lists empty

2. **cBioPortal Platform Papers** (the tool itself):
Check if this paper cites any of the 3 cBioPortal platform reference papers:
- PMID 22588877: Cerami et al. 2012 (Cancer Discovery)
- PMID 23550210: Gao et al. 2013 (Science Signaling)
- PMID 37668528: de Bruijn et al. 2023 (Cancer Cell)

- Set `cites_cbioportal_platform_paper` to true if ANY platform papers are cited
- List matching PMIDs in `cbioportal_platform_pmids_cited`
- List paper names in `cbioportal_platform_papers_cited` (e.g., "Cerami et al. 2012", "Gao et al. 2013")
- If no matches, set to false and leave lists empty

**Confidence**: Set to "high" if clear usage is described with specific details, "medium" if mentioned but details unclear, "low" if only cited in references

Remember: Be specific and conservative. Only include information that is explicitly stated in the paper."""

        return prompt

    def classify_paper(self, paper_data: Dict, paper_text: Optional[str] = None, text_source: str = "abstract", reference_pmids: List[str] = None) -> Dict:
        """Classify a single paper using Claude via Bedrock with instructor.

        Args:
            paper_data: Paper metadata dictionary
            paper_text: Optional pre-extracted text. If None, uses abstract from metadata
            text_source: Source of the text ("pdf", "abstract", or "none")
            reference_pmids: List of PMIDs this paper cites (bibliography)

        Returns:
            Classification results dictionary
        """
        if not paper_text:
            paper_text = paper_data.get("abstract", "")
            text_source = "abstract" if paper_text else "none"

        if not paper_text:
            logger.warning(f"No text available for paper: {paper_data.get('title', 'Unknown')}")
            return {"error": "No text available"}

        if reference_pmids is None:
            reference_pmids = paper_data.get("reference_pmids", [])

        prompt = self.build_classification_prompt(paper_data, paper_text, text_source=text_source, reference_pmids=reference_pmids)

        try:
            # Use instructor for automatic structured output extraction
            # Get thread-local client for parallel processing
            client = self._get_client()
            classification_obj = client.messages.create(
                model=BEDROCK_MODEL_ID,
                max_tokens=2000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_model=PaperClassification,
            )

            # Convert to dict and add metadata
            classification = classification_obj.model_dump()
            classification["schema_version"] = CLASSIFICATION_SCHEMA_VERSION
            classification["paper_id"] = paper_data.get("paper_id")
            classification["title"] = paper_data.get("title")
            classification["year"] = paper_data.get("year")
            classification["text_source"] = text_source
            classification["classified_date"] = datetime.now().isoformat()

            return classification

        except Exception as e:
            logger.error(f"Error classifying paper {paper_data.get('paper_id')}: {e}")
            return {"error": str(e), "paper_id": paper_data.get("paper_id")}

    def get_citation_sentences(self, paper_id: str) -> Optional[str]:
        """Get citation sentences for a paper.

        Args:
            paper_id: Paper ID

        Returns:
            Formatted citation sentences or None if not available
        """
        if paper_id not in self.citation_sentences:
            return None

        data = self.citation_sentences[paper_id]

        # Combine all citation sentences
        sentences = []

        # Add cBioPortal paper citations
        paper_cites = data.get("cbioportal_paper_citations", [])
        if paper_cites:
            sentences.append("=== Sentences citing cBioPortal papers ===")
            sentences.extend(paper_cites)
            sentences.append("")

        # Add platform mentions
        platform_mentions = data.get("cbioportal_platform_mentions", [])
        if platform_mentions:
            sentences.append("=== Sentences mentioning cBioPortal platform ===")
            sentences.extend(platform_mentions)
            sentences.append("")

        # Add data citations
        data_cites = data.get("data_publication_citations", {})
        if data_cites:
            sentences.append("=== Sentences citing underlying data sources ===")
            for pmid, pmid_sentences in data_cites.items():
                sentences.append(f"Data PMID {pmid}:")
                sentences.extend(pmid_sentences)
                sentences.append("")

        if not sentences:
            return None

        return "\n".join(sentences)

    def _classify_single_paper(self, paper: Dict, is_upgrade: bool = False, source: str = "auto") -> Dict:
        """Classify a single paper (helper for parallel processing).

        Args:
            paper: Paper metadata dictionary
            is_upgrade: Whether this is an upgrade classification
            source: Classification source - "auto", "pdf", "sentences", or "both"
                - auto: sentences → PDF → abstract (prioritizes most focused source)
                - pdf: PDF only
                - sentences: citation sentences only
                - both: PDF + sentences combined

        Returns:
            Result dictionary with classification or error info
        """
        paper_id = paper.get("paper_id")

        # Get paper text based on source parameter
        paper_text = None
        text_source = "none"
        pdf_text = None
        sentences_text = None

        # Collect PDF text if needed
        if source in ("auto", "pdf", "both"):
            if paper.get("pdf_downloaded") and paper.get("pdf_path"):
                pdf_path = Path(paper["pdf_path"])
                if pdf_path.exists():
                    pdf_text = self.extract_text_from_pdf(pdf_path)

        # Collect citation sentences if needed
        if source in ("auto", "sentences", "both"):
            sentences_text = self.get_citation_sentences(paper_id)

        # Determine what text to use based on source mode
        if source == "auto":
            # New priority: sentences → PDF → abstract
            if sentences_text:
                paper_text = sentences_text
                text_source = "sentences"
            elif pdf_text:
                paper_text = pdf_text
                text_source = "pdf"
            else:
                paper_text = paper.get("abstract", "")
                if paper_text:
                    text_source = "abstract"

        elif source == "pdf":
            # PDF only
            if pdf_text:
                paper_text = pdf_text
                text_source = "pdf"

        elif source == "sentences":
            # Citation sentences only
            if sentences_text:
                paper_text = sentences_text
                text_source = "sentences"

        elif source == "both":
            # Combine PDF and sentences
            parts = []
            if pdf_text:
                parts.append("=== Paper Text (Abstract and Introduction) ===\n" + pdf_text[:4000])
            if sentences_text:
                parts.append("=== Citation Contexts ===\n" + sentences_text)

            if parts:
                paper_text = "\n\n".join(parts)
                text_source = "pdf+sentences" if pdf_text and sentences_text else ("pdf" if pdf_text else "sentences")

        # Get reference PMIDs for data source citation tracking
        reference_pmids = paper.get("reference_pmids", [])

        # Classify
        classification = self.classify_paper(paper, paper_text, text_source=text_source, reference_pmids=reference_pmids)

        # Add upgrade info to result
        classification["is_upgrade"] = is_upgrade
        classification["paper_id"] = paper_id

        # Rate limiting
        time.sleep(CLASSIFICATION_RATE_LIMIT_DELAY)

        return classification

    def classify_all_papers(
        self,
        citations_data: Dict,
        max_papers: Optional[int] = None,
        paper_ids: Optional[Set[str]] = None,
        skip_existing: bool = True,
        source: str = "auto"
    ) -> pd.DataFrame:
        """Classify all papers in the citations database using parallel processing.

        Args:
            citations_data: Citations metadata dictionary
            paper_ids: If provided, only classify papers with these PMIDs
            max_papers: Maximum number of papers to classify (None for all)
            skip_existing: Skip papers that have already been classified
            source: Classification source - "auto" (default), "pdf", "sentences", or "both"

        Returns:
            DataFrame with classification results
        """
        all_papers = []

        # Collect all papers from all PMIDs
        for pmid, pmid_data in citations_data["papers"].items():
            for citation in pmid_data["citations"]:
                citation["citing_pmid"] = pmid
                all_papers.append(citation)

        # Filter by paper_ids if provided
        if paper_ids:
            all_papers = [p for p in all_papers if p.get("paper_id") in paper_ids]
            logger.info(f"Filtered to {len(all_papers)} papers matching specified PMIDs")

        logger.info(f"Found {len(all_papers)} total papers to potentially classify")

        # Determine which papers need classification
        papers_to_classify = []
        skipped_count = 0

        for paper in all_papers:
            paper_id = paper.get("paper_id")
            should_classify = False
            is_upgrade = False

            if paper_id in self.classifications:
                existing = self.classifications[paper_id]

                # Check for failed attempts - skip those
                if existing.get("classification_failed"):
                    skipped_count += 1
                    continue

                # Re-classify if schema version is outdated
                existing_version = existing.get("schema_version", 0)
                if existing_version < CLASSIFICATION_SCHEMA_VERSION:
                    logger.info(f"Schema outdated for {paper_id} (v{existing_version} -> v{CLASSIFICATION_SCHEMA_VERSION}), re-classifying")
                    should_classify = True
                    is_upgrade = True

                # Check if we can upgrade from abstract to PDF (only if schema is current)
                elif existing.get('text_source') == 'abstract':
                    if paper.get("pdf_downloaded") and paper.get("pdf_path"):
                        pdf_path = Path(paper["pdf_path"])
                        if pdf_path.exists():
                            logger.info(f"PDF now available for {paper_id}, upgrading classification")
                            should_classify = True
                            is_upgrade = True

                # Skip if already classified and no upgrade available
                if not should_classify and skip_existing:
                    skipped_count += 1
                    continue
            else:
                # New paper, needs classification
                should_classify = True

            if should_classify:
                paper["_is_upgrade"] = is_upgrade  # Store for later use
                papers_to_classify.append(paper)

            # Check limit
            if max_papers and len(papers_to_classify) >= max_papers:
                logger.info(f"Reached classification limit of {max_papers}")
                break

        logger.info(f"Classifying {len(papers_to_classify)} papers using {CLASSIFICATION_MAX_WORKERS} parallel workers")
        logger.info(f"Skipped {skipped_count} papers (already classified or previously failed)")

        # Parallel classification using ThreadPoolExecutor
        classified_count = 0
        upgraded_count = 0

        with ThreadPoolExecutor(max_workers=CLASSIFICATION_MAX_WORKERS) as executor:
            # Submit all classification tasks
            future_to_paper = {
                executor.submit(self._classify_single_paper, paper, paper.get("_is_upgrade", False), source): paper
                for paper in papers_to_classify
            }

            # Process results as they complete with progress bar
            from concurrent.futures import as_completed
            for future in tqdm(as_completed(future_to_paper), total=len(papers_to_classify), desc="Classifying papers"):
                paper = future_to_paper[future]
                paper_id = paper.get("paper_id")

                try:
                    classification = future.result()
                    is_upgrade = classification.pop("is_upgrade", False)

                    if "error" in classification:
                        # Track failed attempt to avoid retrying
                        classification["classification_attempted"] = True
                        classification["classification_failed"] = True
                        classification["classification_error"] = classification.get("error")
                        self.classifications[paper_id] = classification
                        logger.warning(f"Classification failed for {paper_id}: {classification.get('error')}")
                    else:
                        # Successful classification
                        self.classifications[paper_id] = classification
                        classified_count += 1
                        if is_upgrade:
                            upgraded_count += 1

                        # Save periodically
                        if classified_count % 10 == 0:
                            self._save_classifications()
                            self._save_csv()

                except Exception as e:
                    logger.error(f"Unexpected error classifying {paper_id}: {e}")

        # Final save
        self._save_classifications()

        logger.info(f"Classified {classified_count} papers ({upgraded_count} upgraded from abstract to PDF)")
        logger.info(f"Total papers in database: {len(self.classifications)}")

        # Always save complete CSV with ALL classifications
        self._save_csv()

        # Return DataFrame of all classifications
        if self.classifications:
            df = pd.DataFrame(list(self.classifications.values()))
            return df

        return pd.DataFrame()


def main():
    """Main function to run the classifier."""
    logger.info("Starting classification process...")

    # Load citations data
    citations_file = METADATA_DIR / "citations.json"
    if not citations_file.exists():
        logger.error("No citations data found. Run fetcher first!")
        return

    with open(citations_file, "r") as f:
        citations_data = json.load(f)

    # Classify papers
    classifier = PaperClassifier()
    df = classifier.classify_all_papers(citations_data, max_papers=20)  # Limit for testing

    logger.info(f"Classification complete! Processed {len(df)} papers")


if __name__ == "__main__":
    main()
