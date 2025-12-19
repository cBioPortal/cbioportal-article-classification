"""Module for classifying papers using AWS Bedrock (Claude) with instructor."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from anthropic import AnthropicBedrock
import instructor
from pydantic import BaseModel, Field
import pandas as pd
from PyPDF2 import PdfReader
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


class PaperClassification(BaseModel):
    """Structured output for paper classification."""

    analysis_type: List[str] = Field(
        description="Types of analysis performed (e.g., mutation analysis, survival analysis)"
    )
    cancer_type: List[str] = Field(
        description="Cancer types studied in the paper"
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
    confidence: str = Field(
        description="Confidence level in the classification: high, medium, or low"
    )


class PaperClassifier:
    """Classifies papers using AWS Bedrock with Claude and instructor."""

    def __init__(self):
        """Initialize the classifier with instructor-patched Bedrock client."""
        base_client = AnthropicBedrock(
            aws_region=AWS_REGION,
            aws_profile=AWS_PROFILE,
        )
        # Patch the client with instructor for structured outputs
        self.client = instructor.from_anthropic(base_client)
        self.classifications_file = METADATA_DIR / "classifications.json"
        self.classifications = self._load_classifications()

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
        """Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to extract (to avoid huge texts)

        Returns:
            Extracted text content
        """
        try:
            reader = PdfReader(pdf_path)
            text_parts = []

            # Extract from first N pages (usually intro/methods are most relevant)
            for page_num in range(min(len(reader.pages), max_pages)):
                page = reader.pages[page_num]
                text_parts.append(page.extract_text())

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def build_classification_prompt(self, paper_data: Dict, paper_text: str) -> str:
        """Build the prompt for Claude to classify the paper.

        Args:
            paper_data: Paper metadata
            paper_text: Extracted text from paper

        Returns:
            Formatted prompt for the LLM
        """
        categories_desc = "\n\n".join([
            f"**{cat_name.replace('_', ' ').title()}**:\n" +
            ", ".join(f'"{opt}"' for opt in options)
            for cat_name, options in CLASSIFICATION_CATEGORIES.items()
        ])

        prompt = f"""You are analyzing a scientific paper that cites cBioPortal, a cancer genomics data portal and analysis platform.

Paper Title: {paper_data.get('title', 'N/A')}
Authors: {paper_data.get('authors', 'N/A')}
Year: {paper_data.get('year', 'N/A')}
Venue: {paper_data.get('venue', 'N/A')}

Paper Abstract and Introduction:
{paper_text[:8000]}

Please classify this paper according to the following categories. For each category, select the most appropriate option(s).

{categories_desc}

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

**Confidence**: Set to "high" if clear usage is described with specific details, "medium" if mentioned but details unclear, "low" if only cited in references

Remember: Be specific and conservative. Only include information that is explicitly stated in the paper."""

        return prompt

    def classify_paper(self, paper_data: Dict, paper_text: Optional[str] = None, text_source: str = "abstract") -> Dict:
        """Classify a single paper using Claude via Bedrock with instructor.

        Args:
            paper_data: Paper metadata dictionary
            paper_text: Optional pre-extracted text. If None, uses abstract from metadata
            text_source: Source of the text ("pdf", "abstract", or "none")

        Returns:
            Classification results dictionary
        """
        if not paper_text:
            paper_text = paper_data.get("abstract", "")
            text_source = "abstract" if paper_text else "none"

        if not paper_text:
            logger.warning(f"No text available for paper: {paper_data.get('title', 'Unknown')}")
            return {"error": "No text available"}

        prompt = self.build_classification_prompt(paper_data, paper_text)

        try:
            # Use instructor for automatic structured output extraction
            classification_obj = self.client.messages.create(
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

    def _classify_single_paper(self, paper: Dict, is_upgrade: bool = False) -> Dict:
        """Classify a single paper (helper for parallel processing).

        Args:
            paper: Paper metadata dictionary
            is_upgrade: Whether this is an upgrade classification

        Returns:
            Result dictionary with classification or error info
        """
        paper_id = paper.get("paper_id")

        # Get paper text and determine source
        paper_text = None
        text_source = "none"

        if paper.get("pdf_downloaded") and paper.get("pdf_path"):
            pdf_path = Path(paper["pdf_path"])
            if pdf_path.exists():
                paper_text = self.extract_text_from_pdf(pdf_path)
                if paper_text:
                    text_source = "pdf"

        # If no PDF, use abstract
        if not paper_text:
            paper_text = paper.get("abstract", "")
            if paper_text:
                text_source = "abstract"

        # Classify
        classification = self.classify_paper(paper, paper_text, text_source=text_source)

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
        skip_existing: bool = True
    ) -> pd.DataFrame:
        """Classify all papers in the citations database using parallel processing.

        Args:
            citations_data: Citations metadata dictionary
            max_papers: Maximum number of papers to classify (None for all)
            skip_existing: Skip papers that have already been classified

        Returns:
            DataFrame with classification results
        """
        all_papers = []

        # Collect all papers from all PMIDs
        for pmid, pmid_data in citations_data["papers"].items():
            for citation in pmid_data["citations"]:
                citation["citing_pmid"] = pmid
                all_papers.append(citation)

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
                executor.submit(self._classify_single_paper, paper, paper.get("_is_upgrade", False)): paper
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
