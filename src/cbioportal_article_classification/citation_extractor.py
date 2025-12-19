"""Module for extracting citation sentences from PDFs."""
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import pdfplumber

from .config import METADATA_DIR, CBIOPORTAL_PMIDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pdfminer warnings for malformed PDFs
logging.getLogger('pdfminer').setLevel(logging.ERROR)


class CitationExtractor:
    """Extracts citation sentences from PDFs."""

    def __init__(self):
        """Initialize the citation extractor."""
        self.citations_file = METADATA_DIR / "citation_sentences.json"
        self.citation_data = self._load_citations()
        self.cbioportal_pmids = set(CBIOPORTAL_PMIDS)
        self.data_pmids = set()  # Will be populated from studies metadata
        self.data_pmids_version: Optional[str] = None

        # Pre-compile regex patterns for performance
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        # Sentence splitting pattern
        self.sentence_split_pattern = re.compile(r'[.!?]\s+')

        # Abbreviation pattern to avoid false splits
        self.abbrev_pattern = re.compile(r'\b(Dr|Mr|Mrs|Ms|Prof|vs|etc|e\.g|i\.e|al)\.\s')

        # PMID pattern cache (will be populated as needed)
        self.pmid_patterns = {}

    def _get_pmid_pattern(self, pmid: str):
        """Get or create compiled pattern for PMID.

        Args:
            pmid: PMID string

        Returns:
            Compiled regex pattern
        """
        if pmid not in self.pmid_patterns:
            # Match PMID in various formats: PMID:12345, PMID 12345, [12345]
            self.pmid_patterns[pmid] = re.compile(
                rf'\b(PMID[\s:]*{pmid}|{pmid}\b)',
                re.IGNORECASE
            )
        return self.pmid_patterns[pmid]

    def _load_citations(self) -> Dict:
        """Load existing citation sentences from disk."""
        if self.citations_file.exists():
            with open(self.citations_file, "r") as f:
                return json.load(f)
        return {}

    def _save_citations(self):
        """Save citation sentences to disk."""
        with open(self.citations_file, "w") as f:
            json.dump(self.citation_data, indent=2, fp=f)
        logger.info(f"Saved citation sentences to {self.citations_file}")

    def load_data_pmids_from_studies(self) -> int:
        """Load PMIDs from cBioPortal studies metadata.

        Returns:
            Number of PMIDs loaded
        """
        studies_file = METADATA_DIR / "cbioportal_studies.json"
        if not studies_file.exists():
            logger.warning("Studies metadata not found. Run fetch-studies first.")
            self.data_pmids_version = "missing"
            return 0

        with open(studies_file, "r") as f:
            studies_data = json.load(f)

        self.data_pmids = set(studies_data.get("pmid_to_studies", {}).keys())
        # Use studies timestamp to detect when metadata changes
        self.data_pmids_version = (
            studies_data.get("last_updated")
            or str(int(studies_file.stat().st_mtime))
        )
        logger.info(f"Loaded {len(self.data_pmids)} data PMIDs from studies metadata")
        return len(self.data_pmids)

    def _current_data_pmids_version(self) -> str:
        """Return the current data PMID version string."""
        return self.data_pmids_version or "missing"

    def _should_skip_paper(
        self,
        paper_id: str,
        pdf_path: Path,
        force_reextract: bool
    ) -> bool:
        """Determine if an already extracted paper can be skipped."""
        if force_reextract:
            return False

        existing = self.citation_data.get(paper_id)
        if not existing:
            return False

        stored_path = existing.get("pdf_path")
        if not stored_path:
            return False

        try:
            if Path(stored_path).resolve() != pdf_path.resolve():
                return False
        except Exception:
            if str(pdf_path) != stored_path:
                return False

        stored_version = existing.get("data_pmids_version", "missing")
        if stored_version != self._current_data_pmids_version():
            return False

        return True

    def extract_text_from_pdf(self, pdf_path: Path, max_pages: int = 20, extract_all: bool = False) -> str:
        """Extract text from PDF file (first + last pages, or all pages).

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to extract (default: 20)
            extract_all: If True, extract all pages regardless of max_pages

        Returns:
            Extracted text content
        """
        try:
            text_parts = []

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                if extract_all:
                    # Extract ALL pages
                    pages_to_extract = list(range(total_pages))
                else:
                    # Extract first half + last half to capture both in-text citations
                    # (in intro/methods) and references section (at end)
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

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (optimized with pre-compiled patterns).

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Handle common abbreviations to avoid false splits
        text = self.abbrev_pattern.sub(r'\1<DOT> ', text)

        # Split on sentence boundaries using pre-compiled pattern
        sentences = self.sentence_split_pattern.split(text)

        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def find_citation_sentences(
        self,
        text: str,
        target_pmids: Set[str],
        context_keywords: Optional[List[str]] = None
    ) -> List[str]:
        """Find sentences that cite specific PMIDs or keywords (optimized two-pass).

        Args:
            text: Full text to search
            target_pmids: Set of PMIDs to search for
            context_keywords: Optional keywords to search for (e.g., "cBioPortal")

        Returns:
            List of sentences containing citations
        """
        # Two-pass strategy: quick check first to avoid expensive sentence splitting
        # Pass 1: Check if any target PMIDs or keywords exist in text
        found_pmids = set()
        found_keywords = []

        for pmid in target_pmids:
            # Quick existence check (much faster than regex)
            if pmid in text or f"PMID{pmid}" in text or f"PMID:{pmid}" in text or f"PMID {pmid}" in text:
                found_pmids.add(pmid)

        if context_keywords:
            for keyword in context_keywords:
                if keyword.lower() in text.lower():
                    found_keywords.append(keyword)

        # If nothing found, return early
        if not found_pmids and not found_keywords:
            return []

        # Pass 2: Extract sentences only if we found relevant content
        sentences = self.split_into_sentences(text)
        matching_sentences = []

        for sentence in sentences:
            # Check for PMID mentions using pre-compiled patterns
            for pmid in found_pmids:
                pattern = self._get_pmid_pattern(pmid)
                if pattern.search(sentence):
                    matching_sentences.append(sentence)
                    break

            # Check for keyword mentions
            for keyword in found_keywords:
                if re.search(rf'\b{re.escape(keyword)}\b', sentence, re.IGNORECASE):
                    matching_sentences.append(sentence)
                    break

        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for s in matching_sentences:
            if s not in seen:
                seen.add(s)
                unique_sentences.append(s)

        return unique_sentences

    def extract_citations_from_paper(
        self,
        paper_id: str,
        pdf_path: Path,
        force_reextract: bool = False,
        paper_metadata: Optional[Dict] = None,
    ) -> Dict:
        """Extract citation sentences from a single paper.

        Args:
            paper_id: Paper ID
            pdf_path: Path to PDF file
            force_reextract: Force re-extraction even if data exists

        Returns:
            Dictionary with extracted citation sentences
        """
        start_time = time.time()
        step_times = {}

        # Check if already extracted
        if paper_id in self.citation_data and not force_reextract:
            logger.debug(f"Citation sentences already extracted for {paper_id}")
            return self.citation_data[paper_id]

        # Pass 1: Extract first + last pages (fast)
        step_start = time.time()
        text = self.extract_text_from_pdf(pdf_path, max_pages=20, extract_all=False)
        step_times['pdf_extraction_pass1'] = time.time() - step_start

        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
            return {"error": "No text extracted"}

        # Separate cBioPortal paper citations from platform mentions
        step_start = time.time()
        cbioportal_paper_citations = self.find_citation_sentences(
            text,
            self.cbioportal_pmids,
            context_keywords=None  # Only PMID citations
        )
        step_times['cbioportal_papers'] = time.time() - step_start

        # Pass 2: If no cBioPortal citations found, try full extraction
        # (We KNOW the paper cites cBioPortal - that's how we found it!)
        if not cbioportal_paper_citations:
            logger.debug(f"No cBioPortal citations in first/last pages for {paper_id}, trying full extraction")
            step_start = time.time()
            text_full = self.extract_text_from_pdf(pdf_path, extract_all=True)
            step_times['pdf_extraction_pass2'] = time.time() - step_start

            if text_full:
                step_start = time.time()
                cbioportal_paper_citations = self.find_citation_sentences(
                    text_full,
                    self.cbioportal_pmids,
                    context_keywords=None
                )
                step_times['cbioportal_papers_pass2'] = time.time() - step_start

                # Use full text for remaining searches
                text = text_full
                step_times['used_full_extraction'] = True
            else:
                step_times['used_full_extraction'] = False
        else:
            step_times['used_full_extraction'] = False

        # Update total PDF extraction time
        step_times['pdf_extraction'] = (
            step_times.get('pdf_extraction_pass1', 0) +
            step_times.get('pdf_extraction_pass2', 0)
        )

        step_start = time.time()
        cbioportal_platform_mentions = self.find_citation_sentences(
            text,
            set(),  # No PMIDs
            context_keywords=["cBioPortal", "cbioportal.org"]
        )
        step_times['platform_mentions'] = time.time() - step_start

        # Remove duplicates - if a sentence has both PMID and platform mention,
        # keep it only in paper_citations
        paper_citation_set = set(cbioportal_paper_citations)
        cbioportal_platform_mentions = [
            s for s in cbioportal_platform_mentions
            if s not in paper_citation_set
        ]

        # Find data publication citations (only if we have data PMIDs loaded)
        step_start = time.time()
        data_publication_citations = {}
        pmids_to_check: Set[str] = set()
        references_known = False

        if paper_metadata:
            reference_pmids = paper_metadata.get("reference_pmids") or []
            pmids_to_check = {
                str(pmid)
                for pmid in reference_pmids
            }
            references_known = bool(paper_metadata.get("reference_pmids_last_updated"))

        if self.data_pmids:
            if pmids_to_check:
                pmids_to_check = pmids_to_check & self.data_pmids
            elif references_known:
                pmids_to_check = set()
            else:
                pmids_to_check = set(self.data_pmids)

            num_data_pmids = len(pmids_to_check)
            for pmid in pmids_to_check:
                pmid_sentences = self.find_citation_sentences(text, {pmid})
                if pmid_sentences:
                    data_publication_citations[pmid] = pmid_sentences
        else:
            num_data_pmids = 0
        step_times['data_citations'] = time.time() - step_start
        step_times['num_data_pmids_checked'] = num_data_pmids

        total_time = time.time() - start_time

        # Log timing for slow papers (>2 seconds)
        if total_time > 2.0:
            full_extract_marker = " [FULL]" if step_times.get('used_full_extraction') else ""
            logger.warning(
                f"Slow extraction for {paper_id}: {total_time:.2f}s{full_extract_marker} - "
                f"pdf={step_times['pdf_extraction']:.2f}s, "
                f"cb_papers={step_times['cbioportal_papers']:.2f}s, "
                f"platform={step_times['platform_mentions']:.2f}s, "
                f"data={step_times['data_citations']:.2f}s ({num_data_pmids} PMIDs)"
            )

        # Store results
        result = {
            "paper_id": paper_id,
            "cbioportal_paper_citations": cbioportal_paper_citations,
            "cbioportal_platform_mentions": cbioportal_platform_mentions,
            "data_publication_citations": data_publication_citations,
            "extraction_date": datetime.now().isoformat(),
            "pdf_path": str(pdf_path),
            "total_cbioportal_paper_citations": len(cbioportal_paper_citations),
            "total_cbioportal_platform_mentions": len(cbioportal_platform_mentions),
            "total_data_pmids_cited": len(data_publication_citations),
            "data_pmids_version": self._current_data_pmids_version(),
            "_timing": step_times  # Store timing for analysis
        }

        return result

    def _extract_single_paper(self, paper: Dict, force_reextract: bool) -> Dict:
        """Extract citations from a single paper (helper for parallel processing).

        Args:
            paper: Paper metadata dictionary
            force_reextract: Force re-extraction

        Returns:
            Result dictionary with extracted citations
        """
        paper_id = paper.get("paper_id")
        pdf_path = Path(paper.get("pdf_path", ""))

        if not pdf_path.exists():
            return {"paper_id": paper_id, "error": "PDF not found"}

        return self.extract_citations_from_paper(
            paper_id,
            pdf_path,
            force_reextract=force_reextract,
            paper_metadata=paper,
        )

    def extract_all_citations(
        self,
        citations_data: Dict,
        max_papers: Optional[int] = None,
        force_reextract: bool = False,
        max_workers: int = 10
    ) -> Dict:
        """Extract citation sentences from all papers with PDFs.

        Args:
            citations_data: Citations metadata dictionary
            max_papers: Maximum number of papers to process (None for all)
            force_reextract: Force re-extraction even if data exists
            max_workers: Number of parallel workers

        Returns:
            Dictionary with extraction statistics
        """
        # Load data PMIDs first
        self.load_data_pmids_from_studies()

        # Collect papers with PDFs
        papers_to_process = []
        skipped_existing = 0
        for pmid_data in citations_data.get("papers", {}).values():
            for citation in pmid_data.get("citations", []):
                if citation.get("pdf_downloaded") and citation.get("pdf_path"):
                    pdf_path = Path(citation["pdf_path"])
                    if pdf_path.exists():
                        if self._should_skip_paper(citation.get("paper_id"), pdf_path, force_reextract):
                            skipped_existing += 1
                            continue

                        papers_to_process.append(citation)

                # Check limit
                if max_papers and len(papers_to_process) >= max_papers:
                    break

            if max_papers and len(papers_to_process) >= max_papers:
                break

        logger.info(f"Processing {len(papers_to_process)} papers with PDFs using {max_workers} workers")
        if skipped_existing:
            logger.info(
                f"Skipping {skipped_existing} papers with existing extraction data "
                f"(use --force to reprocess)"
            )

        if not papers_to_process:
            logger.info("All available PDFs already have up-to-date citation data.")
            stats = self.get_summary_stats()
            stats["extraction_run"] = {
                "extracted_count": 0,
                "error_count": 0,
                "total_processed": 0,
                "skipped_existing": skipped_existing,
            }
            return stats

        # Parallel extraction using ThreadPoolExecutor
        extracted_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all extraction tasks
            future_to_paper = {
                executor.submit(self._extract_single_paper, paper, force_reextract): paper
                for paper in papers_to_process
            }

            # Process results as they complete with progress bar
            from concurrent.futures import as_completed
            for future in tqdm(as_completed(future_to_paper), total=len(papers_to_process), desc="Extracting citations"):
                paper = future_to_paper[future]
                paper_id = paper.get("paper_id")

                try:
                    result = future.result()

                    if "error" in result:
                        error_count += 1
                        logger.warning(f"Extraction failed for {paper_id}: {result.get('error')}")
                    else:
                        # Successful extraction
                        self.citation_data[paper_id] = result
                        extracted_count += 1

                        # Save periodically
                        if extracted_count % 50 == 0:
                            self._save_citations()

                except Exception as e:
                    error_count += 1
                    logger.error(f"Unexpected error extracting citations from {paper_id}: {e}")

        # Final save
        self._save_citations()

        # Generate statistics
        stats = self.get_summary_stats()
        stats["extraction_run"] = {
            "extracted_count": extracted_count,
            "error_count": error_count,
            "total_processed": len(papers_to_process),
            "skipped_existing": skipped_existing,
        }

        logger.info(f"Extracted citations from {extracted_count} papers ({error_count} errors)")
        return stats

    def get_summary_stats(self) -> Dict:
        """Get summary statistics about extracted citations.

        Returns:
            Dictionary with summary statistics
        """
        if not self.citation_data:
            return {"error": "No citation data loaded"}

        total_papers = len(self.citation_data)
        papers_with_paper_citations = sum(
            1 for c in self.citation_data.values()
            if c.get("total_cbioportal_paper_citations", 0) > 0
        )
        papers_with_platform_mentions = sum(
            1 for c in self.citation_data.values()
            if c.get("total_cbioportal_platform_mentions", 0) > 0
        )
        papers_with_data = sum(
            1 for c in self.citation_data.values()
            if c.get("total_data_pmids_cited", 0) > 0
        )

        # Count papers with any cBioPortal mention (paper citation OR platform mention)
        papers_with_any_cbioportal = sum(
            1 for c in self.citation_data.values()
            if c.get("total_cbioportal_paper_citations", 0) > 0
            or c.get("total_cbioportal_platform_mentions", 0) > 0
        )

        # Count co-citations
        papers_citing_both = sum(
            1 for c in self.citation_data.values()
            if (c.get("total_cbioportal_paper_citations", 0) > 0
                or c.get("total_cbioportal_platform_mentions", 0) > 0)
            and c.get("total_data_pmids_cited", 0) > 0
        )

        return {
            "total_papers_extracted": total_papers,
            "papers_with_cbioportal_paper_citations": papers_with_paper_citations,
            "papers_with_cbioportal_platform_mentions": papers_with_platform_mentions,
            "papers_with_any_cbioportal_mention": papers_with_any_cbioportal,
            "papers_with_data_citations": papers_with_data,
            "papers_citing_both": papers_citing_both,
        }


def main():
    """Main function for testing."""
    from .fetcher import CitationFetcher

    # Load citations data
    citations_file = METADATA_DIR / "citations.json"
    if not citations_file.exists():
        logger.error("No citations data found. Run fetcher first!")
        return

    with open(citations_file, "r") as f:
        citations_data = json.load(f)

    # Extract citations
    extractor = CitationExtractor()
    stats = extractor.extract_all_citations(citations_data, max_papers=10)

    logger.info(f"Summary: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
