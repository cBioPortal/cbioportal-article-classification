"""Module for fetching citations from PubMed and downloading PDFs."""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
import logging

import requests
from Bio import Entrez
from tqdm import tqdm

from .config import (
    CBIOPORTAL_PMIDS,
    METADATA_DIR,
    PDF_DIR,
    FETCH_DELAY_SECONDS,
    MAX_CITATIONS_PER_PAPER,
    NCBI_EMAIL,
    NCBI_API_KEY,
    UNPAYWALL_EMAIL,
    PDF_SOURCE_PRIORITY,
    PDF_DOWNLOAD_TIMEOUT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Entrez
Entrez.email = NCBI_EMAIL or "user@example.com"
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY


class CitationFetcher:
    """Fetches citations to cBioPortal papers from PubMed."""

    def __init__(self):
        self.metadata_file = METADATA_DIR / "citations.json"
        self.last_run_file = METADATA_DIR / "last_run.json"
        self.citations_data = self._load_metadata()
        self.last_run_info = self._load_last_run()

    def _load_metadata(self) -> Dict:
        """Load existing citations metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"papers": {}, "total_citations": 0, "last_updated": None}

    def _save_metadata(self):
        """Save citations metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.citations_data, indent=2, fp=f)

    def _load_last_run(self) -> Dict:
        """Load information about the last run."""
        if self.last_run_file.exists():
            with open(self.last_run_file, "r") as f:
                return json.load(f)
        return {"timestamp": None, "papers_fetched": 0}

    def _save_last_run(self, papers_count: int):
        """Save information about this run."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "papers_fetched": papers_count,
        }
        with open(self.last_run_file, "w") as f:
            json.dump(info, indent=2, fp=f)

    def get_existing_paper_ids(self) -> Set[str]:
        """Get set of paper IDs we already have."""
        existing_ids = set()
        for pmid_data in self.citations_data.get("papers", {}).values():
            for citation in pmid_data.get("citations", []):
                if "paper_id" in citation:
                    existing_ids.add(citation["paper_id"])
        return existing_ids

    def fetch_citations_for_pmid(self, pmid: str, existing_ids: Set[str]) -> List[Dict]:
        """Fetch citations for a specific PMID from PubMed.

        Args:
            pmid: PubMed ID of the paper
            existing_ids: Set of paper IDs we already have

        Returns:
            List of citation metadata dictionaries
        """
        logger.info(f"Fetching citations for PMID: {pmid}")

        try:
            # Use elink to find papers that cite this PMID
            time.sleep(FETCH_DELAY_SECONDS)
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pubmed",
                id=pmid,
                linkname="pubmed_pubmed_citedin"
            )
            record = Entrez.read(handle)
            handle.close()

            if not record or not record[0].get("LinkSetDb"):
                logger.info(f"No citations found for PMID {pmid}")
                return []

            # Get list of citing PMIDs
            citing_pmids = [link["Id"] for link in record[0]["LinkSetDb"][0]["Link"]]
            logger.info(f"Found {len(citing_pmids)} citations for PMID {pmid}")

            # Limit to avoid overwhelming requests
            citing_pmids = citing_pmids[:MAX_CITATIONS_PER_PAPER]

            citations = []
            citation_count = 0

            # Fetch metadata in batches of 200 (PubMed limit)
            batch_size = 200
            for i in tqdm(range(0, len(citing_pmids), batch_size),
                         desc=f"Fetching citation metadata for PMID {pmid}"):
                batch = citing_pmids[i:i + batch_size]

                time.sleep(FETCH_DELAY_SECONDS)
                handle = Entrez.efetch(
                    db="pubmed",
                    id=batch,
                    rettype="medline",
                    retmode="xml"
                )
                records = Entrez.read(handle)
                handle.close()

                for article in records["PubmedArticle"]:
                    try:
                        medline = article["MedlineCitation"]
                        citing_pmid = str(medline["PMID"])

                        # Skip if we already have this paper
                        if citing_pmid in existing_ids:
                            continue

                        # Extract article metadata
                        article_data = medline["Article"]

                        # Get title
                        title = str(article_data.get("ArticleTitle", ""))

                        # Get authors
                        authors_list = article_data.get("AuthorList", [])
                        authors = ", ".join([
                            f"{a.get('LastName', '')} {a.get('Initials', '')}".strip()
                            for a in authors_list if "LastName" in a
                        ])

                        # Get publication year
                        pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                        year = pub_date.get("Year", "")

                        # Get journal name
                        venue = str(article_data.get("Journal", {}).get("Title", ""))

                        # Get abstract
                        abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
                        if isinstance(abstract_parts, list):
                            abstract = " ".join([str(part) for part in abstract_parts])
                        else:
                            abstract = str(abstract_parts) if abstract_parts else ""

                        # Get DOI if available
                        doi = ""
                        article_ids = article.get("PubmedData", {}).get("ArticleIdList", [])
                        for aid in article_ids:
                            if aid.attributes.get("IdType") == "doi":
                                doi = str(aid)
                                break

                        # Build URL
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{citing_pmid}/"
                        if doi:
                            url = f"https://doi.org/{doi}"

                        citation_data = {
                            "paper_id": citing_pmid,
                            "title": title,
                            "authors": authors,
                            "year": year,
                            "venue": venue,
                            "url": url,
                            "doi": doi,
                            "abstract": abstract,
                            "fetched_date": datetime.now().isoformat(),
                            "pdf_downloaded": False,
                            "pdf_path": None,
                        }

                        citations.append(citation_data)
                        existing_ids.add(citing_pmid)
                        citation_count += 1

                    except Exception as e:
                        logger.warning(f"Error processing citation: {e}")
                        continue

            logger.info(f"Fetched {citation_count} new citations for PMID {pmid}")
            return citations

        except Exception as e:
            logger.error(f"Error fetching citations for PMID {pmid}: {e}")
            return []

    def fetch_all_citations(self, force_refresh: bool = False) -> Dict:
        """Fetch citations for all cBioPortal papers.

        Args:
            force_refresh: If True, re-fetch all citations even if we have them

        Returns:
            Dictionary with updated citations data
        """
        existing_ids = set() if force_refresh else self.get_existing_paper_ids()
        total_new = 0

        for pmid in CBIOPORTAL_PMIDS:
            if pmid not in self.citations_data["papers"]:
                self.citations_data["papers"][pmid] = {
                    "citations": [],
                    "last_updated": None,
                }

            new_citations = self.fetch_citations_for_pmid(pmid, existing_ids)

            if new_citations:
                self.citations_data["papers"][pmid]["citations"].extend(new_citations)
                self.citations_data["papers"][pmid]["last_updated"] = datetime.now().isoformat()
                total_new += len(new_citations)

        self.citations_data["total_citations"] = sum(
            len(p["citations"]) for p in self.citations_data["papers"].values()
        )
        self.citations_data["last_updated"] = datetime.now().isoformat()

        self._save_metadata()
        self._save_last_run(total_new)

        logger.info(f"Fetched {total_new} new citations. Total: {self.citations_data['total_citations']}")
        return self.citations_data


class PDFDownloader:
    """Downloads PDFs for papers from various sources."""

    def __init__(self, citation_fetcher=None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        self.stats = {"pmc": 0, "biorxiv": 0, "unpaywall": 0, "doi": 0, "failed": 0}
        self.citation_fetcher = citation_fetcher  # Reference to save metadata

    def _sync_existing_pdfs(self, citations_data: Dict) -> int:
        """Check for existing PDFs on disk and sync metadata.

        Args:
            citations_data: Citations metadata dictionary

        Returns:
            Number of PDFs found on disk
        """
        synced = 0

        for pmid, pmid_data in citations_data["papers"].items():
            for citation in pmid_data["citations"]:
                pdf_filename = f"{citation['paper_id']}.pdf"
                pdf_path = PDF_DIR / pdf_filename

                # If PDF exists on disk but metadata says not downloaded
                if pdf_path.exists() and not citation.get("pdf_downloaded"):
                    citation["pdf_downloaded"] = True
                    citation["pdf_path"] = str(pdf_path)
                    synced += 1
                    logger.debug(f"Synced existing PDF: {pdf_filename}")

        if synced > 0:
            logger.info(f"Found {synced} existing PDFs on disk, synced metadata")

        return synced

    def _try_pmc_pdf(self, paper_data: Dict, output_path: Path) -> bool:
        """Try to download PDF from PubMed Central.

        Args:
            paper_data: Dictionary containing paper metadata
            output_path: Where to save the PDF

        Returns:
            True if download successful, False otherwise
        """
        pmid = paper_data.get("paper_id", "")
        if not pmid:
            return False

        try:
            # Check if article is available in PMC
            handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
            record = Entrez.read(handle)
            handle.close()

            if not record or not record[0].get("LinkSetDb"):
                return False

            # Get PMC ID
            pmc_links = record[0]["LinkSetDb"][0]["Link"]
            if not pmc_links:
                return False

            pmc_id = pmc_links[0]["Id"]

            # Fetch PDF from PMC
            time.sleep(0.1)  # Be respectful
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"

            response = self.session.get(pdf_url, timeout=PDF_DOWNLOAD_TIMEOUT, allow_redirects=True)

            if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
                output_path.write_bytes(response.content)
                logger.debug(f"Downloaded from PMC: {paper_data['title'][:50]}")
                self.stats["pmc"] += 1
                return True

        except Exception as e:
            logger.debug(f"PMC download failed: {e}")

        return False

    def _try_biorxiv_medrxiv(self, paper_data: Dict, output_path: Path) -> bool:
        """Try to download PDF from bioRxiv or medRxiv.

        Args:
            paper_data: Dictionary containing paper metadata
            output_path: Where to save the PDF

        Returns:
            True if download successful, False otherwise
        """
        doi = paper_data.get("doi", "")
        if not doi:
            return False

        # Check if it's a bioRxiv or medRxiv DOI
        if "biorxiv" not in doi.lower() and "medrxiv" not in doi.lower():
            return False

        try:
            # bioRxiv/medRxiv DOIs like: 10.1101/2023.01.01.522534
            # PDF URL: https://www.biorxiv.org/content/10.1101/2023.01.01.522534v1.full.pdf
            server = "medrxiv" if "medrxiv" in doi.lower() else "biorxiv"
            pdf_url = f"https://www.{server}.org/content/{doi}v1.full.pdf"

            response = self.session.get(pdf_url, timeout=PDF_DOWNLOAD_TIMEOUT, allow_redirects=True)

            if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
                output_path.write_bytes(response.content)
                logger.debug(f"Downloaded from {server}: {paper_data['title'][:50]}")
                self.stats["biorxiv"] += 1
                return True

        except Exception as e:
            logger.debug(f"bioRxiv/medRxiv download failed: {e}")

        return False

    def _try_unpaywall(self, paper_data: Dict, output_path: Path) -> bool:
        """Try to download PDF using Unpaywall API.

        Args:
            paper_data: Dictionary containing paper metadata
            output_path: Where to save the PDF

        Returns:
            True if download successful, False otherwise
        """
        doi = paper_data.get("doi", "")
        if not doi or not UNPAYWALL_EMAIL:
            return False

        try:
            # Query Unpaywall API
            api_url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
            response = self.session.get(api_url, timeout=10)

            if response.status_code != 200:
                return False

            data = response.json()

            # Check if there's an open access PDF
            if not data.get("is_oa"):
                return False

            # Try to get best OA location
            oa_location = data.get("best_oa_location")
            if not oa_location or not oa_location.get("url_for_pdf"):
                return False

            pdf_url = oa_location["url_for_pdf"]

            # Download the PDF
            pdf_response = self.session.get(pdf_url, timeout=PDF_DOWNLOAD_TIMEOUT, allow_redirects=True)

            if pdf_response.status_code == 200 and "application/pdf" in pdf_response.headers.get("Content-Type", ""):
                output_path.write_bytes(pdf_response.content)
                logger.debug(f"Downloaded via Unpaywall: {paper_data['title'][:50]}")
                self.stats["unpaywall"] += 1
                return True

        except Exception as e:
            logger.debug(f"Unpaywall download failed: {e}")

        return False

    def _try_doi_url(self, paper_data: Dict, output_path: Path) -> bool:
        """Try to download PDF from DOI or direct URL.

        Args:
            paper_data: Dictionary containing paper metadata
            output_path: Where to save the PDF

        Returns:
            True if download successful, False otherwise
        """
        urls = []

        # Try DOI first
        if paper_data.get("doi"):
            urls.append(f"https://doi.org/{paper_data['doi']}")

        # Try direct URL
        if paper_data.get("url"):
            urls.append(paper_data["url"])

        for url in urls:
            try:
                response = self.session.get(url, timeout=PDF_DOWNLOAD_TIMEOUT, allow_redirects=True)

                if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
                    output_path.write_bytes(response.content)
                    logger.debug(f"Downloaded from DOI/URL: {paper_data['title'][:50]}")
                    self.stats["doi"] += 1
                    return True

            except Exception as e:
                logger.debug(f"DOI/URL download failed from {url}: {e}")
                continue

        return False

    def download_pdf(self, paper_data: Dict, output_path: Path) -> bool:
        """Attempt to download PDF for a paper from multiple sources.

        Args:
            paper_data: Dictionary containing paper metadata
            output_path: Where to save the PDF

        Returns:
            True if download successful, False otherwise
        """
        # Try sources in priority order
        source_methods = {
            "pmc": self._try_pmc_pdf,
            "biorxiv": self._try_biorxiv_medrxiv,
            "unpaywall": self._try_unpaywall,
            "doi": self._try_doi_url,
        }

        for source in PDF_SOURCE_PRIORITY:
            if source in source_methods:
                if source_methods[source](paper_data, output_path):
                    return True

        # All sources failed
        self.stats["failed"] += 1
        logger.debug(f"All sources failed for: {paper_data['title'][:50]}")
        return False

    def download_all_pdfs(
        self,
        citations_data: Dict,
        max_downloads: Optional[int] = None,
        checkpoint_frequency: int = 10
    ) -> int:
        """Download PDFs for all citations that don't have them yet.

        Args:
            citations_data: Citations metadata dictionary
            max_downloads: Maximum number of PDFs to download (None for unlimited)
            checkpoint_frequency: Save metadata every N successful downloads

        Returns:
            Number of PDFs successfully downloaded
        """
        # First, sync existing PDFs on disk
        self._sync_existing_pdfs(citations_data)

        downloaded = 0
        attempted = 0

        for pmid, pmid_data in citations_data["papers"].items():
            for citation in tqdm(pmid_data["citations"], desc=f"Downloading PDFs for PMID {pmid}"):
                # Skip if already downloaded
                if citation.get("pdf_downloaded"):
                    continue

                # Skip if previously attempted and failed (unless force retry)
                if citation.get("download_attempted") and not citation.get("pdf_downloaded"):
                    continue

                if max_downloads and downloaded >= max_downloads:
                    logger.info(f"Reached download limit of {max_downloads}")
                    if self.citation_fetcher:
                        self.citation_fetcher._save_metadata()
                    self._print_stats()
                    return downloaded

                pdf_filename = f"{citation['paper_id']}.pdf"
                pdf_path = PDF_DIR / pdf_filename

                # Mark as attempted
                citation["download_attempted"] = True
                citation["download_attempt_date"] = datetime.now().isoformat()
                attempted += 1

                if self.download_pdf(citation, pdf_path):
                    citation["pdf_downloaded"] = True
                    citation["pdf_path"] = str(pdf_path)
                    downloaded += 1

                    # Periodic checkpoint
                    if checkpoint_frequency and downloaded % checkpoint_frequency == 0:
                        if self.citation_fetcher:
                            self.citation_fetcher._save_metadata()
                            logger.info(f"Checkpoint: Saved metadata after {downloaded} downloads")

                time.sleep(FETCH_DELAY_SECONDS)

            # Save after each PMID batch
            if self.citation_fetcher and attempted > 0:
                self.citation_fetcher._save_metadata()
                logger.info(f"Saved metadata after completing PMID {pmid}")

        # Final save
        if self.citation_fetcher:
            self.citation_fetcher._save_metadata()

        self._print_stats()
        return downloaded

    def _print_stats(self):
        """Print PDF download statistics."""
        total_attempted = sum(self.stats.values())
        if total_attempted == 0:
            return

        logger.info("PDF Download Statistics:")
        logger.info(f"  PMC: {self.stats['pmc']} PDFs")
        logger.info(f"  bioRxiv/medRxiv: {self.stats['biorxiv']} PDFs")
        logger.info(f"  Unpaywall: {self.stats['unpaywall']} PDFs")
        logger.info(f"  DOI/Direct: {self.stats['doi']} PDFs")
        logger.info(f"  Failed: {self.stats['failed']} papers")

        total_success = self.stats['pmc'] + self.stats['biorxiv'] + self.stats['unpaywall'] + self.stats['doi']
        if total_attempted > 0:
            success_rate = (total_success / total_attempted) * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")


def main():
    """Main function to run the fetcher."""
    logger.info("Starting citation fetch process...")

    # Fetch citations
    fetcher = CitationFetcher()
    citations_data = fetcher.fetch_all_citations()

    logger.info(f"Total citations in database: {citations_data['total_citations']}")

    # Download PDFs (limited to 10 for initial testing)
    downloader = PDFDownloader()
    downloaded = downloader.download_all_pdfs(citations_data, max_downloads=10)

    # Save updated metadata
    fetcher._save_metadata()

    logger.info("Fetch process complete!")


if __name__ == "__main__":
    main()
