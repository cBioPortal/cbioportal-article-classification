"""Module for fetching and managing cBioPortal study metadata."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import requests
import time

from .config import METADATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudyFetcher:
    """Fetches and manages cBioPortal study metadata."""

    def __init__(self):
        """Initialize the study fetcher."""
        self.studies_file = METADATA_DIR / "cbioportal_studies.json"
        self.api_url = "https://www.cbioportal.org/api/studies"
        self.studies_data = self._load_studies()

    def _load_studies(self) -> Dict:
        """Load existing studies metadata from disk."""
        if self.studies_file.exists():
            with open(self.studies_file, "r") as f:
                return json.load(f)
        return {}

    def _save_studies(self):
        """Save studies metadata to disk."""
        with open(self.studies_file, "w") as f:
            json.dump(self.studies_data, indent=2, fp=f)
        logger.info(f"Saved studies metadata to {self.studies_file}")

    def fetch_all_studies(self, force_refresh: bool = False) -> Dict:
        """Fetch all cBioPortal studies from the API.

        Args:
            force_refresh: Force re-fetch even if data exists

        Returns:
            Dictionary with studies metadata
        """
        if self.studies_data and not force_refresh:
            logger.info("Studies metadata already exists. Use force_refresh=True to re-fetch.")
            return self.studies_data

        logger.info(f"Fetching studies from {self.api_url}...")

        try:
            response = requests.get(self.api_url, timeout=30)
            response.raise_for_status()
            studies_list = response.json()

            logger.info(f"Fetched {len(studies_list)} studies from cBioPortal")

            # Build structured data
            self.studies_data = {
                "last_updated": datetime.now().isoformat(),
                "total_studies": len(studies_list),
                "studies": {},
                "pmid_to_studies": {}  # Index: PMID -> list of study IDs
            }

            # Process each study
            for study in studies_list:
                study_id = study.get("studyId")
                if not study_id:
                    continue

                # Extract relevant fields
                study_metadata = {
                    "studyId": study_id,
                    "name": study.get("name", ""),
                    "description": study.get("description", ""),
                    "pmid": study.get("pmid", ""),
                    "citation": study.get("citation", ""),
                    "cancerTypeId": study.get("cancerTypeId", ""),
                    "allSampleCount": study.get("allSampleCount", 0),
                    "publicStudy": study.get("publicStudy", False),
                    "referenceGenome": study.get("referenceGenome", ""),
                }

                self.studies_data["studies"][study_id] = study_metadata

                # Build PMID index (PMIDs can be comma-separated)
                pmid_str = study.get("pmid", "")
                if pmid_str:
                    # Split by comma and clean whitespace
                    pmids = [p.strip() for p in pmid_str.split(",") if p.strip()]
                    for pmid in pmids:
                        if pmid not in self.studies_data["pmid_to_studies"]:
                            self.studies_data["pmid_to_studies"][pmid] = []
                        self.studies_data["pmid_to_studies"][pmid].append(study_id)

            logger.info(f"Indexed {len(self.studies_data['pmid_to_studies'])} unique PMIDs across studies")

            # Save to disk
            self._save_studies()

            return self.studies_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching studies from API: {e}")
            return {}

    def get_studies_by_pmid(self, pmid: str) -> List[Dict]:
        """Get all studies associated with a PMID.

        Args:
            pmid: PubMed ID to search for

        Returns:
            List of study metadata dictionaries
        """
        if not self.studies_data:
            logger.warning("No studies data loaded. Run fetch_all_studies() first.")
            return []

        study_ids = self.studies_data.get("pmid_to_studies", {}).get(pmid, [])
        return [
            self.studies_data["studies"][sid]
            for sid in study_ids
            if sid in self.studies_data["studies"]
        ]

    def get_all_data_pmids(self) -> List[str]:
        """Get all unique PMIDs from underlying data studies.

        Returns:
            List of PMIDs
        """
        if not self.studies_data:
            logger.warning("No studies data loaded. Run fetch_all_studies() first.")
            return []

        return list(self.studies_data.get("pmid_to_studies", {}).keys())

    def fetch_citation_counts(self) -> Dict:
        """Fetch PubMed citation counts for all studies with PMIDs.

        Uses PubMed eutils API to get the number of papers citing each study.
        Rate-limited to respect NCBI's usage guidelines (max 3 requests/second).

        Returns:
            Dictionary with citation count statistics
        """
        if not self.studies_data or "studies" not in self.studies_data:
            logger.warning("No studies data loaded. Run fetch_all_studies() first.")
            return {}

        logger.info("Fetching citation counts from PubMed for studies with PMIDs...")

        total_studies = 0
        studies_with_citations = 0
        failed_fetches = 0

        for study_id, study in self.studies_data["studies"].items():
            pmid_str = study.get("pmid", "")
            if not pmid_str:
                continue

            # Handle comma-separated PMIDs - use the first one for citation count
            pmids = [p.strip() for p in pmid_str.split(",") if p.strip()]
            if not pmids:
                continue

            primary_pmid = pmids[0]
            total_studies += 1

            try:
                # Fetch citations from PubMed
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
                params = {
                    "dbfrom": "pubmed",
                    "id": primary_pmid,
                    "linkname": "pubmed_pubmed_citedin",
                    "retmode": "json"
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                # Extract citation count
                citation_count = 0
                if "linksets" in data and len(data["linksets"]) > 0:
                    linkset = data["linksets"][0]
                    if "linksetdbs" in linkset and len(linkset["linksetdbs"]) > 0:
                        links = linkset["linksetdbs"][0].get("links", [])
                        citation_count = len(links)

                # Store in study metadata
                study["total_pubmed_citations"] = citation_count
                study["citations_last_updated"] = datetime.now().isoformat()

                studies_with_citations += 1

                if citation_count > 0:
                    logger.info(f"  {study_id} (PMID {primary_pmid}): {citation_count:,} citations")

                # Rate limiting: 3 requests per second max
                time.sleep(0.34)

            except Exception as e:
                logger.warning(f"  Failed to fetch citations for {study_id} (PMID {primary_pmid}): {e}")
                failed_fetches += 1
                # Store error info
                study["total_pubmed_citations"] = None
                study["citations_fetch_error"] = str(e)

        # Update last_updated timestamp
        self.studies_data["citations_last_updated"] = datetime.now().isoformat()

        # Save updated data
        self._save_studies()

        stats = {
            "total_studies_with_pmid": total_studies,
            "successfully_fetched": studies_with_citations,
            "failed_fetches": failed_fetches
        }

        logger.info(f"Citation fetch complete: {studies_with_citations}/{total_studies} successful")
        return stats

    def get_summary_stats(self) -> Dict:
        """Get summary statistics about studies.

        Returns:
            Dictionary with summary statistics
        """
        if not self.studies_data:
            return {"error": "No studies data loaded"}

        total_studies = self.studies_data.get("total_studies", 0)
        studies_with_pmid = sum(
            1 for s in self.studies_data.get("studies", {}).values()
            if s.get("pmid")
        )
        unique_pmids = len(self.studies_data.get("pmid_to_studies", {}))
        last_updated = self.studies_data.get("last_updated", "Never")

        # Cancer type distribution
        cancer_types = {}
        for study in self.studies_data.get("studies", {}).values():
            cancer_type = study.get("cancerTypeId", "unknown")
            cancer_types[cancer_type] = cancer_types.get(cancer_type, 0) + 1

        return {
            "total_studies": total_studies,
            "studies_with_pmid": studies_with_pmid,
            "unique_data_pmids": unique_pmids,
            "last_updated": last_updated,
            "top_cancer_types": sorted(
                cancer_types.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


def main():
    """Main function for testing."""
    fetcher = StudyFetcher()
    studies = fetcher.fetch_all_studies(force_refresh=True)

    stats = fetcher.get_summary_stats()
    logger.info(f"Summary: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
