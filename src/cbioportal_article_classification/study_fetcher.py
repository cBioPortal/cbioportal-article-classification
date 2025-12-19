"""Module for fetching and managing cBioPortal study metadata."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import requests

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
