"""Module for parsing JATS XML files to extract citation contexts."""
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JATSParser:
    """Parser for JATS XML files to extract citation contexts."""

    # Common JATS XML namespaces used by different publishers
    NAMESPACES = {
        'jats': 'http://jats.nlm.nih.gov',
        'xlink': 'http://www.w3.org/1999/xlink',
        'mml': 'http://www.w3.org/1998/Math/MathML',
    }

    def __init__(self):
        """Initialize the JATS parser."""
        pass

    def _register_namespaces(self):
        """Register XML namespaces for cleaner output."""
        for prefix, uri in self.NAMESPACES.items():
            ET.register_namespace(prefix, uri)

    def _find_with_fallback(self, element: ET.Element, tag: str, namespaces: Optional[Dict] = None) -> List[ET.Element]:
        """Find elements with namespace fallback.

        Tries with namespace first, then without if not found.

        Args:
            element: XML element to search in
            tag: Tag name to find
            namespaces: Namespace dict

        Returns:
            List of matching elements
        """
        if namespaces:
            # Try with namespace
            for ns_prefix in ['', 'jats:']:
                try:
                    elements = element.findall(f'.//{ns_prefix}{tag}', namespaces)
                    if elements:
                        return elements
                except Exception:
                    continue

        # Fallback: try without namespace
        try:
            return element.findall(f'.//{tag}')
        except Exception:
            return []

    def _get_text_content(self, element: ET.Element, recursive: bool = True) -> str:
        """Extract all text content from an element.

        Args:
            element: XML element
            recursive: If True, get text from all child elements

        Returns:
            Concatenated text content
        """
        if element is None:
            return ""

        if recursive:
            # Get all text including nested elements
            return ''.join(element.itertext()).strip()
        else:
            # Get only direct text
            return (element.text or "").strip()

    def parse_references(self, xml_path: Path) -> Dict[str, str]:
        """Parse reference list and build citation ID to PMID mapping.

        Args:
            xml_path: Path to JATS XML file

        Returns:
            Dict mapping citation IDs (e.g., 'ref1', 'bib23') to PMIDs
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            citation_map = {}

            # Find reference list (usually in <back><ref-list>)
            ref_lists = self._find_with_fallback(root, 'ref-list', self.NAMESPACES)

            for ref_list in ref_lists:
                refs = self._find_with_fallback(ref_list, 'ref', self.NAMESPACES)

                for ref in refs:
                    # Get reference ID (used in <xref> tags)
                    ref_id = ref.get('id')
                    if not ref_id:
                        continue

                    # Find PMID in various possible locations
                    pmid = None

                    # Look for <pub-id pub-id-type="pmid">
                    pub_ids = self._find_with_fallback(ref, 'pub-id', self.NAMESPACES)
                    for pub_id in pub_ids:
                        if pub_id.get('pub-id-type') == 'pmid':
                            pmid = self._get_text_content(pub_id)
                            break

                    # Alternative: look for <article-id pub-id-type="pmid">
                    if not pmid:
                        article_ids = self._find_with_fallback(ref, 'article-id', self.NAMESPACES)
                        for article_id in article_ids:
                            if article_id.get('pub-id-type') == 'pmid':
                                pmid = self._get_text_content(article_id)
                                break

                    if pmid:
                        # Clean PMID (remove any non-digit characters)
                        pmid_clean = re.sub(r'\D', '', pmid)
                        if pmid_clean:
                            citation_map[ref_id] = pmid_clean

            logger.debug(f"Found {len(citation_map)} references with PMIDs in {xml_path.name}")
            return citation_map

        except Exception as e:
            logger.error(f"Error parsing references from {xml_path}: {e}")
            return {}

    def extract_citation_contexts(
        self,
        xml_path: Path,
        target_pmids: Set[str],
        context_keywords: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """Extract citation contexts for specific PMIDs or keywords.

        Args:
            xml_path: Path to JATS XML file
            target_pmids: Set of PMIDs to find citations for
            context_keywords: Optional keywords to search for (e.g., "cBioPortal")

        Returns:
            Tuple of (pmid_citation_contexts, keyword_contexts)
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # First, build the reference map
            citation_map = self.parse_references(xml_path)

            # Reverse map: PMID -> ref IDs
            pmid_to_refs = {}
            for ref_id, pmid in citation_map.items():
                if pmid not in pmid_to_refs:
                    pmid_to_refs[pmid] = []
                pmid_to_refs[pmid].append(ref_id)

            # Find which target PMIDs are in this paper
            relevant_ref_ids = set()
            for pmid in target_pmids:
                if pmid in pmid_to_refs:
                    relevant_ref_ids.update(pmid_to_refs[pmid])

            pmid_contexts = []
            keyword_contexts = []

            # Find all paragraphs in the body
            body = self._find_with_fallback(root, 'body', self.NAMESPACES)
            if not body:
                return (pmid_contexts, keyword_contexts)

            # Get all paragraphs (<p> tags)
            paragraphs = []
            for body_elem in body:
                paragraphs.extend(self._find_with_fallback(body_elem, 'p', self.NAMESPACES))

            # Search each paragraph for citations and keywords
            for para in paragraphs:
                para_text = self._get_text_content(para)

                # Skip empty paragraphs
                if not para_text or len(para_text) < 20:
                    continue

                # Check for PMID citations via <xref> tags
                xrefs = self._find_with_fallback(para, 'xref', self.NAMESPACES)
                has_target_citation = False

                for xref in xrefs:
                    ref_type = xref.get('ref-type')
                    rid = xref.get('rid')

                    # Check if this is a bibliographic reference to one of our target PMIDs
                    if ref_type == 'bibr' and rid in relevant_ref_ids:
                        has_target_citation = True
                        break

                if has_target_citation:
                    # Clean up text (remove extra whitespace)
                    clean_text = ' '.join(para_text.split())
                    pmid_contexts.append(clean_text)

                # Check for keyword mentions
                if context_keywords:
                    for keyword in context_keywords:
                        if re.search(rf'\b{re.escape(keyword)}\b', para_text, re.IGNORECASE):
                            clean_text = ' '.join(para_text.split())
                            # Avoid duplicates
                            if clean_text not in pmid_contexts:
                                keyword_contexts.append(clean_text)
                            break

            logger.debug(
                f"Extracted {len(pmid_contexts)} PMID citations and {len(keyword_contexts)} keyword mentions from {xml_path.name}"
            )
            return (pmid_contexts, keyword_contexts)

        except Exception as e:
            logger.error(f"Error extracting citation contexts from {xml_path}: {e}")
            return ([], [])

    def extract_full_text(self, xml_path: Path) -> str:
        """Extract full text content from JATS XML.

        Args:
            xml_path: Path to JATS XML file

        Returns:
            Full text of the article
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            text_parts = []

            # Extract abstract
            abstracts = self._find_with_fallback(root, 'abstract', self.NAMESPACES)
            for abstract in abstracts:
                abstract_text = self._get_text_content(abstract)
                if abstract_text:
                    text_parts.append("=== Abstract ===")
                    text_parts.append(abstract_text)

            # Extract body
            body_elements = self._find_with_fallback(root, 'body', self.NAMESPACES)
            for body in body_elements:
                # Get all sections
                sections = self._find_with_fallback(body, 'sec', self.NAMESPACES)
                if sections:
                    for section in sections:
                        # Get section title
                        titles = self._find_with_fallback(section, 'title', self.NAMESPACES)
                        for title in titles:
                            title_text = self._get_text_content(title)
                            if title_text:
                                text_parts.append(f"\n=== {title_text} ===")

                        # Get paragraphs in section
                        paragraphs = self._find_with_fallback(section, 'p', self.NAMESPACES)
                        for para in paragraphs:
                            para_text = self._get_text_content(para)
                            if para_text:
                                text_parts.append(para_text)
                else:
                    # No sections, just get all paragraphs
                    paragraphs = self._find_with_fallback(body, 'p', self.NAMESPACES)
                    for para in paragraphs:
                        para_text = self._get_text_content(para)
                        if para_text:
                            text_parts.append(para_text)

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting full text from {xml_path}: {e}")
            return ""


def main():
    """Main function for testing JATS parser."""
    # This would be used for testing with a sample XML file
    parser = JATSParser()

    # Example: Parse a sample XML file
    xml_file = Path("sample.xml")
    if xml_file.exists():
        citation_map = parser.parse_references(xml_file)
        print(f"Found {len(citation_map)} references with PMIDs")

        target_pmids = {"23550210"}  # cBioPortal paper
        contexts, keywords = parser.extract_citation_contexts(
            xml_file,
            target_pmids,
            context_keywords=["cBioPortal"]
        )
        print(f"Found {len(contexts)} citation contexts")
        print(f"Found {len(keywords)} keyword mentions")
    else:
        print("No sample XML file found")


if __name__ == "__main__":
    main()
