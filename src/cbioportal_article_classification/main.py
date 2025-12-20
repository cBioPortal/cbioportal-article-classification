#!/usr/bin/env python3
"""Main CLI for cBioPortal article classification tool."""
import json
import logging
from pathlib import Path

import click

from .fetcher import CitationFetcher, PDFDownloader
from .classifier import PaperClassifier
from .analyzer import UsageAnalyzer
from .study_fetcher import StudyFetcher
from .citation_extractor import CitationExtractor
from .config import METADATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """cBioPortal Article Classification Tool

    Fetch, classify, and analyze papers that cite cBioPortal.
    """
    pass


@cli.command()
@click.option('--force', is_flag=True, help='Force re-fetch of all citations')
@click.option('--max-downloads', type=int, help='Maximum number of PDFs to download')
@click.option('--pmid', multiple=True, help='Specify PMIDs: with citing-article-metadata refreshes metadata, with citing-article-files forces download (repeatable)')
@click.option('--seed-only', is_flag=True, help='Refresh seed PMIDs list without fetching citation metadata')
@click.option('--retry-failed', is_flag=True, help='Retry all previously failed PDF downloads')
@click.option(
    '--mode',
    type=click.Choice(['all', 'citing-article-metadata', 'citing-article-files'], case_sensitive=False),
    default='all',
    help='Choose what to fetch: citing-article-metadata (PubMed data), citing-article-files (PDFs/XMLs), or all',
)
def fetch(force, max_downloads, pmid, seed_only, retry_failed, mode):
    """Fetch citing article metadata and/or files (PDFs/XMLs) based on mode."""
    mode = mode.lower()

    if mode in ('all', 'citing-article-metadata'):
        logger.info("Starting citation fetch process...")
        fetcher = CitationFetcher()
        if seed_only:
            fetcher.refresh_seed_only(force_refresh=force)
            citations_data = fetcher.citations_data
        elif pmid:
            citations_data = fetcher.refresh_specific_papers(set(pmid), force=force)
        else:
            citations_data = fetcher.fetch_all_citations(force_refresh=force)
        logger.info(f"Total citations: {citations_data['total_citations']}")
    else:
        fetcher = CitationFetcher()
        citations_data = fetcher.citations_data

    if mode in ('all', 'citing-article-files'):
        citations_file = METADATA_DIR / "citations.json"
        if not citations_file.exists():
            click.echo(click.style("Error: No citations data found. Run 'fetch --mode citing-article-metadata' first.", fg='red'))
            return

        logger.info("Downloading PDFs and JATS XML files...")

        downloader = PDFDownloader(citation_fetcher=fetcher)
        downloaded = downloader.download_all_pdfs(
            citations_data,
            max_downloads=max_downloads,
            force_paper_ids=set(pmid) if pmid else None,
            retry_failed=retry_failed,
            download_xml=True
        )
        logger.info(f"Downloaded {downloaded} PDFs")


@cli.command(name='download-pdfs')
@click.option('--max-downloads', type=int, help='Maximum number of PDFs to download')
@click.option('--pmid', multiple=True, help='Force download for specific PubMed IDs (repeatable)')
@click.option('--retry-failed', is_flag=True, help='Retry all previously failed PDF downloads')
def download_pdfs(max_downloads, pmid, retry_failed):
    """Download PDFs and XML files using existing citation metadata."""
    citations_file = METADATA_DIR / "citations.json"
    if not citations_file.exists():
        click.echo(click.style("Error: No citations data found. Run 'fetch' first to build metadata.", fg='red'))
        return

    fetcher = CitationFetcher()
    citations_data = fetcher.citations_data

    downloader = PDFDownloader(citation_fetcher=fetcher)
    downloaded = downloader.download_all_pdfs(
        citations_data,
        max_downloads=max_downloads,
        force_paper_ids=set(pmid) if pmid else None,
        retry_failed=retry_failed,
        download_xml=True
    )

    logger.info(f"Downloaded {downloaded} PDFs")


@cli.command()
@click.option('--max-papers', type=int, help='Maximum number of papers to classify')
@click.option('--pmid', multiple=True, help='Classify specific PMIDs only (repeatable)')
@click.option('--reclassify', is_flag=True, help='Re-classify existing papers')
@click.option(
    '--source',
    type=click.Choice(['auto', 'pdf', 'sentences', 'both'], case_sensitive=False),
    default='auto',
    help='Text source: auto (sentences→PDF→abstract, prioritizes focused context), pdf (full PDF only), sentences (citation contexts only), both (PDF+sentences combined)'
)
def classify(max_papers, pmid, reclassify, source):
    """Classify papers using LLM via AWS Bedrock."""
    logger.info(f"Starting classification process (source: {source})...")

    # Load citations data
    citations_file = METADATA_DIR / "citations.json"
    if not citations_file.exists():
        click.echo(click.style("Error: No citations data found. Run 'fetch' command first!", fg='red'))
        return

    # Check if citation sentences are needed
    if source in ('auto', 'sentences', 'both'):
        citation_sentences_file = METADATA_DIR / "citation_sentences.json"
        if not citation_sentences_file.exists():
            if source == 'sentences':
                click.echo(click.style("Error: No citation sentences found. Run 'extract-citations' first!", fg='red'))
                return
            elif source == 'auto':
                click.echo(click.style("Warning: No citation sentences found. Will fall back to PDF/abstract. Run 'extract-citations' for better results.", fg='yellow'))
            else:  # both
                click.echo(click.style("Warning: No citation sentences found. Will use PDF only. Run 'extract-citations' to combine both sources.", fg='yellow'))

    with open(citations_file, "r") as f:
        citations_data = json.load(f)

    # Classify
    classifier = PaperClassifier()
    df = classifier.classify_all_papers(
        citations_data,
        max_papers=max_papers,
        paper_ids=set(pmid) if pmid else None,
        skip_existing=not reclassify,
        source=source
    )

    logger.info(f"Classification complete! Processed {len(df)} papers")


@cli.command()
@click.option('--skip-plots', is_flag=True, help='Skip generating plots')
def analyze(skip_plots):
    """Analyze and visualize classification results."""
    logger.info("Starting analysis...")

    analyzer = UsageAnalyzer()

    # Generate visualizations
    plot_filename = None
    research_plot_filename = None
    citation_plot_filename = None
    if not skip_plots:
        plot_filename = analyzer.create_visualizations()
        research_plot_filename = analyzer.create_research_area_plot()
        citation_plot_filename = analyzer.create_citation_attribution_plot()

    # Generate report with plot references
    report_path = analyzer.generate_report(
        plot_filename=plot_filename,
        research_plot_filename=research_plot_filename,
        citation_plot_filename=citation_plot_filename
    )

    click.echo(click.style("Analysis complete!", fg='green'))
    click.echo(f"Report: {report_path}")
    if not skip_plots:
        click.echo(f"Plots saved to: {analyzer.PLOTS_DIR if hasattr(analyzer, 'PLOTS_DIR') else 'output/plots'}")


@cli.command(name='run-all')
@click.option('--download-pdfs', is_flag=True, help='Download PDFs during fetch')
@click.option('--max-downloads', type=int, help='Maximum PDFs to download')
@click.option('--max-papers', type=int, help='Maximum papers to classify')
@click.option('--force-pmid', multiple=True, help='Force PDF re-download for specified PubMed IDs (repeatable)')
def run_all(download_pdfs, max_downloads, max_papers, force_pmid):
    """Run the complete pipeline: fetch, classify, analyze."""
    click.echo(click.style("Running complete pipeline...", fg='blue', bold=True))

    # Step 1: Fetch
    click.echo("\n" + click.style("Step 1/3: Fetching citations", fg='cyan'))
    from click.testing import CliRunner
    runner = CliRunner()

    ctx = click.get_current_context()
    ctx.invoke(
        fetch,
        force=False,
        pmid=(),
        download_pdfs=download_pdfs,
        max_downloads=max_downloads,
        force_pmid=force_pmid,
    )

    # Step 2: Classify
    click.echo("\n" + click.style("Step 2/3: Classifying papers", fg='cyan'))
    ctx.invoke(classify, max_papers=max_papers, reclassify=False, source='auto')

    # Step 3: Analyze
    click.echo("\n" + click.style("Step 3/3: Analyzing results", fg='cyan'))
    ctx.invoke(analyze, skip_plots=False)

    click.echo("\n" + click.style("Pipeline complete!", fg='green', bold=True))


@cli.command()
def status():
    """Show current status of the database."""
    from .config import CLASSIFICATION_SCHEMA_VERSION

    citations_file = METADATA_DIR / "citations.json"
    classifications_file = METADATA_DIR / "classifications.json"
    last_run_file = METADATA_DIR / "last_run.json"

    click.echo("\n" + click.style("=== cBioPortal Article Classification Status ===", bold=True) + "\n")

    # Citations
    if citations_file.exists():
        with open(citations_file, "r") as f:
            citations_data = json.load(f)

        total_citations = citations_data.get("total_citations", 0)
        last_updated = citations_data.get("last_updated", "Never")

        pdfs_downloaded = 0
        xmls_downloaded = 0
        pmc_papers = 0
        oa_open = 0
        oa_closed = 0
        oa_unknown = 0
        for pmid_data in citations_data.get("papers", {}).values():
            for citation in pmid_data.get("citations", []):
                if citation.get("pdf_downloaded"):
                    pdfs_downloaded += 1
                if citation.get("xml_downloaded"):
                    xmls_downloaded += 1
                if citation.get("pmc_id"):
                    pmc_papers += 1
                    xml_oa_state = citation.get("xml_open_access")
                    if xml_oa_state is True or (xml_oa_state is None and citation.get("xml_downloaded")):
                        oa_open += 1
                    elif xml_oa_state is False:
                        oa_closed += 1
                    else:
                        oa_unknown += 1

        click.echo(click.style("Citations:", fg='cyan', bold=True))
        click.echo(f"  Total papers: {click.style(str(total_citations), fg='green')}")
        click.echo(f"  PDFs downloaded: {click.style(str(pdfs_downloaded), fg='green')} ({pdfs_downloaded/total_citations*100:.1f}%)" if total_citations > 0 else f"  PDFs downloaded: {click.style(str(pdfs_downloaded), fg='green')}")
        click.echo(f"  XMLs downloaded: {click.style(str(xmls_downloaded), fg='green')} ({xmls_downloaded/total_citations*100:.1f}%)" if total_citations > 0 else f"  XMLs downloaded: {click.style(str(xmls_downloaded), fg='green')}")
        pmc_pct = f" ({pmc_papers/total_citations*100:.1f}%)" if total_citations > 0 and pmc_papers > 0 else ""
        click.echo(f"  Papers with PMC IDs: {click.style(str(pmc_papers), fg='green')}{pmc_pct}")
        if pmc_papers > 0:
            click.echo(
                f"    Open Access: {click.style(str(oa_open), fg='green')} "
                f"({oa_open/pmc_papers*100:.1f}% of PMC)"
            )
            click.echo(
                f"    Not Open Access: {click.style(str(oa_closed), fg='yellow')} "
                f"({oa_closed/pmc_papers*100:.1f}% of PMC)"
            )
            click.echo(
                f"    Unknown: {click.style(str(oa_unknown), fg='yellow')} "
                f"({oa_unknown/pmc_papers*100:.1f}% of PMC)"
            )
        click.echo(f"  Last updated: {last_updated}")

        # Show breakdown by PMID
        click.echo(f"\n  Papers citing each cBioPortal paper:")
        for pmid, pmid_data in citations_data.get("papers", {}).items():
            citation_count = len(pmid_data.get("citations", []))
            click.echo(f"    PMID {pmid}: {click.style(str(citation_count), fg='green')} citations")
    else:
        click.echo(click.style("Citations:", fg='cyan', bold=True) + " No data (run 'fetch' command)")

    click.echo()

    # Classifications
    if classifications_file.exists():
        with open(classifications_file, "r") as f:
            classifications = json.load(f)

        total = len(classifications)

        # Count by schema version and text source
        version_counts = {}
        text_source_counts = {"pdf": 0, "abstract": 0, "sentences": 0, "none": 0, "unknown": 0}
        # Track schema version per text source
        outdated_by_source = {"pdf": 0, "abstract": 0, "sentences": 0, "none": 0, "unknown": 0}
        current_by_source = {"pdf": 0, "abstract": 0, "sentences": 0, "none": 0, "unknown": 0}

        for paper_id, classification in classifications.items():
            version = classification.get("schema_version", 0)
            version_counts[version] = version_counts.get(version, 0) + 1

            text_source = classification.get("text_source")
            if text_source in text_source_counts:
                text_source_counts[text_source] += 1
                if version < CLASSIFICATION_SCHEMA_VERSION:
                    outdated_by_source[text_source] += 1
                else:
                    current_by_source[text_source] += 1
            elif text_source is None:
                text_source_counts["unknown"] += 1
                if version < CLASSIFICATION_SCHEMA_VERSION:
                    outdated_by_source["unknown"] += 1
                else:
                    current_by_source["unknown"] += 1

        click.echo(click.style("Classifications:", fg='cyan', bold=True))
        click.echo(f"  Total classified: {click.style(str(total), fg='green')}")
        click.echo(f"  Current schema version: {click.style(f'v{CLASSIFICATION_SCHEMA_VERSION}', fg='yellow')}")

        # Show version breakdown if there are outdated papers
        outdated = sum(count for version, count in version_counts.items() if version < CLASSIFICATION_SCHEMA_VERSION)
        current_schema = version_counts.get(CLASSIFICATION_SCHEMA_VERSION, 0)
        if outdated > 0:
            click.echo(f"  Schema v{CLASSIFICATION_SCHEMA_VERSION}: {click.style(str(current_schema), fg='green')} papers")
            click.echo(f"  Outdated schemas: {click.style(str(outdated), fg='yellow')} papers total")

        # Show text source breakdown with schema status
        click.echo(f"\n  Text sources:")
        for source in ["sentences", "pdf", "abstract", "unknown"]:
            total_source = text_source_counts[source]
            if total_source > 0:
                current_count = current_by_source[source]
                outdated_count = outdated_by_source[source]
                source_display = source.capitalize() if source != "pdf" else "PDF"

                pct = total_source/total*100
                status_str = ""
                if outdated_count > 0:
                    status_str = f" ({click.style(str(current_count), fg='green')} current, {click.style(str(outdated_count), fg='yellow')} outdated)"
                else:
                    status_str = f" ({click.style('all current', fg='green')})"

                click.echo(f"    {source_display}: {click.style(str(total_source), fg='green')} ({pct:.1f}%){status_str}")
    else:
        click.echo(click.style("Classifications:", fg='cyan', bold=True) + " No data (run 'classify' command)")

    click.echo()

    # Last run
    if last_run_file.exists():
        with open(last_run_file, "r") as f:
            last_run = json.load(f)

        click.echo(click.style("Last Run:", fg='cyan', bold=True))
        click.echo(f"  Timestamp: {last_run.get('timestamp', 'N/A')}")
        click.echo(f"  Papers fetched: {click.style(str(last_run.get('papers_fetched', 0)), fg='green')}")
    else:
        click.echo(click.style("Last Run:", fg='cyan', bold=True) + " No data")

    click.echo()


@cli.command(name='fetch-reference-data')
@click.option('--force', is_flag=True, help='Force re-fetch of all reference data')
def fetch_reference_data(force):
    """Fetch cBioPortal studies and OncoTree tumor types (reference data for classification)."""
    import requests
    import time
    from Bio import Entrez
    from .config import NCBI_EMAIL, NCBI_API_KEY, FETCH_DELAY_SECONDS

    # Configure Entrez
    Entrez.email = NCBI_EMAIL or "user@example.com"
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY

    click.echo(click.style("\n=== Fetching Reference Data ===\n", bold=True))

    # 1. Fetch cBioPortal studies
    logger.info("Fetching cBioPortal study metadata...")
    fetcher = StudyFetcher()
    studies_data = fetcher.fetch_all_studies(force_refresh=force)
    stats = fetcher.get_summary_stats()

    click.echo(click.style("cBioPortal Studies:", fg='cyan', bold=True))
    click.echo(f"  Total studies: {click.style(str(stats.get('total_studies', 0)), fg='green')}")
    click.echo(f"  Studies with PMID: {click.style(str(stats.get('studies_with_pmid', 0)), fg='green')}")
    click.echo(f"  Unique data PMIDs: {click.style(str(stats.get('unique_data_pmids', 0)), fg='green')}")
    click.echo()

    # 2. Fetch OncoTree tumor types
    logger.info("Fetching OncoTree tumor types...")
    oncotree_file = METADATA_DIR / "oncotree_tumor_types.json"

    try:
        response = requests.get("https://oncotree.info/api/tumorTypes/tree", timeout=30)
        response.raise_for_status()

        oncotree_data = response.json()

        # Count tumor types
        def count_codes(node):
            count = 1 if node.get('code') != 'TISSUE' else 0
            for child in node.get('children', {}).values():
                count += count_codes(child)
            return count

        total_codes = count_codes(oncotree_data.get('TISSUE', {}))

        # Save
        with open(oncotree_file, 'w') as f:
            json.dump(oncotree_data, f, indent=2)

        click.echo(click.style("OncoTree Tumor Types:", fg='cyan', bold=True))
        click.echo(f"  Total tumor type codes: {click.style(str(total_codes), fg='green')}")
        click.echo(f"  Saved to: {oncotree_file}")

    except Exception as e:
        click.echo(click.style(f"Error fetching OncoTree data: {e}", fg='red'))

    click.echo()

    # 3. Fetch metadata for data source papers (referenced by cBioPortal studies)
    logger.info("Fetching metadata for data source papers...")
    data_source_file = METADATA_DIR / "data_source_papers.json"

    # Skip if already exists and not forcing
    if data_source_file.exists() and not force:
        click.echo(click.style("Data Source Papers:", fg='cyan', bold=True))
        with open(data_source_file) as f:
            existing_data = json.load(f)
        click.echo(f"  Cached data source papers: {click.style(str(len(existing_data)), fg='green')}")
        click.echo(f"  Use --force to re-fetch")
    else:
        try:
            # Load pmid_to_studies mapping from studies data
            pmid_to_studies = studies_data.get('pmid_to_studies', {})
            data_source_pmids = list(pmid_to_studies.keys())

            click.echo(click.style("Data Source Papers:", fg='cyan', bold=True))
            click.echo(f"  Fetching metadata for {click.style(str(len(data_source_pmids)), fg='yellow')} data source papers...")

            data_source_papers = {}
            batch_size = 100  # PubMed allows up to 200, but 100 is safer

            for i in range(0, len(data_source_pmids), batch_size):
                batch = data_source_pmids[i:i+batch_size]

                try:
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
                            pmid = str(medline["PMID"])
                            article_data = medline["Article"]

                            # Extract metadata
                            title = str(article_data.get("ArticleTitle", ""))

                            # Authors
                            authors_list = article_data.get("AuthorList", [])
                            authors = ", ".join([
                                f"{a.get('LastName', '')} {a.get('Initials', '')}".strip()
                                for a in authors_list if "LastName" in a
                            ])

                            # Year
                            year = ""
                            article_dates = article_data.get("ArticleDate", [])
                            if article_dates and isinstance(article_dates, list) and len(article_dates) > 0:
                                year = article_dates[0].get("Year", "")
                            if not year:
                                pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                                year = pub_date.get("Year", "")
                                if not year and "MedlineDate" in pub_date:
                                    import re
                                    year_match = re.search(r'\d{4}', str(pub_date["MedlineDate"]))
                                    if year_match:
                                        year = year_match.group()

                            # Journal
                            journal = str(article_data.get("Journal", {}).get("Title", ""))

                            # Store with studies reference
                            data_source_papers[pmid] = {
                                "pmid": pmid,
                                "title": title,
                                "authors": authors,
                                "year": year,
                                "journal": journal,
                                "studies": pmid_to_studies.get(pmid, [])
                            }

                        except Exception as e:
                            logger.warning(f"Error processing article {pmid}: {e}")

                except Exception as e:
                    logger.warning(f"Error fetching batch {i//batch_size + 1}: {e}")

            # Save
            with open(data_source_file, 'w') as f:
                json.dump(data_source_papers, f, indent=2)

            click.echo(f"  Fetched metadata for {click.style(str(len(data_source_papers)), fg='green')} papers")
            click.echo(f"  Saved to: {data_source_file}")

        except Exception as e:
            click.echo(click.style(f"Error fetching data source papers: {e}", fg='red'))

    click.echo()
    click.echo(click.style("Reference data fetch complete!", fg='green', bold=True))


@cli.command(name='fetch-studies')
@click.option('--force', is_flag=True, help='Force re-fetch of studies metadata')
@click.option('--skip-citations', is_flag=True, help='Skip fetching PubMed citation counts')
def fetch_studies(force, skip_citations):
    """Fetch cBioPortal study metadata from API (legacy - use fetch-reference-data instead)."""
    logger.info("Fetching cBioPortal study metadata...")

    fetcher = StudyFetcher()
    studies_data = fetcher.fetch_all_studies(force_refresh=force)

    stats = fetcher.get_summary_stats()

    click.echo(click.style("\n=== cBioPortal Studies Summary ===\n", bold=True))
    click.echo(f"Total studies: {click.style(str(stats.get('total_studies', 0)), fg='green')}")
    click.echo(f"Studies with PMID: {click.style(str(stats.get('studies_with_pmid', 0)), fg='green')}")
    click.echo(f"Unique data PMIDs: {click.style(str(stats.get('unique_data_pmids', 0)), fg='green')}")
    click.echo(f"Last updated: {stats.get('last_updated', 'Never')}")

    click.echo(f"\nTop cancer types:")
    for cancer_type, count in stats.get('top_cancer_types', [])[:5]:
        click.echo(f"  {cancer_type}: {click.style(str(count), fg='green')}")

    # Fetch citation counts unless skipped
    if not skip_citations:
        click.echo(click.style("\n=== Fetching PubMed Citation Counts ===\n", bold=True))
        citation_stats = fetcher.fetch_citation_counts()

        if citation_stats:
            click.echo(f"Successfully fetched: {click.style(str(citation_stats.get('successfully_fetched', 0)), fg='green')}/{citation_stats.get('total_studies_with_pmid', 0)} studies")
            if citation_stats.get('failed_fetches', 0) > 0:
                click.echo(f"Failed: {click.style(str(citation_stats.get('failed_fetches', 0)), fg='red')}")
    else:
        click.echo(click.style("\nSkipped citation count fetching (use without --skip-citations to fetch)", fg='yellow'))


@cli.command(name='extract-citations')
@click.option('--max-papers', type=int, help='Maximum number of papers to process')
@click.option('--pmid', multiple=True, help='Extract citations for specific PMIDs only (repeatable)')
@click.option('--force', is_flag=True, help='Force re-extraction even if data exists')
@click.option('--workers', type=int, default=10, help='Number of parallel workers (default: 10)')
def extract_citations(max_papers, pmid, force, workers):
    """Extract citation sentences from PDFs."""
    logger.info("Starting citation extraction process...")

    # Load citations data
    citations_file = METADATA_DIR / "citations.json"
    if not citations_file.exists():
        click.echo(click.style("Error: No citations data found. Run 'fetch' command first!", fg='red'))
        return

    # Check if studies metadata exists
    studies_file = METADATA_DIR / "cbioportal_studies.json"
    if not studies_file.exists():
        click.echo(click.style("Warning: No studies metadata found. Run 'fetch-studies' first for data citation tracking.", fg='yellow'))
        click.echo("Proceeding with cBioPortal citation extraction only...\n")

    with open(citations_file, "r") as f:
        citations_data = json.load(f)

    # Extract citations
    extractor = CitationExtractor()
    stats = extractor.extract_all_citations(
        citations_data,
        max_papers=max_papers,
        paper_ids=set(pmid) if pmid else None,
        force_reextract=force,
        max_workers=workers
    )

    click.echo(click.style("\n=== Citation Extraction Summary ===\n", bold=True))

    if "extraction_run" in stats:
        run_stats = stats["extraction_run"]
        click.echo(f"Processed: {click.style(str(run_stats['total_processed']), fg='green')} papers")
        click.echo(f"Extracted: {click.style(str(run_stats['extracted_count']), fg='green')} papers")
        if run_stats['error_count'] > 0:
            click.echo(f"Errors: {click.style(str(run_stats['error_count']), fg='yellow')}")

    click.echo(f"\nTotal papers with extractions: {click.style(str(stats.get('total_papers_extracted', 0)), fg='green')}")
    click.echo(f"Papers with cBioPortal paper citations: {click.style(str(stats.get('papers_with_cbioportal_paper_citations', 0)), fg='green')}")
    click.echo(f"Papers with cBioPortal platform mentions: {click.style(str(stats.get('papers_with_cbioportal_platform_mentions', 0)), fg='green')}")
    click.echo(f"Papers citing underlying data: {click.style(str(stats.get('papers_with_data_citations', 0)), fg='green')}")
    click.echo(f"Papers citing both cBioPortal and data: {click.style(str(stats.get('papers_citing_both', 0)), fg='green')}")


if __name__ == "__main__":
    cli()
