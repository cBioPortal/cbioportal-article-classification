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
@click.option('--force-pmid', multiple=True, help='Force PDF re-download for specified PubMed IDs (repeatable)')
@click.option('--retry-failed', is_flag=True, help='Retry all previously failed PDF downloads')
@click.option(
    '--mode',
    type=click.Choice(['all', 'citations', 'pdfs'], case_sensitive=False),
    default='all',
    help='Choose whether to fetch citations, PDFs, or both',
)
def fetch(force, max_downloads, force_pmid, retry_failed, mode):
    """Fetch citations and/or PDFs based on mode."""
    mode = mode.lower()

    if mode in ('all', 'citations'):
        logger.info("Starting citation fetch process...")
        fetcher = CitationFetcher()
        citations_data = fetcher.fetch_all_citations(force_refresh=force)
        logger.info(f"Total citations: {citations_data['total_citations']}")
    else:
        fetcher = CitationFetcher()
        citations_data = fetcher.citations_data

    if mode in ('all', 'pdfs'):
        citations_file = METADATA_DIR / "citations.json"
        if not citations_file.exists():
            click.echo(click.style("Error: No citations data found. Run 'fetch --mode citations' first.", fg='red'))
            return

        logger.info("Downloading PDFs...")
        downloader = PDFDownloader(citation_fetcher=fetcher)
        downloaded = downloader.download_all_pdfs(
            citations_data,
            max_downloads=max_downloads,
            force_paper_ids=set(force_pmid) if force_pmid else None,
            retry_failed=retry_failed
        )
        logger.info(f"Downloaded {downloaded} PDFs")


@cli.command(name='download-pdfs')
@click.option('--max-downloads', type=int, help='Maximum number of PDFs to download')
@click.option('--force-pmid', multiple=True, help='Force PDF re-download for specified PubMed IDs (repeatable)')
@click.option('--retry-failed', is_flag=True, help='Retry all previously failed PDF downloads')
def download_pdfs(max_downloads, force_pmid, retry_failed):
    """Download PDFs using existing citation metadata."""
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
        force_paper_ids=set(force_pmid) if force_pmid else None,
        retry_failed=retry_failed
    )

    logger.info(f"Downloaded {downloaded} PDFs")


@cli.command()
@click.option('--max-papers', type=int, help='Maximum number of papers to classify')
@click.option('--reclassify', is_flag=True, help='Re-classify existing papers')
@click.option(
    '--source',
    type=click.Choice(['auto', 'pdf', 'sentences', 'both'], case_sensitive=False),
    default='auto',
    help='Text source for classification: auto (PDF→abstract), pdf (PDF only), sentences (citation sentences), or both (PDF+sentences)'
)
def classify(max_papers, reclassify, source):
    """Classify papers using LLM via AWS Bedrock."""
    logger.info(f"Starting classification process (source: {source})...")

    # Load citations data
    citations_file = METADATA_DIR / "citations.json"
    if not citations_file.exists():
        click.echo(click.style("Error: No citations data found. Run 'fetch' command first!", fg='red'))
        return

    # Check if citation sentences are needed
    if source in ('sentences', 'both'):
        citation_sentences_file = METADATA_DIR / "citation_sentences.json"
        if not citation_sentences_file.exists():
            click.echo(click.style(f"Warning: No citation sentences found. Run 'extract-citations' first!", fg='yellow'))
            if source == 'sentences':
                click.echo(click.style("Cannot proceed with --source=sentences without citation data.", fg='red'))
                return

    with open(citations_file, "r") as f:
        citations_data = json.load(f)

    # Classify
    classifier = PaperClassifier()
    df = classifier.classify_all_papers(
        citations_data,
        max_papers=max_papers,
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
    if not skip_plots:
        plot_filename = analyzer.create_visualizations()
        research_plot_filename = analyzer.create_research_area_plot()

    # Generate report with plot references
    report_path = analyzer.generate_report(
        plot_filename=plot_filename,
        research_plot_filename=research_plot_filename
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
        for pmid_data in citations_data.get("papers", {}).values():
            for citation in pmid_data.get("citations", []):
                if citation.get("pdf_downloaded"):
                    pdfs_downloaded += 1

        click.echo(click.style("Citations:", fg='cyan', bold=True))
        click.echo(f"  Total papers: {click.style(str(total_citations), fg='green')}")
        click.echo(f"  PDFs downloaded: {click.style(str(pdfs_downloaded), fg='green')} ({pdfs_downloaded/total_citations*100:.1f}%)" if total_citations > 0 else f"  PDFs downloaded: {click.style(str(pdfs_downloaded), fg='green')}")
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

        # Count by schema version
        version_counts = {}
        text_source_counts = {"pdf": 0, "abstract": 0, "none": 0, "unknown": 0}

        for paper_id, classification in classifications.items():
            version = classification.get("schema_version", 0)
            version_counts[version] = version_counts.get(version, 0) + 1

            text_source = classification.get("text_source")
            if text_source in text_source_counts:
                text_source_counts[text_source] += 1
            elif text_source is None:
                text_source_counts["unknown"] += 1

        click.echo(click.style("Classifications:", fg='cyan', bold=True))
        click.echo(f"  Total classified: {click.style(str(total), fg='green')}")
        click.echo(f"  Current schema version: {click.style(f'v{CLASSIFICATION_SCHEMA_VERSION}', fg='yellow')}")

        # Show version breakdown if there are outdated papers
        outdated = sum(count for version, count in version_counts.items() if version < CLASSIFICATION_SCHEMA_VERSION)
        if outdated > 0:
            click.echo(f"  Outdated schema: {click.style(str(outdated), fg='yellow')} papers (will be re-classified on next run)")

        # Show text source breakdown
        if text_source_counts["pdf"] > 0 or text_source_counts["abstract"] > 0:
            click.echo(f"\n  Text sources:")
            if text_source_counts["pdf"] > 0:
                click.echo(f"    From PDF: {click.style(str(text_source_counts['pdf']), fg='green')} ({text_source_counts['pdf']/total*100:.1f}%)")
            if text_source_counts["abstract"] > 0:
                click.echo(f"    From Abstract: {click.style(str(text_source_counts['abstract']), fg='green')} ({text_source_counts['abstract']/total*100:.1f}%)")
            if text_source_counts["unknown"] > 0:
                click.echo(f"    Unknown: {click.style(str(text_source_counts['unknown']), fg='yellow')} ({text_source_counts['unknown']/total*100:.1f}%)")
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


@cli.command(name='fetch-studies')
@click.option('--force', is_flag=True, help='Force re-fetch of studies metadata')
def fetch_studies(force):
    """Fetch cBioPortal study metadata from API."""
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


@cli.command(name='extract-citations')
@click.option('--max-papers', type=int, help='Maximum number of papers to process')
@click.option('--force', is_flag=True, help='Force re-extraction even if data exists')
@click.option('--workers', type=int, default=10, help='Number of parallel workers (default: 10)')
def extract_citations(max_papers, force, workers):
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
