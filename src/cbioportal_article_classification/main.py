#!/usr/bin/env python3
"""Main CLI for cBioPortal article classification tool."""
import json
import logging
from pathlib import Path

import click

from .fetcher import CitationFetcher, PDFDownloader
from .classifier import PaperClassifier
from .analyzer import UsageAnalyzer
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
def classify(max_papers, reclassify):
    """Classify papers using LLM via AWS Bedrock."""
    logger.info("Starting classification process...")

    # Load citations data
    citations_file = METADATA_DIR / "citations.json"
    if not citations_file.exists():
        click.echo(click.style("Error: No citations data found. Run 'fetch' command first!", fg='red'))
        return

    with open(citations_file, "r") as f:
        citations_data = json.load(f)

    # Classify
    classifier = PaperClassifier()
    df = classifier.classify_all_papers(
        citations_data,
        max_papers=max_papers,
        skip_existing=not reclassify
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
    ctx.invoke(classify, max_papers=max_papers, reclassify=False)

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


if __name__ == "__main__":
    cli()
