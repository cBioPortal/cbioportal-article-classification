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
@click.option('--download-pdfs', is_flag=True, help='Download PDFs after fetching')
@click.option('--max-downloads', type=int, help='Maximum number of PDFs to download')
def fetch(force, download_pdfs, max_downloads):
    """Fetch citations from Google Scholar."""
    logger.info("Starting citation fetch process...")

    fetcher = CitationFetcher()
    citations_data = fetcher.fetch_all_citations(force_refresh=force)

    logger.info(f"Total citations: {citations_data['total_citations']}")

    if download_pdfs:
        logger.info("Downloading PDFs...")
        downloader = PDFDownloader()
        downloaded = downloader.download_all_pdfs(citations_data, max_downloads=max_downloads)
        logger.info(f"Downloaded {downloaded} PDFs")

        # Save updated metadata with PDF info
        fetcher._save_metadata()


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
    if not skip_plots:
        analyzer.create_visualizations()
        analyzer.create_research_area_plot()

    # Generate report
    report_path = analyzer.generate_report()

    click.echo(click.style("Analysis complete!", fg='green'))
    click.echo(f"Report: {report_path}")
    if not skip_plots:
        click.echo(f"Plots saved to: {analyzer.PLOTS_DIR if hasattr(analyzer, 'PLOTS_DIR') else 'output/plots'}")


@cli.command(name='run-all')
@click.option('--download-pdfs', is_flag=True, help='Download PDFs during fetch')
@click.option('--max-downloads', type=int, help='Maximum PDFs to download')
@click.option('--max-papers', type=int, help='Maximum papers to classify')
def run_all(download_pdfs, max_downloads, max_papers):
    """Run the complete pipeline: fetch, classify, analyze."""
    click.echo(click.style("Running complete pipeline...", fg='blue', bold=True))

    # Step 1: Fetch
    click.echo("\n" + click.style("Step 1/3: Fetching citations", fg='cyan'))
    from click.testing import CliRunner
    runner = CliRunner()

    ctx = click.get_current_context()
    ctx.invoke(fetch, force=False, download_pdfs=download_pdfs, max_downloads=max_downloads)

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
        click.echo(f"  PDFs downloaded: {click.style(str(pdfs_downloaded), fg='green')}")
        click.echo(f"  Last updated: {last_updated}")
    else:
        click.echo(click.style("Citations:", fg='cyan', bold=True) + " No data (run 'fetch' command)")

    click.echo()

    # Classifications
    if classifications_file.exists():
        with open(classifications_file, "r") as f:
            classifications = json.load(f)

        click.echo(click.style("Classifications:", fg='cyan', bold=True))
        click.echo(f"  Total classified: {click.style(str(len(classifications)), fg='green')}")
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
