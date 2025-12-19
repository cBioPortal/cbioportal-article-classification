"""Module for analyzing and visualizing classification results."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    METADATA_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    CLASSIFICATION_CATEGORIES,
    CBIOPORTAL_PMIDS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
sns.set_palette("husl")


class UsageAnalyzer:
    """Analyzes cBioPortal usage patterns from classified papers."""

    def __init__(self):
        """Initialize analyzer with classification data."""
        self.classifications_file = METADATA_DIR / "classifications.json"
        self.classifications_csv = METADATA_DIR / "classifications.csv"
        self.citations_file = METADATA_DIR / "citations.json"
        self.df = self._load_classifications()
        self.citations_df = self._load_citations_metadata()

    def _load_classifications(self) -> pd.DataFrame:
        """Load classifications from CSV or JSON."""
        if self.classifications_csv.exists():
            return pd.read_csv(self.classifications_csv)
        elif self.classifications_file.exists():
            with open(self.classifications_file, "r") as f:
                data = json.load(f)
            return pd.DataFrame(list(data.values()))
        else:
            logger.warning("No classification data found")
            return pd.DataFrame()

    def _load_citations_metadata(self) -> pd.DataFrame:
        """Load citations metadata for bibliometric analysis (deduplicated)."""
        if not self.citations_file.exists():
            logger.warning("No citations metadata found")
            return pd.DataFrame()

        with open(self.citations_file, "r") as f:
            citations_data = json.load(f)

        # Flatten citations from all PMIDs into a single list, tracking which PMIDs each paper cites
        all_citations = []
        paper_to_pmids = {}  # Track which cBioPortal PMIDs each paper cites

        for pmid, pmid_data in citations_data.get("papers", {}).items():
            for citation in pmid_data.get("citations", []):
                paper_id = citation.get("paper_id")

                # Track which cBioPortal papers this cites
                if paper_id not in paper_to_pmids:
                    paper_to_pmids[paper_id] = []
                paper_to_pmids[paper_id].append(pmid)

                # Only add paper once (deduplicate)
                if len(paper_to_pmids[paper_id]) == 1:
                    all_citations.append(citation)

        # Store the mapping for overlap analysis
        self.paper_to_pmids = paper_to_pmids

        return pd.DataFrame(all_citations)

    def explode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Explode list-type category columns for analysis.

        Args:
            df: DataFrame with classification data

        Returns:
            DataFrame with exploded categories
        """
        df_exploded = df.copy()

        for category in CLASSIFICATION_CATEGORIES.keys():
            if category in df_exploded.columns:
                # Convert string representation of lists to actual lists
                if df_exploded[category].dtype == object:
                    df_exploded[category] = df_exploded[category].apply(
                        lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [x]
                    )

        return df_exploded

    def analyze_analysis_types(self) -> pd.DataFrame:
        """Analyze distribution of analysis types.

        Returns:
            DataFrame with analysis type counts
        """
        if self.df.empty or 'analysis_type' not in self.df.columns:
            return pd.DataFrame()

        # Explode the analysis_type lists
        df_temp = self.df.copy()
        df_temp['analysis_type'] = df_temp['analysis_type'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

        analysis_counts = df_temp.explode('analysis_type')['analysis_type'].value_counts()
        return pd.DataFrame({'Analysis Type': analysis_counts.index, 'Count': analysis_counts.values})

    def analyze_cancer_types(self) -> pd.DataFrame:
        """Analyze distribution of cancer types studied.

        Returns:
            DataFrame with cancer type counts
        """
        if self.df.empty or 'cancer_type' not in self.df.columns:
            return pd.DataFrame()

        df_temp = self.df.copy()
        df_temp['cancer_type'] = df_temp['cancer_type'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

        cancer_counts = df_temp.explode('cancer_type')['cancer_type'].value_counts()
        return pd.DataFrame({'Cancer Type': cancer_counts.index, 'Count': cancer_counts.values})

    def analyze_data_sources(self) -> pd.DataFrame:
        """Analyze distribution of data sources used.

        Returns:
            DataFrame with data source counts
        """
        if self.df.empty or 'data_source' not in self.df.columns:
            return pd.DataFrame()

        df_temp = self.df.copy()
        df_temp['data_source'] = df_temp['data_source'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

        source_counts = df_temp.explode('data_source')['data_source'].value_counts()
        return pd.DataFrame({'Data Source': source_counts.index, 'Count': source_counts.values})

    def analyze_text_sources(self) -> pd.DataFrame:
        """Analyze distribution of text sources used for classification.

        Returns:
            DataFrame with text source counts
        """
        if self.df.empty or 'text_source' not in self.df.columns:
            return pd.DataFrame()

        text_source_counts = self.df['text_source'].value_counts()
        return pd.DataFrame({'Text Source': text_source_counts.index, 'Count': text_source_counts.values})

    def analyze_usage_modes(self) -> pd.DataFrame:
        """Analyze how cBioPortal is being used (v4 schema).

        Returns:
            DataFrame with usage mode counts
        """
        if self.df.empty or 'cbioportal_usage_mode' not in self.df.columns:
            return pd.DataFrame()

        df_temp = self.df.copy()
        df_temp['cbioportal_usage_mode'] = df_temp['cbioportal_usage_mode'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

        usage_counts = df_temp.explode('cbioportal_usage_mode')['cbioportal_usage_mode'].value_counts()
        return pd.DataFrame({'Usage Mode': usage_counts.index, 'Count': usage_counts.values})

    def analyze_features_used(self) -> pd.DataFrame:
        """Analyze which cBioPortal features are being used (v4 schema).

        Returns:
            DataFrame with feature usage counts
        """
        if self.df.empty or 'cbioportal_features_used' not in self.df.columns:
            return pd.DataFrame()

        df_temp = self.df.copy()
        df_temp['cbioportal_features_used'] = df_temp['cbioportal_features_used'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

        feature_counts = df_temp.explode('cbioportal_features_used')['cbioportal_features_used'].value_counts()
        return pd.DataFrame({'Feature': feature_counts.index, 'Count': feature_counts.values})

    def analyze_genes_queried(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze most commonly queried genes (v4 schema).

        Args:
            top_n: Number of top genes to return

        Returns:
            DataFrame with gene query counts
        """
        if self.df.empty or 'specific_genes_queried' not in self.df.columns:
            return pd.DataFrame()

        df_temp = self.df.copy()
        df_temp['specific_genes_queried'] = df_temp['specific_genes_queried'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else (x if isinstance(x, list) else [])
        )

        gene_counts = df_temp.explode('specific_genes_queried')['specific_genes_queried'].value_counts()
        # Filter out empty strings and None
        gene_counts = gene_counts[gene_counts.index.notna()]
        gene_counts = gene_counts[gene_counts.index != '']

        return pd.DataFrame({'Gene': gene_counts.head(top_n).index, 'Count': gene_counts.head(top_n).values})

    def analyze_analysis_location(self) -> pd.DataFrame:
        """Analyze where analysis was performed (v4 schema).

        Returns:
            DataFrame with analysis location counts
        """
        if self.df.empty or 'analysis_location' not in self.df.columns:
            return pd.DataFrame()

        location_counts = self.df['analysis_location'].value_counts()
        return pd.DataFrame({'Location': location_counts.index, 'Count': location_counts.values})

    def analyze_author_countries(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze geographic distribution of citing papers.

        Args:
            top_n: Number of top countries to return

        Returns:
            DataFrame with country counts
        """
        if self.citations_df.empty or 'author_countries' not in self.citations_df.columns:
            return pd.DataFrame()

        # Explode author_countries lists
        df_temp = self.citations_df.copy()
        df_temp['author_countries'] = df_temp['author_countries'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else (x if isinstance(x, list) else [])
        )

        country_counts = df_temp.explode('author_countries')['author_countries'].value_counts()
        # Filter out empty strings
        country_counts = country_counts[country_counts.index != '']

        return pd.DataFrame({'Country': country_counts.head(top_n).index, 'Count': country_counts.head(top_n).values})

    def analyze_journals(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze which journals cite cBioPortal most.

        Args:
            top_n: Number of top journals to return

        Returns:
            DataFrame with journal counts
        """
        if self.citations_df.empty or 'venue' not in self.citations_df.columns:
            return pd.DataFrame()

        journal_counts = self.citations_df['venue'].value_counts()
        # Filter out empty strings
        journal_counts = journal_counts[journal_counts.index != '']

        return pd.DataFrame({'Journal': journal_counts.head(top_n).index, 'Count': journal_counts.head(top_n).values})

    def analyze_publication_types(self) -> pd.DataFrame:
        """Analyze types of publications citing cBioPortal.

        Returns:
            DataFrame with publication type counts
        """
        if self.citations_df.empty or 'publication_types' not in self.citations_df.columns:
            return pd.DataFrame()

        df_temp = self.citations_df.copy()
        df_temp['publication_types'] = df_temp['publication_types'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else (x if isinstance(x, list) else [])
        )

        type_counts = df_temp.explode('publication_types')['publication_types'].value_counts()
        # Filter out empty strings
        type_counts = type_counts[type_counts.index != '']

        return pd.DataFrame({'Publication Type': type_counts.index, 'Count': type_counts.values})

    def analyze_funding_agencies(self, top_n: int = 15) -> pd.DataFrame:
        """Analyze funding sources of citing papers.

        Args:
            top_n: Number of top agencies to return

        Returns:
            DataFrame with funding agency counts
        """
        if self.citations_df.empty or 'grants' not in self.citations_df.columns:
            return pd.DataFrame()

        # Extract agencies from grants
        agencies = []
        for grants_list in self.citations_df['grants']:
            if isinstance(grants_list, str):
                try:
                    grants_list = eval(grants_list)
                except:
                    continue
            if isinstance(grants_list, list):
                for grant in grants_list:
                    if isinstance(grant, dict) and grant.get('agency'):
                        agencies.append(grant['agency'])

        if not agencies:
            return pd.DataFrame()

        agency_series = pd.Series(agencies)
        agency_counts = agency_series.value_counts()

        return pd.DataFrame({'Funding Agency': agency_counts.head(top_n).index, 'Count': agency_counts.head(top_n).values})

    def analyze_mesh_terms(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze research topics via MeSH terms.

        Args:
            top_n: Number of top MeSH terms to return

        Returns:
            DataFrame with MeSH term counts
        """
        if self.citations_df.empty or 'mesh_terms' not in self.citations_df.columns:
            return pd.DataFrame()

        df_temp = self.citations_df.copy()
        df_temp['mesh_terms'] = df_temp['mesh_terms'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else (x if isinstance(x, list) else [])
        )

        mesh_counts = df_temp.explode('mesh_terms')['mesh_terms'].value_counts()
        # Filter out empty strings
        mesh_counts = mesh_counts[mesh_counts.index != '']

        return pd.DataFrame({'MeSH Term': mesh_counts.head(top_n).index, 'Count': mesh_counts.head(top_n).values})

    def analyze_citation_overlap(self) -> Dict:
        """Analyze how many papers cite multiple cBioPortal publications.

        Returns:
            Dictionary with overlap statistics
        """
        if not hasattr(self, 'paper_to_pmids') or not self.paper_to_pmids:
            return {}

        # Count how many cBioPortal papers each citing paper references
        cite_counts = {}
        for paper_id, pmids in self.paper_to_pmids.items():
            count = len(pmids)
            cite_counts[count] = cite_counts.get(count, 0) + 1

        # Calculate totals
        total_papers = len(self.paper_to_pmids)
        citing_one = cite_counts.get(1, 0)
        citing_two = cite_counts.get(2, 0)
        citing_three = cite_counts.get(3, 0)

        return {
            "total_unique_papers": total_papers,
            "citing_one_paper": citing_one,
            "citing_two_papers": citing_two,
            "citing_three_papers": citing_three,
            "cross_citations": total_papers - citing_one  # Papers citing 2 or more
        }

    def analyze_temporal_trends(self) -> pd.DataFrame:
        """Analyze usage trends over time.

        Returns:
            DataFrame with yearly publication counts
        """
        if self.df.empty or 'year' not in self.df.columns:
            return pd.DataFrame()

        # Convert year to numeric, handling any non-numeric values
        df_temp = self.df.copy()
        df_temp['year'] = pd.to_numeric(df_temp['year'], errors='coerce')
        df_temp = df_temp.dropna(subset=['year'])

        yearly_counts = df_temp.groupby('year').size().reset_index(name='count')
        return yearly_counts.sort_values('year')

    def get_recent_papers(self, n: int = 10) -> pd.DataFrame:
        """Get most recent papers using cBioPortal.

        Args:
            n: Number of recent papers to return

        Returns:
            DataFrame with recent papers
        """
        if self.df.empty:
            return pd.DataFrame()

        df_temp = self.df.copy()
        df_temp['year'] = pd.to_numeric(df_temp['year'], errors='coerce')

        # Sort by year and get top N
        recent = df_temp.nlargest(n, 'year')[
            ['title', 'year', 'research_area', 'cancer_type', 'cbioportal_usage_summary']
        ]

        return recent

    def create_visualizations(self) -> Optional[str]:
        """Create all visualization plots.

        Returns:
            Filename of the generated plot (without directory path)
        """
        if self.df.empty:
            logger.warning("No data available for visualization")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('cBioPortal Usage Analysis', fontsize=16, fontweight='bold')

        # 1. Analysis Types
        analysis_df = self.analyze_analysis_types()
        if not analysis_df.empty:
            axes[0, 0].barh(analysis_df['Analysis Type'][:10], analysis_df['Count'][:10])
            axes[0, 0].set_xlabel('Number of Papers')
            axes[0, 0].set_title('Top Analysis Types')
            axes[0, 0].invert_yaxis()

        # 2. Cancer Types
        cancer_df = self.analyze_cancer_types()
        if not cancer_df.empty:
            axes[0, 1].barh(cancer_df['Cancer Type'][:10], cancer_df['Count'][:10])
            axes[0, 1].set_xlabel('Number of Papers')
            axes[0, 1].set_title('Top Cancer Types Studied')
            axes[0, 1].invert_yaxis()

        # 3. Data Sources
        data_source_df = self.analyze_data_sources()
        if not data_source_df.empty:
            axes[1, 0].bar(range(len(data_source_df[:8])), data_source_df['Count'][:8])
            axes[1, 0].set_xticks(range(len(data_source_df[:8])))
            axes[1, 0].set_xticklabels(data_source_df['Data Source'][:8], rotation=45, ha='right')
            axes[1, 0].set_ylabel('Number of Papers')
            axes[1, 0].set_title('Data Sources Used')

        # 4. Temporal Trends
        temporal_df = self.analyze_temporal_trends()
        if not temporal_df.empty:
            axes[1, 1].bar(temporal_df['year'], temporal_df['count'], width=0.6,
                          color='steelblue', edgecolor='darkblue', linewidth=1.5)
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Number of Papers')
            axes[1, 1].set_title('cBioPortal Citations Over Time')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            # Format x-axis to show years as integers
            axes[1, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            axes[1, 1].set_xlim(temporal_df['year'].min() - 0.7, temporal_df['year'].max() + 0.7)

        plt.tight_layout()
        filename = "usage_analysis.png"
        plot_path = PLOTS_DIR / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {plot_path}")
        plt.close()

        return filename

    def create_research_area_plot(self) -> Optional[str]:
        """Create detailed plot for research areas.

        Returns:
            Filename of the generated plot (without directory path)
        """
        if self.df.empty or 'research_area' not in self.df.columns:
            return None

        df_temp = self.df.copy()
        df_temp['research_area'] = df_temp['research_area'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

        research_counts = df_temp.explode('research_area')['research_area'].value_counts()

        plt.figure(figsize=(10, 8))
        plt.barh(research_counts.index, research_counts.values, color='steelblue')
        plt.xlabel('Number of Papers', fontsize=12)
        plt.title('Research Areas Using cBioPortal', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        filename = "research_areas.png"
        plot_path = PLOTS_DIR / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved research areas plot to {plot_path}")
        plt.close()

        return filename

    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        if self.df.empty:
            return {}

        stats = {
            "total_papers": len(self.df),
            "date_generated": datetime.now().isoformat(),
        }

        # Year range
        if 'year' in self.df.columns:
            years = pd.to_numeric(self.df['year'], errors='coerce').dropna()
            if not years.empty:
                stats["year_range"] = f"{int(years.min())} - {int(years.max())}"

        # Text source statistics
        if 'text_source' in self.df.columns:
            text_source_counts = self.df['text_source'].value_counts()
            stats["papers_from_pdf"] = int(text_source_counts.get('pdf', 0))
            stats["papers_from_abstract"] = int(text_source_counts.get('abstract', 0))
            stats["papers_no_text"] = int(text_source_counts.get('none', 0))

        # Most common categories
        for category in ['analysis_type', 'cancer_type', 'research_area', 'data_source']:
            if category in self.df.columns:
                df_temp = self.df.copy()
                df_temp[category] = df_temp[category].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                )
                counts = df_temp.explode(category)[category].value_counts()
                if not counts.empty:
                    stats[f"most_common_{category}"] = counts.index[0]

        return stats

    def generate_report(self, plot_filename: Optional[str] = None, research_plot_filename: Optional[str] = None) -> str:
        """Generate a comprehensive markdown report.

        Args:
            plot_filename: Filename of main usage analysis plot
            research_plot_filename: Filename of research areas plot

        Returns:
            Path to generated report
        """
        report_lines = [
            "# cBioPortal Usage Analysis Report",
            f"\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "---\n",
        ]

        # Summary Statistics
        stats = self.generate_summary_stats()
        report_lines.append("## Summary Statistics\n")
        report_lines.append(f"- **Total Papers Analyzed**: {stats.get('total_papers', 0)}")
        report_lines.append(f"- **Year Range**: {stats.get('year_range', 'N/A')}")

        # Text source breakdown
        if 'papers_from_pdf' in stats:
            total = stats.get('total_papers', 0)
            pdf_count = stats.get('papers_from_pdf', 0)
            abstract_count = stats.get('papers_from_abstract', 0)
            none_count = stats.get('papers_no_text', 0)
            pdf_pct = (pdf_count / total * 100) if total > 0 else 0
            abstract_pct = (abstract_count / total * 100) if total > 0 else 0

            report_lines.append(f"- **Classified from Full PDF**: {pdf_count} ({pdf_pct:.1f}%)")
            report_lines.append(f"- **Classified from Abstract Only**: {abstract_count} ({abstract_pct:.1f}%)")
            if none_count > 0:
                report_lines.append(f"- **No Text Available**: {none_count}")

        report_lines.append(f"- **Most Common Analysis Type**: {stats.get('most_common_analysis_type', 'N/A')}")
        report_lines.append(f"- **Most Common Cancer Type**: {stats.get('most_common_cancer_type', 'N/A')}")
        report_lines.append(f"- **Most Common Data Source**: {stats.get('most_common_data_source', 'N/A')}\n")

        # Add Data Collection & Limitations section
        report_lines.append("## Data Collection & Limitations\n")
        report_lines.append("This analysis is based on papers that cite the three main cBioPortal publications:\n")
        for pmid in CBIOPORTAL_PMIDS:
            report_lines.append(f"- PMID {pmid}")
        report_lines.append("")

        # Load citation counts
        if self.citations_file.exists():
            with open(self.citations_file, 'r') as f:
                citations_data = json.load(f)
            total_citations = sum(len(p['citations']) for p in citations_data.get('papers', {}).values())

            # Get overlap analysis
            overlap_stats = self.analyze_citation_overlap()
            unique_papers = overlap_stats.get('total_unique_papers', 0)
            cross_citations = overlap_stats.get('cross_citations', 0)

            report_lines.append(f"**Total citations in database**: {total_citations:,} papers")
            report_lines.append(f"**Unique papers** (deduplicated): {unique_papers:,} papers")
            report_lines.append(f"**Papers citing multiple cBioPortal publications**: {cross_citations:,} papers\n")

        report_lines.append("**Important limitation**: The PubMed eutils API returns fewer citations than shown on the PubMed website. ")
        report_lines.append("For example, PMID 37668528 shows 750 citations on PubMed's website but the API only returns 407 citations. ")
        report_lines.append("This is a known limitation of the eutils citation indexing system. ")
        report_lines.append("Our analysis is based on the subset of citations available through the API.\n")

        # Add detailed citation overlap section
        overlap_stats = self.analyze_citation_overlap()
        if overlap_stats:
            report_lines.append("## Citation Overlap Analysis\n")
            report_lines.append(f"- Papers citing only 1 cBioPortal publication: {overlap_stats['citing_one_paper']:,}")
            report_lines.append(f"- Papers citing 2 cBioPortal publications: {overlap_stats['citing_two_papers']:,}")
            report_lines.append(f"- Papers citing all 3 cBioPortal publications: {overlap_stats['citing_three_papers']:,}\n")

        # Add Visualizations section if plots exist
        if plot_filename or research_plot_filename:
            report_lines.append("## Visualizations\n")

            if plot_filename:
                report_lines.append("### Usage Analysis Overview\n")
                report_lines.append(f"![Usage Analysis](../plots/usage_analysis.png)\n")
                report_lines.append("*Four-panel visualization showing analysis types, cancer types, data sources, and temporal trends.*\n")

            if research_plot_filename:
                report_lines.append("### Research Areas\n")
                report_lines.append(f"![Research Areas](../plots/research_areas.png)\n")
                report_lines.append("*Distribution of research areas utilizing cBioPortal.*\n")

            report_lines.append("---\n")

        # Top Analysis Types
        analysis_df = self.analyze_analysis_types()
        if not analysis_df.empty:
            report_lines.append("## Top Analysis Types\n")
            for idx, row in analysis_df.head(10).iterrows():
                report_lines.append(f"{idx+1}. {row['Analysis Type']}: {row['Count']} papers")
            report_lines.append("")

        # Top Cancer Types
        cancer_df = self.analyze_cancer_types()
        if not cancer_df.empty:
            report_lines.append("## Top Cancer Types\n")
            for idx, row in cancer_df.head(10).iterrows():
                report_lines.append(f"{idx+1}. {row['Cancer Type']}: {row['Count']} papers")
            report_lines.append("")

        # Data Sources
        data_source_df = self.analyze_data_sources()
        if not data_source_df.empty:
            report_lines.append("## Data Sources Used\n")
            for idx, row in data_source_df.head(10).iterrows():
                report_lines.append(f"{idx+1}. {row['Data Source']}: {row['Count']} papers")
            report_lines.append("")

        # cBioPortal Usage Patterns (v4 schema fields)
        usage_modes_df = self.analyze_usage_modes()
        if not usage_modes_df.empty:
            report_lines.append("## How cBioPortal is Being Used\n")
            for idx, row in usage_modes_df.iterrows():
                report_lines.append(f"{idx+1}. {row['Usage Mode']}: {row['Count']} papers")
            report_lines.append("")

        features_df = self.analyze_features_used()
        if not features_df.empty:
            report_lines.append("## cBioPortal Features Used\n")
            for idx, row in features_df.head(10).iterrows():
                report_lines.append(f"{idx+1}. {row['Feature']}: {row['Count']} papers")
            report_lines.append("")

        location_df = self.analyze_analysis_location()
        if not location_df.empty:
            report_lines.append("## Where Analysis Was Performed\n")
            for idx, row in location_df.iterrows():
                report_lines.append(f"{idx+1}. {row['Location']}: {row['Count']} papers")
            report_lines.append("")

        genes_df = self.analyze_genes_queried(top_n=20)
        if not genes_df.empty:
            report_lines.append("## Most Frequently Queried Genes\n")
            for idx, row in genes_df.iterrows():
                report_lines.append(f"{idx+1}. {row['Gene']}: {row['Count']} papers")
            report_lines.append("")

        # Bibliometric analyses
        report_lines.append("---\n")
        report_lines.append("# Bibliometric Analysis\n")
        report_lines.append("*Analysis of metadata from citing papers*\n")

        countries_df = self.analyze_author_countries(top_n=20)
        if not countries_df.empty:
            report_lines.append("## Geographic Distribution\n")
            for idx, row in countries_df.iterrows():
                report_lines.append(f"{idx+1}. {row['Country']}: {row['Count']} papers")
            report_lines.append("")

        journals_df = self.analyze_journals(top_n=15)
        if not journals_df.empty:
            report_lines.append("## Top Journals Citing cBioPortal\n")
            for idx, row in journals_df.head(15).iterrows():
                report_lines.append(f"{idx+1}. {row['Journal']}: {row['Count']} papers")
            report_lines.append("")

        pub_types_df = self.analyze_publication_types()
        if not pub_types_df.empty:
            report_lines.append("## Publication Types\n")
            for idx, row in pub_types_df.head(10).iterrows():
                report_lines.append(f"{idx+1}. {row['Publication Type']}: {row['Count']} papers")
            report_lines.append("")

        funding_df = self.analyze_funding_agencies(top_n=15)
        if not funding_df.empty:
            report_lines.append("## Funding Agencies\n")
            for idx, row in funding_df.iterrows():
                report_lines.append(f"{idx+1}. {row['Funding Agency']}: {row['Count']} papers")
            report_lines.append("")

        mesh_df = self.analyze_mesh_terms(top_n=20)
        if not mesh_df.empty:
            report_lines.append("## Research Topics (MeSH Terms)\n")
            for idx, row in mesh_df.iterrows():
                report_lines.append(f"{idx+1}. {row['MeSH Term']}: {row['Count']} papers")
            report_lines.append("")

        # Recent Papers
        recent_df = self.get_recent_papers(10)
        if not recent_df.empty:
            report_lines.append("## Recent Papers Using cBioPortal\n")
            for idx, row in recent_df.iterrows():
                report_lines.append(f"### {row['title']} ({row['year']})\n")
                report_lines.append(f"- **Research Area**: {row.get('research_area', 'N/A')}")
                report_lines.append(f"- **Cancer Type**: {row.get('cancer_type', 'N/A')}")
                if pd.notna(row.get('cbioportal_usage_summary')):
                    report_lines.append(f"- **Usage**: {row['cbioportal_usage_summary']}")
                report_lines.append("")

        # Save report
        filename = "usage_report.md"
        report_path = REPORTS_DIR / filename
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        logger.info(f"Generated report at {report_path}")

        return str(report_path)


def main():
    """Main function to run the analyzer."""
    logger.info("Starting analysis...")

    analyzer = UsageAnalyzer()

    # Generate visualizations
    plot_filename = analyzer.create_visualizations()
    research_plot_filename = analyzer.create_research_area_plot()

    # Generate report with plot references
    report_path = analyzer.generate_report(
        plot_filename=plot_filename,
        research_plot_filename=research_plot_filename
    )

    logger.info(f"Analysis complete! Report saved to: {report_path}")


if __name__ == "__main__":
    main()
