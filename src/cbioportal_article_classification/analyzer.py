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
        self.df = self._load_classifications()

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
            axes[1, 1].plot(temporal_df['year'], temporal_df['count'], marker='o', linewidth=2, markersize=8)
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Number of Papers')
            axes[1, 1].set_title('cBioPortal Citations Over Time')
            axes[1, 1].grid(True, alpha=0.3)
            # Format x-axis to show years as integers
            axes[1, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            axes[1, 1].set_xlim(temporal_df['year'].min() - 0.5, temporal_df['year'].max() + 0.5)

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
