import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CreatePlotsandTables:
    def __init__(self, metadata_path: Path, output_path: Path):
        self.metadata_path = metadata_path
        self.output_path = output_path

        if not self.metadata_path.exists():
            print(f"Metadata file not found: {self.metadata_path}")
            return

        self.tables_dir = output_path / 'tables'
        self.plots_dir = output_path / 'plots'
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.by_label, self.by_type, self.by_corpus, self.by_speaker = self.load_metadata()

    def load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        by_label = metadata.get('by_label', {})
        by_type = metadata.get('by_type', {})
        by_corpus = metadata.get('by_corpus', {})
        by_speaker  = metadata.get('by_speaker', {})

        return by_label, by_type, by_corpus, by_speaker

    def generate_summary_plots_and_tables(self):
        self.create_label_distributions()
        self.create_types_distributions()
        self.create_video_distributions()
        self.create_speaker_distributions()
        self.create_avg_time_distributions()

    # ============================================================================
    # DISTRIBUTION FUNCTIONS
    # ============================================================================

    def create_avg_time_distributions(self):
        avg_time_distribution_corpora_rows = [ # combined
            {
                'Corpus': corpus,
                'Average Duration (s)': corpus_data.get('avg_clip_duration', 0.0)
            }
            for corpus, corpus_data in self.by_corpus.items()
        ]
        avg_time_distribution_corpora = pd.DataFrame(avg_time_distribution_corpora_rows)
        avg_time_distribution_corpora.to_csv(self.tables_dir / 'avg_time_distribution_corpora.csv', index=False)
        self.plot_bar(
            x=avg_time_distribution_corpora['Corpus'],
            y=avg_time_distribution_corpora['Average Duration (s)'],
            xlabel='Corpus',
            ylabel='Average Duration (s)',
            title='Average Clip Duration Distribution Across All Corpora',
            output_path=self.plots_dir / 'avg_time_distribution_corpora.png'
        )

        # per label
        avg_time_distribution_label_rows = [ # combined
            {
                'Label': label,
                'Average Duration (s)': label_data.get('avg_clip_duration', 0.0)
            }
            for label, label_data in self.by_label.items()
        ]
        avg_time_distribution_label = pd.DataFrame(avg_time_distribution_label_rows)
        avg_time_distribution_label.to_csv(self.tables_dir / 'avg_time_distribution_label.csv', index=False)
        self.plot_bar(
            x=avg_time_distribution_label['Label'],
            y=avg_time_distribution_label['Average Duration (s)'],
            xlabel='Label',
            ylabel='Average Duration (s)',
            title='Average Clip Duration Distribution Across All Labels',
            output_path=self.plots_dir / 'avg_time_distribution_label.png'
        )

        # per type
        avg_time_distribution_type_rows = [ # combined
            {
                'Type': type,
                'Average Duration (s)': type_data.get('avg_clip_duration', 0.0)
            }
            for type, type_data in self.by_type.items()
        ]
        avg_time_distribution_type = pd.DataFrame(avg_time_distribution_type_rows)
        avg_time_distribution_type.to_csv(self.tables_dir / 'avg_time_distribution_type.csv', index=False)
        self.plot_bar(
            x=avg_time_distribution_type['Type'],
            y=avg_time_distribution_type['Average Duration (s)'],
            xlabel='Type',
            ylabel='Average Duration (s)',
            title='Average Clip Duration Distribution Across All Types',
            output_path=self.plots_dir / 'avg_time_distribution_type.png'
        )

    def create_video_distributions(self):
        # Combined video distribution across all corpora
        video_distribution_corpora_rows = [ # combined
            {
                'Corpus': corpus,
                'Clip Count': corpus_data.get('num_clips', 0),
            }
            for corpus, corpus_data in self.by_corpus.items()
        ]

        video_distribution_corpora = pd.DataFrame(video_distribution_corpora_rows)
        video_distribution_corpora.to_csv(self.tables_dir / 'video_distribution_corpora.csv', index=False)

        self.plot_bar(
            x=video_distribution_corpora['Corpus'],
            y=video_distribution_corpora['Clip Count'],
            xlabel='Corpus',
            ylabel='Clip Count',
            title='Combined Video Clip Count Distribution Across All Corpora',
            output_path= self.plots_dir / 'video_distribution_clip_count_corpora.png'
        )
        self.plot_pie_chart(
            labels=video_distribution_corpora['Corpus'],
            sizes=video_distribution_corpora['Clip Count'],
            title='Video Clip Count Distribution Across All Corpora',
            output_path=self.plots_dir / 'video_distribution_clip_count_corpora_pie.png'
        )

    def create_label_distributions(self):
        # Combined label distribution across all corpora
        label_distribution_corpora_rows = [ # combined
            {
                'Label': label,
                'Clip Count': label_data.get('num_clips', 0),
                'Average Duration (s)': label_data.get('avg_clip_duration', 0.0)
            }
            for label, label_data in self.by_label.items()
        ]

        label_distribution_corpora = pd.DataFrame(label_distribution_corpora_rows)
        label_distribution_corpora['Clip Count Percentage'] = (label_distribution_corpora['Clip Count'] / label_distribution_corpora['Clip Count'].sum() * 100).round(2)
        label_distribution_corpora.to_csv(self.tables_dir / 'label_distribution_corpora.csv', index=False)

        self.plot_bar(
            x=label_distribution_corpora['Label'],
            y=label_distribution_corpora['Clip Count'],
            xlabel='Label',
            ylabel='Clip Count',
            title='Combined Label Clip Count Distribution Across All Corpora',
            output_path=self.plots_dir / 'label_distribution_clip_count_corpora.png'
        )
        
        # Per corpus label distribution 
        label_distribution_corpus_rows = [ # per corpus
            {
                'Corpus': corpus,
                'Label': label,
                'Clip Count': label_data.get('num_clips', 0),
                'Average Duration (s)': label_data.get('avg_clip_duration', 0.0)
            }
            for corpus, corpus_data in self.by_corpus.items()
            for label, label_data in corpus_data.get('by_label', {}).items()
        ]
        label_distribution_corpus = pd.DataFrame(label_distribution_corpus_rows)
        label_distribution_corpus.to_csv(self.tables_dir / 'label_distribution_corpus.csv', index=False)

        clip_count_pivot = label_distribution_corpus.pivot(index='Corpus', columns='Label', values='Clip Count').fillna(0)
        self.plot_bars(clip_count_pivot, 'Corpus', 'Clip Count', 'Per Corpus Label Distribution', self.plots_dir / 'label_distribution_corpus.png')
        
    def create_types_distributions(self):
    # Combined label distribution across all corpora
        type_distribution_corpora_rows = [ # combined
            {
                'Type': type,
                'Clip Count': type_data.get('num_clips', 0),
                'Average Duration (s)': type_data.get('avg_clip_duration', 0.0)
            }
            for type, type_data in self.by_type.items()
        ]

        type_distribution_corpora = pd.DataFrame(type_distribution_corpora_rows)
        type_distribution_corpora['Clip Count Percentage'] = (type_distribution_corpora['Clip Count'] / type_distribution_corpora['Clip Count'].sum() * 100).round(2)
        type_distribution_corpora.to_csv(self.tables_dir / 'type_distribution_corpora.csv', index=False)
        n = 10
        top_n = (
            type_distribution_corpora
            .query('Type != "None"')
            .nlargest(n, 'Clip Count')
        )

        self.plot_bar(
            x=top_n['Type'],
            y=top_n['Clip Count'],
            xlabel='Type',
            ylabel='Clip Count',
            title=f'Combined Type Clip Count Distribution Across All Corpora (Top {n} ignoring None)',
            output_path=self.plots_dir / 'type_distribution_clip_count_corpora.png'
        )

        # Per corpus type distribution 
        type_distribution_corpus_rows = [ # per corpus
            {
                'Corpus': corpus,
                'Type': type,
                'Clip Count': type_data.get('num_clips', 0),
                'Average Duration (s)': type_data.get('avg_clip_duration', 0.0)
            }
            for corpus, corpus_data in self.by_corpus.items()
            for type, type_data in corpus_data.get('by_type', {}).items()
        ]
        type_distribution_corpus = pd.DataFrame(type_distribution_corpus_rows)     
        type_distribution_corpus.to_csv(self.tables_dir / 'type_distribution_corpus.csv', index=False)

    def create_speaker_distributions(self):
        unique_speakers_per_corpus_rows = [
            {
                'Corpus': corpus,
                'Clip Count': corpus_data.get('num_clips', 0),
                'Unique Speakers Count': corpus_data.get('total_unique_speakers', 0),
                'Average Videos per Speaker': corpus_data.get('avg_videos_per_speaker', 0.0)
            }
            for corpus, corpus_data in self.by_corpus.items()
        ]
        unique_speakers_per_corpus = pd.DataFrame(unique_speakers_per_corpus_rows)

        # TEMPORARY SOLUTION -- zhubo only has 1 speaker. but labelled as multiple
        unique_speakers_per_corpus.loc[unique_speakers_per_corpus['Corpus'] == 'ZHUBO', 'Unique Speakers Count'] = 1
        unique_speakers_per_corpus.to_csv(self.tables_dir / 'unique_speakers_per_corpus.csv', index=False)

        self.plot_bar(
            x=unique_speakers_per_corpus['Corpus'],
            y=unique_speakers_per_corpus['Unique Speakers Count'],
            xlabel='Corpus',
            ylabel='Unique Speakers Count',
            title='Unique Speakers Count Per Corpus',
            output_path=self.plots_dir / 'unique_speakers_count.png'
        )

        speaker_clip_count_rows = [
            {
                'Speaker': speaker,
                'Clip Count': speaker_data.get('num_clips', 0),
            }
            for speaker, speaker_data in self.by_speaker.items()
        ]
        speaker_clip_count = pd.DataFrame(speaker_clip_count_rows)
        speaker_clip_count.to_csv(self.tables_dir / 'speaker_clip_count.csv', index=False)

        banner_data = { # use SPACE_{index} to add spacing between lines in the banner
            'Total Unique Speakers': speaker_clip_count['Speaker'].nunique(),
            'Total Clips': speaker_clip_count['Clip Count'].sum(),
            'SPACE_1': None,
            'Average Clips per Speaker': speaker_clip_count['Clip Count'].mean(),
            'Median Clips per Speaker': speaker_clip_count['Clip Count'].median(),
            'SPACE_2': None,
            'Min Clips per Speaker': speaker_clip_count['Clip Count'].min(),
            'Max Clips per Speaker': speaker_clip_count['Clip Count'].max(),
        }

        self.plot_histogram(
            data=speaker_clip_count['Clip Count'],
            bins=30,
            xlabel='Clip Count',
            ylabel='Number of Speakers',
            title='Distribution of Clip Count Per Speaker',
            banner_data=banner_data,
            output_path=self.plots_dir / 'speaker_clip_count_distribution.png'
        )

    # ============================================================================
    # PLOTTING FUNCTIONS
    # ============================================================================

    def plot_pie_chart(self, labels: list, sizes: list, title: str, output_path: str):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%', 
            startangle=90,
        )
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(title)
        plt.savefig(output_path, dpi=150)
        plt.close()

    def plot_bars(self, pivot: pd.DataFrame, x_label: str, y_label: str, title: str, output_path: str):
        x_values = pivot.index.tolist()
        label_values = pivot.columns.tolist()
        x = np.arange(len(x_values))
        n_labels = len(label_values)
        width = 0.8 / n_labels
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, label in enumerate(label_values):
            offset = (i - n_labels / 2) * width + width / 2
            ax_bar = ax.bar(x + offset, pivot[label], width, label=label, edgecolor='black')
            ax.bar_label(ax_bar, fmt='{:,.0f}', padding=3, fontsize=9, fontweight='bold')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(x)
        ax.set_xticklabels(x_values, rotation=45, ha='right')
        ax.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title(title)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(output_path, dpi=150)
        plt.close()

    def plot_bar(self, x: list, y: list, xlabel: str, ylabel: str, title: str, output_path: str):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10.colors[:len(x)]  # auto-generate distinct colors
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        bars = ax1.bar(x, y, color=colors, alpha=0.6, label=ylabel, edgecolor='black')

        if len(y) > 10:
            ax1.set_xticks(range(len(x)))
            ax1.set_xticklabels(x, rotation=45, ha='right', fontsize=8)
        else:
            ax1.set_xticks(range(len(x)))
            ax1.set_xticklabels(x, rotation=0, ha='center', fontsize=10)

        if isinstance(y[0], float): # 2 decimal places for float values
            ax1.bar_label(bars, fmt='{:,.2f}', padding=3, fontsize=9, fontweight='bold')
        else:
            ax1.bar_label(bars, fmt='{:,.0f}', padding=3, fontsize=9, fontweight='bold')

        plt.title(title)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(output_path, dpi=150)
        plt.close()

    def format_banner_text(self, banner_data: dict) -> str:
        lines = []
        for key, value in banner_data.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            elif value is None:
                lines.append("\n")  # Add a blank line for spacing
            else:
                lines.append(f"{key}: {value:,}")

        return "\n".join(lines)

    def plot_histogram(self, data: list, bins: int, xlabel: str, ylabel: str, title: str, banner_data: dict, output_path: str):
        colors = plt.cm.tab10.colors[2]  # auto-generate distinct colors

        fig, ax = plt.subplots(figsize=(10, 6))
        n, bin_edges, patches = ax.hist(data, bins=bins, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(bin_edges)
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(title)
        ax.bar_label(patches, fmt='{:,.0f}', padding=3, fontsize=9, fontweight='bold')

        if banner_data is not None:
            banner_text = self.format_banner_text(banner_data)
            ax.text(0.98, 0.98, banner_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', multialignment='left', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        plt.savefig(output_path, dpi=150)
        plt.close()
