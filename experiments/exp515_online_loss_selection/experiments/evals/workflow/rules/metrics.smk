# Metric function definitions
def metric_auprc(y_true, y_pred):
    """Compute Area Under Precision-Recall Curve."""
    return average_precision_score(y_true, y_pred)


def metric_auroc(y_true, y_pred):
    """Compute Area Under ROC Curve."""
    return roc_auc_score(y_true, y_pred)


def metric_spearman(y_true, y_pred):
    """Compute Spearman correlation coefficient."""
    return spearmanr(y_true, y_pred)[0]


# Metric registry - maps metric names to functions
METRIC_FUNCTIONS = {
    "AUPRC": metric_auprc,
    "AUROC": metric_auroc,
    "Spearman": metric_spearman,
}


def get_combined_group(dataset_name):
    """Return combined group name if dataset belongs to one, else None."""
    for group_name, group_config in config.get("combined_dataset_groups", {}).items():
        if dataset_name in group_config["datasets"]:
            return group_name
    return None


def get_dataset_input(wildcards):
    """Get dataset input path - combined or individual."""
    combined_group = get_combined_group(wildcards.dataset)
    if combined_group:
        return f"results/dataset/{combined_group}.parquet"
    else:
        return f"results/dataset/{wildcards.dataset}.parquet"


def get_prediction_input(wildcards):
    """Get prediction input path - combined or individual."""
    combined_group = get_combined_group(wildcards.dataset)
    if combined_group:
        return f"results/prediction/{combined_group}/{wildcards.model}.parquet"
    else:
        return f"results/prediction/{wildcards.dataset}/{wildcards.model}.parquet"


rule metrics:
    """Unified metrics rule - handles AUPRC, AUROC, Spearman for all datasets."""
    input:
        dataset=get_dataset_input,
        prediction=get_prediction_input,
    output:
        "results/metrics/{dataset}/{metric}/{model}.tsv",
    wildcard_constraints:
        dataset="|".join(get_all_datasets()),
    run:
        # Load data
        df_dataset = pd.read_parquet(input.dataset)
        df_pred = pd.read_parquet(input.prediction)

        # Filter to specific benchmark if using combined dataset (positional filtering)
        if 'dataset' in df_dataset.columns:
            mask = df_dataset['dataset'] == wildcards.dataset
            df_dataset = df_dataset[mask]
            df_pred = df_pred[mask]  # Apply same positional mask

        # Extract labels and scores
        y_true = df_dataset["label"]
        y_pred = df_pred["score"]

        # Compute metric using registry
        metric_func = METRIC_FUNCTIONS[wildcards.metric]
        value = metric_func(y_true, y_pred)

        # Save result
        pd.DataFrame({wildcards.metric: [value]}).to_csv(
            output[0], sep="\t", index=False, float_format="%.3f"
        )


rule aggregate_metrics:
    """Aggregate all metrics into long and wide format tables."""
    input:
        get_all_metric_files()
    output:
        wide="results/correlations/metrics_wide.parquet",
        long="results/correlations/metrics_long.parquet"
    run:
        import re
        from pathlib import Path

        records = []
        for file in input:
            # Parse: results/metrics/{dataset}/{metric}/{model}.tsv
            parts = Path(file).parts
            dataset = parts[2]
            metric_type = parts[3]
            model = parts[4].replace('.tsv', '')

            # Extract step and scoring from model name (e.g., "10000_LLR.minus.score")
            match = re.match(r'(\d+)_(.*)', model)
            if match:
                step = int(match.group(1))
                scoring = match.group(2)
            else:
                continue

            # Read metric value
            df = pd.read_csv(file, sep='\t')
            value = df.iloc[0, 0]

            records.append({
                'step': step,
                'dataset': dataset,
                'metric_type': metric_type,
                'scoring': scoring,
                'value': value
            })

        # Long format
        df_long = pd.DataFrame(records)
        df_long.to_parquet(output.long, index=False)

        # Wide format: pivot so each column is {dataset}_{metric_type}_{scoring}
        df_long['column_name'] = (
            df_long['dataset'] + '_' +
            df_long['metric_type'] + '_' +
            df_long['scoring']
        )
        df_wide = df_long.pivot(index='step', columns='column_name', values='value')
        df_wide = df_wide.reset_index()
        df_wide.to_parquet(output.wide, index=False)


rule compute_correlations:
    """Compute Pearson and Spearman correlation matrices."""
    input:
        "results/correlations/metrics_wide.parquet"
    output:
        pearson="results/correlations/pearson.tsv",
        spearman="results/correlations/spearman.tsv"
    run:
        df = pd.read_parquet(input[0])

        # Compute Pearson correlation
        pearson_corr = df.corr(method='pearson')
        pearson_corr.to_csv(output.pearson, sep='\t', float_format='%.3f')

        # Compute Spearman correlation
        spearman_corr = df.corr(method='spearman')
        spearman_corr.to_csv(output.spearman, sep='\t', float_format='%.3f')


rule plot_correlation_heatmaps:
    """Plot correlation matrices as annotated heatmaps."""
    input:
        pearson="results/correlations/pearson.tsv",
        spearman="results/correlations/spearman.tsv"
    output:
        pearson_png="results/correlations/pearson_heatmap.png",
        spearman_png="results/correlations/spearman_heatmap.png",
        pearson_pdf="results/correlations/pearson_heatmap.pdf",
        spearman_pdf="results/correlations/spearman_heatmap.pdf"
    run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Read correlation matrices
        pearson = pd.read_csv(input.pearson, sep='\t', index_col=0)
        spearman = pd.read_csv(input.spearman, sep='\t', index_col=0)

        # Reorder columns: step first, then by correlation with step (descending)
        step_corr_pearson = pearson.loc['step'].drop('step').sort_values(ascending=False)
        new_order = ['step'] + step_corr_pearson.index.tolist()
        pearson = pearson.loc[new_order, new_order]

        step_corr_spearman = spearman.loc['step'].drop('step').sort_values(ascending=False)
        new_order_spearman = ['step'] + step_corr_spearman.index.tolist()
        spearman = spearman.loc[new_order_spearman, new_order_spearman]

        # Plot Pearson
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(pearson, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax,
                    cbar_kws={'label': 'Pearson Correlation'},
                    linewidths=0.5, linecolor='gray')
        ax.set_title('Pearson Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig.savefig(output.pearson_png, dpi=300, bbox_inches='tight')
        fig.savefig(output.pearson_pdf, bbox_inches='tight')
        plt.close()

        # Plot Spearman
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(spearman, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax,
                    cbar_kws={'label': 'Spearman Correlation'},
                    linewidths=0.5, linecolor='gray')
        ax.set_title('Spearman Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig.savefig(output.spearman_png, dpi=300, bbox_inches='tight')
        fig.savefig(output.spearman_pdf, bbox_inches='tight')
        plt.close()


rule plot_metrics_vs_step:
    """Plot line plots showing how each metric changes with training step."""
    input:
        "results/correlations/metrics_long.parquet"
    output:
        png="results/correlations/metrics_vs_step.png",
        pdf="results/correlations/metrics_vs_step.pdf"
    run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.read_parquet(input[0])

        # Create faceted line plot
        df['label'] = df['dataset'] + '\n' + df['metric_type'] + '\n' + df['scoring']

        # Count unique combinations to determine grid size
        n_combos = df['label'].nunique()
        ncols = min(3, n_combos)
        nrows = (n_combos + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (label, group) in enumerate(df.groupby('label')):
            ax = axes[idx]
            group = group.sort_values('step')
            ax.plot(group['step'], group['value'], marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Training Step', fontsize=10)
            ax.set_ylabel('Metric Value', fontsize=10)
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_combos, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        fig.savefig(output.png, dpi=300, bbox_inches='tight')
        fig.savefig(output.pdf, bbox_inches='tight')
        plt.close()


rule plot_metric_pairs:
    """Plot scatter plots for pairs of metrics to visualize correlations."""
    input:
        wide="results/correlations/metrics_wide.parquet",
        pearson="results/correlations/pearson.tsv"
    output:
        png="results/correlations/metric_pairs.png",
        pdf="results/correlations/metric_pairs.pdf"
    run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.read_parquet(input.wide)
        corr = pd.read_csv(input.pearson, sep='\t', index_col=0)

        # Get metric columns (exclude 'step')
        metric_cols = [col for col in df.columns if col != 'step']

        # Find top correlations (excluding self-correlations and duplicates)
        corr_pairs = []
        for i, col1 in enumerate(metric_cols):
            for j, col2 in enumerate(metric_cols):
                if i < j:  # Upper triangle only
                    corr_val = corr.loc[col1, col2]
                    corr_pairs.append((col1, col2, abs(corr_val), corr_val))

        # Sort by absolute correlation and take top 9
        corr_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = corr_pairs[:min(9, len(corr_pairs))]

        # Create 3x3 grid
        ncols = 3
        nrows = (len(top_pairs) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)

        for idx, (col1, col2, abs_corr, corr_val) in enumerate(top_pairs):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]

            # Scatter plot
            ax.scatter(df[col1], df[col2], alpha=0.6, s=100, edgecolors='black', linewidth=1)

            # Add regression line
            z = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[col1].min(), df[col1].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            ax.set_xlabel(col1.replace('_', ' '), fontsize=9)
            ax.set_ylabel(col2.replace('_', ' '), fontsize=9)
            ax.set_title(f'r = {corr_val:.3f}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(top_pairs), nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')

        plt.suptitle('Top Correlated Metric Pairs', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        fig.savefig(output.png, dpi=300, bbox_inches='tight')
        fig.savefig(output.pdf, bbox_inches='tight')
        plt.close()


rule plot_scorings_vs_step:
    """Plot comparison of scoring methods across training steps for each dataset."""
    input:
        "results/correlations/metrics_long.parquet"
    output:
        png="results/correlations/scorings_vs_step.png",
        pdf="results/correlations/scorings_vs_step.pdf"
    run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        # Load data
        df = pd.read_parquet(input[0])

        # Helper function to get primary metric for each dataset
        def get_primary_metric(dataset_name):
            """Return primary metric type for a given dataset."""
            if dataset_name.startswith('sat_mut_mpra_promoter'):
                return 'Spearman'
            elif dataset_name == 'gnomad_promoter':
                return 'AUROC'
            else:  # TraitGym and PromoterAI
                return 'AUPRC'

        # Filter to primary metrics only
        df['primary_metric'] = df['dataset'].apply(get_primary_metric)
        df_primary = df[df['metric_type'] == df['primary_metric']].copy()

        # Define dataset ordering (logical grouping)
        dataset_order = [
            # Classification tasks
            'traitgym_mendelian_promoter',
            'traitgym_complex_promoter',
            'gnomad_promoter',
            'promoterai_gtex_outlier',
            'promoterai_cagi5_saturation',
            'promoterai_mpra_saturation',
            'promoterai_gtex_eqtl',
            'promoterai_mpra_eqtl',
            'promoterai_ukbb_proteome',
            'promoterai_gel_rna',
            # Regression tasks
            'sat_mut_mpra_promoter_F9',
            'sat_mut_mpra_promoter_GP1BA',
            'sat_mut_mpra_promoter_HBB',
            'sat_mut_mpra_promoter_HBG1',
            'sat_mut_mpra_promoter_HNF4A',
            'sat_mut_mpra_promoter_LDLR',
            'sat_mut_mpra_promoter_MSMB',
            'sat_mut_mpra_promoter_PKLR',
            'sat_mut_mpra_promoter_TERT',
        ]

        # Define colors and markers for scoring methods
        scoring_styles = {
            'LLR.minus.score': {'color': '#1f77b4', 'marker': 'o', 'label': 'LLR−'},
            'absLLR.plus.score': {'color': '#ff7f0e', 'marker': 's', 'label': 'absLLR'},
            'L2.plus.score': {'color': '#2ca02c', 'marker': '^', 'label': 'L2'},
        }

        # Helper function to clean dataset names for display
        def clean_dataset_name(name):
            """Convert dataset name to readable title."""
            # Special cases
            if name == 'traitgym_mendelian_promoter':
                return 'TraitGym Mendelian Promoter'
            elif name == 'traitgym_complex_promoter':
                return 'TraitGym Complex Promoter'
            elif name == 'gnomad_promoter':
                return 'gnomAD Promoter'
            elif name.startswith('sat_mut_mpra_promoter_'):
                promoter = name.replace('sat_mut_mpra_promoter_', '')
                return f'Sat Mut MPRA {promoter}'
            elif name.startswith('promoterai_'):
                suffix = name.replace('promoterai_', '').replace('_', ' ').title()
                return f'PromoterAI {suffix}'
            else:
                return name.replace('_', ' ').title()

        # Create figure with subplots
        nrows, ncols = 4, 5
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 16))
        axes = axes.flatten()

        # Plot each dataset
        for idx, dataset in enumerate(dataset_order):
            ax = axes[idx]

            # Get data for this dataset
            df_dataset = df_primary[df_primary['dataset'] == dataset].copy()

            if df_dataset.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(clean_dataset_name(dataset), fontsize=10, fontweight='bold')
                continue

            # Get metric type for y-axis label
            metric_type = df_dataset['metric_type'].iloc[0]

            # Plot each scoring method
            for scoring in df_dataset['scoring'].unique():
                if scoring in scoring_styles:
                    df_scoring = df_dataset[df_dataset['scoring'] == scoring].sort_values('step')
                    style = scoring_styles[scoring]
                    ax.plot(df_scoring['step'], df_scoring['value'],
                           color=style['color'], marker=style['marker'],
                           linewidth=2, markersize=6, label=style['label'])

            # Styling
            ax.set_title(f"{clean_dataset_name(dataset)} ({metric_type})",
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.legend(fontsize=8, loc='best', framealpha=0.9)

            # X-axis: show labels only on bottom row
            if idx >= (nrows - 1) * ncols:
                ax.set_xlabel('Training Step', fontsize=9)
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.set_xticklabels([])

            # Y-axis
            ax.set_ylabel('Metric Value', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)

        # Hide unused subplot (bottom-right)
        axes[-1].axis('off')

        # Overall title
        fig.suptitle('Scoring Method Comparison Across Training Steps',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        fig.savefig(output.png, dpi=300, bbox_inches='tight')
        fig.savefig(output.pdf, bbox_inches='tight')
        plt.close()
