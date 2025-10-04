"""
Model Comparison Script
Compare XGBoost and Kumo AI fraud detection results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_results():
    """Load results from both models"""
    xgboost_results_path = 'results/xgboost_results.csv'
    kumo_results_path = 'results/kumo_results.csv'

    results = {}

    if os.path.exists(xgboost_results_path):
        xgboost_df = pd.read_csv(xgboost_results_path, index_col=0)
        results['XGBoost'] = xgboost_df.to_dict('index')
        print("XGBoost results loaded")
    else:
        print(f"Warning: {xgboost_results_path} not found")
        results['XGBoost'] = None

    if os.path.exists(kumo_results_path):
        kumo_df = pd.read_csv(kumo_results_path)
        results['Kumo'] = kumo_df.to_dict('records')[0]
        print("Kumo results loaded")
    else:
        print(f"Warning: {kumo_results_path} not found")
        results['Kumo'] = None

    return results


def create_comparison_table(results):
    """Create a comparison table"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)

    # Extract metrics
    comparison_data = []

    if results['XGBoost']:
        for dataset, metrics in results['XGBoost'].items():
            comparison_data.append({
                'Model': 'XGBoost',
                'Dataset': dataset,
                'ROC AUC': metrics['roc_auc'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1']
            })

    if results['Kumo']:
        comparison_data.append({
            'Model': 'Kumo AI',
            'Dataset': 'overall',
            'ROC AUC': results['Kumo']['roc_auc'],
            'Precision': results['Kumo']['precision'],
            'Recall': results['Kumo']['recall'],
            'F1 Score': results['Kumo']['f1']
        })

    comparison_df = pd.DataFrame(comparison_data)

    print("\n### Detailed Metrics Comparison ###\n")
    print(comparison_df.to_string(index=False))

    # Save comparison
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("\nComparison saved to results/model_comparison.csv")

    return comparison_df


def create_comparison_plots(comparison_df):
    """Create visualization plots"""
    print("\nGenerating comparison plots...")

    metrics = ['ROC AUC', 'Precision', 'Recall', 'F1 Score']

    # Set style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fraud Detection Model Comparison', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Filter data for this metric
        plot_data = comparison_df[['Model', 'Dataset', metric]].copy()

        # Create bar plot
        x_labels = plot_data.apply(lambda row: f"{row['Model']}\n({row['Dataset']})", axis=1)
        colors = ['#1f77b4' if 'XGBoost' in model else '#ff7f0e'
                  for model in plot_data['Model']]

        bars = ax.bar(range(len(plot_data)), plot_data[metric], color=colors, alpha=0.7)

        # Customize plot
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(x_labels, rotation=0, ha='center')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved to results/model_comparison.png")
    plt.close()


def generate_summary_report(comparison_df):
    """Generate a text summary report"""
    print("\n" + "=" * 80)
    print("SUMMARY ANALYSIS")
    print("=" * 80)

    # Get test set results for XGBoost (if available)
    xgboost_test = comparison_df[
        (comparison_df['Model'] == 'XGBoost') &
        (comparison_df['Dataset'] == 'test')
    ]

    kumo_results = comparison_df[comparison_df['Model'] == 'Kumo AI']

    report = []
    report.append("\n### Key Findings ###\n")

    if not xgboost_test.empty:
        xgb_auc = xgboost_test['ROC AUC'].values[0]
        xgb_precision = xgboost_test['Precision'].values[0]
        xgb_recall = xgboost_test['Recall'].values[0]
        xgb_f1 = xgboost_test['F1 Score'].values[0]

        report.append(f"XGBoost Performance (Test Set):")
        report.append(f"  - ROC AUC: {xgb_auc:.4f}")
        report.append(f"  - Precision: {xgb_precision:.4f}")
        report.append(f"  - Recall: {xgb_recall:.4f}")
        report.append(f"  - F1 Score: {xgb_f1:.4f}")
        report.append("")

    if not kumo_results.empty:
        kumo_auc = kumo_results['ROC AUC'].values[0]
        kumo_precision = kumo_results['Precision'].values[0]
        kumo_recall = kumo_results['Recall'].values[0]
        kumo_f1 = kumo_results['F1 Score'].values[0]

        report.append(f"Kumo AI Performance:")
        report.append(f"  - ROC AUC: {kumo_auc:.4f}")
        report.append(f"  - Precision: {kumo_precision:.4f}")
        report.append(f"  - Recall: {kumo_recall:.4f}")
        report.append(f"  - F1 Score: {kumo_f1:.4f}")
        report.append("")

    # Comparison
    if not xgboost_test.empty and not kumo_results.empty:
        report.append("### Direct Comparison ###\n")
        report.append(f"ROC AUC Difference: {abs(xgb_auc - kumo_auc):.4f}")
        report.append(f"  - Winner: {'XGBoost' if xgb_auc > kumo_auc else 'Kumo AI'}")
        report.append("")
        report.append(f"Precision Difference: {abs(xgb_precision - kumo_precision):.4f}")
        report.append(f"  - Winner: {'XGBoost' if xgb_precision > kumo_precision else 'Kumo AI'}")
        report.append("")
        report.append(f"Recall Difference: {abs(xgb_recall - kumo_recall):.4f}")
        report.append(f"  - Winner: {'XGBoost' if xgb_recall > kumo_recall else 'Kumo AI'}")
        report.append("")
        report.append(f"F1 Score Difference: {abs(xgb_f1 - kumo_f1):.4f}")
        report.append(f"  - Winner: {'XGBoost' if xgb_f1 > kumo_f1 else 'Kumo AI'}")
        report.append("")

    report.append("\n### Recommendations ###\n")
    report.append("1. Model Selection:")
    if not xgboost_test.empty and not kumo_results.empty:
        if xgb_auc > kumo_auc:
            report.append("   - XGBoost shows better overall performance (higher ROC AUC)")
        else:
            report.append("   - Kumo AI shows better overall performance (higher ROC AUC)")

    report.append("\n2. Implementation Considerations:")
    report.append("   - XGBoost: Requires extensive feature engineering, longer training time")
    report.append("   - Kumo AI: Leverages relational data directly, faster prototyping")

    report.append("\n3. Production Deployment:")
    report.append("   - Consider ensemble approach combining both models")
    report.append("   - Monitor model performance over time for drift detection")
    report.append("   - Implement A/B testing for real-world validation")

    report_text = "\n".join(report)
    print(report_text)

    # Save report
    with open('results/comparison_report.txt', 'w') as f:
        f.write(report_text)
    print("\nReport saved to results/comparison_report.txt")


def main():
    """Main comparison function"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Load results
    results = load_results()

    if results['XGBoost'] is None and results['Kumo'] is None:
        print("\nError: No results found. Please run the models first:")
        print("  python xgboost_fraud_detection.py")
        print("  python kumo_fraud_detection.py")
        return

    # Create comparison table
    comparison_df = create_comparison_table(results)

    # Create plots
    if not comparison_df.empty:
        create_comparison_plots(comparison_df)

    # Generate summary report
    generate_summary_report(comparison_df)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
