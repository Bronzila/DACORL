#!/usr/bin/env python3
"""Compare TD3+BC agents vs original teacher performance on NanoGPT Shakespeare."""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style with better defaults
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Define paths
base_path = Path("/home/hasana/cloud_dacorl/data/nanoGPT_shakespeare")
viz_path = Path("/home/hasana/cloud_dacorl/viz")
viz_path.mkdir(exist_ok=True)
teachers = ["constant", "exponential_decay", "step_decay", "sgdr"]

# Deep color palettes
COLORS_TEACHER = ['#8B0000', '#006B3D', '#00308F', '#CC5500']  # Deep red, green, blue, orange
COLORS_AGENT = ['#DC143C', '#228B22', '#4169E1', '#FF8C00']    # Crimson, forest green, royal blue, dark orange

def load_teacher_data(teacher):
    """Load original teacher training data."""
    teacher_path = base_path / teacher / "LayerwiseNanoGPT" / teacher / "0" / "aggregated_run_data.csv"
    
    if not teacher_path.exists():
        print(f"⚠️  Teacher data not found for {teacher}")
        return None
    
    df = pd.read_csv(teacher_path)
    print(f"✓ Loaded teacher {teacher}: {len(df)} rows")
    return df

def load_agent_data(teacher):
    """Load agent evaluation data."""
    agent_path = base_path / teacher / "LayerwiseNanoGPT" / teacher / "0" / "results" / "td3_bc" / "3653403230" / "60000" / "eval_data_train.csv"
    
    if not agent_path.exists():
        print(f"⚠️  Agent data not found for {teacher}")
        return None
    
    df = pd.read_csv(agent_path)
    print(f"✓ Loaded agent {teacher}: {len(df)} rows")
    return df

def analyze_performance(teacher, teacher_df, agent_df):
    """Compare teacher vs agent performance."""
    
    # Teacher final metrics (per run)
    teacher_final = teacher_df.groupby('run_idx').agg({
        'train_loss': 'last',
        'validation_loss': 'last',
        'test_loss': 'last',
        'batch_idx': 'max'
    }).reset_index()
    
    # Agent final metrics (per run)
    agent_final = agent_df.groupby('run_idx').agg({
        'train_loss': 'last',
        'validation_loss': 'last',
        'test_loss': 'last',
        'batch_idx': 'max'
    }).reset_index()
    
    # Calculate statistics
    comparison = {
        'teacher_name': teacher,
        
        # Teacher stats
        'teacher_runs': len(teacher_final),
        'teacher_mean_test_loss': teacher_final['test_loss'].mean(),
        'teacher_std_test_loss': teacher_final['test_loss'].std(),
        'teacher_min_test_loss': teacher_final['test_loss'].min(),
        'teacher_max_test_loss': teacher_final['test_loss'].max(),
        'teacher_mean_val_loss': teacher_final['validation_loss'].mean(),
        'teacher_mean_train_loss': teacher_final['train_loss'].mean(),
        
        # Agent stats
        'agent_runs': len(agent_final),
        'agent_mean_test_loss': agent_final['test_loss'].mean(),
        'agent_std_test_loss': agent_final['test_loss'].std(),
        'agent_min_test_loss': agent_final['test_loss'].min(),
        'agent_max_test_loss': agent_final['test_loss'].max(),
        'agent_mean_val_loss': agent_final['validation_loss'].mean(),
        'agent_mean_train_loss': agent_final['train_loss'].mean(),
    }
    
    # Calculate improvement
    comparison['improvement_test_loss'] = comparison['teacher_mean_test_loss'] - comparison['agent_mean_test_loss']
    comparison['improvement_pct'] = (comparison['improvement_test_loss'] / comparison['teacher_mean_test_loss']) * 100
    
    return comparison, teacher_final, agent_final

def plot_test_loss_comparison(all_comparisons):
    """Plot 1: Test Loss Comparison Bar Chart."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    teachers_list = [c['teacher_name'].replace('_', ' ').title() for c in all_comparisons]
    x = np.arange(len(teachers_list))
    width = 0.38
    
    teacher_means = [c['teacher_mean_test_loss'] for c in all_comparisons]
    teacher_stds = [c['teacher_std_test_loss'] for c in all_comparisons]
    agent_means = [c['agent_mean_test_loss'] for c in all_comparisons]
    agent_stds = [c['agent_std_test_loss'] for c in all_comparisons]
    
    bars1 = ax.bar(x - width/2, teacher_means, width, yerr=teacher_stds, 
                   label='Teacher', capsize=6, color=COLORS_TEACHER, alpha=0.85,
                   edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, agent_means, width, yerr=agent_stds,
                   label='TD3+BC Agent', capsize=6, color=COLORS_AGENT, alpha=0.85,
                   edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Final Test Loss', fontsize=15, fontweight='bold')
    ax.set_xlabel('Scheduler Type', fontsize=15, fontweight='bold')
    ax.set_title('Test Loss: Teachers vs TD3+BC Agents', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(teachers_list, fontsize=13)
    ax.legend(fontsize=13, frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
    
    # Add value labels on bars with more space
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(viz_path / '01_test_loss_comparison.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: viz/01_test_loss_comparison.png")
    plt.close()

def plot_improvement_chart(all_comparisons):
    """Plot 2: Improvement Percentage Chart."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    teachers_list = [c['teacher_name'].replace('_', ' ').title() for c in all_comparisons]
    improvements = [c['improvement_pct'] for c in all_comparisons]
    colors = ['#228B22' if imp > 0 else '#DC143C' for imp in improvements]
    
    bars = ax.barh(teachers_list, improvements, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Performance Change (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Scheduler Type', fontsize=15, fontweight='bold')
    ax.set_title('Agent Improvement over Teacher\n(Positive = Agent Better)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=1)
    
    # Set wider x-axis limits from left to right
    min_imp = min(improvements)
    ax.set_xlim(min_imp - 1, 1)  # Extend from leftmost negative to slightly positive
    
    # Add value labels with better spacing
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        label = f'{imp:+.2f}%'
        x_pos = imp + (0.3 if imp > 0 else -0.3)
        ax.text(x_pos, i, label, va='center', ha='left' if imp > 0 else 'right',
               fontsize=13, fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.8))
    
    plt.tight_layout(pad=2.0)
    plt.savefig(viz_path / '02_improvement_percentage.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: viz/02_improvement_percentage.png")
    plt.close()

def plot_validation_comparison(all_comparisons):
    """Plot 3: Validation Loss Comparison."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    teachers_list = [c['teacher_name'].replace('_', ' ').title() for c in all_comparisons]
    x = np.arange(len(teachers_list))
    width = 0.38
    
    teacher_val = [c['teacher_mean_val_loss'] for c in all_comparisons]
    agent_val = [c['agent_mean_val_loss'] for c in all_comparisons]
    
    bars1 = ax.bar(x - width/2, teacher_val, width, label='Teacher',
                  color=COLORS_TEACHER, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, agent_val, width, label='TD3+BC Agent',
                  color=COLORS_AGENT, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Final Validation Loss', fontsize=15, fontweight='bold')
    ax.set_xlabel('Scheduler Type', fontsize=15, fontweight='bold')
    ax.set_title('Validation Loss: Teachers vs TD3+BC Agents', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(teachers_list, fontsize=13)
    ax.legend(fontsize=13, frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
    
    # Set y-axis from 0 to 12
    ax.set_ylim(0, 12)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(viz_path / '03_validation_loss_comparison.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: viz/03_validation_loss_comparison.png")
    plt.close()

def plot_train_comparison(all_comparisons):
    """Plot 4: Train Loss Comparison."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    teachers_list = [c['teacher_name'].replace('_', ' ').title() for c in all_comparisons]
    x = np.arange(len(teachers_list))
    width = 0.38
    
    teacher_train = [c['teacher_mean_train_loss'] for c in all_comparisons]
    agent_train = [c['agent_mean_train_loss'] for c in all_comparisons]
    
    bars1 = ax.bar(x - width/2, teacher_train, width, label='Teacher',
                  color=COLORS_TEACHER, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, agent_train, width, label='TD3+BC Agent',
                  color=COLORS_AGENT, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Final Train Loss', fontsize=15, fontweight='bold')
    ax.set_xlabel('Scheduler Type', fontsize=15, fontweight='bold')
    ax.set_title('Train Loss: Teachers vs TD3+BC Agents', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(teachers_list, fontsize=13)
    ax.legend(fontsize=13, frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
    
    # Set y-axis from 0 to 12
    ax.set_ylim(0, 12)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(viz_path / '04_train_loss_comparison.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: viz/04_train_loss_comparison.png")
    plt.close()

def plot_learning_curves(all_comparisons):
    """Plot 5: Learning Curves for each teacher/agent pair."""
    
    for idx, comp in enumerate(all_comparisons):
        teacher = comp['teacher_name']
        fig, ax = plt.subplots(figsize=(14, 8))
        
        teacher_df = load_teacher_data(teacher)
        agent_df = load_agent_data(teacher)
        
        if teacher_df is not None:
            teacher_curve = teacher_df.groupby('batch_idx')['validation_loss'].mean()
            ax.plot(teacher_curve.index, teacher_curve.values,
                   label='Teacher', color=COLORS_TEACHER[idx],
                   linewidth=3, linestyle='--', alpha=0.9)
        
        if agent_df is not None:
            agent_curve = agent_df.groupby('batch_idx')['validation_loss'].mean()
            ax.plot(agent_curve.index, agent_curve.values,
                   label='TD3+BC Agent', color=COLORS_AGENT[idx],
                   linewidth=3, alpha=0.9)
        
        ax.set_xlabel('Training Step', fontsize=15, fontweight='bold')
        ax.set_ylabel('Validation Loss', fontsize=15, fontweight='bold')
        ax.set_title(f'Learning Curve: {teacher.replace("_", " ").title()}',
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(fontsize=13, frameon=True, shadow=True, fancybox=True, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        plt.tight_layout(pad=2.0)
        plt.savefig(viz_path / f'05_{teacher}_learning_curve.png', dpi=200, bbox_inches='tight')
        print(f"✓ Saved: viz/05_{teacher}_learning_curve.png")
        plt.close()

def plot_all_learning_curves(all_comparisons):
    """Plot 6: All learning curves combined."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for idx, comp in enumerate(all_comparisons):
        teacher = comp['teacher_name']
        teacher_df = load_teacher_data(teacher)
        agent_df = load_agent_data(teacher)
        
        label_base = teacher.replace('_', ' ').title()
        
        if teacher_df is not None:
            teacher_curve = teacher_df.groupby('batch_idx')['validation_loss'].mean()
            ax.plot(teacher_curve.index, teacher_curve.values,
                   label=f'{label_base} (Teacher)', color=COLORS_TEACHER[idx],
                   linewidth=2.5, linestyle='--', alpha=0.8)
        
        if agent_df is not None:
            agent_curve = agent_df.groupby('batch_idx')['validation_loss'].mean()
            ax.plot(agent_curve.index, agent_curve.values,
                   label=f'{label_base} (Agent)', color=COLORS_AGENT[idx],
                   linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel('Training Step', fontsize=15, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=15, fontweight='bold')
    ax.set_title('All Learning Curves: Teachers vs TD3+BC Agents',
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=11, frameon=True, shadow=True, fancybox=True, 
             loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(viz_path / '06_all_learning_curves.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: viz/06_all_learning_curves.png")
    plt.close()

def print_comparison_table(all_comparisons):
    """Print a formatted comparison table."""
    
    print("\n" + "=" * 120)
    print("DETAILED COMPARISON: TEACHERS vs TD3+BC AGENTS")
    print("=" * 120)
    print()
    
    # Summary table header
    print(f"{'Teacher':<20} {'Type':<10} {'Test Loss':<15} {'Val Loss':<15} {'Train Loss':<15} {'Std Dev':<12} {'Runs':<8}")
    print("-" * 120)
    
    for comp in all_comparisons:
        name = comp['teacher_name'].replace('_', ' ').title()
        
        # Teacher row
        print(f"{name:<20} {'Teacher':<10} "
              f"{comp['teacher_mean_test_loss']:>7.4f} ± {comp['teacher_std_test_loss']:<4.4f} "
              f"{comp['teacher_mean_val_loss']:>7.4f}          "
              f"{comp['teacher_mean_train_loss']:>7.4f}          "
              f"{comp['teacher_std_test_loss']:>7.4f}    "
              f"{comp['teacher_runs']:>4}")
        
        # Agent row
        print(f"{'':<20} {'Agent':<10} "
              f"{comp['agent_mean_test_loss']:>7.4f} ± {comp['agent_std_test_loss']:<4.4f} "
              f"{comp['agent_mean_val_loss']:>7.4f}          "
              f"{comp['agent_mean_train_loss']:>7.4f}          "
              f"{comp['agent_std_test_loss']:>7.4f}    "
              f"{comp['agent_runs']:>4}")
        
        # Improvement row
        improvement_sign = "✓" if comp['improvement_pct'] > 0 else "✗"
        print(f"{'':<20} {'→ Δ':<10} "
              f"{comp['improvement_test_loss']:>+7.4f} ({comp['improvement_pct']:>+6.2f}%) {improvement_sign}")
        print()
    
    print("=" * 120)
    
    # Summary statistics
    print("\nSUMMARY:")
    total_improvements = sum(1 for c in all_comparisons if c['improvement_pct'] > 0)
    avg_improvement = np.mean([c['improvement_pct'] for c in all_comparisons])
    
    print(f"  • Agents outperformed teachers: {total_improvements}/{len(all_comparisons)} cases")
    print(f"  • Average improvement: {avg_improvement:+.2f}%")
    
    best_combo = max(all_comparisons, key=lambda x: x['improvement_pct'])
    worst_combo = min(all_comparisons, key=lambda x: x['improvement_pct'])
    
    print(f"  • Best agent improvement: {best_combo['teacher_name']} ({best_combo['improvement_pct']:+.2f}%)")
    print(f"  • Worst agent performance: {worst_combo['teacher_name']} ({worst_combo['improvement_pct']:+.2f}%)")
    
    # Overall best performer
    best_overall = min(all_comparisons,
                      key=lambda x: min(x['teacher_mean_test_loss'], x['agent_mean_test_loss']))
    is_agent = best_overall['agent_mean_test_loss'] < best_overall['teacher_mean_test_loss']
    best_type = "Agent" if is_agent else "Teacher"
    best_loss = min(best_overall['teacher_mean_test_loss'], best_overall['agent_mean_test_loss'])
    
    print(f"\n🏆 Overall Best: {best_overall['teacher_name'].upper()} {best_type} (Test Loss: {best_loss:.4f})")
    print("=" * 120)

def main():
    print("=" * 80)
    print("COMPARING TD3+BC AGENTS vs ORIGINAL TEACHERS")
    print("=" * 80)
    
    # Load all data
    print("\n📂 Loading teacher and agent data...")
    all_comparisons = []
    
    for teacher in teachers:
        print(f"\nProcessing {teacher}...")
        teacher_df = load_teacher_data(teacher)
        agent_df = load_agent_data(teacher)
        
        if teacher_df is not None and agent_df is not None:
            comparison, teacher_final, agent_final = analyze_performance(
                teacher, teacher_df, agent_df
            )
            all_comparisons.append(comparison)
            print(f"  Teacher: {len(teacher_final)} runs, Agent: {len(agent_final)} runs")
    
    if not all_comparisons:
        print("❌ No data found!")
        return
    
    # Print comparison table
    print_comparison_table(all_comparisons)
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_test_loss_comparison(all_comparisons)
    plot_improvement_chart(all_comparisons)
    plot_validation_comparison(all_comparisons)
    plot_train_comparison(all_comparisons)
    plot_learning_curves(all_comparisons)
    plot_all_learning_curves(all_comparisons)
    
    # Save to CSV
    comparison_df = pd.DataFrame(all_comparisons)
    comparison_df.to_csv(viz_path / 'teacher_vs_agent_comparison.csv', index=False)
    print(f"\n✓ Saved detailed comparison to viz/teacher_vs_agent_comparison.csv")
    
    print(f"\n✅ Comparison complete! All visualizations saved to: {viz_path}/")

if __name__ == "__main__":
    main()
