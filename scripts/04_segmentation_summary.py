#!/usr/bin/env python3
"""
Generate comprehensive summary of segmentation data by modality.

This script reads the enhanced training dataset and creates detailed summaries
and visualizations of segmentation data availability across different imaging
modalities, aneurysm presence, and other factors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional


def load_enhanced_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load the enhanced training dataset with segmentation information.
    
    Args:
        data_path: Path to train_with_dcm_and_segmentation.csv
        
    Returns:
        DataFrame with training data, DICOM counts, and segmentation info
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Enhanced dataset not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded enhanced dataset: {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    return df


def generate_segmentation_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate detailed segmentation summary table by modality.
    
    Args:
        df: Enhanced DataFrame with segmentation info
        
    Returns:
        Summary DataFrame with segmentation statistics by modality
    """
    # Group by modality and calculate statistics
    summary = df.groupby('Modality').agg({
        'SeriesInstanceUID': 'count',
        'has_segmentation': ['sum', 'mean'],
        'dcm_file_count': ['mean', 'median', 'std'],
        'Aneurysm Present': 'sum'
    }).round(3)
    
    # Flatten column names
    summary.columns = [
        'Total_Series',
        'Series_With_Segmentation', 'Segmentation_Percentage',
        'Avg_DICOM_Files', 'Median_DICOM_Files', 'Std_DICOM_Files',
        'Series_With_Aneurysm'
    ]
    
    # Calculate additional metrics
    summary['Series_Without_Segmentation'] = summary['Total_Series'] - summary['Series_With_Segmentation']
    summary['Segmentation_Percentage'] = (summary['Segmentation_Percentage'] * 100).round(1)
    
    # Reorder columns
    column_order = [
        'Total_Series', 'Series_With_Segmentation', 'Series_Without_Segmentation', 
        'Segmentation_Percentage', 'Series_With_Aneurysm',
        'Avg_DICOM_Files', 'Median_DICOM_Files', 'Std_DICOM_Files'
    ]
    summary = summary[column_order]
    
    return summary


def create_segmentation_charts(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create visualizations for segmentation data availability.
    
    Args:
        df: Enhanced DataFrame with segmentation info
        output_dir: Directory to save charts
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Bar chart of segmentation counts by modality
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    seg_counts = df.groupby('Modality')['has_segmentation'].agg(['sum', 'count'])
    seg_counts.columns = ['With_Segmentation', 'Total']
    seg_counts['Without_Segmentation'] = seg_counts['Total'] - seg_counts['With_Segmentation']
    
    # Stacked bar chart
    seg_counts[['With_Segmentation', 'Without_Segmentation']].plot(
        kind='bar', stacked=True, ax=ax1, color=['skyblue', 'lightcoral']
    )
    ax1.set_title('Segmentation Availability by Modality', fontweight='bold')
    ax1.set_xlabel('Imaging Modality')
    ax1.set_ylabel('Number of Series')
    ax1.legend(['With Segmentation', 'Without Segmentation'])
    ax1.tick_params(axis='x', rotation=45)
    
    # Percentage bar chart
    seg_percentages = (seg_counts['With_Segmentation'] / seg_counts['Total'] * 100).round(1)
    seg_percentages.plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_title('Segmentation Availability Percentage by Modality', fontweight='bold')
    ax2.set_xlabel('Imaging Modality')
    ax2.set_ylabel('Percentage of Series with Segmentation')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for i, v in enumerate(seg_percentages.values):
        ax2.text(i, v + 1, f'{v}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'segmentation_availability_by_modality.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved segmentation availability chart: {output_file}")
    
    # 2. Pie chart of overall segmentation distribution
    plt.figure(figsize=(10, 8))
    
    # Overall segmentation distribution
    seg_overall = df['has_segmentation'].value_counts()
    labels = ['Without Segmentation', 'With Segmentation']
    colors = ['lightcoral', 'skyblue']
    
    plt.pie(seg_overall.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Overall Distribution of Segmentation Data', fontsize=14, fontweight='bold')
    
    output_file = output_dir / 'overall_segmentation_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall segmentation distribution: {output_file}")
    
    # 3. Segmentation by aneurysm presence
    plt.figure(figsize=(12, 6))
    
    # Cross-tabulation
    crosstab = pd.crosstab(df['Aneurysm Present'], df['has_segmentation'], normalize='index') * 100
    crosstab.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'skyblue'])
    plt.title('Segmentation Availability by Aneurysm Presence', fontweight='bold')
    plt.xlabel('Aneurysm Present')
    plt.ylabel('Percentage of Series')
    plt.legend(['Without Segmentation', 'With Segmentation'])
    plt.xticks([0, 1], ['No Aneurysm', 'Has Aneurysm'], rotation=0)
    
    output_file = output_dir / 'segmentation_by_aneurysm_presence.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved segmentation by aneurysm chart: {output_file}")


def create_detailed_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create detailed breakdown of segmentation data by modality and aneurysm presence.
    
    Args:
        df: Enhanced DataFrame with segmentation info
        
    Returns:
        Detailed breakdown DataFrame
    """
    # Multi-level groupby
    detailed = df.groupby(['Modality', 'Aneurysm Present']).agg({
        'SeriesInstanceUID': 'count',
        'has_segmentation': ['sum', 'mean'],
        'dcm_file_count': 'mean'
    }).round(3)
    
    # Flatten column names
    detailed.columns = [
        'Total_Series', 'Series_With_Segmentation', 'Segmentation_Percentage', 'Avg_DICOM_Files'
    ]
    
    # Convert percentage to %
    detailed['Segmentation_Percentage'] = (detailed['Segmentation_Percentage'] * 100).round(1)
    
    return detailed


def save_summary_tables(summary_df: pd.DataFrame, detailed_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save summary tables to CSV files.
    
    Args:
        summary_df: Summary statistics by modality
        detailed_df: Detailed breakdown by modality and aneurysm presence
        output_dir: Directory to save CSV files
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary table
    summary_output = output_dir / 'segmentation_summary_by_modality.csv'
    summary_df.to_csv(summary_output)
    print(f"Saved segmentation summary: {summary_output}")
    
    # Save detailed breakdown
    detailed_output = output_dir / 'segmentation_detailed_breakdown.csv'
    detailed_df.to_csv(detailed_output)
    print(f"Saved detailed breakdown: {detailed_output}")


def print_summary_statistics(df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """
    Print comprehensive summary statistics.
    
    Args:
        df: Enhanced DataFrame with segmentation info
        summary_df: Summary statistics by modality
    """
    print("\n" + "="*70)
    print("SEGMENTATION DATA SUMMARY BY MODALITY")
    print("="*70)
    
    # Overall statistics
    total_series = len(df)
    total_with_seg = df['has_segmentation'].sum()
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total series: {total_series:,}")
    print(f"Series with segmentation: {total_with_seg:,} ({total_with_seg/total_series*100:.1f}%)")
    print(f"Series without segmentation: {total_series-total_with_seg:,} ({(total_series-total_with_seg)/total_series*100:.1f}%)")
    
    print(f"\nSUMMARY BY MODALITY:")
    print(summary_df.to_string())
    
    # Additional insights
    print(f"\nKEY INSIGHTS:")
    best_modality = summary_df.loc[summary_df['Segmentation_Percentage'].idxmax()]
    worst_modality = summary_df.loc[summary_df['Segmentation_Percentage'].idxmin()]
    
    print(f"• Modality with highest segmentation rate: {best_modality.name} ({best_modality['Segmentation_Percentage']:.1f}%)")
    print(f"• Modality with lowest segmentation rate: {worst_modality.name} ({worst_modality['Segmentation_Percentage']:.1f}%)")
    print(f"• Most common modality: {summary_df.loc[summary_df['Total_Series'].idxmax()].name} ({summary_df['Total_Series'].max()} series)")


def main():
    """Main function to generate segmentation summary by modality."""
    # Define paths
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    enhanced_data_path = processed_dir / "train_with_dcm_and_segmentation.csv"
    reports_dir = Path("reports")
    
    print("Generating segmentation summary by modality...")
    print(f"Enhanced dataset: {enhanced_data_path}")
    print(f"Output directories: {processed_dir}, {reports_dir}")
    
    # Check if input file exists
    if not enhanced_data_path.exists():
        print(f"Error: Enhanced dataset not found at {enhanced_data_path}")
        print("Please run 03_add_segmentation_info.py first to generate the enhanced dataset.")
        return
    
    # Load enhanced dataset
    df = load_enhanced_dataset(enhanced_data_path)
    
    # Generate summary table
    summary_df = generate_segmentation_summary_table(df)
    
    # Create detailed breakdown
    detailed_df = create_detailed_breakdown(df)
    
    # Print statistics
    print_summary_statistics(df, summary_df)
    
    # Save summary tables
    save_summary_tables(summary_df, detailed_df, processed_dir)
    
    # Create visualizations
    create_segmentation_charts(df, reports_dir)
    
    print(f"\nAll summary files and charts saved to: {processed_dir} and {reports_dir}")


if __name__ == "__main__":
    main()