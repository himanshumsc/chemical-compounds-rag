#!/usr/bin/env python3
"""
Script to analyze text lengths from all compound JSON files.
Calculates total and average lengths for main_entry_length and comprehensive_text_length.
"""

import json
import os
from pathlib import Path
import statistics

def analyze_text_lengths():
    """Analyze text lengths from all compound JSON files."""
    
    # Path to the individual_compounds directory
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    
    if not compounds_dir.exists():
        print(f"❌ Directory not found: {compounds_dir}")
        return
    
    # Lists to store the lengths
    main_entry_lengths = []
    comprehensive_text_lengths = []
    
    # Get all JSON files
    json_files = list(compounds_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {compounds_dir}")
        return
    
    print(f"📁 Found {len(json_files)} JSON files")
    print("=" * 60)
    
    # Process each JSON file
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract lengths
            main_entry_len = data.get('main_entry_length', 0)
            comprehensive_len = data.get('comprehensive_text_length', 0)
            
            # Add to lists
            main_entry_lengths.append(main_entry_len)
            comprehensive_text_lengths.append(comprehensive_len)
            
            # Display individual file info
            compound_name = data.get('name', 'Unknown')
            print(f"📄 {json_file.name:<40} | Main: {main_entry_len:>6} | Comprehensive: {comprehensive_len:>6} | {compound_name}")
            
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
    
    print("=" * 60)
    
    # Calculate statistics
    if main_entry_lengths and comprehensive_text_lengths:
        # Main Entry Length Statistics
        main_total = sum(main_entry_lengths)
        main_avg = statistics.mean(main_entry_lengths)
        main_median = statistics.median(main_entry_lengths)
        main_min = min(main_entry_lengths)
        main_max = max(main_entry_lengths)
        
        # Comprehensive Text Length Statistics
        comp_total = sum(comprehensive_text_lengths)
        comp_avg = statistics.mean(comprehensive_text_lengths)
        comp_median = statistics.median(comprehensive_text_lengths)
        comp_min = min(comprehensive_text_lengths)
        comp_max = max(comprehensive_text_lengths)
        
        # Display results
        print("📊 MAIN ENTRY LENGTH STATISTICS:")
        print(f"   Total Length:     {main_total:,} characters")
        print(f"   Average Length:   {main_avg:.1f} characters")
        print(f"   Median Length:    {main_median:.1f} characters")
        print(f"   Min Length:       {main_min:,} characters")
        print(f"   Max Length:       {main_max:,} characters")
        print()
        
        print("📊 COMPREHENSIVE TEXT LENGTH STATISTICS:")
        print(f"   Total Length:     {comp_total:,} characters")
        print(f"   Average Length:   {comp_avg:.1f} characters")
        print(f"   Median Length:    {comp_median:.1f} characters")
        print(f"   Min Length:       {comp_min:,} characters")
        print(f"   Max Length:       {comp_max:,} characters")
        print()
        
        # Summary comparison
        print("📈 SUMMARY COMPARISON:")
        print(f"   Files processed:  {len(json_files)}")
        print(f"   Main entry avg:   {main_avg:.1f} chars")
        print(f"   Comprehensive avg: {comp_avg:.1f} chars")
        print(f"   Ratio (comp/main): {comp_avg/main_avg:.2f}x")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        if main_avg < 2000:
            print("   ✅ Main entry length is good for GPT-4o input (under 2K chars)")
        else:
            print("   ⚠️  Main entry length might be too long for optimal GPT-4o processing")
            
        if comp_avg > main_avg * 2:
            print("   📝 Comprehensive text is significantly longer - good for detailed analysis")
        else:
            print("   📝 Comprehensive text length is reasonable")
    
    else:
        print("❌ No valid data found to analyze")

if __name__ == "__main__":
    print("🔍 Analyzing text lengths from compound JSON files...")
    print()
    analyze_text_lengths()
