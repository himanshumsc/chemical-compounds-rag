#!/usr/bin/env python3
"""
Script to find last modified dates of Python, JSON, .sh, and .md files
Collects file data to JSON and generates reports on demand
"""
import os
import stat
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def get_file_times(filepath):
    """Get file modification and creation times"""
    try:
        stat_info = os.stat(filepath)
        
        # Use modification time as primary
        mod_time = datetime.fromtimestamp(stat_info.st_mtime)
        
        # Try to get birth time (creation time) - available on some filesystems
        try:
            birth_time = stat_info.st_birthtime
            creation_time = datetime.fromtimestamp(birth_time)
        except AttributeError:
            # Fallback to modification time if birth time not available
            creation_time = datetime.fromtimestamp(stat_info.st_mtime)
        
        file_size = stat_info.st_size
        
        return mod_time, creation_time, file_size
    except Exception as e:
        return None, None, 0

def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def is_in_hidden_folder(filepath, base_path):
    """Check if file is in any hidden folder (folder starting with .)"""
    try:
        # Get relative path from base
        rel_path = filepath.relative_to(base_path)
        # Check all parts of the path
        for part in rel_path.parts:
            if part.startswith('.'):
                return True
        return False
    except ValueError:
        # If file is not relative to base_path, check absolute path
        for part in filepath.parts:
            if part.startswith('.') and part != '.' and part != '..':
                return True
        return False

def group_by_date_range(all_files, days=30):
    """Group files by date ranges"""
    now = datetime.now()
    groups = {
        'Last 7 days': [],
        'Last 30 days': [],
        'Last 90 days': [],
        'Last 6 months': [],
        'Last year': [],
        'Older than 1 year': []
    }
    
    for file_info in all_files:
        mod_time = file_info['modified']
        days_ago = (now - mod_time).days
        
        if days_ago <= 7:
            groups['Last 7 days'].append(file_info)
        elif days_ago <= 30:
            groups['Last 30 days'].append(file_info)
        elif days_ago <= 90:
            groups['Last 90 days'].append(file_info)
        elif days_ago <= 180:
            groups['Last 6 months'].append(file_info)
        elif days_ago <= 365:
            groups['Last year'].append(file_info)
        else:
            groups['Older than 1 year'].append(file_info)
    
    return groups

def group_by_directory(all_files, base_path):
    """Group files by top-level directory"""
    dir_groups = defaultdict(list)
    
    for file_info in all_files:
        rel_path = file_info['path'].relative_to(base_path)
        # Get top-level directory (first part of path)
        if len(rel_path.parts) > 1:
            top_dir = rel_path.parts[0]
        else:
            top_dir = "root"
        dir_groups[top_dir].append(file_info)
    
    return dir_groups

def save_data_to_json(all_files, base_path, json_file):
    """Save file data to JSON file with datetime serialization"""
    # Convert datetime objects to ISO format strings for JSON serialization
    json_data = {
        'metadata': {
            'base_path': str(base_path),
            'collected_at': datetime.now().isoformat(),
            'total_files': len(all_files)
        },
        'files': []
    }
    
    for file_info in all_files:
        json_data['files'].append({
            'path': str(file_info['path']),
            'ext': file_info['ext'],
            'modified': file_info['modified'].isoformat(),
            'creation': file_info['creation'].isoformat(),
            'size': file_info['size']
        })
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Data saved to JSON: {json_file}")
    print(f"📊 Total files: {len(all_files)}")
    return json_file

def load_data_from_json(json_file, base_path):
    """Load file data from JSON file and convert back to datetime objects"""
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    all_files = []
    for file_data in json_data['files']:
        all_files.append({
            'path': Path(file_data['path']),
            'ext': file_data['ext'],
            'modified': datetime.fromisoformat(file_data['modified']),
            'creation': datetime.fromisoformat(file_data['creation']),
            'size': file_data['size']
        })
    
    metadata = json_data.get('metadata', {})
    print(f"✅ Loaded {len(all_files)} files from JSON")
    print(f"📅 Data collected at: {metadata.get('collected_at', 'unknown')}")
    return all_files, metadata

def generate_report(all_files, base_path, extensions, report_file, metadata=None):
    """Generate a comprehensive Markdown report with analysis and grouping"""
    report_lines = []
    
    # Markdown Header
    report_lines.append("# File Last Modified Date Report")
    report_lines.append("")
    report_lines.append(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    if metadata and 'collected_at' in metadata:
        collected_at = datetime.fromisoformat(metadata['collected_at'])
        report_lines.append(f"**Data Collected:** {collected_at.strftime('%Y-%m-%d %H:%M:%S')}  ")
    report_lines.append(f"**Base Path:** `{base_path}`  ")
    report_lines.append(f"**File Types:** {', '.join(extensions)}  ")
    report_lines.append(f"**Excluded:** Files in hidden folders (starting with `.`)  ")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Summary Statistics
    total_files = len(all_files)
    total_size = sum(f.get('size', 0) for f in all_files)
    ext_counts = defaultdict(int)
    ext_sizes = defaultdict(int)
    
    for file_info in all_files:
        ext = file_info['ext']
        ext_counts[ext] += 1
        ext_sizes[ext] += file_info.get('size', 0)
    
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    report_lines.append(f"- **Total Files Found:** {total_files}")
    report_lines.append(f"- **Total Size:** {format_size(total_size)}")
    report_lines.append("")
    report_lines.append("### Files by Extension")
    report_lines.append("")
    report_lines.append("| Extension | Count | Total Size |")
    report_lines.append("|-----------|-------|------------|")
    for ext in extensions:
        count = ext_counts.get(ext, 0)
        size = ext_sizes.get(ext, 0)
        if count > 0:
            report_lines.append(f"| `{ext}` | {count} | {format_size(size)} |")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Analysis Section
    report_lines.append("## Analysis")
    report_lines.append("")
    
    if all_files:
        # Sort files by modification date
        sorted_files = sorted(all_files, key=lambda x: x['modified'])
        oldest_file = sorted_files[0]
        newest_file = sorted_files[-1]
        
        # Date range analysis
        date_groups = group_by_date_range(all_files)
        
        report_lines.append("### Modification Date Analysis")
        report_lines.append("")
        report_lines.append("| Time Period | File Count |")
        report_lines.append("|-------------|------------|")
        for period, files in date_groups.items():
            report_lines.append(f"| {period} | {len(files)} |")
        report_lines.append("")
        
        report_lines.append("### Most Recently Modified Files")
        report_lines.append("")
        report_lines.append("| Last Modified | Size | File Path |")
        report_lines.append("|---------------|------|-----------|")
        for file_info in sorted(all_files, key=lambda x: x['modified'], reverse=True)[:10]:
            rel_path = file_info['path'].relative_to(base_path)
            mod_str = file_info['modified'].strftime("%Y-%m-%d %H:%M:%S")
            size_str = format_size(file_info.get('size', 0))
            report_lines.append(f"| {mod_str} | {size_str} | `{rel_path}` |")
        report_lines.append("")
        
        report_lines.append("### Oldest Modified Files")
        report_lines.append("")
        report_lines.append("| Last Modified | Size | File Path |")
        report_lines.append("|---------------|------|-----------|")
        for file_info in sorted_files[:10]:
            rel_path = file_info['path'].relative_to(base_path)
            mod_str = file_info['modified'].strftime("%Y-%m-%d %H:%M:%S")
            size_str = format_size(file_info.get('size', 0))
            report_lines.append(f"| {mod_str} | {size_str} | `{rel_path}` |")
        report_lines.append("")
        
        report_lines.append("### Largest Files")
        report_lines.append("")
        report_lines.append("| Size | Last Modified | File Path |")
        report_lines.append("|------|---------------|-----------|")
        for file_info in sorted(all_files, key=lambda x: x.get('size', 0), reverse=True)[:10]:
            rel_path = file_info['path'].relative_to(base_path)
            mod_str = file_info['modified'].strftime("%Y-%m-%d %H:%M:%S")
            size_str = format_size(file_info.get('size', 0))
            report_lines.append(f"| {size_str} | {mod_str} | `{rel_path}` |")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Grouping by Directory
    report_lines.append("## Files Grouped by Directory")
    report_lines.append("")
    dir_groups = group_by_directory(all_files, base_path)
    
    for dir_name in sorted(dir_groups.keys()):
        dir_files = dir_groups[dir_name]
        dir_files.sort(key=lambda x: x['modified'], reverse=True)
        total_dir_size = sum(f.get('size', 0) for f in dir_files)
        
        report_lines.append(f"### `{dir_name}/` ({len(dir_files)} files, {format_size(total_dir_size)})")
        report_lines.append("")
        report_lines.append("| Last Modified | Size | File Path |")
        report_lines.append("|---------------|------|-----------|")
        
        for file_info in dir_files:
            rel_path = file_info['path'].relative_to(base_path)
            mod_str = file_info['modified'].strftime("%Y-%m-%d %H:%M:%S")
            size_str = format_size(file_info.get('size', 0))
            report_lines.append(f"| {mod_str} | {size_str} | `{rel_path}` |")
        
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Grouping by File Extension
    report_lines.append("## Files Grouped by Extension")
    report_lines.append("")
    
    for ext in extensions:
        ext_files = [f for f in all_files if f['ext'] == ext]
        ext_files.sort(key=lambda x: x['modified'], reverse=True)
        
        if ext_files:
            total_ext_size = sum(f.get('size', 0) for f in ext_files)
            report_lines.append(f"### {ext.upper()} Files ({len(ext_files)} files, {format_size(total_ext_size)})")
            report_lines.append("")
            report_lines.append("| Last Modified | Creation Date | Size | File Path |")
            report_lines.append("|---------------|---------------|------|-----------|")
            
            for file_info in ext_files:
                rel_path = file_info['path'].relative_to(base_path)
                mod_str = file_info['modified'].strftime("%Y-%m-%d %H:%M:%S")
                creation_str = file_info['creation'].strftime("%Y-%m-%d %H:%M:%S")
                size_str = format_size(file_info.get('size', 0))
                report_lines.append(f"| {mod_str} | {creation_str} | {size_str} | `{rel_path}` |")
            
            report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Grouping by Date Range
    report_lines.append("## Files Grouped by Modification Date Range")
    report_lines.append("")
    
    date_groups = group_by_date_range(all_files)
    for period in ['Last 7 days', 'Last 30 days', 'Last 90 days', 'Last 6 months', 'Last year', 'Older than 1 year']:
        period_files = date_groups[period]
        if period_files:
            period_files.sort(key=lambda x: x['modified'], reverse=True)
            total_period_size = sum(f.get('size', 0) for f in period_files)
            
            report_lines.append(f"### {period} ({len(period_files)} files, {format_size(total_period_size)})")
            report_lines.append("")
            report_lines.append("| Last Modified | Size | File Path |")
            report_lines.append("|---------------|------|-----------|")
            
            for file_info in period_files:
                rel_path = file_info['path'].relative_to(base_path)
                mod_str = file_info['modified'].strftime("%Y-%m-%d %H:%M:%S")
                size_str = format_size(file_info.get('size', 0))
                report_lines.append(f"| {mod_str} | {size_str} | `{rel_path}` |")
            
            report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Footer notes
    report_lines.append("## Notes")
    report_lines.append("")
    report_lines.append("- Files are sorted by last modified date (newest first in tables).")
    report_lines.append("- Creation time shows birth time if available, otherwise modification time.")
    report_lines.append("- File sizes are shown in human-readable format (B, KB, MB, GB).")
    report_lines.append("- Files in hidden folders (starting with `.`) are excluded.")
    report_lines.append("")
    
    # Write report to file
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Print summary to console
    print(f"✅ Report generated successfully!")
    print(f"📄 Report saved to: {report_file}")
    print(f"📊 Total files analyzed: {total_files}")
    print(f"📦 Total size: {format_size(total_size)}")
    
    return report_lines

def collect_file_data(base_path, extensions, json_file):
    """Collect file data and save to JSON"""
    print("Finding all Python, JSON, .sh, and .md files and their last modified dates...")
    print("This may take a moment...")
    print()
    
    all_files = []
    
    # Find all files
    for ext in extensions:
        pattern = f"**/*{ext}"
        for filepath in base_path.rglob(pattern):
            if filepath.is_file():
                # Skip files in hidden folders
                if is_in_hidden_folder(filepath, base_path):
                    continue
                
                try:
                    mod_time, creation_time, file_size = get_file_times(filepath)
                    if mod_time:
                        all_files.append({
                            'path': filepath,
                            'ext': ext,
                            'modified': mod_time,
                            'creation': creation_time,
                            'size': file_size
                        })
                except Exception as e:
                    # Skip files that can't be accessed
                    continue
    
    # Save to JSON
    save_data_to_json(all_files, base_path, json_file)
    return all_files

def main():
    base_path = Path("/home/himanshu")
    extensions = [".py", ".json", ".sh", ".md"]
    # Save files in the same directory as the script
    script_dir = Path(__file__).parent
    json_file = script_dir / "file_modified_data.json"
    report_file = script_dir / "file_modified_report.md"
    
    parser = argparse.ArgumentParser(
        description='Collect file modification data and generate reports'
    )
    parser.add_argument(
        '--collect',
        action='store_true',
        help='Collect file data and save to JSON (default: generate report from existing JSON)'
    )
    parser.add_argument(
        '--json',
        type=str,
        default=str(json_file),
        help=f'Path to JSON data file (default: {json_file})'
    )
    parser.add_argument(
        '--report',
        type=str,
        default=str(report_file),
        help=f'Path to output report file (default: {report_file})'
    )
    
    args = parser.parse_args()
    json_file = Path(args.json)
    report_file = Path(args.report)
    
    if args.collect:
        # Collect data and save to JSON
        all_files = collect_file_data(base_path, extensions, json_file)
        print(f"\n💡 To generate report, run: python3 {__file__} --json {json_file} --report {report_file}")
    else:
        # Generate report from JSON
        if not json_file.exists():
            print(f"❌ JSON file not found: {json_file}")
            print(f"💡 Run with --collect first: python3 {__file__} --collect")
            return
        
        print(f"Loading data from JSON: {json_file}")
        all_files, metadata = load_data_from_json(json_file, base_path)
        print(f"\nGenerating Markdown report with analysis...")
        generate_report(all_files, base_path, extensions, report_file, metadata)

if __name__ == "__main__":
    main()

