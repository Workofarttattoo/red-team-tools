#!/usr/bin/env python3
"""
Corporate Legal Team Caller - Standalone Legal Analysis
Lightweight wrapper for TheGAVLSuite Corporate Legal Team module
"""

import sys
import argparse
import json
from pathlib import Path

# Add TheGAVLSuite to path
GAVL_ROOT = Path.home() / 'TheGAVLSuite'
sys.path.insert(0, str(GAVL_ROOT))

def run_legal(analysis_type, document=None, output_file=None):
    """
    Run corporate legal analysis

    Args:
        analysis_type: Type of analysis (contract, compliance, risk, ip)
        document: Optional document path to analyze
        output_file: Optional JSON output file

    Returns:
        dict: Legal analysis results
    """
    try:
        # Import only Legal module (not full suite)
        from modules.corporate_legal_team.runner import LegalRunner

        # Initialize with minimal context
        runner = LegalRunner()

        # Execute legal analysis
        print(f"[*] Starting Legal Analysis: {analysis_type}")
        if document:
            print(f"[*] Document: {document}")

        results = runner.run(
            analysis_type=analysis_type,
            document=document
        )

        print(f"[+] Risk level: {results.get('risk_level', 'N/A')}")
        print(f"[+] Recommendations: {len(results.get('recommendations', []))}")

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"[+] Results saved to: {output_file}")

        return results

    except ImportError as e:
        print(f"[!] Error: GAVL module not found: {e}")
        print(f"[!] Ensure TheGAVLSuite is at: {GAVL_ROOT}")
        return {'error': str(e)}
    except Exception as e:
        print(f"[!] Error during Legal Analysis: {e}")
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Standalone Corporate Legal Analysis'
    )
    parser.add_argument('--analysis', required=True,
                       choices=['contract', 'compliance', 'risk', 'ip'],
                       help='Type of legal analysis')
    parser.add_argument('--document', help='Document path to analyze')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--json', action='store_true', help='Output JSON to stdout')

    args = parser.parse_args()

    results = run_legal(args.analysis, args.document, args.output)

    if args.json:
        print(json.dumps(results, indent=2))

    return 0 if 'error' not in results else 1


if __name__ == '__main__':
    sys.exit(main())
