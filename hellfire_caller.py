#!/usr/bin/env python3
"""
Hellfire Recon Caller - Standalone Physical Security Assessment
Lightweight wrapper for TheGAVLSuite Hellfire Recon module
"""

import sys
import argparse
import json
from pathlib import Path

# Add TheGAVLSuite to path
GAVL_ROOT = Path.home() / 'TheGAVLSuite'
sys.path.insert(0, str(GAVL_ROOT))

def run_hellfire(address, output_file=None, include_streetview=True):
    """
    Run Hellfire physical security reconnaissance

    Args:
        address: Physical address to assess
        output_file: Optional JSON output file
        include_streetview: Include street view analysis

    Returns:
        dict: Reconnaissance results
    """
    try:
        # Import only Hellfire module (not full suite)
        from modules.hellfire_recon.runner import HellfireRunner

        # Initialize with minimal context
        runner = HellfireRunner()

        # Execute recon
        print(f"[*] Starting Hellfire Recon on: {address}")
        print(f"[*] Street view analysis: {include_streetview}")

        results = runner.run(
            address=address,
            street_view=include_streetview
        )

        print(f"[+] Identified {len(results.get('entry_points', []))} entry points")
        print(f"[+] Security rating: {results.get('security_rating', 'N/A')}")

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
        print(f"[!] Error during Hellfire Recon: {e}")
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Standalone Hellfire Physical Security Recon'
    )
    parser.add_argument('--address', required=True, help='Physical address to assess')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--no-streetview', action='store_true',
                       help='Disable street view analysis')
    parser.add_argument('--json', action='store_true', help='Output JSON to stdout')

    args = parser.parse_args()

    results = run_hellfire(
        args.address,
        args.output,
        include_streetview=not args.no_streetview
    )

    if args.json:
        print(json.dumps(results, indent=2))

    return 0 if 'error' not in results else 1


if __name__ == '__main__':
    sys.exit(main())
