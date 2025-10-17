#!/usr/bin/env python3
"""
OSINT Caller - Standalone Intelligence Gathering Tool
Lightweight wrapper for TheGAVLSuite OSINT module
"""

import sys
import argparse
import json
from pathlib import Path

# Add TheGAVLSuite to path
GAVL_ROOT = Path.home() / 'TheGAVLSuite'
sys.path.insert(0, str(GAVL_ROOT))

def run_osint(target, output_file=None, profile='standard'):
    """
    Run OSINT intelligence gathering on target

    Args:
        target: Domain, IP, or entity to investigate
        output_file: Optional JSON output file
        profile: Analysis profile (quick, standard, deep)

    Returns:
        dict: OSINT results
    """
    try:
        # Import only OSINT module (not full suite)
        from modules.osint.runner import OSINTRunner

        # Initialize with minimal context
        runner = OSINTRunner(profile=profile)

        # Execute intelligence gathering
        print(f"[*] Starting OSINT on target: {target}")
        print(f"[*] Profile: {profile}")

        results = runner.run(target)

        print(f"[+] Gathered {len(results.get('sources', []))} intelligence sources")
        print(f"[+] Confidence: {results.get('confidence', 0):.2%}")

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
        print(f"[!] Error during OSINT: {e}")
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Standalone OSINT Intelligence Gathering'
    )
    parser.add_argument('--target', required=True, help='Target to investigate')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--profile', default='standard',
                       choices=['quick', 'standard', 'deep'],
                       help='Analysis profile')
    parser.add_argument('--json', action='store_true', help='Output JSON to stdout')

    args = parser.parse_args()

    results = run_osint(args.target, args.output, args.profile)

    if args.json:
        print(json.dumps(results, indent=2))

    return 0 if 'error' not in results else 1


if __name__ == '__main__':
    sys.exit(main())
