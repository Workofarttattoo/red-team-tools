#!/usr/bin/env python3
"""
Bayesian Sophiarch Caller - Standalone Probabilistic Reasoning
Lightweight wrapper for TheGAVLSuite Bayesian Sophiarch module
"""

import sys
import argparse
import json
from pathlib import Path

# Add TheGAVLSuite to path
GAVL_ROOT = Path.home() / 'TheGAVLSuite'
sys.path.insert(0, str(GAVL_ROOT))

def run_bayesian(model_type, data=None, output_file=None):
    """
    Run Bayesian probabilistic reasoning

    Args:
        model_type: Model to use (inference, prediction, optimization)
        data: Optional data file or dict
        output_file: Optional JSON output file

    Returns:
        dict: Bayesian analysis results
    """
    try:
        # Import only Bayesian module (not full suite)
        from modules.bayesian_sophiarch.runner import BayesianRunner

        # Initialize with minimal context
        runner = BayesianRunner()

        # Execute Bayesian analysis
        print(f"[*] Starting Bayesian Analysis: {model_type}")

        results = runner.run(
            model=model_type,
            data=data
        )

        print(f"[+] Posterior probability: {results.get('posterior', 0):.2%}")
        print(f"[+] Confidence intervals: {results.get('confidence', {})}")

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
        print(f"[!] Error during Bayesian Analysis: {e}")
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Standalone Bayesian Probabilistic Reasoning'
    )
    parser.add_argument('--model', required=True,
                       choices=['inference', 'prediction', 'optimization'],
                       help='Bayesian model type')
    parser.add_argument('--data', help='Data file (JSON)')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--json', action='store_true', help='Output JSON to stdout')

    args = parser.parse_args()

    # Load data if provided
    data = None
    if args.data:
        with open(args.data) as f:
            data = json.load(f)

    results = run_bayesian(args.model, data, args.output)

    if args.json:
        print(json.dumps(results, indent=2))

    return 0 if 'error' not in results else 1


if __name__ == '__main__':
    sys.exit(main())
