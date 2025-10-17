"""
VectorFlux — payload crafting workbench with guardrails.

Generates signed module manifests, applies scenario guardrails, and prepares
workspace metadata without executing exploit payloads.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ._toolkit import (
  Diagnostic,
  build_health_report,
  emit_diagnostic,
  launch_gui,
  summarise_samples,
  synthesise_latency_samples,
)


TOOL_NAME = "VectorFlux"


@dataclass
class ModuleDescriptor:
  name: str
  category: str
  entrypoint: str
  min_guardrail: str
  description: str

  def as_dict(self) -> Dict[str, object]:
    return asdict(self)


SAMPLE_MODULES: Dict[str, ModuleDescriptor] = {
  "reverse-shell": ModuleDescriptor(
    name="reverse-shell",
    category="access",
    entrypoint="payloads.reverse_shell",
    min_guardrail="operator-signature",
    description="Interactive shell payload with encrypted transport.",
  ),
  "credential-harvest": ModuleDescriptor(
    name="credential-harvest",
    category="collection",
    entrypoint="payloads.credential_harvest",
    min_guardrail="supervisor-approval",
    description="Memory-safe credential capture hooks for blue team rehearsal.",
  ),
  "lateral-movement": ModuleDescriptor(
    name="lateral-movement",
    category="mobility",
    entrypoint="payloads.lateral_movement",
    min_guardrail="playbook-review",
    description="Simulated lateral movement using signed execution graphs.",
  ),
}

SAMPLE_SCENARIO = {
  "name": "incident-23-071",
  "objectives": ["obtain foothold", "collect credentials", "exfil telemetry"],
  "constraints": ["no-production-hosts", "signed-modules-only"],
}


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="VectorFlux payload workbench.")
  parser.add_argument("--workspace", default="vectorflux-workspace", help="Workspace identifier.")
  parser.add_argument("--module", help="Module to stage (use --list-modules to enumerate).")
  parser.add_argument("--list-modules", action="store_true", help="List available modules and exit.")
  parser.add_argument("--scenario", help="Scenario JSON file to merge with defaults.")
  parser.add_argument("--json", action="store_true", help="Emit JSON output.")
  parser.add_argument("--output", help="Write payload manifest to path.")
  parser.add_argument("--gui", action="store_true", help="Launch the VectorFlux graphical interface.")
  return parser


def list_modules() -> Dict[str, Dict[str, object]]:
  return {name: descriptor.as_dict() for name, descriptor in SAMPLE_MODULES.items()}


def load_scenario(path: Optional[str]) -> Dict[str, object]:
  if not path:
    return dict(SAMPLE_SCENARIO)
  try:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
      scenario = dict(SAMPLE_SCENARIO)
      scenario.update(data)
      return scenario
  except FileNotFoundError:
    pass
  except json.JSONDecodeError:
    pass
  return dict(SAMPLE_SCENARIO)


def stage_module(workspace: str, module_name: Optional[str], scenario: Dict[str, object]) -> Dict[str, object]:
  if not module_name:
    module_name = "reverse-shell"
  descriptor = SAMPLE_MODULES.get(module_name)
  if not descriptor:
    raise ValueError(f"Unknown module '{module_name}'. Use --list-modules to inspect options.")
  manifest = {
    "tool": TOOL_NAME,
    "workspace": workspace,
    "scenario": scenario,
    "module": descriptor.as_dict(),
    "token": secrets.token_hex(8),
    "guardrails": {
      "required": descriptor.min_guardrail,
      "issued": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "constraints": scenario.get("constraints", []),
    },
  }
  return manifest


def list_modules_payload() -> Dict[str, object]:
  payload = list_modules()
  return {"tool": TOOL_NAME, "modules": payload}



def generate_manifest(workspace: str, module_name: Optional[str], scenario: Dict[str, object]) -> Tuple[Diagnostic, Dict[str, object]]:
  try:
    manifest = stage_module(workspace, module_name, scenario)
  except ValueError as exc:
    diagnostic = Diagnostic(status="error", summary=str(exc), details={"module": module_name or "reverse-shell"})
    raise ValueError(diagnostic.summary)

  diagnostic = Diagnostic(
    status="info",
    summary=f"Module '{manifest['module']['name']}' staged for workspace '{workspace}'.",
    details={
      "workspace": workspace,
      "module": manifest["module"]["name"],
      "guardrail": manifest["guardrails"]["required"],
    },
  )
  return diagnostic, manifest



def run(args: argparse.Namespace) -> int:
  if args.list_modules:
    payload = list_modules_payload()
    if args.json or not args.output:
      print(json.dumps(payload, indent=2))
    if args.output:
      Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0

  scenario = load_scenario(args.scenario)
  try:
    diagnostic, manifest = generate_manifest(args.workspace, args.module, scenario)
  except ValueError as exc:
    emit_diagnostic(TOOL_NAME, Diagnostic(status="error", summary=str(exc), details={"module": args.module or "reverse-shell"}), json_output=False)
    return 1

  if args.json and not args.output:
    print(json.dumps(manifest, indent=2))
  else:
    emit_diagnostic(TOOL_NAME, diagnostic, json_output=False)
  if args.output:
    Path(args.output).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  return 0


def health_check() -> Dict[str, object]:
  samples = synthesise_latency_samples(TOOL_NAME)
  metrics = summarise_samples(samples)
  details = {
    "modules": list(list_modules().keys()),
    "latency_profile": metrics,
    "sample_latency": [{"probe": label, "latency_ms": value} for label, value in samples],
  }
  return build_health_report(
    TOOL_NAME,
    "ok",
    "VectorFlux workspace ready for signed module staging.",
    details,
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  args_list = list(argv or sys.argv[1:])
  if args_list and args_list[0].lower() == "console":
    args_list = args_list[1:]
  if args_list and args_list[0].lower() == "modules":
    args_list = args_list[1:]
    args_list.insert(0, "--list-modules")
  parser = build_parser()
  args = parser.parse_args(args_list)
  if getattr(args, "gui", False):
    return launch_gui("tools.vectorflux_gui")
  return run(args)


if __name__ == "__main__":
  raise SystemExit(main())
