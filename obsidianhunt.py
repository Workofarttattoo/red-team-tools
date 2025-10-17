"""
ObsidianHunt — host hardening assessor for forensic baselines.

The utility inspects local system artefacts, validates the presence of baseline
controls, and emits a remediation manifest compatible with AgentaOS supervisors.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import time
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ._toolkit import (
  Diagnostic,
  build_health_report,
  emit_diagnostic,
  launch_gui,
  summarise_samples,
  synthesise_latency_samples,
)


TOOL_NAME = "ObsidianHunt"

LINUX_CHECKS = [
  ("auditd-enabled", Path("/etc/audit/auditd.conf")),
  ("secure-tty", Path("/etc/securetty")),
  ("pam-d-conf", Path("/etc/pam.d/system-auth")),
  ("firewall-nftables", Path("/etc/nftables.conf")),
]

MAC_CHECKS = [
  ("sip-status", Path("/usr/bin/csrutil")),
  ("gatekeeper", Path("/usr/sbin/spctl")),
  ("firewall", Path("/usr/libexec/ApplicationFirewall/socketfilterfw")),
]

WINDOWS_CHECKS = [
  ("defender", Path(r"C:\Program Files\Windows Defender")),
  ("bitlocker", Path(r"C:\Windows\System32\manage-bde.exe")),
  ("appguard", Path(r"C:\Windows\System32\gpresult.exe")),
]


@dataclass
class ControlStatus:
  control: str
  status: str
  evidence: str

  def as_dict(self) -> Dict[str, object]:
    return asdict(self)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="ObsidianHunt hardening assessor.")
  parser.add_argument("--profile", default="workstation", help="Assessment profile (e.g. workstation, server).")
  parser.add_argument("--output", help="Write remediation manifest to path.")
  parser.add_argument("--json", action="store_true", help="Emit JSON payload.")
  parser.add_argument("--checks", help="Optional JSON file describing additional controls.")
  parser.add_argument("--gui", action="store_true", help="Launch the ObsidianHunt graphical interface.")
  return parser


def load_additional_checks(path: Optional[str]) -> List[tuple[str, Path]]:
  if not path:
    return []
  try:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
  except FileNotFoundError:
    return []
  except json.JSONDecodeError:
    return []
  checks: List[tuple[str, Path]] = []
  for entry in data:
    if isinstance(entry, dict) and "control" in entry and "path" in entry:
      checks.append((str(entry["control"]), Path(str(entry["path"]))))
  return checks


def platform_checks() -> List[tuple[str, Path]]:
  system = platform.system().lower()
  if system.startswith("linux"):
    return LINUX_CHECKS
  if system == "darwin":
    return MAC_CHECKS
  if system.startswith("windows"):
    return WINDOWS_CHECKS
  return []


def evaluate_control(control: str, path: Path) -> ControlStatus:
  if path.is_dir() or path.is_file():
    return ControlStatus(control=control, status="pass", evidence=str(path))
  if shutil.which(path.name):
    return ControlStatus(control=control, status="pass", evidence=str(shutil.which(path.name)))
  return ControlStatus(control=control, status="warn", evidence=f"missing:{path}")


def gather_baseline(profile: str, extra_checks: List[tuple[str, Path]]) -> Dict[str, object]:
  controls = [evaluate_control(control, path) for control, path in platform_checks()]
  controls.extend(evaluate_control(control, path) for control, path in extra_checks)
  warnings = [control for control in controls if control.status != "pass"]
  hostname = socket.gethostname()
  return {
    "profile": profile,
    "hostname": hostname,
    "platform": platform.platform(),
    "controls": [control.as_dict() for control in controls],
    "warnings": len(warnings),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "env": {
      "AGENTA_FORENSIC_MODE": os.environ.get("AGENTA_FORENSIC_MODE", ""),
    },
  }


def evaluate_baseline(profile: str, extra_checks: List[tuple[str, Path]]) -> Tuple[Diagnostic, Dict[str, object]]:
  baseline = gather_baseline(profile, extra_checks)
  status = "info"
  summary = f"Captured {len(baseline['controls'])} control(s); {baseline['warnings']} require attention."
  if baseline["warnings"]:
    status = "warn"
  diagnostic = Diagnostic(status=status, summary=summary, details={
    "profile": profile,
    "controls": len(baseline["controls"]),
    "warnings": baseline["warnings"],
  })
  payload = {"tool": TOOL_NAME, **baseline}
  return diagnostic, payload


def run(args: argparse.Namespace) -> int:
  extra_checks = load_additional_checks(args.checks)
  diagnostic, payload = evaluate_baseline(args.profile, extra_checks)
  if args.json and not args.output:
    print(json.dumps(payload, indent=2))
  else:
    emit_diagnostic(TOOL_NAME, diagnostic, json_output=False)
  if args.output:
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
  return 0


def health_check() -> Dict[str, object]:
  samples = synthesise_latency_samples(TOOL_NAME)
  metrics = summarise_samples(samples)
  details = {
    "platform": platform.system(),
    "latency_profile": metrics,
    "sample_latency": [{"probe": label, "latency_ms": value} for label, value in samples],
  }
  return build_health_report(
    TOOL_NAME,
    "ok",
    "ObsidianHunt baseline collector ready.",
    details,
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  args_list = list(argv or sys.argv[1:])
  if args_list and args_list[0].lower() == "audit":
    args_list = args_list[1:]
  parser = build_parser()
  args = parser.parse_args(args_list)
  if getattr(args, "gui", False):
    return launch_gui("tools.obsidianhunt_gui")
  return run(args)


if __name__ == "__main__":
  raise SystemExit(main())
