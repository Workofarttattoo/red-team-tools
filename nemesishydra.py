"""
NemesisHydra — distributed authentication rehearsal coordinator.

Consumes target definitions and wordlists, estimates spray durations, and
emits structured telemetry for supervisor review without performing live
authentication attempts.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from ._toolkit import (
  Diagnostic,
  build_health_report,
  emit_diagnostic,
  launch_gui,
  summarise_samples,
  synthesise_latency_samples,
)


TOOL_NAME = "NemesisHydra"

SAMPLE_TARGETS: Sequence[str] = (
  "ssh://guardian.agenta",
  "rdp://192.0.2.55",
  "https://auth.agenta/login",
)


@dataclass
class TargetAssessment:
  target: str
  service: str
  host: str
  port: int
  wordlist_size: int
  estimated_minutes: float
  throttle_risk: str
  resolution_status: str

  def as_dict(self) -> Dict[str, object]:
    payload = asdict(self)
    payload["estimated_minutes"] = round(payload["estimated_minutes"], 2)
    return payload


SERVICE_PORTS = {
  "ssh": 22,
  "rdp": 3389,
  "ftp": 21,
  "http": 80,
  "https": 443,
  "smb": 445,
}


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="NemesisHydra authentication rehearsal utility.")
  parser.add_argument("primary", nargs="?", help="Primary target URI (alias for --target).")
  parser.add_argument("--targets", help="Path to target definitions (one URI per line).")
  parser.add_argument("--target", action="append", help="Inline target URI (repeatable).")
  parser.add_argument("--wordlist", help="Path to credential candidates (username:password or password).")
  parser.add_argument("--rate-limit", type=int, default=12, help="Attempts per minute per target.")
  parser.add_argument("--json", action="store_true", help="Emit JSON payload.")
  parser.add_argument("--output", help="Write JSON payload to path.")
  parser.add_argument("--demo", action="store_true", help="Use built-in demo targets and wordlist.")
  parser.add_argument("--gui", action="store_true", help="Launch the NemesisHydra graphical interface.")
  return parser


def load_targets(path: Optional[str], inline: Optional[Sequence[str]], demo: bool) -> List[str]:
  targets: List[str] = []
  if path:
    try:
      targets.extend(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    except FileNotFoundError:
      pass
  if inline:
    targets.extend(item.strip() for item in inline if item and item.strip())
  if demo or not targets:
    targets.extend(SAMPLE_TARGETS)
  return list(dict.fromkeys(targets))


def load_wordlist(path: Optional[str], demo: bool) -> List[str]:
  words: List[str] = []
  if path:
    try:
      words.extend(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    except FileNotFoundError:
      pass
  if demo or not words:
    words.extend(["admin:admin", "root:toor", "service:Welcome1!", "password"])
  return list(dict.fromkeys(words))


def resolve_host(host: str) -> str:
  try:
    socket.gethostbyname(host)
    return "resolved"
  except socket.gaierror:
    return "unresolved"
  except Exception:
    return "unknown"


def estimate_duration(wordlist_size: int, rate_limit: int) -> float:
  if rate_limit <= 0:
    return float("inf")
  attempts_per_minute = max(1, rate_limit)
  minutes = wordlist_size / attempts_per_minute
  return minutes


def throttle_risk(service: str, rate_limit: int) -> str:
  if service in {"ssh", "rdp"} and rate_limit > 20:
    return "high"
  if rate_limit > 40:
    return "high"
  if rate_limit > 15:
    return "medium"
  return "low"


def assess_target(target: str, wordlist_size: int, rate_limit: int) -> TargetAssessment:
  parsed = urlparse(target)
  service = parsed.scheme or "ssh"
  host = parsed.hostname or target
  port = parsed.port or SERVICE_PORTS.get(service, SERVICE_PORTS["ssh"])
  resolution = resolve_host(host)
  duration = estimate_duration(wordlist_size, rate_limit)
  risk = throttle_risk(service, rate_limit)
  return TargetAssessment(
    target=target,
    service=service,
    host=host,
    port=port,
    wordlist_size=wordlist_size,
    estimated_minutes=duration,
    throttle_risk=risk,
    resolution_status=resolution,
  )


def humanise_minutes(minutes: float) -> str:
  if math.isinf(minutes):
    return "infinite (rate limit disabled)"
  if minutes < 1:
    return f"{minutes * 60:.0f} seconds"
  return f"{minutes:.1f} minutes"


def evaluate_targets(
  targets: Sequence[str],
  wordlist_size: int,
  rate_limit: int,
) -> Tuple[Diagnostic, Dict[str, object]]:
  assessments = [assess_target(target, wordlist_size, rate_limit) for target in targets]
  high_risk = [item for item in assessments if item.throttle_risk == "high"]

  status = "info"
  summary = f"Prepared rehearsal against {len(targets)} target(s) with {wordlist_size} credential candidates."
  if high_risk:
    status = "warn"
    summary = f"{len(high_risk)} target(s) exceed safe throttle thresholds."

  diagnostic = Diagnostic(status=status, summary=summary, details={
    "targets": len(assessments),
    "wordlist_entries": wordlist_size,
    "high_risk": len(high_risk),
    "rate_limit_per_minute": rate_limit,
  })

  payload = {
    "tool": TOOL_NAME,
    "rate_limit": rate_limit,
    "wordlist": wordlist_size,
    "targets": [assessment.as_dict() for assessment in assessments],
    "durations_human": [humanise_minutes(assessment.estimated_minutes) for assessment in assessments],
  }
  return diagnostic, payload


def run(args: argparse.Namespace) -> int:
  inline_targets = list(args.target or [])
  if args.primary:
    inline_targets.append(args.primary)
  targets = load_targets(args.targets, inline_targets, args.demo)
  wordlist = load_wordlist(args.wordlist, args.demo)
  diagnostic, payload = evaluate_targets(targets, len(wordlist), args.rate_limit)

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
    "supported_services": list(SERVICE_PORTS.keys()),
    "latency_profile": metrics,
    "sample_latency": [{"probe": label, "latency_ms": value} for label, value in samples],
  }
  return build_health_report(
    TOOL_NAME,
    "ok",
    "NemesisHydra orchestration pipeline staged.",
    details,
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  if getattr(args, "gui", False):
    return launch_gui("tools.nemesishydra_gui")
  return run(args)


if __name__ == "__main__":
  raise SystemExit(main())
