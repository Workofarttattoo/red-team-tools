"""
CipherSpear — precision database exploitation rehearsal engine.

This reimagined utility inspects candidate injection vectors, evaluates them
against the selected rehearsal techniques, and surfaces telemetry suited for
safe, read-only validation of database exposure.  The implementation focuses on
rapid heuristics instead of live exploitation so operators can triage findings
without mutating the target system.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlparse

from ._toolkit import (
  Diagnostic,
  build_health_report,
  emit_diagnostic,
  launch_gui,
  summarise_samples,
  synthesise_latency_samples,
)


TOOL_NAME = "CipherSpear"

SUSPICIOUS_PATTERNS: Sequence[Tuple[re.Pattern[str], str, int]] = [
  (re.compile(r"(?:'|\")\s*(?:or|and)\s+1\s*=\s*1", re.IGNORECASE), "boolean-tautology", 4),
  (re.compile(r"union\s+select", re.IGNORECASE), "union-select", 3),
  (re.compile(r"sleep\(\s*\d+\s*\)", re.IGNORECASE), "time-based-delay", 2),
  (re.compile(r"benchmark\(", re.IGNORECASE), "timing-benchmark", 2),
  (re.compile(r"(?:--|#)\s*$"), "inline-comment", 1),
  (re.compile(r"/\*.*\*/", re.IGNORECASE | re.DOTALL), "multiline-comment", 1),
  (re.compile(r"load_file\s*\(", re.IGNORECASE), "file-read", 3),
  (re.compile(r"into\s+outfile", re.IGNORECASE), "file-write", 3),
  (re.compile(r"xp_cmdshell", re.IGNORECASE), "command-exec", 4),
  (re.compile(r"regexp\s+'[^']*'", re.IGNORECASE), "pattern-match", 1),
]

SAMPLE_VECTORS: Sequence[str] = (
  "GET /search?term=pulse' OR 1=1-- HTTP/1.1",
  "POST /api/user HTTP/1.1\\n{id: 4, \"username\": \"admin\" }",
  "username=guest&password=guest",
  "GET /inventory?category=electronics&sort=price%20desc HTTP/1.1",
  "https://example.internal/login?user=admin&pass='; sleep(5); --",
)


@dataclass
class VectorAssessment:
  vector: str
  risk_score: int
  risk_label: str
  findings: List[str]
  parameters: List[Tuple[str, str]]
  recommendation: str

  def as_dict(self) -> Dict[str, object]:
    payload = asdict(self)
    payload["parameter_count"] = len(self.parameters)
    return payload


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="CipherSpear precision rehearsal utility.")
  parser.add_argument("--dsn", required=True, help="Target database DSN (e.g. postgresql://user@host/db).")
  parser.add_argument("--tech", default="", help="Comma separated rehearsal techniques (e.g. blind,bool,time).")
  parser.add_argument("--queries", help="Path to file containing candidate injection vectors.")
  parser.add_argument("--vector", action="append", help="Inline injection vector (repeatable).")
  parser.add_argument("--demo", action="store_true", help="Analyse built-in sample vectors.")
  parser.add_argument("--json", action="store_true", help="Emit JSON output.")
  parser.add_argument("--output", help="Write the detailed JSON payload to the specified path.")
  parser.add_argument("--gui", action="store_true", help="Launch the CipherSpear graphical interface.")
  return parser


def load_vectors(path: Optional[str], inline: Optional[Sequence[str]], demo: bool) -> List[str]:
  vectors: List[str] = []
  if path:
    entries = Path(path).read_text(encoding="utf-8").splitlines()
    for entry in entries:
      entry = entry.strip()
      if entry and not entry.startswith("#"):
        vectors.append(entry)
  if inline:
    vectors.extend(item for item in inline if item)
  if demo or not vectors:
    vectors.extend(SAMPLE_VECTORS)
  return vectors


def extract_parameters(vector: str) -> List[Tuple[str, str]]:
  """
  Extract parameter key/value pairs from URLs, query strings, or form bodies.
  """

  pairs: List[Tuple[str, str]] = []
  if "://" in vector:
    parsed = urlparse(vector)
  else:
    parsed = urlparse(f"http://dummy/{vector.lstrip('/')}")
  pairs.extend(parse_qsl(parsed.query, keep_blank_values=True))
  fragments = parsed.fragment.split("&") if parsed.fragment else []
  for fragment in fragments:
    if "=" in fragment:
      key, value = fragment.split("=", maxsplit=1)
      pairs.append((key, value))
  if "=" in vector and not pairs:
    for chunk in vector.split("&"):
      if "=" not in chunk:
        continue
      key, value = chunk.split("=", maxsplit=1)
      pairs.append((key, value))
  return pairs


def assess_vector(vector: str, techniques: Sequence[str]) -> VectorAssessment:
  findings: List[str] = []
  score = 0
  for pattern, label, severity in SUSPICIOUS_PATTERNS:
    if pattern.search(vector):
      findings.append(label)
      score += severity
  params = extract_parameters(vector)
  for _, value in params:
    if "'" in value or "\"" in value:
      findings.append("quote-injection")
      score += 1
    if re.search(r"(select|update|delete)\s+\w+", value, re.IGNORECASE):
      findings.append("embedded-sql")
      score += 2
  if any(t in {"blind", "time"} for t in techniques) and "time-based-delay" in findings:
    score += 2
  risk_label = "low"
  if score >= 8:
    risk_label = "high"
  elif score >= 4:
    risk_label = "medium"
  recommendation = "Review parameterised queries and input validation."
  if "file-write" in findings or "command-exec" in findings:
    recommendation = "Immediate mitigation required; high-impact primitives detected."
  elif "union-select" in findings:
    recommendation = "Verify column alignment and sanitise UNION clauses."
  return VectorAssessment(
    vector=vector,
    risk_score=score,
    risk_label=risk_label,
    findings=sorted(set(findings)),
    parameters=params,
    recommendation=recommendation,
  )


def sanitise_dsn(dsn: str) -> Dict[str, object]:
  parsed = urlparse(dsn)
  redacted_netloc = parsed.netloc
  if "@" in redacted_netloc:
    userinfo, hostinfo = redacted_netloc.split("@", maxsplit=1)
    if ":" in userinfo:
      user, _ = userinfo.split(":", maxsplit=1)
    else:
      user = userinfo
    redacted_netloc = f"{user}@{hostinfo}"
  return {
    "scheme": parsed.scheme or "unknown",
    "netloc": redacted_netloc,
    "path": parsed.path or "/",
    "params": parsed.params,
  }


def render_output(args: argparse.Namespace, diagnostic: Diagnostic, payload: Dict[str, object]) -> int:
  if args.json and not args.output:
    print(json.dumps(payload, indent=2))
  else:
    emit_diagnostic(TOOL_NAME, diagnostic, json_output=False)
  if args.output:
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
  return 0


def evaluate_vectors(
  dsn: str,
  techniques: Sequence[str],
  vectors: Sequence[str],
) -> Tuple[Diagnostic, Dict[str, object]]:
  assessments = [assess_vector(vector, techniques) for vector in vectors]

  totals = {
    "low": sum(1 for item in assessments if item.risk_label == "low"),
    "medium": sum(1 for item in assessments if item.risk_label == "medium"),
    "high": sum(1 for item in assessments if item.risk_label == "high"),
  }
  status = "info"
  summary = "Injection rehearsal completed; no high-risk findings."
  if totals["high"]:
    status = "warn"
    summary = f"{totals['high']} high-risk injection vector(s) detected."
  elif totals["medium"]:
    summary = f"{totals['medium']} medium-risk vector(s) require review."

  payload = {
    "tool": TOOL_NAME,
    "dsn": sanitise_dsn(dsn),
    "techniques": techniques,
    "vectors_processed": len(assessments),
    "findings": totals,
    "assessments": [assessment.as_dict() for assessment in assessments],
  }
  diagnostic = Diagnostic(status=status, summary=summary, details={
    "vectors": len(assessments),
    "high_risk": totals["high"],
    "medium_risk": totals["medium"],
    "techniques": techniques or ["baseline"],
  })
  return diagnostic, payload


def run(args: argparse.Namespace) -> int:
  techniques = [chunk.strip().lower() for chunk in args.tech.split(",") if chunk.strip()]
  vectors = load_vectors(args.queries, args.vector, args.demo)
  diagnostic, payload = evaluate_vectors(args.dsn, techniques, vectors)
  return render_output(args, diagnostic, payload)


def health_check() -> Dict[str, object]:
  samples = synthesise_latency_samples(TOOL_NAME)
  metrics = summarise_samples(samples)
  details = {
    "techniques": ["blind", "bool", "time", "stacked"],
    "latency_profile": metrics,
    "sample_latency": [{"probe": label, "latency_ms": latency} for label, latency in samples],
  }
  return build_health_report(
    TOOL_NAME,
    "ok",
    "CipherSpear parsing and rehearsal engine initialised.",
    details,
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  if args.gui:
    return launch_gui("tools.cipherspear_gui")
  return run(args)


if __name__ == "__main__":
  raise SystemExit(main())
