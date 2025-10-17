"""
SkyBreaker — wireless audit orchestrator with spectrum overlays.

The CLI provides two primary flows:

* ``capture`` ingests scan output (CSV or the built-in demo sample) and emits a
  structured JSON capture suitable for storage in telemetry pipelines.
* ``analyze`` consumes the capture and highlights risky access points, channel
  congestion, and security posture observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ._toolkit import (
  Diagnostic,
  build_health_report,
  emit_diagnostic,
  launch_gui,
  summarise_samples,
  synthesise_latency_samples,
)


TOOL_NAME = "SkyBreaker"

AUDIT_PROFILES: Dict[str, str] = {
  "audit": "Capture live spectrum data for target interface.",
  "recon": "Lightweight survey focusing on nearby AP signal strength.",
  "forensic": "Passive verification without channel hopping.",
}

SAMPLE_SCAN: Sequence[Dict[str, object]] = (
  {"ssid": "Agenta-Lab", "bssid": "de:ad:be:ef:00:01", "signal": -42, "channel": 6, "security": "WPA2-PSK"},
  {"ssid": "GuestNet", "bssid": "de:ad:be:ef:00:02", "signal": -63, "channel": 1, "security": "WPA2-PSK"},
  {"ssid": "ResearchMesh", "bssid": "de:ad:be:ef:00:03", "signal": -55, "channel": 149, "security": "WPA3-SAE"},
  {"ssid": "", "bssid": "de:ad:be:ef:00:04", "signal": -71, "channel": 11, "security": "Hidden-WPA2"},
  {"ssid": "CafeFreeWiFi", "bssid": "de:ad:be:ef:00:05", "signal": -67, "channel": 6, "security": "OPEN"},
)


@dataclass
class NetworkRecord:
  ssid: str
  bssid: str
  signal: int
  channel: int
  security: str
  first_seen: float
  last_seen: float

  def as_dict(self) -> Dict[str, object]:
    payload = asdict(self)
    payload["hidden"] = not bool(self.ssid)
    return payload


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="SkyBreaker wireless audit orchestrator.")
  parser.add_argument("--gui", action="store_true", help="Launch the SkyBreaker graphical interface.")
  sub = parser.add_subparsers(dest="command")

  capture = sub.add_parser("capture", help="Ingest a wireless scan and emit structured telemetry.")
  capture.add_argument("interface", nargs="?", help="Wireless interface identifier.")
  capture.add_argument("--iface", dest="interface", help="Wireless interface identifier.")
  capture.add_argument("--profile", default="audit", choices=list(AUDIT_PROFILES.keys()), help="Capture profile.")
  capture.add_argument("--scan-file", help="CSV scan file (ssid,bssid,signal,channel,security).")
  capture.add_argument("--channel", type=int, default=0, help="Focus channel (0 keeps all).")
  capture.add_argument("--json", action="store_true", help="Emit JSON payload to stdout.")
  capture.add_argument("--output", help="Write capture JSON to path.")

  analyze = sub.add_parser("analyze", help="Evaluate a SkyBreaker capture file.")
  analyze.add_argument("capture", help="Path to capture JSON produced by the capture command.")
  analyze.add_argument("--json", action="store_true", help="Emit findings as JSON.")
  analyze.add_argument("--output", help="Write findings JSON to path.")

  return parser


def _load_scan_csv(path: str) -> List[Dict[str, object]]:
  entries: List[Dict[str, object]] = []
  with open(path, "r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      try:
        entries.append({
          "ssid": row.get("ssid", ""),
          "bssid": row.get("bssid", ""),
          "signal": int(row.get("signal", "-90")),
          "channel": int(row.get("channel", "0")),
          "security": row.get("security", "UNKNOWN"),
        })
      except ValueError:
        continue
  return entries


def _load_scan(path: Optional[str]) -> List[Dict[str, object]]:
  if path:
    try:
      entries = _load_scan_csv(path)
      if entries:
        return entries
    except FileNotFoundError:
      pass
  return [dict(item) for item in SAMPLE_SCAN]


def _normalise_records(entries: Iterable[Dict[str, object]]) -> List[NetworkRecord]:
  now = time.time()
  records: List[NetworkRecord] = []
  for entry in entries:
    records.append(NetworkRecord(
      ssid=str(entry.get("ssid", "")),
      bssid=str(entry.get("bssid", "")).lower(),
      signal=int(entry.get("signal", -90)),
      channel=int(entry.get("channel", 0)),
      security=str(entry.get("security", "UNKNOWN")),
      first_seen=now,
      last_seen=now,
    ))
  return records


def build_capture_payload(
  iface: str,
  profile: str,
  channel: int,
  records: List[NetworkRecord],
) -> Tuple[Diagnostic, Dict[str, object]]:
  open_networks = [record for record in records if record.security.upper() in {"OPEN", "WEP"}]
  congested_channels = _calculate_channel_congestion(records)

  payload = {
    "tool": TOOL_NAME,
    "interface": iface,
    "profile": profile,
    "channel_filter": channel,
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "networks": [record.as_dict() for record in records],
    "open_networks": len(open_networks),
    "channel_congestion": congested_channels,
  }

  status = "info"
  summary = f"Captured {len(records)} access point(s) on interface {iface}."
  if open_networks:
    status = "warn"
    summary = f"Detected {len(open_networks)} open or WEP-secured network(s)."

  diagnostic = Diagnostic(status=status, summary=summary, details={
    "interface": iface,
    "profile": profile,
    "networks": len(records),
    "open_networks": len(open_networks),
  })
  return diagnostic, payload


def run_capture(
  iface: str,
  profile: str,
  channel: int,
  *,
  scan_file: Optional[str] = None,
) -> Tuple[Diagnostic, Dict[str, object]]:
  entries = _load_scan(scan_file)
  records = _normalise_records(entries)
  if channel:
    records = [record for record in records if record.channel == channel]
  return build_capture_payload(iface, profile, channel, records)


def command_capture(args: argparse.Namespace) -> int:
  iface = getattr(args, "interface", None) or getattr(args, "iface", "wlan0")
  diagnostic, payload = run_capture(
    iface,
    args.profile,
    args.channel,
    scan_file=args.scan_file,
  )
  if args.json and not args.output:
    print(json.dumps(payload, indent=2))
  else:
    emit_diagnostic(TOOL_NAME, diagnostic, json_output=False)
  if args.output:
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
  return 0


def _calculate_channel_congestion(records: Iterable[NetworkRecord]) -> Dict[str, object]:
  bucket: Dict[int, int] = {}
  for record in records:
    bucket[record.channel] = bucket.get(record.channel, 0) + 1
  if not bucket:
    return {"max_channel": None, "max_count": 0, "distribution": {}}
  max_channel = max(bucket, key=bucket.get)
  return {
    "max_channel": max_channel,
    "max_count": bucket[max_channel],
    "distribution": bucket,
  }


def build_analysis_payload(data: Dict[str, object]) -> Tuple[Diagnostic, Dict[str, object]]:
  networks = data.get("networks", [])
  open_networks = [net for net in networks if str(net.get("security", "")).upper() in {"OPEN", "WEP"}]
  hidden_networks = [net for net in networks if not net.get("ssid")]
  congested = data.get("channel_congestion", {"max_channel": None, "max_count": 0})

  status = "info"
  summary = "Wireless capture baseline looks steady."
  if open_networks:
    status = "warn"
    summary = f"{len(open_networks)} open or legacy encrypted network(s) detected."
  elif hidden_networks:
    summary = f"{len(hidden_networks)} hidden network(s) present; monitor for rogue APs."

  findings = {
    "interface": data.get("interface", "unknown"),
    "profile": data.get("profile", "unknown"),
    "networks": len(networks),
    "open_networks": len(open_networks),
    "hidden_networks": len(hidden_networks),
    "congested_channel": congested.get("max_channel"),
    "congestion_count": congested.get("max_count", 0),
  }

  diagnostic = Diagnostic(status=status, summary=summary, details=findings)
  payload = {"tool": TOOL_NAME, "findings": findings, "networks": networks}
  return diagnostic, payload


def run_analysis(capture_path: str) -> Tuple[Diagnostic, Dict[str, object]]:
  data = json.loads(Path(capture_path).read_text(encoding="utf-8"))
  return build_analysis_payload(data)


def command_analyze(args: argparse.Namespace) -> int:
  diagnostic, payload = run_analysis(args.capture)
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
    "supported_profiles": list(AUDIT_PROFILES.keys()),
    "latency_profile": metrics,
    "sample_latency": [{"probe": label, "latency_ms": value} for label, value in samples],
  }
  return build_health_report(
    TOOL_NAME,
    "ok",
    "SkyBreaker capture pipeline available.",
    details,
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.gui:
    return launch_gui("tools.skybreaker_gui")

  if args.command == "capture":
    return command_capture(args)
  if args.command == "analyze":
    return command_analyze(args)

  parser.print_help()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
