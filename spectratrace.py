"""
SpectraTrace — protocol inspection studio with redacted capture lanes.

The utility consumes lightweight packet captures (PCAP or JSON) and produces
structured telemetry plus optional policy playbook evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import json
import struct
import sys
import time
from collections import Counter, defaultdict
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


TOOL_NAME = "SpectraTrace"

SAMPLE_PACKETS: Sequence[Dict[str, object]] = (
  {"timestamp": 0.001, "src": "10.0.0.10", "dst": "10.0.0.1", "protocol": "DNS", "length": 96, "info": "Query A guardian.agenta"},
  {"timestamp": 0.004, "src": "10.0.0.1", "dst": "10.0.0.10", "protocol": "DNS", "length": 128, "info": "Response A guardian.agenta 10.0.0.42"},
  {"timestamp": 0.150, "src": "10.0.0.10", "dst": "10.0.0.42", "protocol": "TLS", "length": 512, "info": "ClientHello SNI portal.agenta"},
  {"timestamp": 0.220, "src": "10.0.0.42", "dst": "10.0.0.10", "protocol": "TLS", "length": 1024, "info": "ServerHello certificate wildcard.agenta"},
  {"timestamp": 1.120, "src": "10.0.0.10", "dst": "198.51.100.23", "protocol": "HTTP", "length": 864, "info": "POST /login Content-Length=248"},
  {"timestamp": 1.124, "src": "198.51.100.23", "dst": "10.0.0.10", "protocol": "HTTP", "length": 512, "info": "HTTP/1.1 200 OK"},
)

WORKFLOWS: Dict[str, Dict[str, object]] = {
  "quick-scan": {
    "label": "Quick Scan",
    "description": "Summarise top talkers and protocols with a lightweight capture for rapid situational awareness.",
    "max_packets": 400,
    "playbook_rules": {},
  },
  "latency-troubleshoot": {
    "label": "Latency Troubleshooter",
    "description": "Focus on TCP handshakes, retransmissions, and zero window events to diagnose latency complaints.",
    "max_packets": 1200,
    "playbook_rules": {
      "tcp-retransmit": "retransmission",
      "zero-window": "zero window",
      "tcp-syn": "SYN",
    },
  },
  "suspicious-http": {
    "label": "Suspicious HTTP",
    "description": "Highlight unusual HTTP verbs, large POST bodies, and potential data exfiltration patterns.",
    "max_packets": 1600,
    "playbook_rules": {
      "http-post-large": "POST .*Content-Length",
      "http-put": "PUT ",
      "http-x-custom": "X-",
    },
  },
}

PLUGIN_DIR = Path(__file__).with_name("spectratrace_plugins")


@dataclass
class PacketEntry:
  timestamp: float
  src: str
  dst: str
  protocol: str
  length: int
  info: str

  def as_dict(self) -> Dict[str, object]:
    return asdict(self)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="SpectraTrace protocol inspection utility.")
  parser.add_argument("--iface", default="any", help="Interface associated with the capture (metadata only).")
  parser.add_argument("--capture", help="Path to capture file (pcap, json).")
  parser.add_argument("--playbook", help="Optional YAML/JSON playbook describing alert patterns.")
  parser.add_argument("--max-packets", type=int, default=2000, help="Maximum packets to process from capture.")
  parser.add_argument(
    "--workflow",
    choices=list(WORKFLOWS.keys()),
    help="Apply a guided workflow preset to enrich analysis.",
  )
  parser.add_argument("--json", action="store_true", help="Emit JSON output.")
  parser.add_argument("--output", help="Write JSON payload to path.")
  parser.add_argument("--report", help="Write a formatted analysis report to the given path.")
  parser.add_argument(
    "--report-format",
    choices=["markdown", "html"],
    default="markdown",
    help="Report format when --report is provided (default: markdown).",
  )
  parser.add_argument("--gui", action="store_true", help="Launch the SpectraTrace graphical interface.")
  parser.add_argument(
    "--stream",
    action="store_true",
    help="Read newline-delimited JSON packets from stdin instead of a static capture file.",
  )
  return parser


def load_capture(path: Optional[str], limit: int) -> List[PacketEntry]:
  if not path:
    return [PacketEntry(**packet) for packet in SAMPLE_PACKETS][:limit]
  suffix = Path(path).suffix.lower()
  if suffix in {".json", ".jsonl"}:
    return _load_capture_json(path, limit)
  try:
    return _load_capture_pcap(path, limit)
  except Exception:
    return [PacketEntry(**packet) for packet in SAMPLE_PACKETS][:limit]


def _load_capture_json(path: str, limit: int) -> List[PacketEntry]:
  data = json.loads(Path(path).read_text(encoding="utf-8"))
  packets: List[PacketEntry] = []
  if isinstance(data, dict):
    data = data.get("packets", [])
  for entry in data:
    try:
      packets.append(PacketEntry(
        timestamp=float(entry.get("timestamp", 0.0)),
        src=str(entry.get("src", "0.0.0.0")),
        dst=str(entry.get("dst", "0.0.0.0")),
        protocol=str(entry.get("protocol", "UNKNOWN")),
        length=int(entry.get("length", 0)),
        info=str(entry.get("info", "")),
      ))
    except (TypeError, ValueError):
      continue
    if len(packets) >= limit:
      break
  return packets


def _load_capture_pcap(path: str, limit: int) -> List[PacketEntry]:
  packets: List[PacketEntry] = []
  with open(path, "rb") as handle:
    header = handle.read(24)
    if len(header) < 24:
      return packets
    magic = struct.unpack("I", header[:4])[0]
    if magic == 0xa1b2c3d4:
      endian = ">"
    elif magic == 0xd4c3b2a1:
      endian = "<"
    else:
      return packets
    while len(packets) < limit:
      packet_header = handle.read(16)
      if len(packet_header) < 16:
        break
      ts_sec, ts_usec, incl_len, _ = struct.unpack(f"{endian}IIII", packet_header)
      packet_data = handle.read(incl_len)
      if len(packet_data) < incl_len:
        break
      entry = _parse_packet(packet_data, ts_sec + ts_usec / 1_000_000)
      if entry:
        packets.append(entry)
  return packets


def _parse_packet(data: bytes, timestamp: float) -> Optional[PacketEntry]:
  if len(data) < 14:
    return None
  eth_type = struct.unpack("!H", data[12:14])[0]
  if eth_type == 0x0800 and len(data) >= 34:
    protocol = data[23]
    src = ".".join(str(b) for b in data[26:30])
    dst = ".".join(str(b) for b in data[30:34])
    proto_name = _protocol_name(protocol)
    info = proto_name
  elif eth_type == 0x86DD and len(data) >= 54:
    src = ":".join(f"{data[22 + i*2]:02x}{data[22 + i*2 + 1]:02x}" for i in range(8))
    dst = ":".join(f"{data[38 + i*2]:02x}{data[38 + i*2 + 1]:02x}" for i in range(8))
    proto_name = _protocol_name(data[20])
    info = proto_name
  else:
    src = "00:00:00:00:00:00"
    dst = "00:00:00:00:00:00"
    proto_name = f"ETH(0x{eth_type:04x})"
    info = proto_name
  return PacketEntry(timestamp=timestamp, src=src, dst=dst, protocol=proto_name, length=len(data), info=info)


def _protocol_name(protocol_number: int) -> str:
  mapping = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    50: "ESP",
    51: "AH",
  }
  return mapping.get(protocol_number, f"IP_PROTO_{protocol_number}")


def _packet_from_mapping(packet: Dict[str, object]) -> PacketEntry:
  timestamp = float(packet.get("timestamp") or time.time())
  src = str(packet.get("src", "0.0.0.0"))
  dst = str(packet.get("dst", "0.0.0.0"))
  protocol = str(packet.get("protocol", "UNKNOWN"))
  length = int(packet.get("length") or 0)
  info = str(packet.get("info", ""))
  return PacketEntry(timestamp=timestamp, src=src, dst=dst, protocol=protocol, length=length, info=info)


def ingest_stream(stream: Iterable[str], limit: int) -> List[PacketEntry]:
  packets: List[PacketEntry] = []
  for raw in stream:
    if len(packets) >= limit:
      break
    line = raw.strip()
    if not line:
      continue
    try:
      payload = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(payload, dict):
      try:
        packets.append(_packet_from_mapping(payload))
      except (TypeError, ValueError):
        continue
  return packets


def load_playbook(path: Optional[str]) -> Dict[str, str]:
  if not path:
    return {}
  text = Path(path).read_text(encoding="utf-8")
  try:
    data = json.loads(text)
    if isinstance(data, dict):
      return {str(key): str(value) for key, value in data.items()}
  except json.JSONDecodeError:
    pass
  rules: Dict[str, str] = {}
  for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith("#") or ":" not in line:
      continue
    key, value = line.split(":", maxsplit=1)
    rules[key.strip()] = value.strip()
  return rules


def analyse_packets(packets: Iterable[PacketEntry], playbook: Dict[str, str]) -> Dict[str, object]:
  talkers: Dict[str, int] = defaultdict(int)
  protocols = Counter()
  alerts: List[Dict[str, object]] = []
  for packet in packets:
    talkers[packet.src] += packet.length
    talkers[packet.dst] += packet.length
    protocols[packet.protocol] += 1
    for name, pattern in playbook.items():
      if pattern and pattern.lower() in packet.info.lower():
        alerts.append({
          "rule": name,
          "pattern": pattern,
          "packet": packet.as_dict(),
        })
  top_talkers = sorted(talkers.items(), key=lambda item: item[1], reverse=True)[:5]
  return {
    "packet_count": sum(protocols.values()),
    "protocol_breakdown": dict(protocols),
    "top_talkers": [{"address": address, "bytes": count} for address, count in top_talkers],
    "alerts": alerts,
  }


def annotate_packets(packets: Sequence[Dict[str, object]], playbook: Dict[str, str]) -> List[Dict[str, object]]:
  annotations: List[Dict[str, object]] = []
  for idx, packet in enumerate(packets):
    tags: List[str] = []
    proto = str(packet.get("protocol", "")).upper()
    info = str(packet.get("info", ""))[:256]
    length = int(packet.get("length") or 0)
    timestamp = float(packet.get("timestamp") or 0.0)

    if proto not in {"TCP", "UDP", "ICMP", "DNS", "TLS", "HTTP"}:
      tags.append("unknown-protocol")
    if length > 1400:
      tags.append("mtu-warning")
    try:
      src_ip = ipaddress.ip_address(str(packet.get("src", "0.0.0.0")))
      dst_ip = ipaddress.ip_address(str(packet.get("dst", "0.0.0.0")))
      if src_ip.is_private and not dst_ip.is_private:
        tags.append("egress")
      if dst_ip.is_private and not src_ip.is_private:
        tags.append("ingress")
    except ValueError:
      tags.append("address-parse-error")

    for name, pattern in playbook.items():
      if pattern and pattern.lower() in info.lower():
        tags.append(f"playbook:{name}")

    if tags:
      annotations.append({
        "packet_index": idx,
        "tags": tags,
        "timestamp": timestamp,
        "summary": ", ".join(tags),
      })
  return annotations


def load_plugin_modules() -> List[object]:
  modules: List[object] = []
  if not PLUGIN_DIR.exists():
    return modules
  for plugin_path in sorted(PLUGIN_DIR.glob("*.py")):
    spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
    if not spec or not spec.loader:
      continue
    try:
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)  # type: ignore[attr-defined]
      modules.append(module)
    except Exception:
      continue
  return modules


def generate_heatmap(packets: Sequence[Dict[str, object]], *, buckets: int = 12) -> List[Dict[str, object]]:
  if not packets or buckets <= 0:
    return []
  timestamps = [float(packet.get("timestamp", 0.0)) for packet in packets]
  min_time = min(timestamps)
  max_time = max(timestamps)
  span = max(max_time - min_time, 1e-6)
  bucket_size = span / buckets
  heatmap: List[Dict[str, object]] = []
  for bucket in range(buckets):
    start = min_time + bucket * bucket_size
    end = start + bucket_size
    heatmap.append({
      "index": bucket,
      "start": start,
      "end": end,
      "packet_count": 0,
      "bytes": 0,
    })
  for packet in packets:
    timestamp = float(packet.get("timestamp", 0.0))
    length = int(packet.get("length") or 0)
    bucket_index = int(((timestamp - min_time) / span) * buckets)
    if bucket_index >= buckets:
      bucket_index = buckets - 1
    if bucket_index < 0:
      bucket_index = 0
    heatmap[bucket_index]["packet_count"] += 1
    heatmap[bucket_index]["bytes"] += length
  return heatmap


def evaluate_capture(
  packets: Sequence[PacketEntry],
  playbook: Dict[str, str],
  iface: str,
  *,
  ingest_mode: str,
  workflow: Optional[str] = None,
  workflow_meta: Optional[Dict[str, object]] = None,
  recipe_config: Optional[Dict[str, object]] = None,
) -> Tuple[Diagnostic, Dict[str, object]]:
  analysis = analyse_packets(packets, playbook)
  status = "info"
  summary = f"Captured {analysis['packet_count']} packet(s) across {len(analysis['protocol_breakdown'])} protocol(s)."
  if analysis["alerts"]:
    status = "warn"
    summary = f"{len(analysis['alerts'])} alert(s) fired from playbook."

  diagnostic = Diagnostic(status=status, summary=summary, details={
    "interface": iface,
    "packets": analysis["packet_count"],
    "protocols": list(analysis["protocol_breakdown"].keys()),
    "alerts": len(analysis["alerts"]),
  })
  packet_payload = [packet.as_dict() for packet in packets]
  annotations = annotate_packets(packet_payload, playbook)
  heatmap = generate_heatmap(packet_payload)
  plugin_insights: List[Dict[str, object]] = []
  for module in load_plugin_modules():
    handler = getattr(module, "process_packets", None)
    if callable(handler):
      try:
        result = handler(packet_payload, analysis)
        if isinstance(result, dict):
          plugin_insights.append(result)
      except Exception:
        continue
  payload = {
    "tool": TOOL_NAME,
    "interface": iface,
    "packets": packet_payload,
    "analysis": analysis,
    "annotations": annotations,
    "heatmap": heatmap,
    "ingest": {
      "mode": ingest_mode,
      "count": len(packet_payload),
    },
  }
  if workflow_meta:
    payload["workflow"] = workflow_meta
  elif workflow:
    payload["workflow"] = {"name": workflow}
  if recipe_config:
    payload["recipe"] = recipe_config
  if plugin_insights:
    payload["plugin_insights"] = plugin_insights
  return diagnostic, payload


def generate_report(payload: Dict[str, object], fmt: str) -> str:
  analysis = payload.get("analysis", {})
  workflow = payload.get("workflow") or {}
  annotations = payload.get("annotations", [])
  insights = payload.get("plugin_insights", [])

  if fmt.lower() == "html":
    lines = [
      "<!DOCTYPE html>",
      "<html><head><meta charset='utf-8'><title>SpectraTrace Report</title>",
      "<style>body{font-family:Arial,Helvetica,sans-serif;margin:2rem;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ccc;padding:0.4rem;text-align:left;} h2{margin-top:2rem;}</style>",
      "</head><body>",
      "<h1>SpectraTrace Report</h1>",
      f"<p><strong>Interface:</strong> {payload.get('interface','')}</p>",
      f"<p><strong>Packet count:</strong> {analysis.get('packet_count',0)}</p>",
      f"<p><strong>Workflow:</strong> {workflow.get('label') or workflow.get('name') or 'none'}</p>",
      "<h2>Protocol Breakdown</h2><table><tr><th>Protocol</th><th>Packets</th></tr>",
    ]
    for proto, count in analysis.get("protocol_breakdown", {}).items():
      lines.append(f"<tr><td>{proto}</td><td>{count}</td></tr>")
    lines.append("</table>")
    if annotations:
      lines.append("<h2>Annotations</h2><table><tr><th>Packet</th><th>Summary</th></tr>")
      for annotation in annotations:
        lines.append(f"<tr><td>{annotation.get('packet_index','')}</td><td>{annotation.get('summary','')}</td></tr>")
      lines.append("</table>")
    if insights:
      lines.append("<h2>Plugin Insights</h2>")
      for insight in insights:
        lines.append(f"<p><strong>{insight.get('name','Plugin')}</strong>: {insight.get('summary','')}</p>")
    lines.append("</body></html>")
    return "\n".join(lines)

  lines = [
    "# SpectraTrace Report",
    "",
    f"* **Interface:** {payload.get('interface','')}",
    f"* **Packet count:** {analysis.get('packet_count',0)}",
    f"* **Workflow:** {workflow.get('label') or workflow.get('name') or 'none'}",
    "",
    "## Protocol Breakdown",
  ]
  for proto, count in analysis.get("protocol_breakdown", {}).items():
    lines.append(f"- {proto}: {count}")
  if annotations:
    lines.append("")
    lines.append("## Annotations")
    for annotation in annotations:
      lines.append(f"- Packet {annotation.get('packet_index')}: {annotation.get('summary','')}")
  if insights:
    lines.append("")
    lines.append("## Plugin Insights")
    for insight in insights:
      lines.append(f"- **{insight.get('name','Plugin')}**: {insight.get('summary','')}")
  return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
  workflow_cfg = WORKFLOWS.get(args.workflow) if args.workflow else None
  effective_limit = args.max_packets
  if workflow_cfg and workflow_cfg.get("max_packets"):
    effective_limit = min(effective_limit, int(workflow_cfg["max_packets"]))

  if args.stream:
    packets = ingest_stream(sys.stdin, effective_limit)
    ingest_mode = "stream"
  else:
    packets = load_capture(args.capture, effective_limit)
    ingest_mode = "capture" if args.capture else "sample"
  playbook = load_playbook(args.playbook)
  if workflow_cfg:
    for name, pattern in workflow_cfg.get("playbook_rules", {}).items():
      playbook.setdefault(name, pattern)

  workflow_meta = None
  if workflow_cfg:
    workflow_meta = {
      "name": args.workflow,
      "label": workflow_cfg.get("label", args.workflow),
      "description": workflow_cfg.get("description", ""),
      "max_packets": workflow_cfg.get("max_packets"),
      "extra_rules": list(workflow_cfg.get("playbook_rules", {}).keys()),
    }

  diagnostic, payload = evaluate_capture(
    packets,
    playbook,
    args.iface,
    ingest_mode=ingest_mode,
    workflow=args.workflow,
    workflow_meta=workflow_meta,
    recipe_config=None,
  )

  if args.json and not args.output:
    print(json.dumps(payload, indent=2))
  else:
    emit_diagnostic(TOOL_NAME, diagnostic, json_output=False)
  if args.output:
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
  if args.report:
    report_text = generate_report(payload, args.report_format)
    Path(args.report).write_text(report_text, encoding="utf-8")
  return 0


def health_check() -> Dict[str, object]:
  samples = synthesise_latency_samples(TOOL_NAME)
  metrics = summarise_samples(samples)
  details = {
    "supported_formats": ["pcap", "json"],
    "supported_ingest": ["file", "stream"],
    "latency_profile": metrics,
    "sample_latency": [{"probe": label, "latency_ms": value} for label, value in samples],
  }
  return build_health_report(
    TOOL_NAME,
    "ok",
    "SpectraTrace parser initialised.",
    details,
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  if getattr(args, "gui", False):
    return launch_gui("tools.spectratrace_gui")
  return run(args)


if __name__ == "__main__":
  raise SystemExit(main())
