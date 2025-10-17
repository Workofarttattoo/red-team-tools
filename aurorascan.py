"""
AuroraScan — adaptive network mapper with service fingerprint diffing hooks.

Inspired by nmap, AuroraScan focuses on quick enumeration of accessible TCP
endpoints using lightweight heuristics suited for agent-driven workflows.  The
implementation offered here is intentionally conservative: it performs
non-invasive TCP connect probes with configurable concurrency, honours short
timeouts, and emits structured JSON that downstream agents can store in their
telemetry streams.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import queue
import socket
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ._toolkit import launch_gui, summarise_samples, synthesise_latency_samples


TOOL_NAME = "AuroraScan"


DEFAULT_TIMEOUT = 1.5
DEFAULT_CONCURRENCY = 64
DEFAULT_ZAP_SCHEME = "auto"

PORT_PROFILES: Dict[str, Sequence[int]] = {
  "recon": [
    22, 53, 80, 110, 123, 135, 139, 143, 161, 179, 389, 443,
    445, 465, 502, 512, 513, 514, 554, 587, 631, 636, 8080, 8443,
  ],
  "core": [
    21, 22, 23, 25, 53, 80, 111, 135, 139, 143, 161, 389, 443,
    445, 548, 587, 5900, 8080,
  ],
  "full": list(range(1, 1025)),
}

PROFILE_DESCRIPTIONS: Dict[str, str] = {
  "recon": "High-signal ports for rapid situational awareness.",
  "core": "Essential services commonly exposed by workstations and servers.",
  "full": "Complete TCP sweep across ports 1-1024.",
}


def iter_profiles() -> Iterable[Tuple[str, Sequence[int], str]]:
  for key, ports in PORT_PROFILES.items():
    yield key, ports, PROFILE_DESCRIPTIONS.get(key, "")


@dataclass
class PortObservation:
  port: int
  status: str
  response_time_ms: float
  banner: Optional[str] = None


@dataclass
class TargetReport:
  target: str
  resolved: str
  elapsed_ms: float
  observations: List[PortObservation]

  def as_dict(self) -> Dict[str, object]:
    return {
      "target": self.target,
      "resolved": self.resolved,
      "elapsed_ms": self.elapsed_ms,
      "observations": [asdict(obs) for obs in self.observations],
    }


async def probe_port(host: str, port: int, timeout: float) -> Tuple[int, str, float, Optional[str]]:
  start = time.perf_counter()
  try:
    conn = asyncio.open_connection(host, port)
    reader, writer = await asyncio.wait_for(conn, timeout=timeout)
    elapsed = (time.perf_counter() - start) * 1000
    banner = None
    try:
      writer.write(b"\n")
      await asyncio.wait_for(writer.drain(), timeout=timeout)
      peek = await asyncio.wait_for(reader.read(128), timeout=timeout)
      banner = peek.decode(errors="ignore").strip() if peek else None
    except (asyncio.TimeoutError, ConnectionError):
      banner = None
    finally:
      writer.close()
      with contextlib.suppress(Exception):
        await writer.wait_closed()
    return port, "open", elapsed, banner
  except asyncio.TimeoutError:
    elapsed = (time.perf_counter() - start) * 1000
    return port, "filtered", elapsed, None
  except ConnectionRefusedError:
    elapsed = (time.perf_counter() - start) * 1000
    return port, "closed", elapsed, None
  except OSError:
    elapsed = (time.perf_counter() - start) * 1000
    return port, "error", elapsed, None


async def scan_target(
  host: str,
  ports: Sequence[int],
  timeout: float,
  concurrency: int,
  *,
  progress_queue: Optional[queue.Queue] = None,
  stop_flag: Optional[threading.Event] = None,
) -> TargetReport:
  try:
    resolved = socket.gethostbyname(host)
  except socket.gaierror:
    resolved = "unresolved"

  connect_host = resolved if resolved != "unresolved" else host

  semaphore = asyncio.Semaphore(concurrency)
  observations: List[PortObservation] = []

  async def worker(port: int) -> None:
    if stop_flag and stop_flag.is_set():
      return
    async with semaphore:
      if stop_flag and stop_flag.is_set():
        return
      result_port, status, elapsed, banner = await probe_port(connect_host, port, timeout)
      observations.append(PortObservation(result_port, status, elapsed, banner))
      if progress_queue:
        progress_queue.put(
          (
            host,
            result_port,
            status,
            elapsed,
            banner,
          )
        )

  start = time.perf_counter()
  tasks = [asyncio.create_task(worker(port)) for port in ports]
  try:
    await asyncio.gather(*tasks)
  except asyncio.CancelledError:
    for task in tasks:
      task.cancel()
    raise
  observations.sort(key=lambda item: item.port)
  elapsed_ms = (time.perf_counter() - start) * 1000
  return TargetReport(host, resolved, elapsed_ms, observations)


def parse_ports(port_arg: Optional[str], profile: str) -> Sequence[int]:
  if port_arg:
    ports: List[int] = []
    for chunk in port_arg.split(","):
      chunk = chunk.strip()
      if not chunk:
        continue
      if "-" in chunk:
        start_str, end_str = chunk.split("-", maxsplit=1)
        start_port = int(start_str)
        end_port = int(end_str)
        ports.extend(range(start_port, end_port + 1))
      else:
        ports.append(int(chunk))
    return sorted(set(p for p in ports if 1 <= p <= 65535))
  return PORT_PROFILES.get(profile, PORT_PROFILES["recon"])


def parse_targets(target_arg: str) -> List[str]:
  targets: List[str] = []
  for chunk in target_arg.split(","):
    target = chunk.strip()
    if target:
      targets.append(target)
  return targets


def load_targets_from_file(path: Optional[str]) -> List[str]:
  if not path:
    return []
  try:
    with open(path, "r", encoding="utf-8") as handle:
      return [line.strip() for line in handle if line.strip() and not line.startswith("#")]
  except FileNotFoundError:
    print("[warn] Target file not found; ignoring.")
    return []


def display_profiles() -> None:
  print("[info] Available scan profiles:")
  for name, ports, description in iter_profiles():
    print(f"  - {name:<10} ({len(ports)} ports)  {description}")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="AuroraScan network mapper.")
  parser.add_argument("targets", nargs="?", help="Comma-separated hostnames or IP addresses.")
  parser.add_argument("--targets-file", help="Path to file containing one target per line.")
  parser.add_argument("--list-profiles", action="store_true", help="Show built-in scanning profiles and exit.")
  parser.add_argument("--ports", help="Comma/range list of ports to scan (e.g., 22,80,4000-4010).")
  parser.add_argument("--profile", default="recon", choices=list(PORT_PROFILES.keys()), help="Port profile preset.")
  parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-connection timeout in seconds.")
  parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent connection attempts.")
  parser.add_argument("--json", action="store_true", help="Emit results as JSON instead of human-readable text.")
  parser.add_argument("--output", help="Optional path to write JSON results.")
  parser.add_argument("--tag", default="aurorascan", help="Label included in JSON output.")
  parser.add_argument("--zap-targets", help="Write discovered open services to a file for OWASP ZAP import.")
  parser.add_argument("--zap-scheme", choices=["http", "https", "auto"], default=DEFAULT_ZAP_SCHEME, help="Scheme used when generating ZAP URLs (auto guesses from port).")
  parser.add_argument("--gui", action="store_true", help="Launch the AuroraScan graphical interface.")
  return parser


def print_human(results: Iterable[TargetReport]) -> None:
  for report in results:
    open_observations = [obs for obs in report.observations if obs.status == 'open']
    other_counts: Dict[str, int] = {}
    for obs in report.observations:
      if obs.status != 'open':
        other_counts[obs.status] = other_counts.get(obs.status, 0) + 1
    print(f"[info] Target: {report.target} ({report.resolved}) - {report.elapsed_ms:.2f} ms total")
    if open_observations:
      print('    PORT  STATUS   LAT(ms)  BANNER')
      for obs in open_observations:
        banner = (obs.banner or '')[:48]
        print(f"    {obs.port:>4}/tcp  open    {obs.response_time_ms:>7.2f}  {banner}")
    else:
      print('    No open ports detected under the selected profile.')
    if other_counts:
      summary = ', '.join(f"{status}:{count}" for status, count in sorted(other_counts.items()))
      print(f"    Other responses - {summary}")
    print('')


def write_json(results: Iterable[TargetReport], path: Optional[str], tag: str) -> None:
  payload = {
    "tool": tag or TOOL_NAME.lower(),
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "results": [report.as_dict() for report in results],
  }
  if path:
    with open(path, "w", encoding="utf-8") as handle:
      json.dump(payload, handle, indent=2)
    print(f"[info] Results written to {path}")
  else:
    print(json.dumps(payload, indent=2))


def write_zap_targets(results: Iterable[TargetReport], path: Optional[str], default_scheme: str) -> None:
  if not path:
    return
  entries: List[str] = []
  seen = set()
  for report in results:
    host = report.target
    for obs in report.observations:
      if obs.status != "open":
        continue
      scheme = default_scheme
      if scheme == "auto":
        scheme = "https" if obs.port in {443, 8443, 9443, 9444} else "http"
      if (host, obs.port, scheme) in seen:
        continue
      seen.add((host, obs.port, scheme))
      if (scheme == "http" and obs.port == 80) or (scheme == "https" and obs.port == 443):
        url = f"{scheme}://{host}"
      else:
        url = f"{scheme}://{host}:{obs.port}"
      entries.append(url)
  with open(path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(entries) + ("\n" if entries else ""))
  print(f"[info] ZAP target list written to {path} ({len(entries)} endpoint(s)).")


def run_scan(
  targets: Sequence[str],
  ports: Sequence[int],
  *,
  timeout: float,
  concurrency: int,
  progress_queue: Optional[queue.Queue] = None,
  stop_flag: Optional[threading.Event] = None,
) -> List[TargetReport]:
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  tasks = [
    scan_target(
      target,
      ports,
      timeout,
      concurrency,
      progress_queue=progress_queue,
      stop_flag=stop_flag,
    )
    for target in targets
  ]
  try:
    reports = loop.run_until_complete(asyncio.gather(*tasks))
  finally:
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()
  return reports


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.list_profiles:
    display_profiles()
    return 0

  if getattr(args, "gui", False):
    return launch_gui("tools.aurorascan_gui")

  targets: List[str] = []
  if args.targets:
    targets.extend(parse_targets(args.targets))
  targets.extend(load_targets_from_file(args.targets_file))
  if not targets:
    parser.error("No targets specified. Provide targets argument or --targets-file.")

  ports = parse_ports(args.ports, args.profile)
  if not ports:
    parser.error("No ports selected after parsing profile and overrides.")

  print(f"[info] Starting AuroraScan against {len(targets)} target(s) on {len(ports)} port(s).")
  reports = run_scan(
    targets,
    ports,
    timeout=args.timeout,
    concurrency=args.concurrency,
  )

  if args.json or args.output:
    write_json(reports, args.output, args.tag)
  else:
    print_human(reports)

  if args.zap_targets:
    write_zap_targets(reports, args.zap_targets, args.zap_scheme)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())


def health_check() -> Dict[str, object]:
  """
  Provide a lightweight readiness report used by runtime health checks.
  """

  samples = synthesise_latency_samples(TOOL_NAME)
  sample_payload = [{"probe": label, "latency_ms": value} for label, value in samples]
  metrics = summarise_samples(samples)
  return {
    "tool": TOOL_NAME,
    "status": "ok",
    "summary": "AuroraScan ready to schedule network telemetry probes.",
    "samples": sample_payload,
    "metrics": metrics,
  }
