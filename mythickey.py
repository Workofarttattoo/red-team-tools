"""
MythicKey — credential resilience analyser.

The CLI inspects password hash inventories, attempts lightweight wordlist
replay rehearsal, and surfaces policy guidance without mutating real systems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ._toolkit import (
  Diagnostic,
  build_health_report,
  emit_diagnostic,
  launch_gui,
  summarise_samples,
  synthesise_latency_samples,
)


TOOL_NAME = "MythicKey"

SAMPLE_HASHES: Sequence[str] = (
  hashlib.md5(b"changeme").hexdigest(),
  hashlib.sha1(b"P@ssw0rd").hexdigest(),
  hashlib.sha256(b"winter2024").hexdigest(),
)

SAMPLE_WORDS: Sequence[str] = ("changeme", "password", "winter", "winter2024", "welcome1", "P@ssw0rd")

ALGORITHM_BY_LENGTH = {
  32: "md5",
  40: "sha1",
  56: "sha224",
  64: "sha256",
  96: "sha384",
  128: "sha512",
}


@dataclass
class HashAssessment:
  digest: str
  algorithm: str
  cracked: bool
  plaintext: Optional[str]
  attempts: int

  def as_dict(self) -> Dict[str, object]:
    payload = asdict(self)
    payload["digest_prefix"] = self.digest[:12]
    return payload


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="MythicKey credential resilience analyser.")
  parser.add_argument("--hashes", help="Path to newline-delimited password hashes.")
  parser.add_argument("--wordlist", help="Wordlist for rehearsal (one per line).")
  parser.add_argument("--profile", default="cpu", help="Processing profile hint (e.g. cpu, gpu-balanced).")
  parser.add_argument("--demo", action="store_true", help="Use built-in sample hashes and dictionary.")
  parser.add_argument("--json", action="store_true", help="Emit JSON findings.")
  parser.add_argument("--output", help="Write detailed JSON to path.")
  parser.add_argument("--gui", action="store_true", help="Launch the MythicKey graphical interface.")
  return parser


def load_hashes(path: Optional[str], demo: bool) -> List[str]:
  hashes: List[str] = []
  if path:
    try:
      hashes.extend(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    except FileNotFoundError:
      pass
  if demo or not hashes:
    hashes.extend(SAMPLE_HASHES)
  return hashes


def load_wordlist(path: Optional[str], demo: bool) -> List[str]:
  words: List[str] = []
  if path:
    try:
      words.extend(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    except FileNotFoundError:
      pass
  if demo or not words:
    words.extend(SAMPLE_WORDS)
  return list(dict.fromkeys(words))


def guess_algorithm(digest: str) -> str:
  algorithm = ALGORITHM_BY_LENGTH.get(len(digest), "unknown")
  if digest.startswith("{") and "}" in digest:
    return digest[1:digest.index("}")].lower()
  return algorithm


def mutate_word(word: str, profile: str) -> Iterable[str]:
  yield word
  if not word:
    return
  yield word.capitalize()
  yield word + "123"
  yield word + "!"
  if profile.startswith("gpu"):
    yield word[::-1]
    yield word + "@"
    yield word.replace("a", "@").replace("o", "0").replace("s", "$")


def hash_word(word: str, algorithm: str) -> Optional[str]:
  try:
    h = hashlib.new(algorithm)
  except ValueError:
    return None
  h.update(word.encode("utf-8"))
  return h.hexdigest()


def crack_hash(digest: str, algorithm: str, wordlist: Iterable[str], profile: str) -> HashAssessment:
  attempts = 0
  for word in wordlist:
    for candidate in mutate_word(word, profile):
      attempts += 1
      hashed = hash_word(candidate, algorithm)
      if hashed and hashed.lower() == digest.lower():
        return HashAssessment(digest=digest, algorithm=algorithm, cracked=True, plaintext=candidate, attempts=attempts)
  return HashAssessment(digest=digest, algorithm=algorithm, cracked=False, plaintext=None, attempts=attempts)


def evaluate_hashes(
  digests: Sequence[str],
  wordlist: Sequence[str],
  profile: str,
) -> Tuple[Diagnostic, Dict[str, object]]:
  assessments: List[HashAssessment] = []
  for digest in digests:
    algorithm = guess_algorithm(digest)
    assessments.append(crack_hash(digest, algorithm, wordlist, profile))

  cracked = [item for item in assessments if item.cracked]
  status = "info"
  summary = f"{len(cracked)} of {len(assessments)} hash(es) recovered during rehearsal."
  if cracked:
    status = "warn"
    summary = f"{len(cracked)} hash(es) matched dictionary candidates."

  diagnostic = Diagnostic(status=status, summary=summary, details={
    "hashes": len(assessments),
    "cracked": len(cracked),
    "profile": profile,
  })
  payload = {
    "tool": TOOL_NAME,
    "profile": profile,
    "hashes": [assessment.as_dict() for assessment in assessments],
  }
  return diagnostic, payload


def run(args: argparse.Namespace) -> int:
  hashes = load_hashes(args.hashes, args.demo)
  wordlist = load_wordlist(args.wordlist, args.demo)
  profile = args.profile.lower()
  diagnostic, payload = evaluate_hashes(hashes, wordlist, profile)
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
    "hash_algorithms": sorted(set(ALGORITHM_BY_LENGTH.values())),
    "latency_profile": metrics,
    "sample_latency": [{"probe": label, "latency_ms": value} for label, value in samples],
  }
  return build_health_report(
    TOOL_NAME,
    "ok",
    "MythicKey dictionary rehearsal engine operational.",
    details,
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  args_list = list(argv or sys.argv[1:])
  if args_list and args_list[0].lower() == "crack":
    args_list = args_list[1:]
  parser = build_parser()
  args = parser.parse_args(args_list)
  if getattr(args, "gui", False):
    return launch_gui("tools.mythickey_gui")
  return run(args)


if __name__ == "__main__":
  raise SystemExit(main())
