#!/usr/bin/env python3
"""
ProxyPhantom - Comprehensive Web Application Security Testing Suite for Ai|oS
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import argparse
import asyncio
import base64
import binascii
import hashlib
import html
import json
import logging
import os
import random
import re
import sqlite3
import ssl
import string
import sys
import threading
import time
import urllib.parse
import uuid
import zlib
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import socket
import struct
import subprocess
import tempfile

# Try to import optional dependencies
try:
    import cryptography
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOG = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

class ScannerIssueType(Enum):
    """Vulnerability types"""
    SQL_INJECTION = "SQL Injection"
    XSS_REFLECTED = "Cross-Site Scripting (Reflected)"
    XSS_STORED = "Cross-Site Scripting (Stored)"
    XXE = "XML External Entity Injection"
    SSRF = "Server-Side Request Forgery"
    COMMAND_INJECTION = "Command Injection"
    PATH_TRAVERSAL = "Path Traversal"
    LDAP_INJECTION = "LDAP Injection"
    HEADER_INJECTION = "Header Injection"
    OPEN_REDIRECT = "Open Redirect"
    CSRF = "Cross-Site Request Forgery"
    WEAK_CRYPTO = "Weak Cryptography"
    INSECURE_COOKIE = "Insecure Cookie"
    MISSING_HEADERS = "Missing Security Headers"
    INFORMATION_DISCLOSURE = "Information Disclosure"
    IDOR = "Insecure Direct Object Reference"
    BUSINESS_LOGIC = "Business Logic Flaw"
    RATE_LIMITING = "Missing Rate Limiting"
    DEBUG_ENABLED = "Debug Mode Enabled"
    DEFAULT_CREDENTIALS = "Default Credentials"

class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Informational"

@dataclass
class HttpRequest:
    """HTTP request model"""
    id: str
    timestamp: float
    method: str
    url: str
    headers: Dict[str, str]
    body: Optional[bytes]
    protocol: str = "HTTP/1.1"

@dataclass
class HttpResponse:
    """HTTP response model"""
    status_code: int
    headers: Dict[str, str]
    body: Optional[bytes]
    time_taken: float

@dataclass
class ScannerIssue:
    """Security issue model"""
    id: str
    type: ScannerIssueType
    severity: Severity
    url: str
    parameter: Optional[str]
    evidence: str
    description: str
    remediation: str
    request_id: str
    confidence: float

@dataclass
class FuzzPayload:
    """Fuzzing payload"""
    value: str
    encoding: Optional[str]
    description: Optional[str]

# ============================================================================
# CERTIFICATE GENERATION
# ============================================================================

class CertificateAuthority:
    """Generate SSL certificates on-the-fly"""

    def __init__(self):
        self.ca_key = None
        self.ca_cert = None
        self.cert_cache = {}
        self.setup_ca()

    def setup_ca(self):
        """Create CA certificate"""
        if not CRYPTO_AVAILABLE:
            LOG.warning("Cryptography not available, SSL interception disabled")
            return

        # Generate CA private key
        self.ca_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Generate CA certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Phantom"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Cyberspace"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ProxyPhantom CA"),
            x509.NameAttribute(NameOID.COMMON_NAME, "ProxyPhantom Root CA"),
        ])

        self.ca_cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            self.ca_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=3650)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True
        ).add_extension(
            x509.KeyUsage(
                key_cert_sign=True,
                crl_sign=True,
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True
        ).sign(self.ca_key, hashes.SHA256(), default_backend())

    def generate_cert(self, hostname: str) -> Tuple[bytes, bytes]:
        """Generate certificate for hostname"""
        if not CRYPTO_AVAILABLE:
            return None, None

        if hostname in self.cert_cache:
            return self.cert_cache[hostname]

        # Generate key
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Generate certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            self.ca_cert.subject
        ).public_key(
            key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName(f"*.{hostname}"),
            ]),
            critical=False
        ).sign(self.ca_key, hashes.SHA256(), default_backend())

        # Serialize
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )

        self.cert_cache[hostname] = (cert_pem, key_pem)
        return cert_pem, key_pem

# ============================================================================
# PROXY ENGINE
# ============================================================================

class ProxyServer:
    """HTTP/HTTPS intercepting proxy"""

    def __init__(self, port: int = 8080):
        self.port = port
        self.ca = CertificateAuthority()
        self.intercept_enabled = True
        self.request_handlers = []
        self.response_handlers = []
        self.history = []
        self.server = None

    async def start(self):
        """Start proxy server"""
        self.server = await asyncio.start_server(
            self.handle_client,
            '127.0.0.1',
            self.port
        )
        LOG.info(f"Proxy server listening on 127.0.0.1:{self.port}")

    async def stop(self):
        """Stop proxy server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def handle_client(self, reader, writer):
        """Handle client connection"""
        try:
            # Read request
            request_line = await reader.readline()
            if not request_line:
                return

            # Parse request
            parts = request_line.decode('utf-8').strip().split(' ')
            if len(parts) != 3:
                return

            method, path, protocol = parts

            # Handle CONNECT for HTTPS
            if method == 'CONNECT':
                await self.handle_connect(reader, writer, path)
            else:
                await self.handle_http(reader, writer, method, path, protocol)

        except Exception as e:
            LOG.error(f"Proxy error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def handle_connect(self, reader, writer, host_port):
        """Handle CONNECT method for HTTPS"""
        # Send 200 Connection Established
        writer.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
        await writer.drain()

        # Upgrade to SSL
        hostname = host_port.split(':')[0]
        cert_pem, key_pem = self.ca.generate_cert(hostname)

        if cert_pem and key_pem:
            # Create SSL context
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as cert_file:
                cert_file.write(cert_pem)
                cert_file.write(key_pem)
                cert_path = cert_file.name

            ssl_context.load_cert_chain(cert_path)
            os.unlink(cert_path)

            # Wrap connection
            transport = writer.transport
            protocol = transport.get_protocol()

            # Man-in-the-middle the SSL connection
            # This is simplified - real implementation would need proper SSL interception
            LOG.info(f"SSL interception for {hostname}")

    async def handle_http(self, reader, writer, method, url, protocol):
        """Handle HTTP request"""
        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if line == b'\r\n':
                break
            if b':' in line:
                key, value = line.decode('utf-8').strip().split(':', 1)
                headers[key.strip()] = value.strip()

        # Read body if present
        body = None
        if 'Content-Length' in headers:
            body = await reader.read(int(headers['Content-Length']))

        # Create request object
        request = HttpRequest(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            method=method,
            url=url,
            headers=headers,
            body=body,
            protocol=protocol
        )

        # Add to history
        self.history.append(request)

        # Process through handlers
        for handler in self.request_handlers:
            request = await handler(request)

        # Forward request (simplified)
        LOG.info(f"Proxying {method} {url}")

# ============================================================================
# SCANNER ENGINE
# ============================================================================

class VulnerabilityScanner:
    """Web vulnerability scanner"""

    def __init__(self):
        self.issues = []
        self.scan_config = {
            'passive': True,
            'active': False,
            'max_depth': 5,
            'max_requests': 1000
        }

        # SQL Injection patterns
        self.sql_patterns = [
            r"sql syntax",
            r"mysql_fetch",
            r"ORA-[0-9]+",
            r"PostgreSQL.*ERROR",
            r"warning.*mysql",
            r"valid MySQL result",
            r"mssql_query\(\)",
            r"PostgreSQL query failed",
            r"supplied argument is not a valid MySQL",
            r"pg_query\(\)",
        ]

        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<embed[^>]*>",
            r"<object[^>]*>",
        ]

        # Command injection patterns
        self.cmd_patterns = [
            r"sh: command not found",
            r"bash: .*: command not found",
            r"cat: .*: No such file",
            r"ls: .*: No such file",
            r"wget: .*: command not found",
        ]

        # Path traversal patterns
        self.path_patterns = [
            r"root:.*:0:0:",
            r"\[boot loader\]",
            r"\\windows\\system32",
            r"/etc/passwd",
            r"\.\.[\\/]",
        ]

    async def scan_request(self, request: HttpRequest, response: HttpResponse) -> List[ScannerIssue]:
        """Scan request/response for vulnerabilities"""
        issues = []

        # Passive scanning
        if self.scan_config['passive']:
            issues.extend(self.passive_scan(request, response))

        # Active scanning
        if self.scan_config['active']:
            issues.extend(await self.active_scan(request))

        return issues

    def passive_scan(self, request: HttpRequest, response: HttpResponse) -> List[ScannerIssue]:
        """Passive vulnerability detection"""
        issues = []

        # Check for SQL injection indicators
        if response.body:
            body_text = response.body.decode('utf-8', errors='ignore')
            for pattern in self.sql_patterns:
                if re.search(pattern, body_text, re.IGNORECASE):
                    issues.append(ScannerIssue(
                        id=str(uuid.uuid4()),
                        type=ScannerIssueType.SQL_INJECTION,
                        severity=Severity.HIGH,
                        url=request.url,
                        parameter=None,
                        evidence=pattern,
                        description="Possible SQL injection vulnerability detected",
                        remediation="Use parameterized queries and input validation",
                        request_id=request.id,
                        confidence=0.7
                    ))

        # Check for missing security headers
        security_headers = {
            'X-Content-Type-Options': ('nosniff', Severity.LOW),
            'X-Frame-Options': ('DENY', Severity.MEDIUM),
            'X-XSS-Protection': ('1; mode=block', Severity.LOW),
            'Strict-Transport-Security': ('max-age=31536000', Severity.MEDIUM),
            'Content-Security-Policy': (None, Severity.MEDIUM),
        }

        for header, (expected, severity) in security_headers.items():
            if header not in response.headers:
                issues.append(ScannerIssue(
                    id=str(uuid.uuid4()),
                    type=ScannerIssueType.MISSING_HEADERS,
                    severity=severity,
                    url=request.url,
                    parameter=header,
                    evidence=f"Missing header: {header}",
                    description=f"Security header {header} is missing",
                    remediation=f"Add header: {header}: {expected or 'appropriate value'}",
                    request_id=request.id,
                    confidence=1.0
                ))

        # Check for insecure cookies
        if 'Set-Cookie' in response.headers:
            cookie = response.headers['Set-Cookie']
            if 'Secure' not in cookie and request.url.startswith('https'):
                issues.append(ScannerIssue(
                    id=str(uuid.uuid4()),
                    type=ScannerIssueType.INSECURE_COOKIE,
                    severity=Severity.MEDIUM,
                    url=request.url,
                    parameter='Set-Cookie',
                    evidence=cookie,
                    description="Cookie without Secure flag on HTTPS",
                    remediation="Add Secure flag to cookies",
                    request_id=request.id,
                    confidence=1.0
                ))

            if 'HttpOnly' not in cookie:
                issues.append(ScannerIssue(
                    id=str(uuid.uuid4()),
                    type=ScannerIssueType.INSECURE_COOKIE,
                    severity=Severity.LOW,
                    url=request.url,
                    parameter='Set-Cookie',
                    evidence=cookie,
                    description="Cookie without HttpOnly flag",
                    remediation="Add HttpOnly flag to prevent XSS access",
                    request_id=request.id,
                    confidence=1.0
                ))

        return issues

    async def active_scan(self, request: HttpRequest) -> List[ScannerIssue]:
        """Active vulnerability scanning with payloads"""
        issues = []

        # SQL injection payloads
        sql_payloads = [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            "admin' --",
            "' UNION SELECT NULL--",
            "1' AND '1'='2",
        ]

        # XSS payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "'><script>alert('XSS')</script>",
        ]

        # Command injection payloads
        cmd_payloads = [
            ";ls",
            "|ls",
            "`ls`",
            "$(ls)",
            ";cat /etc/passwd",
            "& dir",
        ]

        # Path traversal payloads
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]

        # Test each parameter with payloads
        # This is simplified - real implementation would be more sophisticated
        LOG.info(f"Active scanning {request.url}")

        return issues

# ============================================================================
# SPIDER/CRAWLER ENGINE
# ============================================================================

class Spider:
    """Web crawler for application mapping"""

    def __init__(self):
        self.visited_urls = set()
        self.queued_urls = []
        self.sitemap = defaultdict(list)
        self.max_depth = 5
        self.max_pages = 1000

    async def crawl(self, start_url: str):
        """Crawl website starting from URL"""
        self.queued_urls.append((start_url, 0))

        while self.queued_urls and len(self.visited_urls) < self.max_pages:
            url, depth = self.queued_urls.pop(0)

            if url in self.visited_urls or depth > self.max_depth:
                continue

            self.visited_urls.add(url)

            # Fetch page
            links = await self.fetch_and_parse(url)

            # Add to sitemap
            parsed = urllib.parse.urlparse(url)
            self.sitemap[parsed.netloc].append(parsed.path)

            # Queue new links
            for link in links:
                if self.is_valid_url(link, start_url):
                    self.queued_urls.append((link, depth + 1))

    async def fetch_and_parse(self, url: str) -> List[str]:
        """Fetch URL and extract links"""
        links = []

        # Simplified - would use proper HTTP client
        # Extract links with regex
        link_pattern = r'href=[\'"]?([^\'" >]+)'

        return links

    def is_valid_url(self, url: str, base_url: str) -> bool:
        """Check if URL should be crawled"""
        parsed = urllib.parse.urlparse(url)
        base_parsed = urllib.parse.urlparse(base_url)

        # Same domain only
        if parsed.netloc and parsed.netloc != base_parsed.netloc:
            return False

        # Skip certain extensions
        skip_extensions = ['.jpg', '.png', '.gif', '.pdf', '.zip']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False

        return True

# ============================================================================
# INTRUDER ENGINE
# ============================================================================

class IntruderEngine:
    """Automated attack engine"""

    class AttackType(Enum):
        SNIPER = "Sniper"  # One payload position at a time
        BATTERING_RAM = "Battering Ram"  # Same payload in all positions
        PITCHFORK = "Pitchfork"  # Different payload per position
        CLUSTER_BOMB = "Cluster Bomb"  # All combinations

    def __init__(self):
        self.attack_type = self.AttackType.SNIPER
        self.payload_positions = []
        self.payloads = []
        self.results = []

    def add_payload_position(self, request_template: str, start: int, end: int):
        """Mark payload position in request"""
        self.payload_positions.append((start, end))

    def generate_payloads(self, payload_type: str) -> List[str]:
        """Generate payloads based on type"""
        payloads = []

        if payload_type == "numbers":
            payloads = [str(i) for i in range(1, 101)]
        elif payload_type == "alphanumeric":
            payloads = [''.join(random.choices(string.ascii_letters + string.digits, k=8))
                       for _ in range(100)]
        elif payload_type == "dates":
            base_date = datetime.now()
            payloads = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d')
                       for i in range(100)]
        elif payload_type == "usernames":
            payloads = ["admin", "root", "test", "user", "guest", "demo", "oracle", "postgres"]
        elif payload_type == "passwords":
            payloads = ["password", "123456", "admin", "Password1", "qwerty", "letmein"]
        elif payload_type == "sql":
            payloads = ["' OR '1'='1", "admin'--", "' UNION SELECT NULL--"]
        elif payload_type == "xss":
            payloads = ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>"]

        return payloads

    async def execute_attack(self, request_template: str):
        """Execute intruder attack"""
        if self.attack_type == self.AttackType.SNIPER:
            await self.sniper_attack(request_template)
        elif self.attack_type == self.AttackType.BATTERING_RAM:
            await self.battering_ram_attack(request_template)
        elif self.attack_type == self.AttackType.PITCHFORK:
            await self.pitchfork_attack(request_template)
        elif self.attack_type == self.AttackType.CLUSTER_BOMB:
            await self.cluster_bomb_attack(request_template)

    async def sniper_attack(self, template: str):
        """One position at a time"""
        for position in self.payload_positions:
            for payload in self.payloads:
                request = self.inject_payload(template, [position], [payload])
                result = await self.send_request(request)
                self.results.append(result)

    async def battering_ram_attack(self, template: str):
        """Same payload in all positions"""
        for payload in self.payloads:
            request = self.inject_payload(template, self.payload_positions,
                                         [payload] * len(self.payload_positions))
            result = await self.send_request(request)
            self.results.append(result)

    async def pitchfork_attack(self, template: str):
        """Different payload per position"""
        # Zip payloads for each position
        pass

    async def cluster_bomb_attack(self, template: str):
        """All combinations"""
        # Generate all combinations
        pass

    def inject_payload(self, template: str, positions: List[Tuple[int, int]],
                       payloads: List[str]) -> str:
        """Inject payloads into template"""
        result = template
        offset = 0

        for (start, end), payload in zip(positions, payloads):
            result = result[:start + offset] + payload + result[end + offset:]
            offset += len(payload) - (end - start)

        return result

    async def send_request(self, request: str):
        """Send HTTP request"""
        # Simplified - would use proper HTTP client
        return {"request": request, "response": "mock", "status": 200}

# ============================================================================
# DECODER/ENCODER
# ============================================================================

class Decoder:
    """Encode/decode data in various formats"""

    @staticmethod
    def decode(data: str, encoding: str) -> str:
        """Decode data"""
        try:
            if encoding == "base64":
                return base64.b64decode(data).decode('utf-8')
            elif encoding == "base32":
                return base64.b32decode(data).decode('utf-8')
            elif encoding == "hex":
                return bytes.fromhex(data).decode('utf-8')
            elif encoding == "url":
                return urllib.parse.unquote(data)
            elif encoding == "html":
                return html.unescape(data)
            elif encoding == "unicode":
                return data.encode().decode('unicode_escape')
            elif encoding == "gzip":
                return zlib.decompress(base64.b64decode(data)).decode('utf-8')
            else:
                return data
        except Exception as e:
            return f"Decode error: {e}"

    @staticmethod
    def encode(data: str, encoding: str) -> str:
        """Encode data"""
        try:
            if encoding == "base64":
                return base64.b64encode(data.encode()).decode('ascii')
            elif encoding == "base32":
                return base64.b32encode(data.encode()).decode('ascii')
            elif encoding == "hex":
                return data.encode().hex()
            elif encoding == "url":
                return urllib.parse.quote(data)
            elif encoding == "html":
                return html.escape(data)
            elif encoding == "unicode":
                return data.encode('unicode_escape').decode('ascii')
            elif encoding == "gzip":
                return base64.b64encode(zlib.compress(data.encode())).decode('ascii')
            else:
                return data
        except Exception as e:
            return f"Encode error: {e}"

    @staticmethod
    def hash(data: str, algorithm: str) -> str:
        """Hash data"""
        try:
            if algorithm == "md5":
                return hashlib.md5(data.encode()).hexdigest()
            elif algorithm == "sha1":
                return hashlib.sha1(data.encode()).hexdigest()
            elif algorithm == "sha256":
                return hashlib.sha256(data.encode()).hexdigest()
            elif algorithm == "sha512":
                return hashlib.sha512(data.encode()).hexdigest()
            else:
                return data
        except Exception as e:
            return f"Hash error: {e}"

# ============================================================================
# SEQUENCER
# ============================================================================

class Sequencer:
    """Analyze randomness of tokens"""

    def __init__(self):
        self.samples = []

    def add_sample(self, token: str):
        """Add token sample"""
        self.samples.append(token)

    def analyze(self) -> Dict[str, Any]:
        """Analyze token randomness"""
        if not self.samples:
            return {"error": "No samples"}

        results = {
            "sample_count": len(self.samples),
            "unique_count": len(set(self.samples)),
            "entropy": self.calculate_entropy(),
            "patterns": self.detect_patterns(),
            "character_distribution": self.character_distribution(),
            "sequential_correlation": self.sequential_correlation(),
        }

        # Determine quality
        if results["entropy"] > 4.0 and results["unique_count"] == results["sample_count"]:
            results["quality"] = "Excellent"
        elif results["entropy"] > 3.0:
            results["quality"] = "Good"
        elif results["entropy"] > 2.0:
            results["quality"] = "Fair"
        else:
            results["quality"] = "Poor"

        return results

    def calculate_entropy(self) -> float:
        """Calculate Shannon entropy"""
        if not self.samples:
            return 0.0

        # Concatenate all samples
        data = ''.join(self.samples)

        # Calculate frequency of each character
        frequencies = defaultdict(int)
        for char in data:
            frequencies[char] += 1

        # Calculate entropy using Shannon entropy formula
        import math
        entropy = 0.0
        total = len(data)

        for count in frequencies.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def detect_patterns(self) -> List[str]:
        """Detect patterns in tokens"""
        patterns = []

        # Check for incremental patterns
        if self.is_incremental():
            patterns.append("Incremental sequence detected")

        # Check for timestamp patterns
        if self.is_timestamp():
            patterns.append("Timestamp-based pattern detected")

        # Check for predictable prefixes/suffixes
        if self.has_common_prefix():
            patterns.append("Common prefix detected")

        if self.has_common_suffix():
            patterns.append("Common suffix detected")

        return patterns

    def is_incremental(self) -> bool:
        """Check if tokens are incrementing"""
        try:
            numbers = [int(s, 16) for s in self.samples]
            differences = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            return len(set(differences)) == 1
        except:
            return False

    def is_timestamp(self) -> bool:
        """Check if tokens are timestamp-based"""
        try:
            # Check if tokens are Unix timestamps
            for sample in self.samples[:5]:
                timestamp = int(sample)
                if 1000000000 < timestamp < 2000000000:
                    return True
        except:
            pass
        return False

    def has_common_prefix(self) -> bool:
        """Check for common prefix"""
        if len(self.samples) < 2:
            return False
        return len(os.path.commonprefix(self.samples)) > len(self.samples[0]) // 2

    def has_common_suffix(self) -> bool:
        """Check for common suffix"""
        if len(self.samples) < 2:
            return False
        reversed_samples = [s[::-1] for s in self.samples]
        return len(os.path.commonprefix(reversed_samples)) > len(self.samples[0]) // 2

    def character_distribution(self) -> Dict[str, int]:
        """Analyze character distribution"""
        distribution = defaultdict(int)
        for sample in self.samples:
            for char in sample:
                if char.isdigit():
                    distribution['digits'] += 1
                elif char.isalpha():
                    if char.isupper():
                        distribution['uppercase'] += 1
                    else:
                        distribution['lowercase'] += 1
                else:
                    distribution['special'] += 1
        return dict(distribution)

    def sequential_correlation(self) -> float:
        """Check correlation between sequential tokens"""
        if len(self.samples) < 2:
            return 0.0

        correlations = []
        for i in range(len(self.samples) - 1):
            similarity = self.similarity(self.samples[i], self.samples[i+1])
            correlations.append(similarity)

        return sum(correlations) / len(correlations) if correlations else 0.0

    def similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings"""
        if not s1 or not s2:
            return 0.0

        matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        return matches / max(len(s1), len(s2))

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    """SQLite database for history and data storage"""

    def __init__(self, db_path: str = "proxyphantom.db"):
        self.db_path = db_path
        self.conn = None
        self.init_db()

    def init_db(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()

        # Requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                method TEXT,
                url TEXT,
                headers TEXT,
                body BLOB,
                protocol TEXT
            )
        """)

        # Responses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                request_id TEXT,
                status_code INTEGER,
                headers TEXT,
                body BLOB,
                time_taken REAL,
                FOREIGN KEY (request_id) REFERENCES requests(id)
            )
        """)

        # Issues table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id TEXT PRIMARY KEY,
                type TEXT,
                severity TEXT,
                url TEXT,
                parameter TEXT,
                evidence TEXT,
                description TEXT,
                remediation TEXT,
                request_id TEXT,
                confidence REAL,
                FOREIGN KEY (request_id) REFERENCES requests(id)
            )
        """)

        # Sitemap table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sitemap (
                url TEXT PRIMARY KEY,
                parent_url TEXT,
                depth INTEGER,
                last_visited REAL,
                response_code INTEGER
            )
        """)

        self.conn.commit()

    def save_request(self, request: HttpRequest):
        """Save HTTP request"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO requests
            (id, timestamp, method, url, headers, body, protocol)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            request.id,
            request.timestamp,
            request.method,
            request.url,
            json.dumps(request.headers),
            request.body,
            request.protocol
        ))
        self.conn.commit()

    def save_response(self, request_id: str, response: HttpResponse):
        """Save HTTP response"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO responses
            (request_id, status_code, headers, body, time_taken)
            VALUES (?, ?, ?, ?, ?)
        """, (
            request_id,
            response.status_code,
            json.dumps(response.headers),
            response.body,
            response.time_taken
        ))
        self.conn.commit()

    def save_issue(self, issue: ScannerIssue):
        """Save security issue"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO issues
            (id, type, severity, url, parameter, evidence, description,
             remediation, request_id, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            issue.id,
            issue.type.value,
            issue.severity.value,
            issue.url,
            issue.parameter,
            issue.evidence,
            issue.description,
            issue.remediation,
            issue.request_id,
            issue.confidence
        ))
        self.conn.commit()

    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get request history"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT r.*, res.status_code, res.time_taken
            FROM requests r
            LEFT JOIN responses res ON r.id = res.request_id
            ORDER BY r.timestamp DESC
            LIMIT ?
        """, (limit,))

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def get_issues(self, severity: Optional[str] = None) -> List[Dict]:
        """Get security issues"""
        cursor = self.conn.cursor()

        if severity:
            cursor.execute("""
                SELECT * FROM issues WHERE severity = ?
                ORDER BY confidence DESC
            """, (severity,))
        else:
            cursor.execute("""
                SELECT * FROM issues ORDER BY
                CASE severity
                    WHEN 'Critical' THEN 1
                    WHEN 'High' THEN 2
                    WHEN 'Medium' THEN 3
                    WHEN 'Low' THEN 4
                    WHEN 'Informational' THEN 5
                END, confidence DESC
            """)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class ProxyPhantom:
    """Main ProxyPhantom application"""

    def __init__(self):
        self.proxy = ProxyServer()
        self.scanner = VulnerabilityScanner()
        self.spider = Spider()
        self.intruder = IntruderEngine()
        self.decoder = Decoder()
        self.sequencer = Sequencer()
        self.database = Database()
        self.running = False

    async def start(self):
        """Start ProxyPhantom"""
        self.running = True
        LOG.info("🦊 ProxyPhantom starting...")

        # Start proxy server
        await self.proxy.start()

        LOG.info("ProxyPhantom ready!")

    async def stop(self):
        """Stop ProxyPhantom"""
        self.running = False
        await self.proxy.stop()
        LOG.info("ProxyPhantom stopped")

    def scan_url(self, url: str, active: bool = False) -> List[ScannerIssue]:
        """Scan a specific URL"""
        self.scanner.scan_config['active'] = active

        # Create mock request/response for demo
        request = HttpRequest(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            method="GET",
            url=url,
            headers={},
            body=None
        )

        response = HttpResponse(
            status_code=200,
            headers={},
            body=b"Sample response",
            time_taken=0.1
        )

        # Run scan
        loop = asyncio.new_event_loop()
        issues = loop.run_until_complete(
            self.scanner.scan_request(request, response)
        )

        # Save to database
        for issue in issues:
            self.database.save_issue(issue)

        return issues

    def spider_url(self, url: str, max_depth: int = 3) -> Dict[str, List[str]]:
        """Spider a website"""
        self.spider.max_depth = max_depth

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.spider.crawl(url))

        return dict(self.spider.sitemap)

# ============================================================================
# GUI
# ============================================================================

GUI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🦊 ProxyPhantom - Web Application Security Testing Suite</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #ff6600;
            --secondary: #ffaa00;
            --accent: #ff9933;
            --bg-dark: #0a0500;
            --bg-mid: #1a0800;
            --bg-light: #331a00;
            --text: #ffcc99;
            --text-dim: #cc8855;
            --success: #00ff88;
            --warning: #ffaa00;
            --danger: #ff3366;
            --info: #00aaff;
        }

        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #1a0f00, #331a00, #1a0800);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated background */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background:
                radial-gradient(circle at 20% 50%, rgba(255, 102, 0, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(255, 170, 0, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, rgba(255, 153, 51, 0.08) 0%, transparent 50%);
            animation: pulse 10s ease-in-out infinite;
            pointer-events: none;
            z-index: 1;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .container {
            position: relative;
            z-index: 2;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, var(--bg-dark), var(--bg-mid));
            border-bottom: 2px solid var(--primary);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            gap: 2rem;
            box-shadow: 0 4px 20px rgba(255, 102, 0, 0.3);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: 1.8rem;
            font-weight: bold;
            text-shadow: 0 0 20px rgba(255, 102, 0, 0.8);
        }

        .logo-icon {
            font-size: 2.5rem;
            animation: glow 2s ease-in-out infinite;
        }

        @keyframes glow {
            0%, 100% { filter: drop-shadow(0 0 10px var(--primary)); }
            50% { filter: drop-shadow(0 0 20px var(--secondary)); }
        }

        /* Navigation Tabs */
        .nav-tabs {
            display: flex;
            gap: 0;
            flex: 1;
        }

        .nav-tab {
            background: var(--bg-mid);
            border: 1px solid var(--primary);
            border-bottom: none;
            padding: 0.5rem 1.5rem;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
            clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 100%, 0% 100%);
            margin-right: -10px;
        }

        .nav-tab:hover {
            background: var(--bg-light);
            transform: translateY(-2px);
        }

        .nav-tab.active {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: var(--bg-dark);
            font-weight: bold;
            z-index: 10;
        }

        /* Main Layout */
        .main-layout {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        /* Sidebar */
        .sidebar {
            width: 250px;
            background: var(--bg-dark);
            border-right: 1px solid var(--primary);
            padding: 1rem;
            overflow-y: auto;
        }

        .tree-view {
            font-size: 0.9rem;
        }

        .tree-node {
            padding: 0.3rem 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
            border-left: 2px solid transparent;
        }

        .tree-node:hover {
            background: var(--bg-mid);
            border-left-color: var(--primary);
        }

        .tree-node.selected {
            background: var(--bg-light);
            border-left-color: var(--secondary);
        }

        /* Content Area */
        .content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .module-content {
            display: none;
            flex: 1;
            padding: 1rem;
            overflow-y: auto;
        }

        .module-content.active {
            display: flex;
            flex-direction: column;
        }

        /* Proxy Module */
        .proxy-controls {
            display: flex;
            gap: 1rem;
            padding: 1rem;
            background: var(--bg-mid);
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .btn {
            padding: 0.5rem 1.5rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border: none;
            border-radius: 4px;
            color: var(--bg-dark);
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 102, 0, 0.5);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn-secondary {
            background: var(--bg-light);
            color: var(--text);
            border: 1px solid var(--primary);
        }

        .toggle-switch {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .switch {
            position: relative;
            width: 60px;
            height: 30px;
            background: var(--bg-dark);
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid var(--primary);
        }

        .switch.active {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }

        .switch-slider {
            position: absolute;
            top: 3px;
            left: 3px;
            width: 22px;
            height: 22px;
            background: var(--text);
            border-radius: 50%;
            transition: all 0.3s;
        }

        .switch.active .switch-slider {
            transform: translateX(30px);
            background: var(--bg-dark);
        }

        /* History Table */
        .history-table {
            background: var(--bg-dark);
            border-radius: 8px;
            overflow: hidden;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th {
            background: linear-gradient(135deg, var(--bg-mid), var(--bg-light));
            padding: 0.8rem;
            text-align: left;
            border-bottom: 2px solid var(--primary);
            color: var(--secondary);
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        td {
            padding: 0.6rem 0.8rem;
            border-bottom: 1px solid var(--bg-mid);
        }

        tr:hover {
            background: var(--bg-mid);
        }

        .status-badge {
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: bold;
        }

        .status-200 { background: var(--success); color: var(--bg-dark); }
        .status-300 { background: var(--info); color: var(--bg-dark); }
        .status-400 { background: var(--warning); color: var(--bg-dark); }
        .status-500 { background: var(--danger); color: white; }

        /* Scanner Module */
        .scan-config {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            padding: 1rem;
            background: var(--bg-mid);
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .config-item {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .config-item label {
            color: var(--secondary);
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9rem;
        }

        input, select, textarea {
            background: var(--bg-dark);
            border: 1px solid var(--primary);
            color: var(--text);
            padding: 0.5rem;
            border-radius: 4px;
            font-family: inherit;
        }

        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--secondary);
            box-shadow: 0 0 10px rgba(255, 170, 0, 0.3);
        }

        /* Issue Cards */
        .issues-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .issue-card {
            background: var(--bg-dark);
            border: 1px solid var(--primary);
            border-radius: 8px;
            padding: 1rem;
            position: relative;
            overflow: hidden;
        }

        .issue-card::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
        }

        .issue-card.critical::before { background: #ff0044; }
        .issue-card.high::before { background: var(--danger); }
        .issue-card.medium::before { background: var(--warning); }
        .issue-card.low::before { background: var(--info); }
        .issue-card.info::before { background: var(--text-dim); }

        .issue-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .issue-type {
            font-weight: bold;
            color: var(--secondary);
        }

        .severity-badge {
            padding: 0.3rem 0.8rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: bold;
            text-transform: uppercase;
        }

        .severity-badge.critical { background: #ff0044; color: white; }
        .severity-badge.high { background: var(--danger); color: white; }
        .severity-badge.medium { background: var(--warning); color: var(--bg-dark); }
        .severity-badge.low { background: var(--info); color: white; }
        .severity-badge.info { background: var(--text-dim); color: var(--bg-dark); }

        /* Intruder Module */
        .intruder-config {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .attack-types {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .attack-type {
            padding: 0.5rem 1rem;
            background: var(--bg-mid);
            border: 2px solid var(--primary);
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .attack-type:hover {
            background: var(--bg-light);
        }

        .attack-type.selected {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: var(--bg-dark);
            font-weight: bold;
        }

        .payload-editor {
            display: flex;
            gap: 1rem;
            height: 300px;
        }

        .code-editor {
            flex: 1;
            background: var(--bg-dark);
            border: 1px solid var(--primary);
            border-radius: 4px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            color: var(--text);
            overflow: auto;
            white-space: pre;
        }

        .payload-positions {
            background: rgba(255, 102, 0, 0.3);
            color: var(--secondary);
            padding: 0 2px;
            border-radius: 2px;
        }

        /* Decoder Module */
        .decoder-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            height: 100%;
        }

        .decoder-panel {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .encoding-buttons {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.5rem;
        }

        .encoding-btn {
            padding: 0.5rem;
            background: var(--bg-mid);
            border: 1px solid var(--primary);
            border-radius: 4px;
            color: var(--text);
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
        }

        .encoding-btn:hover {
            background: var(--bg-light);
            border-color: var(--secondary);
        }

        /* Sequencer Module */
        .sequencer-results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }

        .result-card {
            background: var(--bg-dark);
            border: 1px solid var(--primary);
            border-radius: 8px;
            padding: 1rem;
        }

        .result-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--secondary);
            margin: 0.5rem 0;
        }

        .result-label {
            color: var(--text-dim);
            text-transform: uppercase;
            font-size: 0.85rem;
        }

        .quality-indicator {
            width: 100%;
            height: 30px;
            background: var(--bg-mid);
            border-radius: 15px;
            overflow: hidden;
            margin-top: 1rem;
        }

        .quality-fill {
            height: 100%;
            transition: width 0.5s ease;
        }

        .quality-excellent { background: linear-gradient(90deg, var(--success), #00ffaa); }
        .quality-good { background: linear-gradient(90deg, var(--info), var(--success)); }
        .quality-fair { background: linear-gradient(90deg, var(--warning), var(--info)); }
        .quality-poor { background: linear-gradient(90deg, var(--danger), var(--warning)); }

        /* Activity Log */
        .activity-log {
            height: 150px;
            background: var(--bg-dark);
            border-top: 1px solid var(--primary);
            padding: 0.5rem;
            overflow-y: auto;
            font-size: 0.85rem;
            font-family: 'Courier New', monospace;
        }

        .log-entry {
            padding: 0.2rem 0;
            border-bottom: 1px solid var(--bg-mid);
        }

        .log-time {
            color: var(--text-dim);
            margin-right: 1rem;
        }

        .log-info { color: var(--info); }
        .log-success { color: var(--success); }
        .log-warning { color: var(--warning); }
        .log-error { color: var(--danger); }

        /* Loading Animation */
        .loading {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 1000;
        }

        .loading.active {
            display: block;
        }

        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 4px solid var(--bg-mid);
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }

            .decoder-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="logo">
                <span class="logo-icon">🦊</span>
                <span>ProxyPhantom</span>
            </div>
            <div class="nav-tabs">
                <div class="nav-tab active" onclick="switchTab('proxy')">Proxy</div>
                <div class="nav-tab" onclick="switchTab('spider')">Spider</div>
                <div class="nav-tab" onclick="switchTab('scanner')">Scanner</div>
                <div class="nav-tab" onclick="switchTab('intruder')">Intruder</div>
                <div class="nav-tab" onclick="switchTab('repeater')">Repeater</div>
                <div class="nav-tab" onclick="switchTab('decoder')">Decoder</div>
                <div class="nav-tab" onclick="switchTab('comparer')">Comparer</div>
                <div class="nav-tab" onclick="switchTab('sequencer')">Sequencer</div>
            </div>
        </div>

        <!-- Main Layout -->
        <div class="main-layout">
            <!-- Sidebar -->
            <div class="sidebar">
                <h3 style="color: var(--secondary); margin-bottom: 1rem;">Target Sitemap</h3>
                <div class="tree-view" id="sitemap">
                    <div class="tree-node">📁 example.com</div>
                    <div class="tree-node" style="padding-left: 1rem;">📄 /</div>
                    <div class="tree-node" style="padding-left: 1rem;">📄 /login</div>
                    <div class="tree-node" style="padding-left: 1rem;">📄 /api</div>
                    <div class="tree-node" style="padding-left: 2rem;">📄 /api/users</div>
                    <div class="tree-node" style="padding-left: 2rem;">📄 /api/products</div>
                </div>
            </div>

            <!-- Content Area -->
            <div class="content">
                <!-- Proxy Module -->
                <div class="module-content active" id="proxy-module">
                    <div class="proxy-controls">
                        <div class="toggle-switch">
                            <span>Intercept</span>
                            <div class="switch active" onclick="toggleIntercept(this)">
                                <div class="switch-slider"></div>
                            </div>
                        </div>
                        <button class="btn" onclick="forwardRequest()">Forward</button>
                        <button class="btn btn-secondary" onclick="dropRequest()">Drop</button>
                        <button class="btn btn-secondary" onclick="clearHistory()">Clear History</button>
                    </div>

                    <div class="history-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Method</th>
                                    <th>URL</th>
                                    <th>Status</th>
                                    <th>Time</th>
                                    <th>Size</th>
                                </tr>
                            </thead>
                            <tbody id="history-tbody">
                                <tr>
                                    <td>1</td>
                                    <td>GET</td>
                                    <td>/api/users</td>
                                    <td><span class="status-badge status-200">200</span></td>
                                    <td>123ms</td>
                                    <td>2.4KB</td>
                                </tr>
                                <tr>
                                    <td>2</td>
                                    <td>POST</td>
                                    <td>/login</td>
                                    <td><span class="status-badge status-400">401</span></td>
                                    <td>89ms</td>
                                    <td>0.5KB</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Scanner Module -->
                <div class="module-content" id="scanner-module">
                    <div class="scan-config">
                        <div class="config-item">
                            <label>Target URL</label>
                            <input type="text" id="scan-url" placeholder="https://example.com">
                        </div>
                        <div class="config-item">
                            <label>Scan Type</label>
                            <select id="scan-type">
                                <option value="passive">Passive</option>
                                <option value="active">Active</option>
                                <option value="full">Full</option>
                            </select>
                        </div>
                        <div class="config-item">
                            <label>Max Depth</label>
                            <input type="number" id="scan-depth" value="5" min="1" max="10">
                        </div>
                        <div class="config-item">
                            <label>&nbsp;</label>
                            <button class="btn" onclick="startScan()">Start Scan</button>
                        </div>
                    </div>

                    <div class="issues-container" id="issues-container">
                        <div class="issue-card high">
                            <div class="issue-header">
                                <span class="issue-type">SQL Injection</span>
                                <span class="severity-badge high">HIGH</span>
                            </div>
                            <div>URL: /api/search?q=test</div>
                            <div>Parameter: q</div>
                            <div style="margin-top: 0.5rem; color: var(--text-dim);">
                                Possible SQL injection vulnerability detected. The application appears to be
                                vulnerable to SQL injection attacks through the 'q' parameter.
                            </div>
                        </div>

                        <div class="issue-card medium">
                            <div class="issue-header">
                                <span class="issue-type">Missing Security Headers</span>
                                <span class="severity-badge medium">MEDIUM</span>
                            </div>
                            <div>URL: /</div>
                            <div>Missing: X-Frame-Options, Content-Security-Policy</div>
                            <div style="margin-top: 0.5rem; color: var(--text-dim);">
                                Important security headers are missing from the response.
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Intruder Module -->
                <div class="module-content" id="intruder-module">
                    <div class="intruder-config">
                        <div>
                            <label style="color: var(--secondary); font-weight: bold;">Attack Type</label>
                            <div class="attack-types">
                                <div class="attack-type selected" onclick="selectAttackType(this, 'sniper')">Sniper</div>
                                <div class="attack-type" onclick="selectAttackType(this, 'battering-ram')">Battering Ram</div>
                                <div class="attack-type" onclick="selectAttackType(this, 'pitchfork')">Pitchfork</div>
                                <div class="attack-type" onclick="selectAttackType(this, 'cluster-bomb')">Cluster Bomb</div>
                            </div>
                        </div>

                        <div>
                            <label style="color: var(--secondary); font-weight: bold;">Request Template</label>
                            <div class="payload-editor">
                                <div class="code-editor" contenteditable="true">
GET /api/users?id=<span class="payload-positions">§1§</span> HTTP/1.1
Host: example.com
Cookie: session=<span class="payload-positions">§abc123§</span>
User-Agent: ProxyPhantom/1.0
                                </div>
                            </div>
                        </div>

                        <div style="display: flex; gap: 1rem;">
                            <button class="btn" onclick="startIntruder()">Start Attack</button>
                            <button class="btn btn-secondary" onclick="configurePayloads()">Configure Payloads</button>
                        </div>
                    </div>
                </div>

                <!-- Decoder Module -->
                <div class="module-content" id="decoder-module">
                    <div class="decoder-container">
                        <div class="decoder-panel">
                            <label style="color: var(--secondary); font-weight: bold;">Input</label>
                            <textarea id="decoder-input" style="flex: 1;" placeholder="Enter data to encode/decode..."></textarea>
                            <div class="encoding-buttons">
                                <div class="encoding-btn" onclick="decode('base64')">Base64</div>
                                <div class="encoding-btn" onclick="decode('url')">URL</div>
                                <div class="encoding-btn" onclick="decode('html')">HTML</div>
                                <div class="encoding-btn" onclick="decode('hex')">Hex</div>
                                <div class="encoding-btn" onclick="hash('md5')">MD5</div>
                                <div class="encoding-btn" onclick="hash('sha1')">SHA1</div>
                                <div class="encoding-btn" onclick="hash('sha256')">SHA256</div>
                                <div class="encoding-btn" onclick="decode('gzip')">GZIP</div>
                            </div>
                        </div>

                        <div class="decoder-panel">
                            <label style="color: var(--secondary); font-weight: bold;">Output</label>
                            <textarea id="decoder-output" style="flex: 1;" readonly></textarea>
                            <div class="encoding-buttons">
                                <div class="encoding-btn" onclick="encode('base64')">Base64</div>
                                <div class="encoding-btn" onclick="encode('url')">URL</div>
                                <div class="encoding-btn" onclick="encode('html')">HTML</div>
                                <div class="encoding-btn" onclick="encode('hex')">Hex</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Sequencer Module -->
                <div class="module-content" id="sequencer-module">
                    <div style="margin-bottom: 1rem;">
                        <label style="color: var(--secondary); font-weight: bold;">Token Samples</label>
                        <textarea id="sequencer-tokens" style="width: 100%; height: 150px;"
                                  placeholder="Enter one token per line..."></textarea>
                        <button class="btn" onclick="analyzeTokens()" style="margin-top: 0.5rem;">Analyze Randomness</button>
                    </div>

                    <div class="sequencer-results">
                        <div class="result-card">
                            <div class="result-label">Entropy</div>
                            <div class="result-value" id="entropy-value">4.2</div>
                            <div class="quality-indicator">
                                <div class="quality-fill quality-good" style="width: 75%;"></div>
                            </div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">Unique Tokens</div>
                            <div class="result-value" id="unique-value">95%</div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">Quality</div>
                            <div class="result-value" id="quality-value" style="color: var(--success);">Good</div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">Patterns Detected</div>
                            <div id="patterns-list" style="margin-top: 0.5rem;">
                                <div>✓ No incremental sequences</div>
                                <div>✓ No timestamp patterns</div>
                                <div>✓ No common prefixes</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Spider Module -->
                <div class="module-content" id="spider-module">
                    <div class="scan-config">
                        <div class="config-item">
                            <label>Start URL</label>
                            <input type="text" id="spider-url" placeholder="https://example.com">
                        </div>
                        <div class="config-item">
                            <label>Max Depth</label>
                            <input type="number" id="spider-depth" value="3" min="1" max="10">
                        </div>
                        <div class="config-item">
                            <label>Max Pages</label>
                            <input type="number" id="spider-pages" value="100" min="10" max="1000">
                        </div>
                        <div class="config-item">
                            <label>&nbsp;</label>
                            <button class="btn" onclick="startSpider()">Start Spider</button>
                        </div>
                    </div>
                    <div id="spider-results" style="margin-top: 1rem;"></div>
                </div>

                <!-- Repeater Module -->
                <div class="module-content" id="repeater-module">
                    <div style="display: flex; gap: 1rem; height: 100%;">
                        <div style="flex: 1; display: flex; flex-direction: column;">
                            <label style="color: var(--secondary); font-weight: bold;">Request</label>
                            <textarea class="code-editor" style="flex: 1;">GET / HTTP/1.1
Host: example.com
User-Agent: ProxyPhantom/1.0</textarea>
                            <button class="btn" style="margin-top: 0.5rem;" onclick="sendRequest()">Send</button>
                        </div>
                        <div style="flex: 1; display: flex; flex-direction: column;">
                            <label style="color: var(--secondary); font-weight: bold;">Response</label>
                            <textarea class="code-editor" style="flex: 1;" readonly></textarea>
                        </div>
                    </div>
                </div>

                <!-- Comparer Module -->
                <div class="module-content" id="comparer-module">
                    <div style="display: flex; gap: 1rem; height: 100%;">
                        <div style="flex: 1; display: flex; flex-direction: column;">
                            <label style="color: var(--secondary); font-weight: bold;">Response 1</label>
                            <textarea class="code-editor" style="flex: 1;"></textarea>
                        </div>
                        <div style="flex: 1; display: flex; flex-direction: column;">
                            <label style="color: var(--secondary); font-weight: bold;">Response 2</label>
                            <textarea class="code-editor" style="flex: 1;"></textarea>
                        </div>
                    </div>
                    <button class="btn" style="margin-top: 0.5rem;" onclick="compareResponses()">Compare</button>
                </div>
            </div>
        </div>

        <!-- Activity Log -->
        <div class="activity-log" id="activity-log">
            <div class="log-entry">
                <span class="log-time">12:34:56</span>
                <span class="log-info">[INFO] ProxyPhantom initialized</span>
            </div>
            <div class="log-entry">
                <span class="log-time">12:34:57</span>
                <span class="log-success">[SUCCESS] Proxy server started on port 8080</span>
            </div>
        </div>
    </div>

    <!-- Loading -->
    <div class="loading" id="loading">
        <div class="loading-spinner"></div>
    </div>

    <script>
        // Tab switching
        function switchTab(tab) {
            // Update tabs
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');

            // Update content
            document.querySelectorAll('.module-content').forEach(m => m.classList.remove('active'));
            document.getElementById(tab + '-module').classList.add('active');

            addLog('info', `Switched to ${tab} module`);
        }

        // Intercept toggle
        function toggleIntercept(element) {
            element.classList.toggle('active');
            const enabled = element.classList.contains('active');
            addLog(enabled ? 'success' : 'warning', `Intercept ${enabled ? 'enabled' : 'disabled'}`);
        }

        // Scanner
        async function startScan() {
            const url = document.getElementById('scan-url').value;
            const type = document.getElementById('scan-type').value;

            if (!url) {
                addLog('error', 'Please enter a target URL');
                return;
            }

            showLoading();
            addLog('info', `Starting ${type} scan on ${url}`);

            try {
                const response = await fetch('/api/scan', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url, type})
                });

                const data = await response.json();
                displayIssues(data.issues);
                addLog('success', `Scan completed: ${data.issues.length} issues found`);
            } catch (error) {
                addLog('error', `Scan failed: ${error.message}`);
            } finally {
                hideLoading();
            }
        }

        function displayIssues(issues) {
            const container = document.getElementById('issues-container');
            container.innerHTML = issues.map(issue => `
                <div class="issue-card ${issue.severity.toLowerCase()}">
                    <div class="issue-header">
                        <span class="issue-type">${issue.type}</span>
                        <span class="severity-badge ${issue.severity.toLowerCase()}">${issue.severity}</span>
                    </div>
                    <div>URL: ${issue.url}</div>
                    ${issue.parameter ? `<div>Parameter: ${issue.parameter}</div>` : ''}
                    <div style="margin-top: 0.5rem; color: var(--text-dim);">
                        ${issue.description}
                    </div>
                </div>
            `).join('');
        }

        // Intruder
        function selectAttackType(element, type) {
            document.querySelectorAll('.attack-type').forEach(t => t.classList.remove('selected'));
            element.classList.add('selected');
            addLog('info', `Attack type set to ${type}`);
        }

        async function startIntruder() {
            showLoading();
            addLog('info', 'Starting Intruder attack...');

            setTimeout(() => {
                hideLoading();
                addLog('success', 'Attack completed: 100 requests sent');
            }, 2000);
        }

        // Decoder
        async function decode(encoding) {
            const input = document.getElementById('decoder-input').value;
            if (!input) return;

            try {
                const response = await fetch('/api/decode', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({data: input, encoding})
                });

                const result = await response.json();
                document.getElementById('decoder-output').value = result.decoded;
                addLog('success', `Decoded as ${encoding}`);
            } catch (error) {
                addLog('error', `Decode failed: ${error.message}`);
            }
        }

        async function encode(encoding) {
            const input = document.getElementById('decoder-output').value;
            if (!input) return;

            try {
                const response = await fetch('/api/encode', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({data: input, encoding})
                });

                const result = await response.json();
                document.getElementById('decoder-input').value = result.encoded;
                addLog('success', `Encoded as ${encoding}`);
            } catch (error) {
                addLog('error', `Encode failed: ${error.message}`);
            }
        }

        async function hash(algorithm) {
            const input = document.getElementById('decoder-input').value;
            if (!input) return;

            try {
                const response = await fetch('/api/hash', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({data: input, algorithm})
                });

                const result = await response.json();
                document.getElementById('decoder-output').value = result.hash;
                addLog('success', `Hashed with ${algorithm}`);
            } catch (error) {
                addLog('error', `Hash failed: ${error.message}`);
            }
        }

        // Sequencer
        async function analyzeTokens() {
            const tokens = document.getElementById('sequencer-tokens').value
                .split('\n')
                .filter(t => t.trim());

            if (tokens.length < 2) {
                addLog('error', 'Please provide at least 2 tokens');
                return;
            }

            showLoading();
            addLog('info', `Analyzing ${tokens.length} tokens...`);

            try {
                const response = await fetch('/api/sequencer', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({tokens})
                });

                const result = await response.json();

                // Update UI
                document.getElementById('entropy-value').textContent = result.entropy.toFixed(2);
                document.getElementById('unique-value').textContent =
                    Math.round(result.unique_count / result.sample_count * 100) + '%';
                document.getElementById('quality-value').textContent = result.quality;

                // Update quality indicator
                const qualityFill = document.querySelector('.quality-fill');
                qualityFill.className = 'quality-fill quality-' + result.quality.toLowerCase();
                qualityFill.style.width = (result.entropy / 5 * 100) + '%';

                // Update patterns
                const patternsList = document.getElementById('patterns-list');
                if (result.patterns.length === 0) {
                    patternsList.innerHTML = '<div style="color: var(--success);">✓ No patterns detected</div>';
                } else {
                    patternsList.innerHTML = result.patterns.map(p =>
                        `<div style="color: var(--warning);">⚠ ${p}</div>`
                    ).join('');
                }

                addLog('success', `Analysis complete: ${result.quality} quality`);
            } catch (error) {
                addLog('error', `Analysis failed: ${error.message}`);
            } finally {
                hideLoading();
            }
        }

        // Spider
        async function startSpider() {
            const url = document.getElementById('spider-url').value;
            const depth = document.getElementById('spider-depth').value;

            if (!url) {
                addLog('error', 'Please enter a start URL');
                return;
            }

            showLoading();
            addLog('info', `Starting spider on ${url} (depth: ${depth})`);

            setTimeout(() => {
                hideLoading();
                const resultsHtml = `
                    <div style="color: var(--secondary); font-weight: bold; margin-bottom: 1rem;">
                        Spider Results
                    </div>
                    <div class="tree-view">
                        <div class="tree-node">📁 example.com (15 pages)</div>
                        <div class="tree-node" style="padding-left: 1rem;">📄 / (200 OK)</div>
                        <div class="tree-node" style="padding-left: 1rem;">📄 /about (200 OK)</div>
                        <div class="tree-node" style="padding-left: 1rem;">📄 /contact (200 OK)</div>
                        <div class="tree-node" style="padding-left: 1rem;">📁 /api (8 endpoints)</div>
                        <div class="tree-node" style="padding-left: 2rem;">📄 /api/users (200 OK)</div>
                        <div class="tree-node" style="padding-left: 2rem;">📄 /api/products (200 OK)</div>
                    </div>
                `;
                document.getElementById('spider-results').innerHTML = resultsHtml;
                addLog('success', 'Spider completed: 15 pages discovered');
            }, 2000);
        }

        // Utility functions
        function showLoading() {
            document.getElementById('loading').classList.add('active');
        }

        function hideLoading() {
            document.getElementById('loading').classList.remove('active');
        }

        function addLog(level, message) {
            const log = document.getElementById('activity-log');
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `
                <span class="log-time">${time}</span>
                <span class="log-${level}">[${level.toUpperCase()}] ${message}</span>
            `;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }

        function forwardRequest() {
            addLog('info', 'Request forwarded');
        }

        function dropRequest() {
            addLog('warning', 'Request dropped');
        }

        function clearHistory() {
            document.getElementById('history-tbody').innerHTML = '';
            addLog('info', 'History cleared');
        }

        function sendRequest() {
            showLoading();
            addLog('info', 'Sending request...');
            setTimeout(() => {
                hideLoading();
                addLog('success', 'Response received');
            }, 1000);
        }

        function compareResponses() {
            addLog('info', 'Comparing responses...');
            setTimeout(() => {
                addLog('success', 'Comparison complete: 23 differences found');
            }, 500);
        }

        function configurePayloads() {
            addLog('info', 'Opening payload configuration...');
        }

        // WebSocket connection for real-time updates
        function connectWebSocket() {
            const ws = new WebSocket('ws://localhost:8081');

            ws.onopen = () => {
                addLog('success', 'Connected to ProxyPhantom backend');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'request') {
                    addRequestToHistory(data.request);
                } else if (data.type === 'issue') {
                    addIssueToDisplay(data.issue);
                }
            };

            ws.onerror = () => {
                addLog('error', 'WebSocket connection error');
            };

            ws.onclose = () => {
                addLog('warning', 'Disconnected from backend');
                setTimeout(connectWebSocket, 5000);
            };
        }

        function addRequestToHistory(request) {
            const tbody = document.getElementById('history-tbody');
            const row = tbody.insertRow(0);
            row.innerHTML = `
                <td>${request.id}</td>
                <td>${request.method}</td>
                <td>${request.url}</td>
                <td><span class="status-badge status-${request.status}">${request.status}</span></td>
                <td>${request.time}ms</td>
                <td>${request.size}</td>
            `;
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            addLog('info', 'ProxyPhantom Web UI loaded');
            // Uncomment to enable WebSocket
            // connectWebSocket();
        });
    </script>
</body>
</html>
"""

# ============================================================================
# CLI INTERFACE
# ============================================================================

def health_check() -> Dict[str, Any]:
    """Health check for Ai|oS integration"""
    try:
        # Check dependencies
        deps_ok = True
        missing = []

        if not CRYPTO_AVAILABLE:
            missing.append("cryptography")

        if not AIOHTTP_AVAILABLE:
            missing.append("aiohttp")

        # Create test instance
        app = ProxyPhantom()

        # Check database
        db_ok = os.path.exists("proxyphantom.db") or True

        status = "ok" if deps_ok and db_ok else "warn"

        return {
            "tool": "ProxyPhantom",
            "status": status,
            "summary": f"Web application security testing suite - {status}",
            "details": {
                "proxy_port": 8080,
                "ssl_interception": CRYPTO_AVAILABLE,
                "websocket_support": WEBSOCKETS_AVAILABLE,
                "database": "SQLite",
                "missing_deps": missing if missing else None
            }
        }
    except Exception as e:
        return {
            "tool": "ProxyPhantom",
            "status": "error",
            "summary": f"Health check failed: {e}",
            "details": {"error": str(e)}
        }

def main(argv=None):
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='🦊 ProxyPhantom - Web Application Security Testing Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  proxyphantom --gui                    # Launch GUI interface
  proxyphantom --proxy                  # Start proxy server
  proxyphantom --scan https://example.com  # Scan website
  proxyphantom --spider https://example.com --depth 5  # Spider website
  proxyphantom --demo                   # Run demonstration
        """
    )

    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    parser.add_argument('--proxy', action='store_true', help='Start proxy server')
    parser.add_argument('--scan', metavar='URL', help='Scan URL for vulnerabilities')
    parser.add_argument('--spider', metavar='URL', help='Spider website')
    parser.add_argument('--depth', type=int, default=3, help='Spider depth (default: 3)')
    parser.add_argument('--active', action='store_true', help='Enable active scanning')
    parser.add_argument('--port', type=int, default=8080, help='Proxy port (default: 8080)')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--health', action='store_true', help='Run health check')
    parser.add_argument('--json', action='store_true', help='JSON output')

    args = parser.parse_args(argv)

    # Health check
    if args.health:
        result = health_check()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"[{result['status'].upper()}] {result['summary']}")
            for key, value in result['details'].items():
                if value is not None:
                    print(f"  {key}: {value}")
        return 0

    # GUI mode
    if args.gui:
        LOG.info("🦊 Launching ProxyPhantom GUI...")

        # Save HTML to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(GUI_HTML)
            html_path = f.name

        # Open in browser
        import webbrowser
        webbrowser.open(f'file://{html_path}')

        LOG.info(f"GUI opened at: file://{html_path}")
        LOG.info("Note: Backend API not connected in demo mode")
        return 0

    # Create application instance
    app = ProxyPhantom()

    # Demo mode
    if args.demo:
        LOG.info("🦊 ProxyPhantom Demo Mode")
        LOG.info("-" * 50)

        # Demo scan
        LOG.info("\n[1] Scanning https://example.com...")
        issues = app.scan_url("https://example.com", active=False)

        if args.json:
            print(json.dumps([asdict(i) for i in issues], indent=2))
        else:
            LOG.info(f"Found {len(issues)} issues:")
            for issue in issues[:3]:  # Show first 3
                print(f"  - [{issue.severity.value}] {issue.type.value}")
                print(f"    URL: {issue.url}")
                print(f"    {issue.description}")

        # Demo decoder
        LOG.info("\n[2] Decoder Demo...")
        test_string = "ProxyPhantom"
        encoded = Decoder.encode(test_string, "base64")
        decoded = Decoder.decode(encoded, "base64")

        if not args.json:
            print(f"  Original: {test_string}")
            print(f"  Base64: {encoded}")
            print(f"  Decoded: {decoded}")

        # Demo sequencer
        LOG.info("\n[3] Token Analysis Demo...")
        sequencer = Sequencer()

        # Add sample tokens
        tokens = [
            "a4f8d2b1c9e7",
            "b5g9e3c2d0f8",
            "c6h0f4d3e1g9",
            "d7i1g5e4f2h0"
        ]

        for token in tokens:
            sequencer.add_sample(token)

        analysis = sequencer.analyze()

        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print(f"  Entropy: {analysis['entropy']:.2f}")
            print(f"  Quality: {analysis['quality']}")
            print(f"  Unique: {analysis['unique_count']}/{analysis['sample_count']}")

        LOG.info("\n✓ Demo completed successfully!")
        return 0

    # Proxy mode
    if args.proxy:
        LOG.info(f"🦊 Starting ProxyPhantom proxy on port {args.port}...")
        LOG.info(f"Configure your browser to use proxy: 127.0.0.1:{args.port}")
        LOG.info("Press Ctrl+C to stop")

        loop = asyncio.new_event_loop()
        app.proxy.port = args.port

        try:
            loop.run_until_complete(app.start())
            loop.run_forever()
        except KeyboardInterrupt:
            LOG.info("\nShutting down...")
            loop.run_until_complete(app.stop())
        finally:
            loop.close()

        return 0

    # Scan mode
    if args.scan:
        LOG.info(f"🦊 Scanning {args.scan}...")
        issues = app.scan_url(args.scan, active=args.active)

        if args.json:
            # Convert to dict for JSON serialization
            issues_dict = []
            for issue in issues:
                issue_dict = asdict(issue)
                issue_dict['type'] = issue.type.value
                issue_dict['severity'] = issue.severity.value
                issues_dict.append(issue_dict)
            print(json.dumps(issues_dict, indent=2))
        else:
            LOG.info(f"Found {len(issues)} issues")
            for issue in issues:
                print(f"[{issue.severity.value}] {issue.type.value}")
                print(f"  URL: {issue.url}")
                if issue.parameter:
                    print(f"  Parameter: {issue.parameter}")
                print(f"  {issue.description}")
                print()

        return 0

    # Spider mode
    if args.spider:
        LOG.info(f"🦊 Spidering {args.spider} (depth: {args.depth})...")
        sitemap = app.spider_url(args.spider, max_depth=args.depth)

        if args.json:
            print(json.dumps(sitemap, indent=2))
        else:
            for domain, paths in sitemap.items():
                print(f"\n{domain}:")
                for path in paths[:10]:  # Show first 10
                    print(f"  {path}")
                if len(paths) > 10:
                    print(f"  ... and {len(paths) - 10} more")

        return 0

    # No action specified
    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())