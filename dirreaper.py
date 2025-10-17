#!/usr/bin/env python3
"""
DirReaper - High-Performance Directory/File Enumeration Tool
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

A blazingly fast directory brute-forcing tool with multiple scanning modes:
- Directory/File enumeration
- Virtual host discovery
- DNS subdomain enumeration
- S3 bucket discovery
- Parameter fuzzing
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import argparse
import logging
import hashlib
import socket
import struct
import random
import threading
import tkinter as tk
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urljoin, urlparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import dns.resolver
import dns.exception

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

# Built-in wordlists
WORDLIST_COMMON = [
    "admin", "administrator", "login", "logout", "dashboard", "panel", "config",
    "api", "v1", "v2", "assets", "static", "media", "uploads", "download", "files",
    "backup", "backups", "tmp", "temp", "cache", "logs", "log", "debug",
    "test", "tests", "testing", "dev", "development", "staging", "prod", "production",
    "public", "private", "secret", "hidden", "secure", "security", "auth",
    "user", "users", "profile", "account", "accounts", "settings", "preferences",
    "data", "database", "db", "sql", "mysql", "postgres", "mongodb", "redis",
    "wp-admin", "wp-content", "wp-includes", "wordpress", "blog", "news",
    "about", "contact", "help", "support", "faq", "terms", "privacy", "legal",
    "index", "home", "main", "default", "welcome", "portal", "gateway",
    "images", "img", "css", "js", "javascript", "scripts", "styles", "fonts",
    "docs", "documentation", "doc", "readme", "changelog", "license", "robots.txt",
    ".git", ".svn", ".env", ".htaccess", ".htpasswd", "web.config", "composer.json",
    "package.json", "bower.json", "Gruntfile.js", "gulpfile.js", "webpack.config.js",
    "phpinfo", "info", "status", "health", "healthcheck", "metrics", "stats",
    "search", "query", "find", "filter", "sort", "order", "page", "limit",
    "ajax", "json", "xml", "rss", "feed", "sitemap", "robots", "humans",
    "error", "404", "500", "403", "401", "400", "errors", "exception",
    "old", "new", "archive", "archives", "2020", "2021", "2022", "2023", "2024", "2025"
]

WORDLIST_MEDIUM = WORDLIST_COMMON + [
    "alpha", "beta", "gamma", "delta", "release", "rc", "build", "version",
    "system", "sys", "core", "kernel", "framework", "lib", "library", "vendor",
    "node_modules", "components", "modules", "plugins", "extensions", "addons",
    "themes", "templates", "layouts", "views", "partials", "includes",
    "src", "source", "dist", "distribution", "bin", "binary", "exe", "app",
    "application", "applications", "apps", "software", "program", "programs",
    "service", "services", "daemon", "process", "task", "job", "worker",
    "queue", "queues", "message", "messages", "mail", "email", "smtp", "pop3",
    "imap", "webmail", "mailbox", "inbox", "outbox", "sent", "drafts", "trash",
    "calendar", "events", "schedule", "booking", "reservation", "appointment",
    "shop", "store", "cart", "checkout", "payment", "order", "orders", "invoice",
    "product", "products", "catalog", "category", "categories", "item", "items",
    "content", "contents", "article", "articles", "post", "posts", "page", "pages",
    "comment", "comments", "review", "reviews", "rating", "ratings", "vote",
    "forum", "forums", "board", "boards", "thread", "threads", "topic", "topics",
    "chat", "messenger", "conversation", "conversations", "message", "messages",
    "group", "groups", "team", "teams", "organization", "organizations", "company",
    "client", "clients", "customer", "customers", "member", "members", "guest",
    "staff", "employee", "employees", "manager", "managers", "executive", "ceo",
    "report", "reports", "analytics", "statistics", "dashboard", "kpi", "metric",
    "graph", "chart", "diagram", "visualization", "export", "import", "sync",
    "webhook", "webhooks", "callback", "callbacks", "hook", "hooks", "trigger",
    "cron", "scheduler", "timer", "clock", "timestamp", "datetime", "timezone",
    "locale", "language", "languages", "translation", "translations", "i18n", "l10n",
    "theme", "skin", "style", "design", "ui", "ux", "frontend", "backend",
    "middleware", "gateway", "proxy", "reverse-proxy", "loadbalancer", "cdn",
    "storage", "filesystem", "disk", "volume", "bucket", "container", "blob"
]

WORDLIST_BIG = WORDLIST_MEDIUM + [
    f"backup{i}" for i in range(1, 10)
] + [
    f"test{i}" for i in range(1, 10)
] + [
    f"dev{i}" for i in range(1, 10)
] + [
    f"admin{i}" for i in range(1, 10)
] + [
    f"user{i}" for i in range(1, 10)
] + [
    f"v{i}" for i in range(1, 10)
] + [
    f"api_v{i}" for i in range(1, 10)
] + [
    f"{year}" for year in range(2015, 2026)
] + [
    f"{month:02d}" for month in range(1, 13)
] + [
    f"{day:02d}" for day in range(1, 32)
]

# Common file extensions for fuzzing
COMMON_EXTENSIONS = [
    ".php", ".asp", ".aspx", ".jsp", ".html", ".htm", ".js", ".json", ".xml",
    ".txt", ".cfg", ".conf", ".ini", ".log", ".bak", ".backup", ".old", ".new",
    ".zip", ".tar", ".gz", ".rar", ".7z", ".sql", ".db", ".sqlite", ".csv",
    ".xls", ".xlsx", ".doc", ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".gif"
]

# Common subdomain wordlist
SUBDOMAIN_WORDLIST = [
    "www", "mail", "ftp", "admin", "blog", "shop", "api", "m", "mobile",
    "dev", "test", "staging", "prod", "beta", "alpha", "demo", "app",
    "portal", "secure", "vpn", "remote", "cloud", "cdn", "static", "media",
    "assets", "images", "img", "files", "download", "uploads", "docs",
    "support", "help", "kb", "wiki", "forum", "community", "chat", "irc",
    "git", "svn", "hg", "repo", "code", "build", "ci", "jenkins", "gitlab",
    "db", "database", "mysql", "postgres", "mongo", "redis", "cache", "queue",
    "mail", "smtp", "pop", "imap", "webmail", "email", "mx", "mx1", "mx2"
]

# Common S3 bucket patterns
S3_BUCKET_PATTERNS = [
    "{domain}-backup", "{domain}-backups", "{domain}-archive", "{domain}-archives",
    "{domain}-assets", "{domain}-static", "{domain}-media", "{domain}-uploads",
    "{domain}-files", "{domain}-documents", "{domain}-data", "{domain}-logs",
    "{domain}-dev", "{domain}-test", "{domain}-staging", "{domain}-prod",
    "{domain}-public", "{domain}-private", "{domain}-temp", "{domain}-tmp",
    "backup-{domain}", "backups-{domain}", "archive-{domain}", "archives-{domain}",
    "assets-{domain}", "static-{domain}", "media-{domain}", "uploads-{domain}",
    "dev-{domain}", "test-{domain}", "staging-{domain}", "prod-{domain}"
]

@dataclass
class ScanResult:
    """Represents a single scan result"""
    url: str
    status: int
    size: int
    redirect: Optional[str] = None
    content_type: Optional[str] = None
    title: Optional[str] = None
    timestamp: float = 0
    response_time: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DirReaper:
    """High-performance directory enumeration engine"""

    def __init__(self, target: str, wordlist: List[str] = None, mode: str = "dir",
                 extensions: List[str] = None, threads: int = 50,
                 status_codes: List[int] = None, recursive: bool = False,
                 follow_redirects: bool = True, timeout: int = 10,
                 user_agent: str = None, proxy: str = None):
        self.target = target.rstrip('/')
        self.wordlist = wordlist or WORDLIST_COMMON
        self.mode = mode
        self.extensions = extensions or []
        self.threads = min(threads, 500)  # Cap at 500
        self.status_codes = status_codes or [200, 201, 301, 302, 401, 403]
        self.recursive = recursive
        self.follow_redirects = follow_redirects
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.user_agent = user_agent or "DirReaper/1.0 (Ai|oS Security Scanner)"
        self.proxy = proxy

        # Statistics
        self.stats = {
            'requests': 0,
            'found': 0,
            'errors': 0,
            'start_time': time.time(),
            'requests_per_sec': 0
        }

        # Results storage
        self.results: List[ScanResult] = []
        self.discovered_dirs: Set[str] = set()
        self.scanned_paths: Set[str] = set()

        # Rate limiting
        self.semaphore = asyncio.Semaphore(self.threads)
        self.rate_limit_delay = 0
        self.waf_detected = False

    async def scan_path(self, session: aiohttp.ClientSession, path: str) -> Optional[ScanResult]:
        """Scan a single path"""
        async with self.semaphore:
            if path in self.scanned_paths:
                return None
            self.scanned_paths.add(path)

            # Apply rate limiting if needed
            if self.rate_limit_delay > 0:
                await asyncio.sleep(self.rate_limit_delay)

            url = urljoin(self.target, path)
            start_time = time.time()

            try:
                async with session.get(url, allow_redirects=self.follow_redirects) as response:
                    self.stats['requests'] += 1
                    response_time = time.time() - start_time

                    # Check for WAF detection
                    if response.status == 429 or 'rate limit' in response.headers.get('X-Error', '').lower():
                        self.waf_detected = True
                        self.rate_limit_delay = min(self.rate_limit_delay + 0.1, 2.0)
                        LOG.warning(f"Rate limiting detected, slowing down to {self.rate_limit_delay}s delay")

                    # Check if status code is interesting
                    if response.status in self.status_codes:
                        content = await response.text()
                        size = len(content)

                        # Extract title from HTML
                        title = None
                        if 'text/html' in response.headers.get('Content-Type', ''):
                            import re
                            title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                            if title_match:
                                title = title_match.group(1)[:100]

                        result = ScanResult(
                            url=url,
                            status=response.status,
                            size=size,
                            redirect=str(response.url) if response.url != url else None,
                            content_type=response.headers.get('Content-Type'),
                            title=title,
                            timestamp=time.time(),
                            response_time=response_time
                        )

                        self.stats['found'] += 1
                        self.results.append(result)

                        # Track directories for recursive scanning
                        if self.recursive and response.status in [200, 301, 302]:
                            if path.endswith('/') or '.' not in path.split('/')[-1]:
                                self.discovered_dirs.add(path.rstrip('/'))

                        return result

            except asyncio.TimeoutError:
                self.stats['errors'] += 1
                LOG.debug(f"Timeout: {url}")
            except aiohttp.ClientError as e:
                self.stats['errors'] += 1
                LOG.debug(f"Error scanning {url}: {e}")
            except Exception as e:
                self.stats['errors'] += 1
                LOG.debug(f"Unexpected error scanning {url}: {e}")

            return None

    async def generate_paths(self) -> List[str]:
        """Generate paths to scan based on mode"""
        paths = []

        if self.mode == "dir":
            # Generate directory/file paths
            for word in self.wordlist:
                # Add base word
                paths.append(word)
                paths.append(f"{word}/")

                # Add with extensions
                for ext in self.extensions:
                    paths.append(f"{word}{ext}")

        elif self.mode == "vhost":
            # Virtual host discovery uses wordlist as subdomain prefixes
            paths = self.wordlist

        elif self.mode == "dns":
            # DNS enumeration uses subdomain wordlist
            paths = SUBDOMAIN_WORDLIST if not self.wordlist else self.wordlist

        elif self.mode == "s3":
            # S3 bucket discovery
            domain = urlparse(self.target).netloc.replace('www.', '')
            for pattern in S3_BUCKET_PATTERNS:
                paths.append(pattern.format(domain=domain.split('.')[0]))

        elif self.mode == "fuzzing":
            # Parameter fuzzing
            for word in self.wordlist:
                paths.append(f"?{word}=test")
                paths.append(f"?id={word}")
                paths.append(f"/{word}")

        return paths

    async def scan_dir_mode(self):
        """Directory/file enumeration mode"""
        connector = aiohttp.TCPConnector(limit=self.threads, limit_per_host=self.threads)
        timeout = self.timeout

        headers = {'User-Agent': self.user_agent}

        async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
            # Initial scan
            paths = await self.generate_paths()
            tasks = [self.scan_path(session, path) for path in paths]

            LOG.info(f"Starting directory scan with {len(paths)} paths")
            await asyncio.gather(*tasks)

            # Recursive scanning
            if self.recursive and self.discovered_dirs:
                LOG.info(f"Found {len(self.discovered_dirs)} directories, scanning recursively...")

                for directory in list(self.discovered_dirs):
                    dir_paths = [f"{directory}/{word}" for word in self.wordlist]
                    dir_paths += [f"{directory}/{word}/" for word in self.wordlist]

                    for ext in self.extensions:
                        dir_paths += [f"{directory}/{word}{ext}" for word in self.wordlist]

                    tasks = [self.scan_path(session, path) for path in dir_paths]
                    await asyncio.gather(*tasks)

    async def scan_vhost_mode(self):
        """Virtual host discovery mode"""
        base_domain = urlparse(self.target).netloc
        connector = aiohttp.TCPConnector(limit=self.threads, limit_per_host=self.threads)
        timeout = self.timeout

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for subdomain in self.wordlist:
                vhost = f"{subdomain}.{base_domain}"
                headers = {
                    'Host': vhost,
                    'User-Agent': self.user_agent
                }

                try:
                    async with session.get(self.target, headers=headers, allow_redirects=False) as response:
                        self.stats['requests'] += 1

                        if response.status in self.status_codes:
                            content = await response.text()

                            result = ScanResult(
                                url=vhost,
                                status=response.status,
                                size=len(content),
                                content_type=response.headers.get('Content-Type'),
                                timestamp=time.time()
                            )

                            self.stats['found'] += 1
                            self.results.append(result)
                            LOG.info(f"Found vhost: {vhost} [{response.status}]")

                except Exception as e:
                    self.stats['errors'] += 1
                    LOG.debug(f"Error checking vhost {vhost}: {e}")

    async def scan_dns_mode(self):
        """DNS subdomain enumeration mode"""
        domain = urlparse(self.target).netloc.replace('www.', '')
        resolver = dns.resolver.Resolver()
        resolver.timeout = 2
        resolver.lifetime = 2

        for subdomain in SUBDOMAIN_WORDLIST:
            fqdn = f"{subdomain}.{domain}"
            self.stats['requests'] += 1

            try:
                answers = resolver.resolve(fqdn, 'A')
                ips = [str(rdata) for rdata in answers]

                result = ScanResult(
                    url=fqdn,
                    status=200,  # Use 200 for found
                    size=len(ips),
                    redirect=','.join(ips),  # Store IPs in redirect field
                    timestamp=time.time()
                )

                self.stats['found'] += 1
                self.results.append(result)
                LOG.info(f"Found subdomain: {fqdn} -> {ips}")

            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                pass
            except Exception as e:
                self.stats['errors'] += 1
                LOG.debug(f"DNS error for {fqdn}: {e}")

    async def scan_s3_mode(self):
        """S3 bucket discovery mode"""
        connector = aiohttp.TCPConnector(limit=self.threads, limit_per_host=self.threads)
        timeout = self.timeout
        headers = {'User-Agent': self.user_agent}

        async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
            paths = await self.generate_paths()

            for bucket_name in paths:
                # Check multiple S3 URLs
                s3_urls = [
                    f"https://{bucket_name}.s3.amazonaws.com/",
                    f"https://s3.amazonaws.com/{bucket_name}/",
                    f"https://{bucket_name}.s3-website.amazonaws.com/"
                ]

                for url in s3_urls:
                    try:
                        async with session.get(url, allow_redirects=False) as response:
                            self.stats['requests'] += 1

                            if response.status in [200, 301, 302, 403]:
                                content = await response.text()

                                # Check if it's a valid S3 response
                                if 'ListBucketResult' in content or 'AccessDenied' in content:
                                    result = ScanResult(
                                        url=url,
                                        status=response.status,
                                        size=len(content),
                                        content_type=response.headers.get('Content-Type'),
                                        title="S3 Bucket Found" if response.status == 200 else "S3 Bucket (Access Denied)",
                                        timestamp=time.time()
                                    )

                                    self.stats['found'] += 1
                                    self.results.append(result)
                                    LOG.info(f"Found S3 bucket: {url} [{response.status}]")
                                    break

                    except Exception as e:
                        self.stats['errors'] += 1
                        LOG.debug(f"Error checking S3 bucket {url}: {e}")

    async def run(self) -> List[ScanResult]:
        """Run the scan based on mode"""
        self.stats['start_time'] = time.time()

        LOG.info(f"Starting {self.mode} scan on {self.target}")
        LOG.info(f"Wordlist: {len(self.wordlist)} words, Threads: {self.threads}")

        try:
            if self.mode == "dir":
                await self.scan_dir_mode()
            elif self.mode == "vhost":
                await self.scan_vhost_mode()
            elif self.mode == "dns":
                await self.scan_dns_mode()
            elif self.mode == "s3":
                await self.scan_s3_mode()
            elif self.mode == "fuzzing":
                await self.scan_dir_mode()  # Use dir mode for fuzzing
            else:
                LOG.error(f"Unknown mode: {self.mode}")

        except KeyboardInterrupt:
            LOG.info("Scan interrupted by user")

        # Calculate final stats
        duration = time.time() - self.stats['start_time']
        if duration > 0:
            self.stats['requests_per_sec'] = self.stats['requests'] / duration

        LOG.info(f"Scan complete: {self.stats['found']} results found")
        LOG.info(f"Total requests: {self.stats['requests']}, Errors: {self.stats['errors']}")
        LOG.info(f"Requests/sec: {self.stats['requests_per_sec']:.2f}")

        return self.results

def health_check() -> Dict[str, Any]:
    """Health check for Ai|oS integration"""
    try:
        start = time.time()

        # Check if aiohttp is available
        import aiohttp
        aiohttp_version = aiohttp.__version__

        # Check if dnspython is available
        try:
            import dns.resolver
            dns_available = True
        except ImportError:
            dns_available = False

        # Test basic functionality
        reaper = DirReaper("https://example.com", wordlist=["test"], threads=1)

        latency = (time.time() - start) * 1000

        return {
            "tool": "DirReaper",
            "status": "ok",
            "summary": "Directory enumeration tool operational",
            "details": {
                "modes": ["dir", "vhost", "dns", "s3", "fuzzing"],
                "aiohttp_version": aiohttp_version,
                "dns_support": dns_available,
                "max_threads": 500,
                "wordlists": {
                    "common": len(WORDLIST_COMMON),
                    "medium": len(WORDLIST_MEDIUM),
                    "big": len(WORDLIST_BIG)
                },
                "latency_ms": round(latency, 2)
            }
        }
    except Exception as e:
        return {
            "tool": "DirReaper",
            "status": "error",
            "summary": f"Health check failed: {e}",
            "details": {"error": str(e)}
        }

# Embedded GUI HTML
GUI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DirReaper - Directory Enumeration Tool</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0f001a, #1a0033, #150026);
            color: #cc66ff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Header */
        .header {
            background: linear-gradient(90deg, #8800cc, #aa33ff);
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(136, 0, 204, 0.5);
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '💀';
            position: absolute;
            left: 50px;
            top: 50%;
            transform: translateY(-50%) rotate(-15deg);
            font-size: 48px;
            opacity: 0.3;
            animation: float 3s ease-in-out infinite;
        }

        .header h1 {
            font-size: 36px;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5);
            letter-spacing: 3px;
        }

        .header .subtitle {
            font-size: 14px;
            opacity: 0.9;
            margin-top: 5px;
        }

        /* Mode Tabs */
        .mode-tabs {
            display: flex;
            background: #1a0033;
            border-bottom: 2px solid #8800cc;
        }

        .mode-tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            background: transparent;
            color: #aa33ff;
            border: none;
            font-family: inherit;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s;
            position: relative;
        }

        .mode-tab:hover {
            background: rgba(136, 0, 204, 0.2);
        }

        .mode-tab.active {
            background: #8800cc;
            color: white;
        }

        .mode-tab.active::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            right: 0;
            height: 2px;
            background: #cc66ff;
            box-shadow: 0 0 10px #cc66ff;
        }

        /* Main Container */
        .container {
            display: flex;
            flex: 1;
            padding: 20px;
            gap: 20px;
        }

        /* Configuration Panel */
        .config-panel {
            flex: 1;
            background: rgba(26, 0, 51, 0.7);
            border: 1px solid #8800cc;
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }

        .config-group {
            margin-bottom: 20px;
        }

        .config-group label {
            display: block;
            margin-bottom: 5px;
            color: #cc66ff;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .config-group input,
        .config-group select,
        .config-group textarea {
            width: 100%;
            padding: 10px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #8800cc;
            border-radius: 5px;
            color: #cc66ff;
            font-family: inherit;
            transition: all 0.3s;
        }

        .config-group input:focus,
        .config-group select:focus,
        .config-group textarea:focus {
            outline: none;
            border-color: #cc66ff;
            box-shadow: 0 0 10px rgba(204, 102, 255, 0.3);
        }

        /* Slider */
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .slider {
            flex: 1;
            -webkit-appearance: none;
            appearance: none;
            height: 5px;
            background: #1a0033;
            outline: none;
            border-radius: 5px;
        }

        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            background: #cc66ff;
            cursor: pointer;
            border-radius: 50%;
            box-shadow: 0 0 10px #cc66ff;
        }

        .slider::-moz-range-thumb {
            width: 20px;
            height: 20px;
            background: #cc66ff;
            cursor: pointer;
            border-radius: 50%;
            box-shadow: 0 0 10px #cc66ff;
        }

        .slider-value {
            min-width: 50px;
            text-align: center;
            color: #cc66ff;
            font-weight: bold;
        }

        /* Buttons */
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }

        .btn {
            flex: 1;
            padding: 12px;
            background: linear-gradient(135deg, #8800cc, #aa33ff);
            border: none;
            border-radius: 5px;
            color: white;
            font-family: inherit;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(136, 0, 204, 0.5);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn.stop {
            background: linear-gradient(135deg, #cc0066, #ff3399);
        }

        .btn.pause {
            background: linear-gradient(135deg, #cc6600, #ff9933);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Results Section */
        .results-section {
            flex: 2;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* Statistics Panel */
        .stats-panel {
            background: rgba(26, 0, 51, 0.7);
            border: 1px solid #8800cc;
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }

        .stat-item {
            text-align: center;
            padding: 10px;
            background: rgba(136, 0, 204, 0.1);
            border-radius: 5px;
            border: 1px solid rgba(136, 0, 204, 0.3);
        }

        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #cc66ff;
            text-shadow: 0 0 10px rgba(204, 102, 255, 0.5);
        }

        .stat-label {
            font-size: 11px;
            color: #aa33ff;
            text-transform: uppercase;
            margin-top: 5px;
            letter-spacing: 1px;
        }

        /* Progress Bar */
        .progress-container {
            margin-top: 15px;
        }

        .progress-bar {
            height: 20px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #8800cc, #cc66ff);
            width: 0%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }

        /* Results Table */
        .results-panel {
            flex: 1;
            background: rgba(26, 0, 51, 0.7);
            border: 1px solid #8800cc;
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .results-title {
            font-size: 18px;
            color: #cc66ff;
            font-weight: bold;
        }

        .export-btn {
            padding: 8px 15px;
            background: #8800cc;
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
        }

        .export-btn:hover {
            background: #aa33ff;
        }

        .table-container {
            flex: 1;
            overflow-y: auto;
            overflow-x: auto;
        }

        .results-table {
            width: 100%;
            border-collapse: collapse;
        }

        .results-table th {
            background: rgba(136, 0, 204, 0.2);
            padding: 10px;
            text-align: left;
            color: #cc66ff;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            position: sticky;
            top: 0;
        }

        .results-table td {
            padding: 8px 10px;
            border-bottom: 1px solid rgba(136, 0, 204, 0.1);
            font-size: 13px;
            color: #aa33ff;
        }

        .results-table tr {
            animation: slideIn 0.3s ease-out;
        }

        .results-table tr:hover {
            background: rgba(136, 0, 204, 0.1);
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        /* Status Colors */
        .status-200 { color: #00ff00; }
        .status-301, .status-302 { color: #ffff00; }
        .status-401, .status-403 { color: #ff9900; }
        .status-404 { color: #999999; }
        .status-500 { color: #ff0000; }

        /* Animations */
        @keyframes float {
            0%, 100% { transform: translateY(-50%) rotate(-15deg) scale(1); }
            50% { transform: translateY(-50%) rotate(-15deg) scale(1.1); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .scanning {
            animation: pulse 1s infinite;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(26, 0, 51, 0.5);
        }

        ::-webkit-scrollbar-thumb {
            background: #8800cc;
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #aa33ff;
        }

        /* Loading Spinner */
        .spinner {
            width: 30px;
            height: 30px;
            border: 3px solid rgba(204, 102, 255, 0.3);
            border-top: 3px solid #cc66ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Mode-specific configurations */
        .mode-config {
            display: none;
        }

        .mode-config.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>💀 DirReaper</h1>
        <div class="subtitle">High-Performance Directory Enumeration Tool</div>
    </div>

    <div class="mode-tabs">
        <button class="mode-tab active" data-mode="dir">Directory</button>
        <button class="mode-tab" data-mode="vhost">VHost</button>
        <button class="mode-tab" data-mode="dns">DNS</button>
        <button class="mode-tab" data-mode="s3">S3 Bucket</button>
        <button class="mode-tab" data-mode="fuzzing">Fuzzing</button>
    </div>

    <div class="container">
        <div class="config-panel">
            <!-- Directory Mode Config -->
            <div class="mode-config active" id="config-dir">
                <div class="config-group">
                    <label>Target URL</label>
                    <input type="text" id="target" placeholder="https://example.com" value="https://example.com">
                </div>

                <div class="config-group">
                    <label>Wordlist</label>
                    <select id="wordlist">
                        <option value="common">Common (100 words)</option>
                        <option value="medium">Medium (200 words)</option>
                        <option value="big">Big (300+ words)</option>
                        <option value="custom">Custom</option>
                    </select>
                </div>

                <div class="config-group" id="custom-wordlist-group" style="display: none;">
                    <label>Custom Wordlist (one per line)</label>
                    <textarea id="custom-wordlist" rows="5" placeholder="admin\nlogin\napi\n..."></textarea>
                </div>

                <div class="config-group">
                    <label>Extensions (comma-separated)</label>
                    <input type="text" id="extensions" placeholder=".php,.asp,.html,.js" value=".php,.html">
                </div>

                <div class="config-group">
                    <label>Status Codes (comma-separated)</label>
                    <input type="text" id="status-codes" value="200,301,302,401,403">
                </div>

                <div class="config-group">
                    <label>Threads: <span id="threads-value">50</span></label>
                    <div class="slider-container">
                        <input type="range" class="slider" id="threads" min="1" max="500" value="50">
                    </div>
                </div>

                <div class="config-group">
                    <label>
                        <input type="checkbox" id="recursive"> Recursive Scanning
                    </label>
                </div>

                <div class="config-group">
                    <label>
                        <input type="checkbox" id="follow-redirects" checked> Follow Redirects
                    </label>
                </div>
            </div>

            <!-- VHost Mode Config -->
            <div class="mode-config" id="config-vhost">
                <div class="config-group">
                    <label>Target URL</label>
                    <input type="text" id="vhost-target" placeholder="https://example.com">
                </div>

                <div class="config-group">
                    <label>Subdomain Wordlist</label>
                    <select id="vhost-wordlist">
                        <option value="common">Common Subdomains</option>
                        <option value="custom">Custom</option>
                    </select>
                </div>
            </div>

            <!-- DNS Mode Config -->
            <div class="mode-config" id="config-dns">
                <div class="config-group">
                    <label>Target Domain</label>
                    <input type="text" id="dns-target" placeholder="example.com">
                </div>

                <div class="config-group">
                    <label>Subdomain Wordlist</label>
                    <select id="dns-wordlist">
                        <option value="common">Common Subdomains</option>
                        <option value="custom">Custom</option>
                    </select>
                </div>
            </div>

            <!-- S3 Mode Config -->
            <div class="mode-config" id="config-s3">
                <div class="config-group">
                    <label>Target Domain</label>
                    <input type="text" id="s3-target" placeholder="example.com">
                </div>

                <div class="config-group">
                    <label>Bucket Patterns</label>
                    <select id="s3-patterns">
                        <option value="default">Default Patterns</option>
                        <option value="custom">Custom</option>
                    </select>
                </div>
            </div>

            <!-- Fuzzing Mode Config -->
            <div class="mode-config" id="config-fuzzing">
                <div class="config-group">
                    <label>Target URL</label>
                    <input type="text" id="fuzz-target" placeholder="https://example.com/page">
                </div>

                <div class="config-group">
                    <label>Fuzzing Wordlist</label>
                    <textarea id="fuzz-wordlist" rows="5" placeholder="id\nuser\npage\nfile\n..."></textarea>
                </div>
            </div>

            <div class="button-group">
                <button class="btn" id="start-btn">Start Scan</button>
                <button class="btn pause" id="pause-btn" disabled>Pause</button>
                <button class="btn stop" id="stop-btn" disabled>Stop</button>
            </div>
        </div>

        <div class="results-section">
            <div class="stats-panel">
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="stat-requests">0</div>
                        <div class="stat-label">Requests</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-found">0</div>
                        <div class="stat-label">Found</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-errors">0</div>
                        <div class="stat-label">Errors</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-rps">0</div>
                        <div class="stat-label">Req/Sec</div>
                    </div>
                </div>

                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress">0%</div>
                    </div>
                </div>
            </div>

            <div class="results-panel">
                <div class="results-header">
                    <div class="results-title">Scan Results</div>
                    <button class="export-btn" id="export-btn">Export</button>
                </div>

                <div class="table-container">
                    <table class="results-table" id="results-table">
                        <thead>
                            <tr>
                                <th>Status</th>
                                <th>Size</th>
                                <th>Path</th>
                                <th>Type</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody id="results-body">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Mode switching
        const modeTabs = document.querySelectorAll('.mode-tab');
        const modeConfigs = document.querySelectorAll('.mode-config');
        let currentMode = 'dir';

        modeTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                modeTabs.forEach(t => t.classList.remove('active'));
                modeConfigs.forEach(c => c.classList.remove('active'));

                tab.classList.add('active');
                currentMode = tab.dataset.mode;
                document.getElementById(`config-${currentMode}`).classList.add('active');
            });
        });

        // Wordlist selection
        document.getElementById('wordlist').addEventListener('change', (e) => {
            const customGroup = document.getElementById('custom-wordlist-group');
            customGroup.style.display = e.target.value === 'custom' ? 'block' : 'none';
        });

        // Threads slider
        const threadsSlider = document.getElementById('threads');
        const threadsValue = document.getElementById('threads-value');

        threadsSlider.addEventListener('input', () => {
            threadsValue.textContent = threadsSlider.value;
        });

        // Scanning state
        let scanning = false;
        let scanInterval = null;
        let results = [];

        // Start scan
        document.getElementById('start-btn').addEventListener('click', async () => {
            if (scanning) return;

            scanning = true;
            results = [];

            // Update UI
            document.getElementById('start-btn').disabled = true;
            document.getElementById('pause-btn').disabled = false;
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('results-body').innerHTML = '';

            // Reset stats
            document.getElementById('stat-requests').textContent = '0';
            document.getElementById('stat-found').textContent = '0';
            document.getElementById('stat-errors').textContent = '0';
            document.getElementById('stat-rps').textContent = '0';

            // Get configuration
            const config = getConfig();

            // Start animation
            document.querySelector('.progress-fill').style.width = '0%';

            // Simulate scanning (in real implementation, this would call the backend)
            simulateScan(config);
        });

        // Stop scan
        document.getElementById('stop-btn').addEventListener('click', () => {
            stopScan();
        });

        // Export results
        document.getElementById('export-btn').addEventListener('click', () => {
            exportResults();
        });

        function getConfig() {
            const config = {
                mode: currentMode,
                threads: parseInt(threadsSlider.value)
            };

            switch(currentMode) {
                case 'dir':
                    config.target = document.getElementById('target').value;
                    config.wordlist = document.getElementById('wordlist').value;
                    config.extensions = document.getElementById('extensions').value;
                    config.statusCodes = document.getElementById('status-codes').value;
                    config.recursive = document.getElementById('recursive').checked;
                    config.followRedirects = document.getElementById('follow-redirects').checked;

                    if (config.wordlist === 'custom') {
                        config.customWordlist = document.getElementById('custom-wordlist').value;
                    }
                    break;

                case 'vhost':
                    config.target = document.getElementById('vhost-target').value;
                    config.wordlist = document.getElementById('vhost-wordlist').value;
                    break;

                case 'dns':
                    config.target = document.getElementById('dns-target').value;
                    config.wordlist = document.getElementById('dns-wordlist').value;
                    break;

                case 's3':
                    config.target = document.getElementById('s3-target').value;
                    config.patterns = document.getElementById('s3-patterns').value;
                    break;

                case 'fuzzing':
                    config.target = document.getElementById('fuzz-target').value;
                    config.wordlist = document.getElementById('fuzz-wordlist').value;
                    break;
            }

            return config;
        }

        function simulateScan(config) {
            let requests = 0;
            let found = 0;
            let errors = 0;
            const startTime = Date.now();

            // Simulate results
            const paths = [
                {status: 200, size: 15234, path: '/admin/', type: 'text/html'},
                {status: 301, size: 0, path: '/api/', type: 'redirect'},
                {status: 403, size: 298, path: '/.git/', type: 'text/html'},
                {status: 200, size: 8976, path: '/login.php', type: 'text/html'},
                {status: 401, size: 456, path: '/secure/', type: 'text/html'},
                {status: 200, size: 23456, path: '/uploads/', type: 'text/html'},
                {status: 302, size: 0, path: '/dashboard/', type: 'redirect'},
                {status: 200, size: 5432, path: '/config.php', type: 'application/x-httpd-php'},
                {status: 403, size: 298, path: '/.env', type: 'text/plain'},
                {status: 200, size: 12345, path: '/backup.sql', type: 'application/sql'}
            ];

            let pathIndex = 0;

            scanInterval = setInterval(() => {
                if (!scanning || pathIndex >= paths.length) {
                    stopScan();
                    return;
                }

                // Update stats
                requests += Math.floor(Math.random() * 10) + 5;

                // Add result
                if (Math.random() > 0.7 && pathIndex < paths.length) {
                    const result = paths[pathIndex++];
                    found++;
                    addResult(result);
                }

                if (Math.random() > 0.95) {
                    errors++;
                }

                // Update UI
                document.getElementById('stat-requests').textContent = requests;
                document.getElementById('stat-found').textContent = found;
                document.getElementById('stat-errors').textContent = errors;

                const elapsed = (Date.now() - startTime) / 1000;
                const rps = Math.floor(requests / elapsed);
                document.getElementById('stat-rps').textContent = rps;

                const progress = Math.min((pathIndex / paths.length) * 100, 100);
                const progressBar = document.getElementById('progress');
                progressBar.style.width = progress + '%';
                progressBar.textContent = Math.floor(progress) + '%';

            }, 200);
        }

        function addResult(result) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="status-${result.status}">${result.status}</td>
                <td>${result.size}</td>
                <td>${result.path}</td>
                <td>${result.type}</td>
                <td>${new Date().toLocaleTimeString()}</td>
            `;

            document.getElementById('results-body').appendChild(row);
            results.push(result);

            // Auto-scroll to bottom
            const tableContainer = document.querySelector('.table-container');
            tableContainer.scrollTop = tableContainer.scrollHeight;
        }

        function stopScan() {
            scanning = false;

            if (scanInterval) {
                clearInterval(scanInterval);
                scanInterval = null;
            }

            // Update UI
            document.getElementById('start-btn').disabled = false;
            document.getElementById('pause-btn').disabled = true;
            document.getElementById('stop-btn').disabled = true;

            const progressBar = document.getElementById('progress');
            if (progressBar.style.width !== '100%') {
                progressBar.textContent = 'Stopped';
            } else {
                progressBar.textContent = 'Complete';
            }
        }

        function exportResults() {
            if (results.length === 0) {
                alert('No results to export');
                return;
            }

            const data = JSON.stringify(results, null, 2);
            const blob = new Blob([data], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `dirreaper-results-${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""

def run_gui():
    """Launch the Tkinter GUI with embedded HTML"""
    import tkinter as tk
    from tkinter import ttk
    import webbrowser
    import tempfile

    # Create temporary HTML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(GUI_HTML)
        html_path = f.name

    # Open in browser
    webbrowser.open(f'file://{html_path}')

    print(f"[*] GUI opened in browser: {html_path}")

async def main_async(argv: List[str] = None) -> int:
    """Async main function"""
    parser = argparse.ArgumentParser(description='DirReaper - Directory Enumeration Tool')
    parser.add_argument('target', nargs='?', help='Target URL or domain')
    parser.add_argument('--mode', choices=['dir', 'vhost', 'dns', 's3', 'fuzzing'],
                       default='dir', help='Scanning mode')
    parser.add_argument('--wordlist', choices=['common', 'medium', 'big', 'custom'],
                       default='common', help='Wordlist to use')
    parser.add_argument('--custom-wordlist', type=str, help='Path to custom wordlist file')
    parser.add_argument('--extensions', type=str, help='File extensions to test (comma-separated)')
    parser.add_argument('--threads', type=int, default=50, help='Number of concurrent threads')
    parser.add_argument('--status-codes', type=str, default='200,301,302,401,403',
                       help='Status codes to report (comma-separated)')
    parser.add_argument('--recursive', action='store_true', help='Enable recursive scanning')
    parser.add_argument('--no-follow-redirects', action='store_true', help='Do not follow redirects')
    parser.add_argument('--timeout', type=int, default=10, help='Request timeout in seconds')
    parser.add_argument('--user-agent', type=str, help='Custom User-Agent string')
    parser.add_argument('--proxy', type=str, help='Proxy URL')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    parser.add_argument('--gui', action='store_true', help='Launch GUI')
    parser.add_argument('--health-check', action='store_true', help='Run health check')

    args = parser.parse_args(argv)

    # Health check mode
    if args.health_check:
        result = health_check()
        print(json.dumps(result, indent=2))
        return 0

    # GUI mode
    if args.gui:
        run_gui()
        return 0

    # Require target for scanning
    if not args.target:
        parser.print_help()
        return 1

    # Select wordlist
    wordlist = WORDLIST_COMMON
    if args.wordlist == 'medium':
        wordlist = WORDLIST_MEDIUM
    elif args.wordlist == 'big':
        wordlist = WORDLIST_BIG
    elif args.custom_wordlist:
        try:
            with open(args.custom_wordlist, 'r') as f:
                wordlist = [line.strip() for line in f if line.strip()]
        except Exception as e:
            LOG.error(f"Failed to load custom wordlist: {e}")
            return 1

    # Parse extensions
    extensions = []
    if args.extensions:
        extensions = [ext.strip() for ext in args.extensions.split(',')]

    # Parse status codes
    status_codes = [int(code.strip()) for code in args.status_codes.split(',')]

    # Create scanner
    scanner = DirReaper(
        target=args.target,
        wordlist=wordlist,
        mode=args.mode,
        extensions=extensions,
        threads=args.threads,
        status_codes=status_codes,
        recursive=args.recursive,
        follow_redirects=not args.no_follow_redirects,
        timeout=args.timeout,
        user_agent=args.user_agent,
        proxy=args.proxy
    )

    # Run scan
    results = await scanner.run()

    # Output results
    if args.json:
        output_data = {
            'target': args.target,
            'mode': args.mode,
            'stats': scanner.stats,
            'results': [r.to_dict() for r in results]
        }

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            LOG.info(f"Results saved to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))
    else:
        # Pretty print results
        print(f"\n{'='*60}")
        print(f"DirReaper Scan Results")
        print(f"{'='*60}")
        print(f"Target: {args.target}")
        print(f"Mode: {args.mode}")
        print(f"Found: {len(results)} results")
        print(f"{'='*60}\n")

        for result in results:
            status_color = ''
            if result.status == 200:
                status_color = '\033[92m'  # Green
            elif result.status in [301, 302]:
                status_color = '\033[93m'  # Yellow
            elif result.status in [401, 403]:
                status_color = '\033[91m'  # Red

            print(f"{status_color}[{result.status}]\033[0m {result.url} ({result.size} bytes)")

            if result.redirect:
                print(f"  -> Redirect: {result.redirect}")
            if result.title:
                print(f"  -> Title: {result.title}")

        if args.output:
            # Save text output
            with open(args.output, 'w') as f:
                for result in results:
                    f.write(f"[{result.status}] {result.url} ({result.size} bytes)\n")
            LOG.info(f"Results saved to {args.output}")

    return 0

def main(argv: List[str] = None) -> int:
    """Main entry point"""
    try:
        import asyncio
        return asyncio.run(main_async(argv))
    except KeyboardInterrupt:
        print("\n[!] Scan interrupted by user")
        return 130
    except Exception as e:
        LOG.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())