# Red Team Tools MCP Server - Quick Start
**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## 📍 Location

```
/Users/noone/red-team-tools/mcp_server.py
```

## 🚀 Quick Install (1 command)

```bash
bash /Users/noone/red-team-tools/install_to_claude.sh
```

Then restart Claude Desktop - done!

## 📝 Manual Setup

### Step 1: Edit Claude Desktop config

```bash
open -e ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

### Step 2: Add this configuration

```json
{
  "mcpServers": {
    "red-team-tools": {
      "command": "python3",
      "args": ["/Users/noone/red-team-tools/mcp_server.py"]
    }
  }
}
```

### Step 3: Restart Claude Desktop

Close and reopen Claude Desktop. The server will connect automatically.

## ✅ Verify Installation

```bash
bash /Users/noone/red-team-tools/test_mcp.sh
```

## 💬 Example Queries (in Claude Desktop)

Once connected, you can ask Claude:

### Tool Discovery
- "What security tools are available for network reconnaissance?"
- "Find tools that can perform SQL injection testing"
- "Show me all payload generation tools"
- "What tools operate during the reconnaissance phase?"

### Capability Search
- "Which tools can crack passwords?"
- "Find tools with directory enumeration capabilities"
- "What tools support traffic interception?"

### Semantic Navigation
- "Show me the semantic lattice graph"
- "What capabilities does AuroraScan have?"
- "Find tools related to CipherSpear"
- "List all vulnerability assessment tools"

## 🛠️ Available Tools (16 total)

### Network Reconnaissance
- AuroraScan
- NmapPro
- SpectraTrace

### Vulnerability Assessment
- CipherSpear
- NemesisHydra
- ObsidianHunt
- VulnHunter
- Sovereign Suite

### Web Application
- DirReaper

### Credential Analysis
- MythicKey

### Wireless Security
- SkyBreaker

### Payload Generation
- VectorFlux
- PayloadForge

### OSINT
- OSINTWorkflows

### Proxy & Traffic
- ProxyPhantom

### Reporting
- Scribe

## 🔧 Troubleshooting

### Server not appearing in Claude Desktop?

1. Check config file exists:
   ```bash
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. Test server directly:
   ```bash
   bash /Users/noone/red-team-tools/test_mcp.sh
   ```

3. Check for errors:
   ```bash
   echo '{"method":"initialize","params":{}}' | python3 /Users/noone/red-team-tools/mcp_server.py
   ```

### Need to reinstall?

```bash
rm ~/Library/Application\ Support/Claude/claude_desktop_config.json
bash /Users/noone/red-team-tools/install_to_claude.sh
```

## 📚 Full Documentation

- **MCP Server Code**: `/Users/noone/red-team-tools/mcp_server.py`
- **Full README**: `/Users/noone/red-team-tools/MCP_SERVER_README.md`
- **Config Template**: `/Users/noone/red-team-tools/mcp-config.json`

## 🎯 What You Can Do

With the MCP server connected, you can:

✅ **Discover tools** by capability, phase, or technique
✅ **Navigate relationships** between tools and concepts
✅ **Explore the semantic lattice** to understand tool ecosystems
✅ **Get detailed info** about any security tool
✅ **Find related tools** based on shared capabilities

## 🔒 Security Note

This MCP server provides **read-only access** to tool information. It does not execute any security tools - it only helps you discover and understand them.

---

**Server Version**: 1.0.0
**Tools**: 16 security tools
**Semantic Nodes**: 27 total
**Protocol**: MCP 2024-11-05
