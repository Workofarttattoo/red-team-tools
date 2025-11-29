# Red Team Tools MCP Server with Semantic Lattice
**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Overview

This MCP (Model Context Protocol) server exposes the Red Team Security Tools through a semantic lattice architecture that provides intelligent tool discovery, capability-based search, and knowledge graph navigation.

## Semantic Lattice Architecture

The semantic lattice organizes security tools in a multi-dimensional knowledge graph:

```
Security Assessment (root)
  ├─ Network Reconnaissance
  │   └─ AuroraScan (port scanning, service detection, OS fingerprinting)
  ├─ Vulnerability Assessment
  │   ├─ CipherSpear (SQL injection, database security)
  │   ├─ NemesisHydra (authentication testing)
  │   └─ ObsidianHunt (configuration audit, hardening)
  ├─ Credential Analysis
  │   └─ MythicKey (password auditing, hash cracking)
  ├─ Traffic Analysis
  │   └─ SpectraTrace (packet inspection, protocol analysis)
  ├─ Wireless Security
  │   └─ SkyBreaker (WiFi auditing, WPA testing)
  └─ Payload Generation
      └─ VectorFlux (payload encoding, obfuscation)
```

### Relationship Types

- **Hierarchical**: `parent`, `child` (taxonomic organization)
- **Capability**: `implements`, `provides` (tool → technique)
- **Temporal**: `operates_in` (tool → attack phase)
- **Cross-domain**: Tools can relate across multiple dimensions

## MCP Resources

The server exposes the following resource types:

### 1. Semantic Graph
```
URI: lattice://semantic-graph
Description: Complete semantic lattice with all nodes and relationships
Format: JSON
```

### 2. Tool Resources
```
URI: lattice://tools/{tool_id}
Description: Detailed information about a specific security tool
Examples:
  - lattice://tools/aurorascan
  - lattice://tools/cipherspear
  - lattice://tools/mythickey
```

### 3. Capability Indexes
```
URI: lattice://capabilities/{capability}
Description: All tools providing a specific capability
Examples:
  - lattice://capabilities/port_scanning
  - lattice://capabilities/hash_cracking
  - lattice://capabilities/sql_injection_detection
```

## MCP Tools

### find_tools_by_capability
Find tools that provide a specific security capability.

**Input:**
```json
{
  "capability": "port_scanning"
}
```

**Output:**
```json
{
  "capability": "port_scanning",
  "tools": [
    {
      "id": "aurorascan",
      "name": "AuroraScan",
      "description": "Network reconnaissance and port scanning...",
      "techniques": ["tcp_syn_scan", "udp_scan", "version_detection"]
    }
  ]
}
```

### find_tools_by_phase
Find tools that operate in a specific attack phase.

**Input:**
```json
{
  "phase": "reconnaissance"
}
```

**Output:**
```json
{
  "phase": "reconnaissance",
  "tools": [
    {
      "id": "aurorascan",
      "name": "AuroraScan",
      "capabilities": ["port_scanning", "service_detection", "os_fingerprinting"]
    },
    {
      "id": "skybreaker",
      "name": "SkyBreaker",
      "capabilities": ["wpa_testing", "handshake_capture"]
    }
  ]
}
```

### get_tool_info
Get comprehensive information about a tool.

**Input:**
```json
{
  "tool_id": "cipherspear"
}
```

**Output:**
```json
{
  "id": "cipherspear",
  "name": "CipherSpear",
  "description": "Database injection vulnerability analysis...",
  "capabilities": ["sql_injection_detection", "nosql_injection", "blind_sqli"],
  "techniques": ["union_based", "error_based", "time_based_blind"],
  "phases": ["vulnerability_discovery", "exploitation"],
  "relationships": {
    "parent": ["vuln_assessment"],
    "implements": ["union_based", "error_based", "time_based_blind"]
  }
}
```

### navigate_lattice
Navigate the semantic lattice via relationships.

**Input:**
```json
{
  "node_id": "network_recon",
  "relationship": "has_parent"
}
```

**Output:**
```json
{
  "source_node": "network_recon",
  "relationship": "has_parent",
  "related_nodes": [
    {
      "id": "aurorascan",
      "type": "tool",
      "name": "AuroraScan",
      "description": "Network reconnaissance..."
    }
  ]
}
```

## Installation

### 1. Make the server executable
```bash
chmod +x /Users/noone/red-team-tools/mcp_server.py
```

### 2. Test the server
```bash
# Test basic functionality
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python3 /Users/noone/red-team-tools/mcp_server.py

# List all resources
echo '{"jsonrpc":"2.0","id":2,"method":"resources/list","params":{}}' | python3 /Users/noone/red-team-tools/mcp_server.py

# Get semantic graph
echo '{"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":"lattice://semantic-graph"}}' | python3 /Users/noone/red-team-tools/mcp_server.py
```

### 3. Configure in Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

### 4. Configure in other MCP clients

Use the provided `mcp-config.json` as a template.

## Usage Examples

### Example 1: Find tools for reconnaissance
```json
{
  "method": "tools/call",
  "params": {
    "name": "find_tools_by_phase",
    "arguments": {
      "phase": "reconnaissance"
    }
  }
}
```

### Example 2: Discover SQL injection tools
```json
{
  "method": "tools/call",
  "params": {
    "name": "find_tools_by_capability",
    "arguments": {
      "capability": "sql_injection_detection"
    }
  }
}
```

### Example 3: Get detailed tool information
```json
{
  "method": "tools/call",
  "params": {
    "name": "get_tool_info",
    "arguments": {
      "tool_id": "aurorascan"
    }
  }
}
```

### Example 4: Explore the semantic graph
```json
{
  "method": "resources/read",
  "params": {
    "uri": "lattice://semantic-graph"
  }
}
```

## Semantic Lattice Features

### Multi-dimensional Navigation

Tools can be discovered through multiple pathways:

1. **By Capability**: "What tools can crack passwords?"
2. **By Technique**: "What tools use time-based blind SQL injection?"
3. **By Phase**: "What tools operate during the reconnaissance phase?"
4. **By Category**: "What vulnerability assessment tools are available?"

### Cross-domain Relationships

Tools are linked across different security domains:
- A tool in "Network Reconnaissance" might relate to "Vulnerability Assessment"
- Techniques from different phases can share common tools
- Capabilities map to multiple attack vectors

### Knowledge Graph Properties

- **Nodes**: Tools, techniques, vulnerabilities, concepts
- **Edges**: Hierarchical, capability-based, temporal, adversarial
- **Attributes**: Metadata about capabilities, techniques, phases
- **Traversal**: Navigate from any node to related nodes

## Tool Capabilities Reference

### AuroraScan
- **Capabilities**: port_scanning, service_detection, os_fingerprinting
- **Techniques**: tcp_syn_scan, udp_scan, version_detection
- **Phases**: reconnaissance, enumeration

### CipherSpear
- **Capabilities**: sql_injection_detection, nosql_injection, blind_sqli
- **Techniques**: union_based, error_based, time_based_blind
- **Phases**: vulnerability_discovery, exploitation

### SkyBreaker
- **Capabilities**: wpa_testing, handshake_capture, deauth_testing
- **Techniques**: packet_injection, monitor_mode, handshake_analysis
- **Phases**: reconnaissance, attack

### MythicKey
- **Capabilities**: hash_cracking, pattern_analysis, entropy_calculation
- **Techniques**: dictionary_attack, rule_based, mask_attack
- **Phases**: credential_access, privilege_escalation

### SpectraTrace
- **Capabilities**: protocol_dissection, anomaly_detection, flow_analysis
- **Techniques**: deep_packet_inspection, statistical_analysis, signature_matching
- **Phases**: collection, analysis

### NemesisHydra
- **Capabilities**: brute_force, credential_stuffing, auth_bypass
- **Techniques**: parallel_testing, intelligent_throttling, session_analysis
- **Phases**: initial_access, credential_access

### ObsidianHunt
- **Capabilities**: config_audit, compliance_checking, baseline_comparison
- **Techniques**: policy_validation, secure_baseline, deviation_detection
- **Phases**: discovery, assessment

### VectorFlux
- **Capabilities**: payload_encoding, obfuscation, staging
- **Techniques**: polymorphic_encoding, encryption, staged_delivery
- **Phases**: weaponization, delivery

## Attack Phase Taxonomy

The semantic lattice organizes tools by MITRE ATT&CK-inspired phases:

1. **Reconnaissance**: Information gathering, passive/active scanning
2. **Enumeration**: Service identification, version detection
3. **Vulnerability Discovery**: Weakness identification, exploit research
4. **Weaponization**: Payload creation, exploit development
5. **Delivery**: Payload staging, delivery mechanisms
6. **Exploitation**: Vulnerability exploitation, initial access
7. **Initial Access**: First foothold establishment
8. **Credential Access**: Password/hash acquisition
9. **Privilege Escalation**: Elevation of privileges
10. **Collection**: Data gathering, exfiltration preparation
11. **Analysis**: Traffic analysis, forensic examination
12. **Assessment**: Security posture evaluation

## Extending the Lattice

To add new tools or relationships, edit `mcp_server.py`:

```python
# Add a new tool
tools.append({
    "id": "newtool",
    "name": "NewTool",
    "description": "Description of the tool",
    "category": "network_recon",  # Parent category
    "capabilities": ["new_capability"],
    "techniques": ["technique1", "technique2"],
    "phases": ["reconnaissance"]
})
```

The semantic lattice will automatically create bidirectional relationships.

## API Compliance

This server implements:
- MCP Protocol Version: 2024-11-05
- Resource Subscription: Supported
- Tool Invocation: Supported
- Prompt Templates: Not implemented (tools-focused server)

## Security Considerations

- **Read-only**: Server provides information only, no tool execution
- **Defensive Use**: Tools are for authorized security assessment only
- **Forensic Mode**: All operations are advisory, no system mutations
- **Access Control**: Implement authentication in production environments

## License

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)
All Rights Reserved. PATENT PENDING.

This software is provided for authorized security assessment and defensive purposes only.

---

**Version**: 1.0.0
**Protocol**: MCP 2024-11-05
**Architecture**: Semantic Lattice
**Last Updated**: 2025-11-08
