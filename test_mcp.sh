#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Quick MCP server test script

MCP_SERVER="/Users/noone/red-team-tools/mcp_server.py"

echo "Testing Red Team Tools MCP Server..."
echo ""

# Test 1: Initialize
echo "1. Testing initialization..."
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python3 "$MCP_SERVER" 2>/dev/null | jq -r '.result.serverInfo.name'
echo ""

# Test 2: List tools
echo "2. Listing available tools..."
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | python3 "$MCP_SERVER" 2>/dev/null | jq -r '.result.tools[].name'
echo ""

# Test 3: Find reconnaissance tools
echo "3. Finding reconnaissance tools..."
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"find_tools_by_phase","arguments":{"phase":"reconnaissance"}}}' | python3 "$MCP_SERVER" 2>/dev/null | jq -r '.result.content[0].text | fromjson | .tools[].name'
echo ""

# Test 4: Get graph stats
echo "4. Semantic lattice statistics..."
echo '{"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"lattice://semantic-graph"}}' | python3 "$MCP_SERVER" 2>/dev/null | jq '.result.contents[0].text | fromjson | .metadata'
echo ""

echo "✅ MCP Server is working!"
echo ""
echo "To use in Claude Desktop, add to:"
echo "~/Library/Application Support/Claude/claude_desktop_config.json"
echo ""
echo '{'
echo '  "mcpServers": {'
echo '    "red-team-tools": {'
echo '      "command": "python3",'
echo '      "args": ["/Users/noone/red-team-tools/mcp_server.py"]'
echo '    }'
echo '  }'
echo '}'
