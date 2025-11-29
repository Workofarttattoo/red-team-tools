#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Install Red Team Tools MCP Server to Claude Desktop

CONFIG_DIR="$HOME/Library/Application Support/Claude"
CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"
MCP_SERVER="/Users/noone/red-team-tools/mcp_server.py"

echo "Installing Red Team Tools MCP Server to Claude Desktop..."
echo ""

# Create config directory if it doesn't exist
mkdir -p "$CONFIG_DIR"

# Check if config file exists
if [ -f "$CONFIG_FILE" ]; then
    echo "✓ Found existing config at: $CONFIG_FILE"
    echo ""
    echo "⚠️  Backing up existing config..."
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✓ Backup created"
    echo ""

    # Check if red-team-tools already configured
    if grep -q "red-team-tools" "$CONFIG_FILE"; then
        echo "⚠️  red-team-tools server already configured!"
        echo ""
        read -p "Overwrite existing configuration? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 0
        fi
    fi

    # Parse existing config and add red-team-tools
    echo "Adding red-team-tools to existing configuration..."

    # Use jq if available, otherwise manual edit
    if command -v jq &> /dev/null; then
        jq '.mcpServers["red-team-tools"] = {
            "command": "python3",
            "args": ["/Users/noone/red-team-tools/mcp_server.py"],
            "env": {}
        }' "$CONFIG_FILE" > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
        echo "✓ Configuration updated using jq"
    else
        echo ""
        echo "⚠️  jq not found. Please manually add this to $CONFIG_FILE:"
        echo ""
        cat << 'EOF'
{
  "mcpServers": {
    "red-team-tools": {
      "command": "python3",
      "args": ["/Users/noone/red-team-tools/mcp_server.py"],
      "env": {}
    }
  }
}
EOF
        echo ""
        exit 1
    fi
else
    echo "Creating new config file..."
    cat > "$CONFIG_FILE" << 'EOF'
{
  "mcpServers": {
    "red-team-tools": {
      "command": "python3",
      "args": ["/Users/noone/red-team-tools/mcp_server.py"],
      "env": {}
    }
  }
}
EOF
    echo "✓ Configuration created"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Installation Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "1. Restart Claude Desktop"
echo "2. The red-team-tools MCP server will connect automatically"
echo "3. You can now use commands like:"
echo "   - 'Find tools for network reconnaissance'"
echo "   - 'What tools can perform SQL injection testing?'"
echo "   - 'Show me all vulnerability assessment tools'"
echo ""
echo "Files created:"
echo "  • MCP Server: $MCP_SERVER"
echo "  • Config: $CONFIG_FILE"
echo "  • Test script: /Users/noone/red-team-tools/test_mcp.sh"
echo ""
