#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Unit tests for Red Team Tools MCP Server
"""

import unittest
import json
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server import (
    RedTeamMCPServer,
    SemanticLattice,
    SemanticNode,
    MCPResource,
    MCPTool
)


class TestSemanticLattice(unittest.TestCase):
    """Tests for SemanticLattice class"""

    def setUp(self):
        self.lattice = SemanticLattice()

    def test_lattice_initialization(self):
        """Test lattice builds with nodes"""
        self.assertGreater(len(self.lattice.nodes), 0)

    def test_get_node_exists(self):
        """Test retrieving existing node"""
        node = self.lattice.get_node("aurorascan")
        self.assertIsNotNone(node)
        self.assertEqual(node.name, "AuroraScan")
        self.assertEqual(node.type, "tool")

    def test_get_node_not_exists(self):
        """Test retrieving non-existent node"""
        node = self.lattice.get_node("nonexistent_tool")
        self.assertIsNone(node)

    def test_find_tools_by_capability(self):
        """Test finding tools by capability"""
        tools = self.lattice.find_tools_by_capability("port_scanning")
        self.assertGreater(len(tools), 0)
        # All returned nodes should have port_scanning capability
        for tool in tools:
            self.assertIn("port_scanning", tool.attributes.get("capabilities", []))

    def test_find_tools_by_capability_none_found(self):
        """Test finding tools with non-existent capability"""
        tools = self.lattice.find_tools_by_capability("quantum_teleportation")
        self.assertEqual(len(tools), 0)

    def test_find_tools_by_phase(self):
        """Test finding tools by attack phase"""
        tools = self.lattice.find_tools_by_phase("reconnaissance")
        self.assertGreater(len(tools), 0)
        # All returned nodes should operate in reconnaissance phase
        for tool in tools:
            self.assertIn("reconnaissance", tool.attributes.get("phases", []))

    def test_find_tools_by_technique(self):
        """Test finding tools by technique"""
        tools = self.lattice.find_tools_by_technique("tcp_syn_scan")
        self.assertGreater(len(tools), 0)

    def test_export_graph(self):
        """Test exporting lattice as graph"""
        graph = self.lattice.export_graph()
        self.assertIn("nodes", graph)
        self.assertIn("metadata", graph)
        self.assertGreater(graph["metadata"]["total_nodes"], 0)
        self.assertGreater(graph["metadata"]["tool_count"], 0)

    def test_tool_categories(self):
        """Test that all tools have valid category relationships"""
        tools = [n for n in self.lattice.nodes.values() if n.type == "tool"]
        for tool in tools:
            self.assertIn("parent", tool.relationships)
            self.assertGreater(len(tool.relationships["parent"]), 0)


class TestRedTeamMCPServer(unittest.TestCase):
    """Tests for RedTeamMCPServer class"""

    def setUp(self):
        self.server = RedTeamMCPServer()

    def test_server_initialization(self):
        """Test server initializes with correct info"""
        self.assertEqual(self.server.server_info["name"], "red-team-tools-mcp")
        self.assertEqual(self.server.server_info["version"], "1.0.0")

    def test_handle_initialize(self):
        """Test initialize handler"""
        result = asyncio.run(self.server.handle_initialize({}))
        self.assertIn("protocolVersion", result)
        self.assertIn("capabilities", result)
        self.assertIn("serverInfo", result)
        self.assertTrue(result["capabilities"]["resources"]["subscribe"])

    def test_handle_list_resources(self):
        """Test list resources handler"""
        result = asyncio.run(self.server.handle_list_resources({}))
        self.assertIn("resources", result)
        self.assertGreater(len(result["resources"]), 0)
        # Check first resource has required fields
        first_resource = result["resources"][0]
        self.assertIn("uri", first_resource)
        self.assertIn("name", first_resource)
        self.assertIn("description", first_resource)

    def test_handle_read_resource_semantic_graph(self):
        """Test reading semantic graph resource"""
        result = asyncio.run(self.server.handle_read_resource({
            "uri": "lattice://semantic-graph"
        }))
        self.assertIn("contents", result)
        self.assertEqual(len(result["contents"]), 1)
        content = json.loads(result["contents"][0]["text"])
        self.assertIn("nodes", content)
        self.assertIn("metadata", content)

    def test_handle_read_resource_tool(self):
        """Test reading tool resource"""
        result = asyncio.run(self.server.handle_read_resource({
            "uri": "lattice://tools/aurorascan"
        }))
        self.assertIn("contents", result)
        content = json.loads(result["contents"][0]["text"])
        self.assertEqual(content["name"], "AuroraScan")
        self.assertIn("capabilities", content)

    def test_handle_read_resource_not_found(self):
        """Test reading non-existent resource"""
        result = asyncio.run(self.server.handle_read_resource({
            "uri": "lattice://tools/nonexistent"
        }))
        self.assertIn("error", result)

    def test_handle_list_tools(self):
        """Test list tools handler"""
        result = asyncio.run(self.server.handle_list_tools({}))
        self.assertIn("tools", result)
        tool_names = [t["name"] for t in result["tools"]]
        self.assertIn("find_tools_by_capability", tool_names)
        self.assertIn("find_tools_by_phase", tool_names)
        self.assertIn("get_tool_info", tool_names)
        self.assertIn("navigate_lattice", tool_names)

    def test_handle_call_tool_find_by_capability(self):
        """Test calling find_tools_by_capability"""
        result = asyncio.run(self.server.handle_call_tool({
            "name": "find_tools_by_capability",
            "arguments": {"capability": "port_scanning"}
        }))
        self.assertIn("content", result)
        content = json.loads(result["content"][0]["text"])
        self.assertEqual(content["capability"], "port_scanning")
        self.assertGreater(len(content["tools"]), 0)

    def test_handle_call_tool_find_by_phase(self):
        """Test calling find_tools_by_phase"""
        result = asyncio.run(self.server.handle_call_tool({
            "name": "find_tools_by_phase",
            "arguments": {"phase": "reconnaissance"}
        }))
        self.assertIn("content", result)
        content = json.loads(result["content"][0]["text"])
        self.assertEqual(content["phase"], "reconnaissance")
        self.assertGreater(len(content["tools"]), 0)

    def test_handle_call_tool_get_info(self):
        """Test calling get_tool_info"""
        result = asyncio.run(self.server.handle_call_tool({
            "name": "get_tool_info",
            "arguments": {"tool_id": "aurorascan"}
        }))
        self.assertIn("content", result)
        content = json.loads(result["content"][0]["text"])
        self.assertEqual(content["name"], "AuroraScan")

    def test_handle_call_tool_unknown(self):
        """Test calling unknown tool"""
        result = asyncio.run(self.server.handle_call_tool({
            "name": "unknown_tool",
            "arguments": {}
        }))
        self.assertIn("error", result)

    def test_handle_message_routing(self):
        """Test message routing"""
        # Test initialize
        result = asyncio.run(self.server.handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }))
        self.assertEqual(result["jsonrpc"], "2.0")
        self.assertEqual(result["id"], 1)
        self.assertIn("result", result)

    def test_handle_message_unknown_method(self):
        """Test handling unknown method"""
        result = asyncio.run(self.server.handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown/method",
            "params": {}
        }))
        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], -32601)


class TestMCPDataclasses(unittest.TestCase):
    """Tests for MCP dataclasses"""

    def test_mcp_resource(self):
        """Test MCPResource dataclass"""
        resource = MCPResource(
            uri="test://resource",
            name="Test Resource",
            description="A test resource"
        )
        self.assertEqual(resource.uri, "test://resource")
        self.assertEqual(resource.mimeType, "application/json")

    def test_mcp_tool(self):
        """Test MCPTool dataclass"""
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            inputSchema={"type": "object"}
        )
        self.assertEqual(tool.name, "test_tool")

    def test_semantic_node(self):
        """Test SemanticNode dataclass"""
        node = SemanticNode(
            id="test",
            type="tool",
            name="Test Tool",
            description="A test tool",
            attributes={"key": "value"},
            relationships={"parent": ["root"]}
        )
        self.assertEqual(node.id, "test")
        self.assertEqual(node.attributes["key"], "value")


class TestToolCoverage(unittest.TestCase):
    """Tests to ensure all expected tools are present"""

    def setUp(self):
        self.lattice = SemanticLattice()
        self.expected_tools = [
            "aurorascan", "cipherspear", "skybreaker", "mythickey",
            "spectratrace", "nemesishydra", "obsidianhunt", "vectorflux",
            "nmappro", "payloadforge", "dirreaper", "proxyphantom",
            "osintworkflows", "scribe", "vulnhunter", "sovereign_suite"
        ]

    def test_all_expected_tools_exist(self):
        """Test all expected tools are in the lattice"""
        for tool_id in self.expected_tools:
            node = self.lattice.get_node(tool_id)
            self.assertIsNotNone(node, f"Tool {tool_id} not found in lattice")
            self.assertEqual(node.type, "tool")

    def test_tool_count(self):
        """Test correct number of tools"""
        tools = [n for n in self.lattice.nodes.values() if n.type == "tool"]
        self.assertEqual(len(tools), len(self.expected_tools))


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
