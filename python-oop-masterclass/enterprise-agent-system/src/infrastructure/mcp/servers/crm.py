"""
CRM MCP Server

Provides tools for customer relationship management operations.

Demonstrates:
- MCP server implementation
- External system integration
- Customer data access
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..base import MCPServer, Tool, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


# ============================================================================
# CRM MCP SERVER
# ============================================================================

class CRMMCPServer(MCPServer):
    """MCP server for CRM operations."""

    def __init__(self):
        """Initialize CRM server."""
        super().__init__(
            name="crm",
            description="Customer relationship management operations"
        )

    def _register_tools(self) -> None:
        """Register CRM tools."""
        # Get customer tool
        self.register_tool(Tool(
            name="get_customer",
            description="Get customer information by ID",
            parameters=[
                ToolParameter("customer_id", ToolParameterType.STRING, "Customer ID")
            ],
            handler=self._get_customer
        ))

        # Get customer history tool
        self.register_tool(Tool(
            name="get_customer_history",
            description="Get customer interaction history",
            parameters=[
                ToolParameter("customer_id", ToolParameterType.STRING, "Customer ID"),
                ToolParameter("limit", ToolParameterType.NUMBER, "Maximum results", required=False, default=10)
            ],
            handler=self._get_customer_history
        ))

        # Update customer tool
        self.register_tool(Tool(
            name="update_customer",
            description="Update customer information",
            parameters=[
                ToolParameter("customer_id", ToolParameterType.STRING, "Customer ID"),
                ToolParameter("updates", ToolParameterType.OBJECT, "Fields to update")
            ],
            handler=self._update_customer
        ))

        # Get customer preferences tool
        self.register_tool(Tool(
            name="get_preferences",
            description="Get customer preferences",
            parameters=[
                ToolParameter("customer_id", ToolParameterType.STRING, "Customer ID")
            ],
            handler=self._get_preferences
        ))

    async def _get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Get customer information.

        Args:
            customer_id: Customer ID

        Returns:
            Customer data
        """
        try:
            logger.info(f"Getting customer: {customer_id}")

            # This would query CRM database/API
            # For now, return mock data
            customer = {
                "id": customer_id,
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1234567890",
                "status": "active",
                "tier": "premium",
                "joined_date": "2023-01-15",
                "total_interactions": 42,
                "satisfaction_score": 4.5,
                "last_contact": datetime.now().isoformat(),
                "metadata": {
                    "preferred_channel": "email",
                    "timezone": "America/New_York"
                }
            }

            return {
                "customer": customer,
                "found": True
            }

        except Exception as e:
            logger.error(f"Error getting customer: {e}")
            return {
                "error": str(e),
                "found": False
            }

    async def _get_customer_history(
        self,
        customer_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get customer interaction history.

        Args:
            customer_id: Customer ID
            limit: Maximum results

        Returns:
            Interaction history
        """
        try:
            logger.info(f"Getting history for customer: {customer_id}")

            # This would query interaction database
            # For now, return mock data
            interactions = [
                {
                    "id": f"int-{i}",
                    "type": "support_request",
                    "category": "account" if i % 2 == 0 else "billing",
                    "status": "completed",
                    "created_at": datetime.now().isoformat(),
                    "resolved_at": datetime.now().isoformat(),
                    "resolution_time_minutes": 15,
                    "satisfaction": 4 + (i % 2)
                }
                for i in range(min(limit, 5))
            ]

            return {
                "interactions": interactions,
                "count": len(interactions),
                "customer_id": customer_id
            }

        except Exception as e:
            logger.error(f"Error getting customer history: {e}")
            return {
                "error": str(e),
                "interactions": [],
                "count": 0
            }

    async def _update_customer(
        self,
        customer_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update customer information.

        Args:
            customer_id: Customer ID
            updates: Fields to update

        Returns:
            Update result
        """
        try:
            logger.info(f"Updating customer: {customer_id} with {updates}")

            # This would update CRM database/API
            # For now, just acknowledge

            return {
                "customer_id": customer_id,
                "updated_fields": list(updates.keys()),
                "success": True,
                "updated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error updating customer: {e}")
            return {
                "error": str(e),
                "success": False
            }

    async def _get_preferences(self, customer_id: str) -> Dict[str, Any]:
        """Get customer preferences.

        Args:
            customer_id: Customer ID

        Returns:
            Customer preferences
        """
        try:
            logger.info(f"Getting preferences for customer: {customer_id}")

            # This would query preferences database
            # For now, return mock data
            preferences = {
                "customer_id": customer_id,
                "communication": {
                    "preferred_channel": "email",
                    "frequency": "weekly",
                    "marketing_opt_in": True
                },
                "support": {
                    "preferred_language": "en",
                    "accessibility_needs": [],
                    "callback_hours": "9am-5pm EST"
                },
                "notifications": {
                    "email": True,
                    "sms": False,
                    "push": True
                }
            }

            return {
                "preferences": preferences,
                "found": True
            }

        except Exception as e:
            logger.error(f"Error getting preferences: {e}")
            return {
                "error": str(e),
                "found": False
            }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=== CRM MCP Server Demo ===\n")

        # Create server
        server = CRMMCPServer()

        # List tools
        print("Available Tools:")
        for tool in server.list_tools():
            print(f"  - {tool['name']}: {tool['description']}")

        # Get customer
        print("\nGetting customer...")
        result = await server.execute_tool(
            "get_customer",
            customer_id="cust-123"
        )
        print(f"Customer: {result['customer']['name']}")

        # Get history
        print("\nGetting customer history...")
        result = await server.execute_tool(
            "get_customer_history",
            customer_id="cust-123",
            limit=5
        )
        print(f"Found {result['count']} interactions")

    asyncio.run(main())
