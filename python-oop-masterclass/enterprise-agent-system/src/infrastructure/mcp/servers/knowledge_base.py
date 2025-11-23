"""
Knowledge Base MCP Server

Provides tools for searching and retrieving knowledge base articles.

Demonstrates:
- MCP server implementation
- Vector DB integration
- Tool pattern
"""

from typing import Dict, Any, List, Optional
import logging

from ..base import MCPServer, Tool, ToolParameter, ToolParameterType
from ....memory.vector_store import VectorStoreFactory

logger = logging.getLogger(__name__)


# ============================================================================
# KNOWLEDGE BASE MCP SERVER
# ============================================================================

class KnowledgeBaseMCPServer(MCPServer):
    """MCP server for knowledge base operations."""

    def __init__(self, vector_store_url: str = "http://localhost:8080"):
        """Initialize knowledge base server.

        Args:
            vector_store_url: Vector store URL
        """
        self.vector_store = VectorStoreFactory.create("weaviate", vector_store_url)
        super().__init__(
            name="knowledge_base",
            description="Search and retrieve knowledge base articles"
        )

    def _register_tools(self) -> None:
        """Register knowledge base tools."""
        # Search articles tool
        self.register_tool(Tool(
            name="search_articles",
            description="Search knowledge base articles using semantic search",
            parameters=[
                ToolParameter("query", ToolParameterType.STRING, "Search query"),
                ToolParameter("limit", ToolParameterType.NUMBER, "Maximum results", required=False, default=5),
                ToolParameter("category", ToolParameterType.STRING, "Filter by category", required=False)
            ],
            handler=self._search_articles
        ))

        # Get article tool
        self.register_tool(Tool(
            name="get_article",
            description="Get a specific article by ID",
            parameters=[
                ToolParameter("article_id", ToolParameterType.STRING, "Article ID")
            ],
            handler=self._get_article
        ))

        # List categories tool
        self.register_tool(Tool(
            name="list_categories",
            description="List all article categories",
            parameters=[],
            handler=self._list_categories
        ))

    async def _search_articles(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search articles.

        Args:
            query: Search query
            limit: Maximum results
            category: Optional category filter

        Returns:
            Search results
        """
        try:
            logger.info(f"Searching KB: {query} (limit={limit}, category={category})")

            # Build filters
            filters = {}
            if category:
                filters["category"] = category

            # Search vector DB
            results = await self.vector_store.similarity_search(
                query,
                k=limit,
                namespace="knowledge_base",
                filters=filters if filters else None
            )

            # Format results
            articles = [
                {
                    "id": r.id,
                    "title": r.metadata.get("title", "Untitled"),
                    "content": r.content,
                    "category": r.metadata.get("category", "general"),
                    "relevance_score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ]

            return {
                "articles": articles,
                "count": len(articles),
                "query": query
            }

        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return {
                "error": str(e),
                "articles": [],
                "count": 0
            }

    async def _get_article(self, article_id: str) -> Dict[str, Any]:
        """Get specific article.

        Args:
            article_id: Article ID

        Returns:
            Article data
        """
        try:
            logger.info(f"Getting article: {article_id}")

            # Get from vector DB
            doc = await self.vector_store.get_document(article_id, namespace="knowledge_base")

            if doc:
                return {
                    "id": doc.id,
                    "title": doc.metadata.get("title", "Untitled"),
                    "content": doc.content,
                    "category": doc.metadata.get("category", "general"),
                    "metadata": doc.metadata,
                    "found": True
                }
            else:
                return {
                    "error": "Article not found",
                    "found": False
                }

        except Exception as e:
            logger.error(f"Error getting article: {e}")
            return {
                "error": str(e),
                "found": False
            }

    async def _list_categories(self) -> Dict[str, Any]:
        """List all categories.

        Returns:
            Category list
        """
        try:
            logger.info("Listing KB categories")

            # This would query unique categories from vector DB
            # For now, return common categories
            categories = [
                {"name": "account", "description": "Account management"},
                {"name": "billing", "description": "Billing and payments"},
                {"name": "technical", "description": "Technical support"},
                {"name": "product", "description": "Product information"},
                {"name": "general", "description": "General questions"}
            ]

            return {
                "categories": categories,
                "count": len(categories)
            }

        except Exception as e:
            logger.error(f"Error listing categories: {e}")
            return {
                "error": str(e),
                "categories": [],
                "count": 0
            }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=== Knowledge Base MCP Server Demo ===\n")

        # Create server
        server = KnowledgeBaseMCPServer()

        # List tools
        print("Available Tools:")
        for tool in server.list_tools():
            print(f"  - {tool['name']}: {tool['description']}")

        # Search articles
        print("\nSearching articles...")
        result = await server.execute_tool(
            "search_articles",
            query="password reset",
            limit=3
        )
        print(f"Found {result['count']} articles")

        # List categories
        print("\nListing categories...")
        result = await server.execute_tool("list_categories")
        print(f"Found {result['count']} categories")

    asyncio.run(main())
