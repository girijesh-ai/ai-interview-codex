#!/usr/bin/env python3
"""Test suite for Weaviate vector database setup and operations.

This module tests the following Weaviate functionality:
    - Connection establishment
    - Collection (schema) creation
    - CRUD operations (Create, Read, Update, Delete)
    - Vector similarity search
    - Filtered queries
    - Batch operations

Run this script to verify your Weaviate instance is working correctly.

Example:
    $ python test_weaviate.py
"""

from typing import List, Dict, Any, Optional
import sys
import traceback

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
import numpy as np


class WeaviateTestSuite:
    """Test suite for Weaviate database operations."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        """Initialize test suite with connection parameters.

        Args:
            host: Weaviate server hostname
            port: Weaviate server port
        """
        self.host = host
        self.port = port
        self.client: Optional[weaviate.WeaviateClient] = None
        self.collection_name = "TestArticle"
        self.vector_dimensions = 384

    def print_section(self, title: str) -> None:
        """Print formatted section header.

        Args:
            title: Section title to display
        """
        separator = "=" * 80
        print(f"\n{separator}")
        print(title)
        print(separator)

    def test_connection(self) -> bool:
        """Test connection to Weaviate instance.

        Returns:
            True if connection successful, False otherwise
        """
        self.print_section("TEST 1: Connection")

        try:
            self.client = weaviate.connect_to_local(host=self.host, port=self.port)

            if self.client.is_ready():
                print("Successfully connected to Weaviate")
                meta = self.client.get_meta()
                print(f"Version: {meta['version']}")
                print(f"Available modules: {list(meta['modules'].keys())}")
                return True
            else:
                print("ERROR: Failed to connect to Weaviate")
                return False

        except Exception as e:
            print(f"ERROR: Connection failed with exception: {e}")
            return False

    def test_create_collection(self) -> bool:
        """Test collection creation with schema definition.

        Returns:
            True if collection created successfully, False otherwise
        """
        self.print_section("TEST 2: Create Collection")

        try:
            if not self.client:
                raise RuntimeError("Client not initialized")

            # Delete existing collection for clean slate
            if self.client.collections.exists(self.collection_name):
                self.client.collections.delete(self.collection_name)
                print(f"Deleted existing '{self.collection_name}' collection")

            # Create new collection
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="views", data_type=DataType.INT)
                ]
            )

            print(f"Created '{self.collection_name}' collection")

            # Verify collection exists
            if self.client.collections.exists(self.collection_name):
                print(f"Verified: '{self.collection_name}' collection exists")
                return True
            else:
                print(f"ERROR: Collection '{self.collection_name}' not found after creation")
                return False

        except Exception as e:
            print(f"ERROR: Collection creation failed: {e}")
            traceback.print_exc()
            return False

    def test_insert_data(self) -> bool:
        """Test batch data insertion.

        Returns:
            True if data inserted successfully, False otherwise
        """
        self.print_section("TEST 3: Insert Data (Batch)")

        try:
            if not self.client:
                raise RuntimeError("Client not initialized")

            collection = self.client.collections.get(self.collection_name)

            sample_data = [
                {
                    "title": "Introduction to Weaviate",
                    "content": "Weaviate is a vector database for semantic search",
                    "category": "Database",
                    "views": 100
                },
                {
                    "title": "HNSW Algorithm Explained",
                    "content": "HNSW is a graph-based algorithm for ANN search",
                    "category": "Algorithms",
                    "views": 200
                },
                {
                    "title": "RAG Systems Guide",
                    "content": "Retrieval-Augmented Generation combines search with LLMs",
                    "category": "AI",
                    "views": 300
                }
            ]

            # Batch insert with random vectors
            with collection.batch.dynamic() as batch:
                for item in sample_data:
                    vector = np.random.rand(self.vector_dimensions).tolist()
                    batch.add_object(properties=item, vector=vector)

            print(f"Inserted {len(sample_data)} objects")

            # Verify count
            result = collection.aggregate.over_all(total_count=True)
            print(f"Total objects in collection: {result.total_count}")

            return result.total_count == len(sample_data)

        except Exception as e:
            print(f"ERROR: Data insertion failed: {e}")
            traceback.print_exc()
            return False

    def test_fetch_data(self) -> bool:
        """Test data retrieval.

        Returns:
            True if data fetched successfully, False otherwise
        """
        self.print_section("TEST 4: Fetch Data")

        try:
            if not self.client:
                raise RuntimeError("Client not initialized")

            collection = self.client.collections.get(self.collection_name)
            response = collection.query.fetch_objects(limit=10)

            print(f"Retrieved {len(response.objects)} objects:\n")

            for i, obj in enumerate(response.objects, 1):
                title = obj.properties['title']
                category = obj.properties['category']
                views = obj.properties['views']
                print(f"{i}. {title}")
                print(f"   Category: {category}, Views: {views}")

            return len(response.objects) > 0

        except Exception as e:
            print(f"ERROR: Data fetch failed: {e}")
            traceback.print_exc()
            return False

    def test_vector_search(self) -> bool:
        """Test vector similarity search.

        Returns:
            True if search executed successfully, False otherwise
        """
        self.print_section("TEST 5: Vector Similarity Search")

        try:
            if not self.client:
                raise RuntimeError("Client not initialized")

            collection = self.client.collections.get(self.collection_name)

            # Generate random query vector
            query_vector = np.random.rand(self.vector_dimensions).tolist()

            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=2,
                return_metadata=['distance']
            )

            print(f"Found {len(response.objects)} similar objects:\n")

            for i, obj in enumerate(response.objects, 1):
                title = obj.properties['title']
                distance = obj.metadata.distance
                print(f"{i}. {title}")
                print(f"   Distance: {distance:.4f}")

            return len(response.objects) > 0

        except Exception as e:
            print(f"ERROR: Vector search failed: {e}")
            traceback.print_exc()
            return False

    def test_filtered_search(self) -> bool:
        """Test filtered queries.

        Returns:
            True if filtered search executed successfully, False otherwise
        """
        self.print_section("TEST 6: Filtered Search")

        try:
            if not self.client:
                raise RuntimeError("Client not initialized")

            collection = self.client.collections.get(self.collection_name)

            # Filter for objects with views > 150
            response = collection.query.fetch_objects(
                filters=Filter.by_property("views").greater_than(150),
                limit=10
            )

            print(f"Found {len(response.objects)} objects with views > 150:\n")

            for obj in response.objects:
                title = obj.properties['title']
                views = obj.properties['views']
                print(f"- {title}: {views} views")

            return True

        except Exception as e:
            print(f"ERROR: Filtered search failed: {e}")
            traceback.print_exc()
            return False

    def test_update(self) -> bool:
        """Test update operations.

        Returns:
            True if update executed successfully, False otherwise
        """
        self.print_section("TEST 7: Update Data")

        try:
            if not self.client:
                raise RuntimeError("Client not initialized")

            collection = self.client.collections.get(self.collection_name)

            # Get first object
            response = collection.query.fetch_objects(limit=1)
            if not response.objects:
                print("ERROR: No objects found to update")
                return False

            obj = response.objects[0]
            original_views = obj.properties['views']

            print(f"Before update: {obj.properties['title']} - {original_views} views")

            # Update views
            new_views = original_views + 50
            collection.data.update(
                uuid=obj.uuid,
                properties={"views": new_views}
            )

            # Fetch again to verify
            updated = collection.query.fetch_object_by_id(obj.uuid)
            updated_views = updated.properties['views']

            print(f"After update: {updated.properties['title']} - {updated_views} views")

            if updated_views == new_views:
                print("Update verified successfully")
                return True
            else:
                print(f"ERROR: Update verification failed. Expected {new_views}, got {updated_views}")
                return False

        except Exception as e:
            print(f"ERROR: Update operation failed: {e}")
            traceback.print_exc()
            return False

    def cleanup(self) -> None:
        """Clean up test resources."""
        self.print_section("CLEANUP")

        try:
            if self.client:
                if self.client.collections.exists(self.collection_name):
                    self.client.collections.delete(self.collection_name)
                    print(f"Deleted '{self.collection_name}' collection")

                self.client.close()
                print("Connection closed")

        except Exception as e:
            print(f"ERROR during cleanup: {e}")

    def run_all_tests(self) -> bool:
        """Run all tests in sequence.

        Returns:
            True if all tests passed, False otherwise
        """
        print("\nStarting Weaviate Test Suite")
        print("=" * 80)

        test_results = []

        try:
            # Test connection
            if not self.test_connection():
                print("\nERROR: Connection test failed. Ensure Weaviate is running:")
                print("  cd gen-ai/weaviate-setup && docker-compose up -d")
                return False

            # Run all tests
            test_results.append(("Create Collection", self.test_create_collection()))
            test_results.append(("Insert Data", self.test_insert_data()))
            test_results.append(("Fetch Data", self.test_fetch_data()))
            test_results.append(("Vector Search", self.test_vector_search()))
            test_results.append(("Filtered Search", self.test_filtered_search()))
            test_results.append(("Update", self.test_update()))

            # Print summary
            self.print_section("TEST SUMMARY")

            all_passed = True
            for test_name, result in test_results:
                status = "PASSED" if result else "FAILED"
                print(f"{test_name}: {status}")
                if not result:
                    all_passed = False

            print("\n" + "=" * 80)
            if all_passed:
                print("ALL TESTS PASSED")
            else:
                print("SOME TESTS FAILED")
            print("=" * 80)

            return all_passed

        except Exception as e:
            print(f"\nERROR: Test suite failed with exception: {e}")
            traceback.print_exc()
            return False

        finally:
            self.cleanup()


def main() -> int:
    """Main entry point for test suite.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    test_suite = WeaviateTestSuite()
    success = test_suite.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
