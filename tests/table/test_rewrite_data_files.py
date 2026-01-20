# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from unittest.mock import Mock

import pyarrow as pa
import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
from pyiceberg.table.update.snapshot import FileGroup, RewriteDataFiles, RewriteDataFilesResult
from pyiceberg.typedef import Record


def test_rewrite_data_files_result_defaults() -> None:
    """Test that RewriteDataFilesResult has correct default values."""
    result = RewriteDataFilesResult()
    assert result.rewritten_data_files_count == 0
    assert result.added_data_files_count == 0
    assert result.rewritten_bytes == 0
    assert result.failed_group_count == 0


def test_file_group_properties() -> None:
    """Test FileGroup properties."""
    # Create mock data files
    file1 = DataFile.from_args(
        file_path="/path/to/file1.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=1000,
        record_count=100,
    )
    file2 = DataFile.from_args(
        file_path="/path/to/file2.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=2000,
        record_count=200,
    )

    group = FileGroup(data_files=[file1, file2])

    assert group.file_count == 2
    assert group.total_size_bytes == 3000


def test_file_group_empty() -> None:
    """Test FileGroup with no files."""
    group = FileGroup()
    assert group.file_count == 0
    assert group.total_size_bytes == 0


def test_rewrite_data_files_option() -> None:
    """Test setting options on RewriteDataFiles."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)

    # Test single option
    result = rewrite.option("target-file-size-bytes", "134217728")
    assert result is rewrite  # Method chaining
    assert rewrite._options["target-file-size-bytes"] == "134217728"


def test_rewrite_data_files_options() -> None:
    """Test setting multiple options on RewriteDataFiles."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)

    # Test multiple options
    result = rewrite.options({
        "target-file-size-bytes": "134217728",
        "min-input-files": "3",
    })
    assert result is rewrite  # Method chaining
    assert rewrite._options["target-file-size-bytes"] == "134217728"
    assert rewrite._options["min-input-files"] == "3"


def test_rewrite_data_files_target_file_size_default() -> None:
    """Test default target file size."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)

    # Default is 512MB
    assert rewrite._target_file_size == 512 * 1024 * 1024


def test_rewrite_data_files_target_file_size_custom() -> None:
    """Test custom target file size."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)
    rewrite.option("target-file-size-bytes", "134217728")

    assert rewrite._target_file_size == 134217728  # 128MB


def test_rewrite_data_files_min_file_size_default() -> None:
    """Test default minimum file size (75% of target)."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)

    # Default is 75% of 512MB = 384MB
    expected = int(512 * 1024 * 1024 * 0.75)
    assert rewrite._min_file_size == expected


def test_rewrite_data_files_min_file_size_custom() -> None:
    """Test custom minimum file size."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)
    rewrite.option("min-file-size-bytes", "100000000")

    assert rewrite._min_file_size == 100000000


def test_rewrite_data_files_max_file_size_default() -> None:
    """Test default maximum file size (180% of target)."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)

    # Default is 180% of 512MB
    expected = int(512 * 1024 * 1024 * 1.8)
    assert rewrite._max_file_size == expected


def test_rewrite_data_files_max_file_size_custom() -> None:
    """Test custom maximum file size."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)
    rewrite.option("max-file-size-bytes", "1000000000")

    assert rewrite._max_file_size == 1000000000


def test_rewrite_data_files_min_input_files_default() -> None:
    """Test default minimum input files."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)

    assert rewrite._min_input_files == 2


def test_rewrite_data_files_min_input_files_custom() -> None:
    """Test custom minimum input files."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)
    rewrite.option("min-input-files", "5")

    assert rewrite._min_input_files == 5


def test_should_rewrite_small_file() -> None:
    """Test that small files are candidates for rewrite."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)
    # Set a small target so our test file is considered "small"
    rewrite.option("target-file-size-bytes", "10000000")  # 10MB target

    small_file = DataFile.from_args(
        file_path="/path/to/small.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=1000000,  # 1MB, below 75% of 10MB (7.5MB)
        record_count=100,
    )

    assert rewrite._should_rewrite(small_file) is True


def test_should_rewrite_large_file() -> None:
    """Test that large files are candidates for rewrite."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)
    # Set a small target so our test file is considered "large"
    rewrite.option("target-file-size-bytes", "10000000")  # 10MB target

    large_file = DataFile.from_args(
        file_path="/path/to/large.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=20000000,  # 20MB, above 180% of 10MB (18MB)
        record_count=2000,
    )

    assert rewrite._should_rewrite(large_file) is True


def test_should_not_rewrite_optimal_file() -> None:
    """Test that optimally-sized files are not candidates for rewrite."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)
    rewrite.option("target-file-size-bytes", "10000000")  # 10MB target

    optimal_file = DataFile.from_args(
        file_path="/path/to/optimal.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=10000000,  # 10MB, exactly at target
        record_count=1000,
    )

    assert rewrite._should_rewrite(optimal_file) is False


def test_group_files_by_partition_unpartitioned() -> None:
    """Test grouping files for unpartitioned table."""
    file1 = DataFile.from_args(
        file_path="/path/to/file1.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=1000,
        record_count=100,
    )
    file2 = DataFile.from_args(
        file_path="/path/to/file2.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=2000,
        record_count=200,
    )

    # Both files have no partition (None)
    groups = RewriteDataFiles._group_files_by_partition([file1, file2])

    # All files should be in one group
    assert len(groups) == 1
    group_files = list(groups.values())[0]
    assert len(group_files) == 2
    assert file1 in group_files
    assert file2 in group_files


def test_group_files_by_partition_partitioned() -> None:
    """Test grouping files for partitioned table."""
    # Create mock files with different partitions
    partition1 = Record(2024, 1)  # year=2024, month=1
    partition2 = Record(2024, 2)  # year=2024, month=2

    file1 = DataFile.from_args(
        file_path="/path/to/file1.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=1000,
        record_count=100,
        partition=partition1,
    )
    file2 = DataFile.from_args(
        file_path="/path/to/file2.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=2000,
        record_count=200,
        partition=partition1,
    )
    file3 = DataFile.from_args(
        file_path="/path/to/file3.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=3000,
        record_count=300,
        partition=partition2,
    )

    groups = RewriteDataFiles._group_files_by_partition([file1, file2, file3])

    # Should have 2 partition groups
    assert len(groups) == 2

    # Find the partition1 group
    partition1_files = groups[partition1]
    assert len(partition1_files) == 2
    assert file1 in partition1_files
    assert file2 in partition1_files

    # Find the partition2 group
    partition2_files = groups[partition2]
    assert len(partition2_files) == 1
    assert file3 in partition2_files


def test_bin_pack_groups_filters_small_groups() -> None:
    """Test that groups with fewer than min_input_files are filtered out."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)
    rewrite.option("min-input-files", "3")  # Require at least 3 files
    rewrite.option("max-file-group-size-bytes", "1000000000")  # 1GB max group size

    # Create a small partition with only 2 files
    file1 = DataFile.from_args(
        file_path="/path/to/file1.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=1000,
        record_count=100,
    )
    file2 = DataFile.from_args(
        file_path="/path/to/file2.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=2000,
        record_count=200,
    )

    partition_groups = {Record(): [file1, file2]}
    file_groups = rewrite._bin_pack_groups(partition_groups)

    # Group should be filtered out since it has only 2 files (< 3)
    assert len(file_groups) == 0


def test_bin_pack_groups_includes_large_groups() -> None:
    """Test that groups with sufficient files are included."""
    mock_transaction = Mock()
    mock_transaction.table_metadata = Mock()

    rewrite = RewriteDataFiles(transaction=mock_transaction)
    rewrite.option("min-input-files", "2")  # Require at least 2 files
    rewrite.option("max-file-group-size-bytes", "1000000000")  # 1GB max group size

    file1 = DataFile.from_args(
        file_path="/path/to/file1.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=1000,
        record_count=100,
    )
    file2 = DataFile.from_args(
        file_path="/path/to/file2.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=2000,
        record_count=200,
    )
    file3 = DataFile.from_args(
        file_path="/path/to/file3.parquet",
        file_format=FileFormat.PARQUET,
        file_size_in_bytes=3000,
        record_count=300,
    )

    partition_groups = {Record(): [file1, file2, file3]}
    file_groups = rewrite._bin_pack_groups(partition_groups)

    # Should have at least one group
    assert len(file_groups) >= 1
    # Total files should be 3
    total_files = sum(g.file_count for g in file_groups)
    assert total_files == 3


def test_commit_returns_empty_result_when_no_snapshot(table_v2: "Table") -> None:
    """Test that commit returns empty result when table has no snapshot."""
    from pyiceberg.table import Table

    # Mock the table metadata to have no current snapshot
    table_v2.metadata = table_v2.metadata.model_copy(
        update={"current_snapshot_id": None, "snapshots": []}
    )

    result = table_v2.maintenance.rewrite_data_files().commit()

    assert result.rewritten_data_files_count == 0
    assert result.added_data_files_count == 0
    assert result.rewritten_bytes == 0


@pytest.fixture
def table_v2():
    """Create a minimal table fixture for testing."""
    from pyiceberg.catalog.noop import NoopCatalog
    from pyiceberg.io import load_file_io
    from pyiceberg.table import Table
    from pyiceberg.table.metadata import TableMetadataV2

    # Minimal v2 metadata
    metadata = TableMetadataV2(
        format_version=2,
        table_uuid="9c12d441-03fe-4693-9a96-a0705ddf69c1",
        location="s3://bucket/test/location",
        last_updated_ms=1602638573590,
        last_column_id=3,
        schemas=[
            {
                "schema_id": 0,
                "type": "struct",
                "fields": [
                    {"id": 1, "name": "x", "required": True, "type": "long"},
                    {"id": 2, "name": "y", "required": True, "type": "long"},
                    {"id": 3, "name": "z", "required": True, "type": "long"},
                ],
            }
        ],
        current_schema_id=0,
        partition_specs=[{"spec_id": 0, "fields": []}],
        default_spec_id=0,
        last_partition_id=1000,
        sort_orders=[{"order_id": 0, "fields": []}],
        default_sort_order_id=0,
        properties={},
        current_snapshot_id=None,
        snapshots=[],
    )

    return Table(
        identifier=("database", "table"),
        metadata=metadata,
        metadata_location="s3://bucket/test/location/metadata/v1.metadata.json",
        io=load_file_io(),
        catalog=NoopCatalog("NoopCatalog"),
    )


def test_maintenance_rewrite_data_files_method_exists(table_v2: "Table") -> None:
    """Test that rewrite_data_files method exists on maintenance."""
    maintenance = table_v2.maintenance
    assert hasattr(maintenance, "rewrite_data_files")


def test_maintenance_rewrite_data_files_returns_builder(table_v2: "Table") -> None:
    """Test that rewrite_data_files returns a RewriteDataFiles builder."""
    builder = table_v2.maintenance.rewrite_data_files()
    assert isinstance(builder, RewriteDataFiles)


def test_maintenance_rewrite_data_files_chaining(table_v2: "Table") -> None:
    """Test method chaining on RewriteDataFiles builder."""
    builder = (
        table_v2.maintenance
        .rewrite_data_files()
        .option("target-file-size-bytes", "134217728")
        .option("min-input-files", "3")
    )
    assert isinstance(builder, RewriteDataFiles)
    assert builder._options["target-file-size-bytes"] == "134217728"
    assert builder._options["min-input-files"] == "3"


# Integration tests that test the full commit path
@pytest.fixture
def catalog_with_table(tmp_path):
    """Create a catalog with a table containing multiple small files."""
    from pyiceberg.catalog.memory import InMemoryCatalog

    catalog = InMemoryCatalog("test_catalog", warehouse=str(tmp_path))
    catalog.create_namespace("default")
    return catalog


def test_rewrite_data_files_full_commit_path(catalog_with_table, tmp_path) -> None:
    """Integration test: create table with small files, run rewrite, verify consolidation."""
    # Create a simple schema
    schema = pa.schema([
        ("id", pa.int64()),
        ("name", pa.string()),
    ])

    # Create table
    table = catalog_with_table.create_table(
        "default.test_rewrite",
        schema=schema,
    )

    # Append multiple small batches to create multiple small files
    for i in range(3):
        small_data = pa.table({
            "id": [i * 10 + j for j in range(10)],
            "name": [f"name_{i * 10 + j}" for j in range(10)],
        })
        table.append(small_data)

    # Verify we have multiple files
    files_before = list(table.scan().plan_files())
    assert len(files_before) == 3, f"Expected 3 files, got {len(files_before)}"

    # Run rewrite with very small target size to ensure files are candidates
    # and min-input-files=2 to allow grouping
    result = (
        table.maintenance
        .rewrite_data_files()
        .option("target-file-size-bytes", "1000000000")  # 1GB target (larger than our files)
        .option("min-file-size-bytes", "100000000")  # 100MB min (larger than our files)
        .option("min-input-files", "2")
        .commit()
    )

    # Verify the result
    assert result.rewritten_data_files_count == 3, f"Expected 3 files rewritten, got {result.rewritten_data_files_count}"
    assert result.added_data_files_count >= 1, f"Expected at least 1 file added, got {result.added_data_files_count}"
    assert result.failed_group_count == 0, f"Expected 0 failed groups, got {result.failed_group_count}"

    # Verify data integrity - should still have all 30 rows
    result_table = table.scan().to_arrow()
    assert result_table.num_rows == 30, f"Expected 30 rows, got {result_table.num_rows}"

    # Verify files were consolidated
    files_after = list(table.scan().plan_files())
    assert len(files_after) < len(files_before), f"Expected fewer files after rewrite: {len(files_after)} vs {len(files_before)}"


def test_rewrite_data_files_no_candidates(catalog_with_table, tmp_path) -> None:
    """Integration test: verify no rewrite when files are within size thresholds."""
    schema = pa.schema([
        ("id", pa.int64()),
        ("name", pa.string()),
    ])

    table = catalog_with_table.create_table(
        "default.test_rewrite_no_candidates",
        schema=schema,
    )

    # Append a single batch
    data = pa.table({
        "id": [i for i in range(100)],
        "name": [f"name_{i}" for i in range(100)],
    })
    table.append(data)

    # Run rewrite with default thresholds - the file should be too small to be a candidate
    # but we only have 1 file so min_input_files won't be met anyway
    result = (
        table.maintenance
        .rewrite_data_files()
        .option("min-input-files", "2")
        .commit()
    )

    # No files should be rewritten because we only have 1 file
    assert result.rewritten_data_files_count == 0
    assert result.added_data_files_count == 0


def test_rewrite_data_files_empty_table(catalog_with_table, tmp_path) -> None:
    """Integration test: verify rewrite handles empty table gracefully."""
    schema = pa.schema([
        ("id", pa.int64()),
    ])

    table = catalog_with_table.create_table(
        "default.test_rewrite_empty",
        schema=schema,
    )

    # Table has no data, no snapshot
    result = table.maintenance.rewrite_data_files().commit()

    assert result.rewritten_data_files_count == 0
    assert result.added_data_files_count == 0
    assert result.failed_group_count == 0
