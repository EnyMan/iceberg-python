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
import functools
import logging
import operator
import time

logger = logging.getLogger(__name__)

import pyarrow as pa
from pyarrow import Table as pyarrow_table
from pyarrow import compute as pc

from pyiceberg.expressions import (
    AlwaysFalse,
    AlwaysTrue,
    And,
    BooleanExpression,
    EqualTo,
    GreaterThanOrEqual,
    In,
    LessThanOrEqual,
    Or,
)

# Threshold for switching from In() predicate to range-based or no filter
# When unique keys exceed this, the In() predicate becomes too expensive to process
LARGE_FILTER_THRESHOLD = 10_000


def create_match_filter(df: pyarrow_table, join_cols: list[str]) -> BooleanExpression:
    """
    Create an Iceberg BooleanExpression filter that exactly matches rows based on join columns.

    For single-column keys, uses an efficient In() predicate.
    For composite keys, creates Or(And(...), And(...), ...) for exact row matching.
    This function should be used when exact matching is required (e.g., overwrite, insert filtering).
    """
    unique_keys = df.select(join_cols).group_by(join_cols).aggregate([])

    if len(unique_keys) == 0:
        return AlwaysFalse()

    if len(join_cols) == 1:
        return In(join_cols[0], unique_keys[0].to_pylist())
    else:
        filters = [
            functools.reduce(operator.and_, [EqualTo(col, row[col]) for col in join_cols]) for row in unique_keys.to_pylist()
        ]

        if len(filters) == 0:
            return AlwaysFalse()
        elif len(filters) == 1:
            return filters[0]
        else:
            return Or(*filters)


def _is_numeric_type(arrow_type: pa.DataType) -> bool:
    """Check if a PyArrow type is numeric (suitable for range filtering)."""
    return pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)


def _create_range_filter(col_name: str, values: pa.Array) -> BooleanExpression:
    """Create a min/max range filter for a numeric column."""
    min_val = pc.min(values).as_py()
    max_val = pc.max(values).as_py()
    return And(GreaterThanOrEqual(col_name, min_val), LessThanOrEqual(col_name, max_val))


def create_coarse_match_filter(df: pyarrow_table, join_cols: list[str]) -> BooleanExpression:
    """
    Create a coarse Iceberg BooleanExpression filter for initial row scanning.

    For small datasets (< LARGE_FILTER_THRESHOLD unique keys):
      - Single-column keys: uses In() predicate
      - Composite keys: uses AND of In() predicates per column

    For large datasets (>= LARGE_FILTER_THRESHOLD unique keys):
      - Single numeric column with dense IDs: uses min/max range filter
      - Otherwise: returns AlwaysTrue() to skip filtering (full scan)

    This function should only be used for initial scans where exact matching happens
    downstream (e.g., in get_rows_to_update() via the join operation).
    """
    unique_keys = df.select(join_cols).group_by(join_cols).aggregate([])
    num_unique_keys = len(unique_keys)

    if num_unique_keys == 0:
        return AlwaysFalse()

    # For small datasets, use the standard In() approach
    if num_unique_keys < LARGE_FILTER_THRESHOLD:
        if len(join_cols) == 1:
            return In(join_cols[0], unique_keys[0].to_pylist())
        else:
            column_filters = []
            for col in join_cols:
                unique_values = pc.unique(unique_keys[col]).to_pylist()
                column_filters.append(In(col, unique_values))
            return functools.reduce(operator.and_, column_filters)

    # For large datasets, use optimized strategies
    logger.info(
        f"Large dataset detected ({num_unique_keys} unique keys >= {LARGE_FILTER_THRESHOLD} threshold), "
        "using optimized filter strategy"
    )

    if len(join_cols) == 1:
        col_name = join_cols[0]
        col_data = unique_keys[col_name]
        col_type = col_data.type

        # For numeric columns, check if range filter is efficient (dense IDs)
        if _is_numeric_type(col_type):
            min_val = pc.min(col_data).as_py()
            max_val = pc.max(col_data).as_py()
            value_range = max_val - min_val + 1
            density = num_unique_keys / value_range if value_range > 0 else 0

            # If IDs are dense (>10% coverage of the range), use range filter
            # Otherwise, range filter would read too much irrelevant data
            if density > 0.1:
                logger.info(
                    f"Using range filter for column '{col_name}': "
                    f"min={min_val}, max={max_val}, density={density:.2%}"
                )
                return _create_range_filter(col_name, col_data)
            else:
                logger.info(
                    f"Skipping filter (sparse IDs, density={density:.2%}): "
                    f"full scan will be performed"
                )
                return AlwaysTrue()
        else:
            # Non-numeric single column with many values - skip filter
            logger.info(
                f"Skipping filter for non-numeric column '{col_name}' with {num_unique_keys} values: "
                "full scan will be performed"
            )
            return AlwaysTrue()
    else:
        # Composite keys with many values - use range filters for numeric columns where possible
        column_filters = []
        for col in join_cols:
            col_data = unique_keys[col]
            col_type = col_data.type
            unique_values = pc.unique(col_data)

            if _is_numeric_type(col_type) and len(unique_values) >= LARGE_FILTER_THRESHOLD:
                # Use range filter for large numeric columns
                min_val = pc.min(unique_values).as_py()
                max_val = pc.max(unique_values).as_py()
                value_range = max_val - min_val + 1
                density = len(unique_values) / value_range if value_range > 0 else 0

                if density > 0.1:
                    logger.info(f"Using range filter for composite key column '{col}': density={density:.2%}")
                    column_filters.append(_create_range_filter(col, unique_values))
                else:
                    # Sparse numeric column - still use In() as it's part of composite key
                    column_filters.append(In(col, unique_values.to_pylist()))
            else:
                # Small column or non-numeric - use In()
                column_filters.append(In(col, unique_values.to_pylist()))

        if len(column_filters) == 0:
            return AlwaysTrue()
        return functools.reduce(operator.and_, column_filters)


def has_duplicate_rows(df: pyarrow_table, join_cols: list[str]) -> bool:
    """Check for duplicate rows in a PyArrow table based on the join columns."""
    return len(df.select(join_cols).group_by(join_cols).aggregate([([], "count_all")]).filter(pc.field("count_all") > 1)) > 0


def _compare_columns_vectorized(
    source_col: pa.Array | pa.ChunkedArray, target_col: pa.Array | pa.ChunkedArray
) -> pa.Array:
    """
    Vectorized comparison of two columns, returning a boolean array where True means values differ.

    Handles struct types recursively by comparing each nested field.
    Handles null values correctly: null != non-null is True, null == null is True (no update needed).
    """
    col_type = source_col.type

    if pa.types.is_struct(col_type):
        # Handle struct-level nulls first
        source_null = pc.is_null(source_col)
        target_null = pc.is_null(target_col)
        struct_null_diff = pc.xor(source_null, target_null)  # Different if exactly one is null

        # PyArrow cannot directly compare struct columns, so we recursively compare each field
        diff_masks = []
        for i, field in enumerate(col_type):
            src_field = pc.struct_field(source_col, [i])
            tgt_field = pc.struct_field(target_col, [i])
            field_diff = _compare_columns_vectorized(src_field, tgt_field)
            diff_masks.append(field_diff)

        if not diff_masks:
            # Empty struct - only null differences matter
            return struct_null_diff

        # Combine field differences with struct-level null differences
        field_diff = functools.reduce(pc.or_, diff_masks)
        return pc.or_(field_diff, struct_null_diff)

    elif pa.types.is_list(col_type) or pa.types.is_large_list(col_type) or pa.types.is_fixed_size_list(col_type) or pa.types.is_map(col_type):
        # For list/map types, fall back to Python comparison as PyArrow doesn't support vectorized comparison
        # This is still faster than the original row-by-row approach since we batch the conversion
        source_py = source_col.to_pylist()
        target_py = target_col.to_pylist()
        return pa.array([s != t for s, t in zip(source_py, target_py, strict=True)], type=pa.bool_())

    else:
        # For primitive types, use vectorized not_equal
        # Handle nulls: not_equal returns null when comparing with null
        # We need: null vs non-null = different (True), null vs null = same (False)
        diff = pc.not_equal(source_col, target_col)
        source_null = pc.is_null(source_col)
        target_null = pc.is_null(target_col)

        # XOR of null masks: True if exactly one is null (meaning they differ)
        null_diff = pc.xor(source_null, target_null)

        # Combine: different if values differ OR exactly one is null
        # Fill null comparison results with False (both non-null but comparison returned null shouldn't happen,
        # but if it does, treat as no difference)
        diff_filled = pc.fill_null(diff, False)
        return pc.or_(diff_filled, null_diff)


def get_rows_to_update(source_table: pa.Table, target_table: pa.Table, join_cols: list[str]) -> pa.Table:
    """
    Return a table with rows that need to be updated in the target table based on the join columns.

    Uses vectorized PyArrow operations for efficient comparison, avoiding row-by-row Python loops.
    The table is joined on the identifier columns, and then checked if there are any updated rows.
    """
    func_start = time.perf_counter()

    all_columns = set(source_table.column_names)
    join_cols_set = set(join_cols)

    non_key_cols = list(all_columns - join_cols_set)

    if has_duplicate_rows(target_table, join_cols):
        raise ValueError("Target table has duplicate rows, aborting upsert")

    if len(target_table) == 0:
        # When the target table is empty, there is nothing to update
        logger.info(f"get_rows_to_update: {time.perf_counter() - func_start:.3f}s (empty target table)")
        return source_table.schema.empty_table()

    if len(non_key_cols) == 0:
        # No non-key columns to compare, all matched rows are "updates" but with no changes
        logger.info(f"get_rows_to_update: {time.perf_counter() - func_start:.3f}s (no non-key columns)")
        return source_table.schema.empty_table()

    SOURCE_INDEX_COLUMN_NAME = "__source_index"
    TARGET_INDEX_COLUMN_NAME = "__target_index"

    if SOURCE_INDEX_COLUMN_NAME in join_cols or TARGET_INDEX_COLUMN_NAME in join_cols:
        raise ValueError(
            f"{SOURCE_INDEX_COLUMN_NAME} and {TARGET_INDEX_COLUMN_NAME} are reserved for joining "
            f"DataFrames, and cannot be used as column names"
        ) from None

    # Step 1: Prepare source index with join keys and a marker index
    # Cast to target table schema, so we can do the join
    # See: https://github.com/apache/arrow/issues/37542
    index_start = time.perf_counter()
    source_index = (
        source_table.cast(target_table.schema)
        .select(join_cols_set)
        .append_column(SOURCE_INDEX_COLUMN_NAME, pa.array(range(len(source_table))))
    )

    # Step 2: Prepare target index with join keys and a marker
    target_index = target_table.select(join_cols_set).append_column(TARGET_INDEX_COLUMN_NAME, pa.array(range(len(target_table))))
    index_end = time.perf_counter()

    # Step 3: Perform an inner join to find which rows from source exist in target
    join_start = time.perf_counter()
    matching_indices = source_index.join(target_index, keys=list(join_cols_set), join_type="inner")
    join_end = time.perf_counter()

    if len(matching_indices) == 0:
        # No matching rows found
        logger.info(
            f"get_rows_to_update: {time.perf_counter() - func_start:.3f}s "
            f"(index prep: {index_end - index_start:.3f}s, join: {join_end - join_start:.3f}s, "
            f"matched: 0, to_update: 0)"
        )
        return source_table.schema.empty_table()

    # Step 4: Take matched rows in batch (vectorized - single operation)
    take_start = time.perf_counter()
    source_indices = matching_indices[SOURCE_INDEX_COLUMN_NAME]
    target_indices = matching_indices[TARGET_INDEX_COLUMN_NAME]

    matched_source = source_table.take(source_indices)
    matched_target = target_table.take(target_indices)
    take_end = time.perf_counter()

    # Step 5: Vectorized comparison per column
    compare_start = time.perf_counter()
    diff_masks = []
    for col in non_key_cols:
        source_col = matched_source.column(col)
        target_col = matched_target.column(col)
        col_diff = _compare_columns_vectorized(source_col, target_col)
        diff_masks.append(col_diff)

    # Step 6: Combine masks with OR (any column different = needs update)
    combined_mask = functools.reduce(pc.or_, diff_masks)
    compare_end = time.perf_counter()

    # Step 7: Filter to get indices of rows that need updating
    to_update_indices = pc.filter(source_indices, combined_mask)

    func_end = time.perf_counter()
    logger.info(
        f"get_rows_to_update: {func_end - func_start:.3f}s "
        f"(index prep: {index_end - index_start:.3f}s, "
        f"join: {join_end - join_start:.3f}s, "
        f"take: {take_end - take_start:.3f}s, "
        f"compare: {compare_end - compare_start:.3f}s, "
        f"matched: {len(matching_indices)}, to_update: {len(to_update_indices)})"
    )

    if len(to_update_indices) > 0:
        return source_table.take(to_update_indices)
    else:
        return source_table.schema.empty_table()
