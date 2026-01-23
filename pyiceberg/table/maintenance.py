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
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pyiceberg.table import Table
    from pyiceberg.table.update.snapshot import ExpireSnapshots, RewriteDataFiles


class MaintenanceTable:
    tbl: Table

    def __init__(self, tbl: Table) -> None:
        self.tbl = tbl

    def expire_snapshots(self) -> ExpireSnapshots:
        """Return an ExpireSnapshots builder for snapshot expiration operations.

        Returns:
            ExpireSnapshots builder for configuring and executing snapshot expiration.
        """
        from pyiceberg.table import Transaction
        from pyiceberg.table.update.snapshot import ExpireSnapshots

        return ExpireSnapshots(transaction=Transaction(self.tbl, autocommit=True))

    def rewrite_data_files(self) -> RewriteDataFiles:
        """Return a RewriteDataFiles builder for compaction operations.

        This operation reads small data files and rewrites them into larger,
        optimally-sized files. Files are selected based on size thresholds and
        grouped by partition for efficient processing.

        Example:
            result = (
                table.maintenance
                .rewrite_data_files()
                .filter("year = 2024")  # Optional: restrict to partitions
                .option("target-file-size-bytes", "134217728")  # Optional: 128MB
                .commit()
            )

            print(f"Rewrote {result.rewritten_data_files_count} files into "
                  f"{result.added_data_files_count} files")

        Returns:
            RewriteDataFiles builder for configuring and executing file compaction.
        """
        from pyiceberg.table import Transaction
        from pyiceberg.table.update.snapshot import RewriteDataFiles

        return RewriteDataFiles(transaction=Transaction(self.tbl, autocommit=True))
