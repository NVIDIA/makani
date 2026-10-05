# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Date range handling shared by the converters that write one makani file per year.

Kept free of MPI and torch so that it can be imported, and tested, anywhere.
"""

from typing import List, Optional, Tuple
import argparse as ap
import datetime as dt


def parse_date(value: str, end_of_day: Optional[bool] = False) -> dt.datetime:
    """Parse an ISO 8601 date or datetime as UTC.

    A bare date such as ``2026-06-30`` means the start of that day, or its last
    hour if ``end_of_day`` is set, so that an end date is inclusive of the day.
    Naive values are taken to be UTC; aware ones are converted to it.
    """
    parsed = dt.datetime.fromisoformat(value)
    if end_of_day and len(value) == 10:
        parsed = parsed.replace(hour=23)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def yearly_sample_times(
    start_date: dt.datetime, end_date: dt.datetime, dhours: int
) -> List[Tuple[int, List[dt.datetime]]]:
    """Split an inclusive date range into per year lists of sample times.

    Samples sit on the ``dhours`` grid anchored at 00Z on January 1st of each
    year, which is the grid the yearly files have always used, so a start date
    off that grid is rounded up to the next sample. Years in the range that end
    up without any sample are left out.

    Returns
    -------
    List of ``(year, times)`` pairs in chronological order.
    """
    if end_date < start_date:
        raise ValueError(f"End date {end_date} lies before start date {start_date}.")

    result = []
    for year in range(start_date.year, end_date.year + 1):
        year_start = dt.datetime(year=year, day=1, month=1, tzinfo=dt.timezone.utc)
        year_end = dt.datetime(year=year, day=31, month=12, hour=23, tzinfo=dt.timezone.utc)
        hours_in_year = int((year_end - year_start).total_seconds() // 3600)
        times = [year_start + h * dt.timedelta(hours=1) for h in range(0, hours_in_year + 1, dhours)]
        times = [t for t in times if start_date <= t <= end_date]
        if times:
            result.append((year, times))

    if not result:
        raise ValueError(f"No samples on the {dhours}h grid between {start_date} and {end_date}.")
    return result


def add_date_range_arguments(parser: ap.ArgumentParser):
    """Add the ``--start_date`` and ``--end_date`` options shared by the yearly converters."""
    parser.add_argument(
        "--start_date",
        type=str,
        help="First date to convert, inclusive, as ISO 8601 in UTC, e.g. 2018-01-01 or 2018-01-01T06.",
        required=True,
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="Last date to convert, inclusive, as ISO 8601 in UTC. A bare date includes the whole day. "
        "One file is written per year in the range, partial at either end if needed.",
        required=True,
    )


def date_range_from_args(args: ap.Namespace) -> Tuple[dt.datetime, dt.datetime]:
    """Return the parsed ``(start_date, end_date)`` from options added by :func:`add_date_range_arguments`."""
    return parse_date(args.start_date), parse_date(args.end_date, end_of_day=True)
