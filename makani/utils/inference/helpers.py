# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from more_itertools import batched, divide
from typing import Optional, List, Iterator
import datetime as dt

import torch.utils.data as tud


# this is a small helper to convert datetime to correct time zone
def date_helper(isostring):
    date = dt.datetime.fromisoformat(isostring)
    try:
        date = date.astimezone(dt.timezone.utc)
    except:
        date = date.replace(tzinfo=dt.timezone.utc)

    return date


def split_list(lst: List[int], nchunks: int) -> List[List[int]]:
    return [list(x) for x in list(divide(nchunks, lst))]


class SortedIndexSampler(tud.Sampler[List[int]]):
    def __init__(self, indices: List[int], maxind: int, batch_size: int, rollout_steps: int, rollout_dt: int, incomplete_rollouts: Optional[bool] = False) -> None:

        # make sure the batch size is sane
        batch_size = min(len(indices), batch_size)
        batches = map(list, batched(indices, batch_size))
        self.indices = []
        for batch in batches:
            rollout = []
            append = True
            for s in range(0, rollout_steps+1):
                shift = [b + rollout_dt * s for b in batch]
                if max(shift) >= maxind:
                    append = False
                    break

                rollout.append(shift)

            if append or incomplete_rollouts:
                self.indices += rollout

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.indices:
            yield batch


class SimpleIndexSampler(tud.Sampler[List[int]]):
    def __init__(self, indices: List[List[int]]) -> None:
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.indices:
            yield batch


def translate_date_sampler_to_timedelta_sampler(sampler, date_dataset, timedelta_dataset):
    indexlist = []
    iterator = iter(sampler)
    for indices in iterator:
        tstamps = [date_dataset.get_time_at_index(idx) for idx in indices]
        timedeltas = [t - dt.datetime(year=t.year, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc) for t in tstamps]
        indexlist.append([timedelta_dataset.get_index_at_time(t) for t in timedeltas])

    return SimpleIndexSampler(indexlist)
