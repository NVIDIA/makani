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

"""The DALI inference dataloader.

Inference does not walk an epoch. It reads the batches a schedule names, one
lead time after the other, and looks samples up by date, see
:class:`~makani.utils.inference.helpers.SortedIndexSampler`. The dataset that
answers all of that is :class:`~makani.utils.dataloaders.data_loader_multifiles.MultifilesDataset`:
it addresses a sample by the first step of its window, reads windows that span
files, and maps between indices and dates. This loader reads through it rather
than through a second implementation of the same arithmetic, so that a sample
means the same thing on either path, and is the same down to the bit.

What DALI adds is the pipeline: the windows are read in its python workers,
batched, moved to the device and converted to the model grid there. Unlike the
training loader there is no epoch, no shuffling and no sharding -- the
inferencer splits the dates across ranks itself -- and nothing to checkpoint.
A pipeline serves one schedule, see :meth:`DaliInferenceLoader.schedule`.
"""

from typing import Sequence

import numpy as np
import torch

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from makani.utils.dataloaders.batch import (
    make_batch,
    INPUT,
    INPUT_DYNAMIC_FEATURES,
    INPUT_TIMESTAMP,
    ZENITH,
)
from makani.utils.dataloaders.sample_source import feature_output_name
from makani.utils.grids import GridConverter


class ScheduledSource(object):
    """DALI batch callback that reads the batches of a schedule from a dataset.

    Called with a ``batch_info``, it returns batch ``batch_info.iteration`` of the
    schedule, each sample read with
    :meth:`~makani.utils.dataloaders.data_loader_multifiles.MultifilesDataset.read_window`,
    and raises ``StopIteration`` past the last one. Only the input window is read;
    the inferencer takes the verification from the samples of later lead times.

    Parameters
    ----------
    dataset : MultifilesDataset
        Where the samples come from, normalized there.
    schedule : sequence of sequence of int
        The batches, as dataset indices. They may differ in size, the last one
        of an inference range usually being shorter.
    """

    def __init__(self, dataset, schedule: Sequence[Sequence[int]]):
        self.dataset = dataset
        self.schedule = [list(batch) for batch in schedule]

    def outputs(self):
        """The outputs of :meth:`__call__`, in the order they are returned, as ``(name, DALI layout)``."""
        outputs = [(INPUT, "FCHW")]
        if self.dataset.add_zenith:
            outputs.append((feature_output_name(INPUT_DYNAMIC_FEATURES, ZENITH), "FCHW"))
        outputs.append((INPUT_TIMESTAMP, "F"))
        return outputs

    def output_names(self):
        return [name for name, _ in self.outputs()]

    def output_layout(self):
        return [layout for _, layout in self.outputs()]

    def __len__(self):
        return len(self.schedule)

    def __call__(self, batch_info):
        if batch_info.iteration >= len(self.schedule):
            raise StopIteration

        windows = [self.dataset.read_window(idx, return_target=False) for idx in self.schedule[batch_info.iteration]]

        # one batch per output, the samples stacked along a new leading dimension
        return tuple(np.stack([window[name] for window in windows], axis=0) for name in self.output_names())


class DaliInferenceLoader(object):
    """Feeds the batches of an inference schedule, read in a DALI pipeline.

    Iterating yields batch dicts without targets, see
    :mod:`makani.utils.dataloaders.batch`, on the device the pipeline runs on.
    The schedule is set with :meth:`schedule`, which builds the pipeline for it;
    until then there is nothing to iterate.

    The dataset is exposed as :attr:`dataset`, as the multifiles path does, since
    the inferencer looks up dates and indices through it.

    Parameters
    ----------
    dataset : MultifilesDataset
        Built without targets. It normalizes, so the pipeline does not.
    batch_size : int
        The largest batch of a schedule.
    num_data_workers : int
        Python workers reading windows, at least one.
    data_grid_type, model_grid_type : str
        The grids the data is stored on and the model works on.
    dali_device : str
        ``"gpu"`` or ``"cpu"``, where the batches come out.
    seed : int
        Seed of the pipeline. Nothing in it is random, but DALI asks for one.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_data_workers: int,
        data_grid_type: str,
        model_grid_type: str,
        dali_device: str = "gpu",
        seed: int = 333,
    ):
        if num_data_workers < 1:
            raise ValueError(f"num_data_workers must be >= 1 for the DALI loader, got {num_data_workers}.")
        if dataset.return_target:
            raise ValueError("The DALI inference loader reads input windows only, build the dataset without targets.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_data_workers = num_data_workers
        self.seed = seed

        self.dali_device = dali_device
        if self.dali_device == "gpu":
            self.device_index = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{self.device_index}")
        elif self.dali_device == "cpu":
            self.device_index = None
            self.device = torch.device("cpu")
        else:
            raise NotImplementedError(f"Makani currently does not support dali_device {self.dali_device}")

        # the dataset converts on the host; the batches here are converted where they land
        self.grid_converter = GridConverter(
            data_grid_type,
            model_grid_type,
            torch.deg2rad(torch.tensor(dataset.lat_lon_local[0])).to(torch.float32).to(self.device),
            torch.deg2rad(torch.tensor(dataset.lat_lon_local[1])).to(torch.float32).to(self.device),
        )

        # what the other loaders expose
        self.data_shapes = dataset.data_shapes
        self.lat_lon = dataset.lat_lon
        self.get_input_normalization = dataset.get_input_normalization
        self.get_output_normalization = dataset.get_output_normalization

        self.source = None
        self.pipeline = None
        self.iterator = None

    def _get_pipeline(self):
        pipeline = Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            device_id=self.device_index,
            py_num_workers=self.num_data_workers,
            py_start_method="spawn",
            seed=self.seed,
            prefetch_queue_depth=2,
        )

        with pipeline:
            data = fn.external_source(
                source=self.source,
                num_outputs=len(self.source.output_names()),
                layout=self.source.output_layout(),
                batch=True,
                batch_info=True,
                parallel=True,
                prefetch_queue_depth=self.num_data_workers,
            )

            # the timestamps go along, so that the whole batch is on one device
            if self.dali_device == "gpu":
                data = [output.gpu() for output in data]

            pipeline.set_outputs(*data)

        return pipeline

    def schedule(self, batches: Sequence[Sequence[int]]):
        """Build the pipeline for ``batches``, dataset indices batched in the order they are to be read.

        The workers hold their own copy of the schedule, so a new one needs a new
        pipeline; the previous one is shut down first. Returns the loader, to be
        iterated once.
        """
        self.close()

        self.source = ScheduledSource(self.dataset, batches)
        if any(len(batch) > self.batch_size for batch in self.source.schedule):
            raise ValueError(f"A batch of the schedule is larger than the batch size {self.batch_size}.")

        # an empty schedule has nothing to read, and DALI would refuse to prefetch it
        if len(self.source) == 0:
            return self

        self.pipeline = self._get_pipeline()
        self.pipeline.start_py_workers()
        self.iterator = DALIGenericIterator(
            [self.pipeline],
            self.source.output_names(),
            auto_reset=False,
            size=-1,
            last_batch_policy=LastBatchPolicy.PARTIAL,
            prepare_first_batch=False,
        )

        return self

    def close(self):
        """Shut the pipeline down and release its workers. A new :meth:`schedule` builds another."""
        self.iterator = None
        if self.pipeline is not None:
            self.pipeline._shutdown()
            self.pipeline = None

    def __len__(self):
        return 0 if self.source is None else len(self.source)

    def __iter__(self):
        if self.source is None:
            raise RuntimeError("No schedule to read, call schedule() first.")
        if self.iterator is None:
            return

        zenith = feature_output_name(INPUT_DYNAMIC_FEATURES, ZENITH)
        for token in self.iterator:
            outputs = token[0]

            with torch.no_grad():
                features = {}
                if zenith in outputs:
                    features[ZENITH] = self.grid_converter(outputs[zenith])

                yield make_batch(
                    inp=self.grid_converter(outputs[INPUT]),
                    inp_timestamp=outputs[INPUT_TIMESTAMP],
                    inp_dynamic_features=features,
                )
