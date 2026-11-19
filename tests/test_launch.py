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

import unittest

from parameterized import parameterized

from makani.distributed.launch import Launch, detect_launch

_OPENMPI = {"OMPI_COMM_WORLD_SIZE": "8", "OMPI_COMM_WORLD_RANK": "5", "OMPI_COMM_WORLD_LOCAL_RANK": "1"}
_SLURM_STEP = {
    "SLURM_STEP_ID": "0",
    "SLURM_STEP_NUM_TASKS": "16",
    "SLURM_NTASKS": "32",
    "SLURM_PROCID": "9",
    "SLURM_LOCALID": "3",
}
_PMI = {"PMI_SIZE": "4", "PMI_RANK": "2", "MPI_LOCALRANKID": "0"}
_ENV = {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}


class TestDetectLaunch(unittest.TestCase):
    """The size, rank and local rank of a process, from the variables its launcher set."""

    def test_no_launcher_is_one_process(self):
        self.assertEqual(detect_launch({}), Launch(size=1, rank=0, local_rank=0, launcher="none"))

    @parameterized.expand(
        [
            ("openmpi", _OPENMPI, Launch(8, 5, 1, "openmpi")),
            ("slurm", _SLURM_STEP, Launch(16, 9, 3, "slurm")),
            ("pmi", _PMI, Launch(4, 2, 0, "pmi")),
            ("env", _ENV, Launch(2, 1, 1, "env")),
        ]
    )
    def test_each_launcher(self, _, environ, expected):
        self.assertEqual(detect_launch(environ), expected)

    def test_the_launchers_are_queried_in_order(self):
        # Open MPI before Slurm before PMI before variables set by hand
        self.assertEqual(detect_launch({**_ENV, **_PMI, **_SLURM_STEP, **_OPENMPI}).launcher, "openmpi")
        self.assertEqual(detect_launch({**_ENV, **_PMI, **_SLURM_STEP}).launcher, "slurm")
        self.assertEqual(detect_launch({**_ENV, **_PMI}).launcher, "pmi")

    def test_a_slurm_batch_script_is_no_launch(self):
        # outside of a step, Slurm's task variables describe the job, not the one process
        environ = {"SLURM_JOB_ID": "1", "SLURM_NTASKS": "32", "SLURM_PROCID": "0", "SLURM_LOCALID": "0"}
        self.assertEqual(detect_launch(environ), Launch())

    def test_the_task_count_of_a_step_falls_back_to_the_job(self):
        environ = {"SLURM_STEP_ID": "0", "SLURM_NTASKS": "4", "SLURM_PROCID": "3"}
        self.assertEqual(detect_launch(environ).size, 4)

    def test_world_rank_names_the_rank_as_well(self):
        self.assertEqual(detect_launch({"WORLD_SIZE": "4", "WORLD_RANK": "3"}).rank, 3)

    def test_an_unknown_local_rank_of_several_processes_stays_unknown(self):
        self.assertIsNone(detect_launch({"PMI_SIZE": "4", "PMI_RANK": "2"}).local_rank)
        self.assertEqual(detect_launch({"WORLD_SIZE": "1"}).local_rank, 0)

    def test_inconsistent_variables_are_refused(self):
        with self.assertRaisesRegex(ValueError, "none of"):
            detect_launch({"OMPI_COMM_WORLD_SIZE": "2"})
        with self.assertRaisesRegex(ValueError, "outside"):
            detect_launch({"PMI_SIZE": "2", "PMI_RANK": "2"})
        with self.assertRaisesRegex(ValueError, "not an integer"):
            detect_launch({"WORLD_SIZE": "two"})


if __name__ == "__main__":
    unittest.main()
