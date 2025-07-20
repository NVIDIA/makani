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

from typing import Union, List


class AWSConnector(object):
    def __init__(self, aws_access_key_id, aws_secret_access_key, aws_region_name, aws_endpoint_url, aws_session_token=None):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region_name = aws_region_name
        self.aws_endpoint_url = aws_endpoint_url
        self.aws_session_token = aws_session_token

        import boto3

        self.s3_resource = boto3.resource("s3")
        self.s3_session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id, aws_secret_access_key=self.aws_secret_access_key, region_name=self.aws_region_name, aws_session_token=self.aws_session_token
        )
        self.s3_client = self.s3_session.client("s3", endpoint_url=self.aws_endpoint_url)

    def list_bucket(self, paths: Union[str, List[str]]):
        s3_paginator = self.s3_client.get_paginator("list_objects_v2")

        def list_keys(bucket, prefix="/", delimiter="/", start_after=""):
            prefix = prefix.lstrip(delimiter)
            start_after = (start_after or prefix) if prefix.endswith(delimiter) else start_after
            for page in s3_paginator.paginate(Bucket=bucket, Prefix=prefix, StartAfter=start_after):
                for content in page.get("Contents", ()):
                    yield content["Key"]

        paths = [paths] if not isinstance(paths, List) else paths
        obj = []
        for path in paths:
            # aws paths always use / as separator afaik
            pathsplit = path.strip("/").split("/")
            bname = pathsplit[0]
            prefix = "/".join(pathsplit[1:])
            for key in list_keys(bucket=bname, prefix=prefix):
                obj.append(bname + "/" + key)

        return obj
