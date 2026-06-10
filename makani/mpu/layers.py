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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from makani.utils import comm

# parallel helpers
from torch_harmonics.distributed import compute_split_shapes
from makani.mpu.mappings import reduce_from_parallel_region
from makani.mpu.mappings import gather_from_parallel_region
from makani.mpu.mappings import copy_to_parallel_region

# transformer engine is an optional dependency: it is only used for the
# (optional) FP8/FP4 path and must not be required for import. availability is
# checked without importing it; the module is imported lazily where used.
from makani.utils.te_helpers import TE_AVAILABLE as _TE_AVAILABLE, get_te


class DistributedMatmul(nn.Module):
    """Megatron-style tensor-parallel matmul over a single feature-parallel group.

    The legacy 2D (fin x fout) decomposition has been retired in favor of a 1D
    column/row fork-join over a single comm group (``comm_name``, "matmul" by
    default):

    - ``parallel_mode="column"`` shards the OUTPUT dimension. The input is
      replicated across the group (``copy_to_parallel_region``: identity in the
      forward, all-reduce in the backward) and there is NO output reduction, so
      the result is sharded along the output dimension.
    - ``parallel_mode="row"`` shards the INPUT dimension. The input is assumed to
      be sharded (e.g. the sharded hidden produced by a preceding column layer),
      and the partial outputs are all-reduced (``reduce_from_parallel_region``)
      into a replicated full output. For a row layer the bias is replicated and
      added AFTER the reduction.
    """

    def __init__(self, inp_dim, out_dim, input_format="nchw", comm_name="matmul", parallel_mode="column", bias=True, use_te=False):
        super(DistributedMatmul, self).__init__()

        if parallel_mode not in ["column", "row"]:
            raise ValueError(f"Error, parallel_mode {parallel_mode} not supported (use 'column' or 'row').")

        if input_format not in ["nchw", "traditional"]:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        self.comm_name = comm_name
        self.parallel_mode = parallel_mode
        self.input_format = input_format

        # only use transformer engine if it was requested and is actually available
        self.use_te = use_te and _TE_AVAILABLE
        if use_te and not _TE_AVAILABLE:
            import warnings

            warnings.warn("use_te=True was requested but transformer_engine is not installed; falling back to the standard matmul.")

        comm_size = comm.get_size(comm_name)

        # column shards the output dim, row shards the input dim
        if parallel_mode == "column":
            if out_dim % comm_size != 0:
                raise ValueError(f"the output feature dim ({out_dim}) has to be evenly divisible by the matmul comm dim ({comm_size})")
            out_dim_local = out_dim // comm_size
            inp_dim_local = inp_dim
        else:
            if inp_dim % comm_size != 0:
                raise ValueError(f"the input feature dim ({inp_dim}) has to be evenly divisible by the matmul comm dim ({comm_size})")
            out_dim_local = out_dim
            inp_dim_local = inp_dim // comm_size

        # the dimension that is sharded over the matmul group (None for the
        # replicated dimension) - used for checkpoint gather/scatter only
        weight_inp_shard = None if parallel_mode == "column" else comm_name
        weight_out_shard = comm_name if parallel_mode == "column" else None

        # weight. transformer engine keeps the (2D) weight inside a te.Linear and
        # performs the local matmul in FP8/FP4; the tensor-parallel communication
        # (copy/reduce, see forward) and the gradient reduction stay in makani, so
        # te.Linear is used purely as a local GEMM (no tp_group). The bias is
        # always owned here (te.Linear bias=False) so the row-parallel bias can be
        # added AFTER the all-reduce, matching the native path.
        if self.use_te:
            te = get_te()
            self.te_linear = te.Linear(inp_dim_local, out_dim_local, bias=False)
            # te weight is always 2D (out_local, in_local) regardless of input_format
            self.weight.is_shared_mp = ["spatial"]
            self.weight.sharded_dims_mp = [weight_out_shard, weight_inp_shard]
        elif input_format == "nchw":
            self.weight = nn.Parameter(torch.ones(out_dim_local, inp_dim_local, 1, 1))
            self.weight.is_shared_mp = ["spatial"]
            self.weight.sharded_dims_mp = [weight_out_shard, weight_inp_shard, None, None]
            self.matmul_handle = F.conv2d
        else:  # traditional
            self.weight = nn.Parameter(torch.ones(out_dim_local, inp_dim_local))
            self.weight.is_shared_mp = ["spatial"]
            self.weight.sharded_dims_mp = [weight_out_shard, weight_inp_shard]
            self.matmul_handle = F.linear

        # bias
        self.bias = None
        if bias:
            # a column bias is sharded over the (sharded) output dim, a row bias
            # is replicated over the matmul group (added after the all-reduce) and
            # must therefore NOT be summed over matmul in the gradient hook
            bias_shard = comm_name if parallel_mode == "column" else None
            if input_format == "nchw":
                self.bias = nn.Parameter(torch.zeros(1, out_dim_local, 1, 1))
                self.bias.is_shared_mp = ["spatial"]
                self.bias.sharded_dims_mp = [None, bias_shard, None, None]
            elif input_format == "traditional":
                self.bias = nn.Parameter(torch.zeros(out_dim_local))
                self.bias.is_shared_mp = ["spatial"]
                self.bias.sharded_dims_mp = [bias_shard]

    @property
    def weight(self):
        # te.Linear owns the weight in the TE path; expose it through the same
        # attribute so external init / annotation / the gradient hook are uniform.
        # raise AttributeError (not KeyError) when the native weight is not yet
        # registered: register_parameter() probes hasattr(self, "weight") while
        # registering, and hasattr only swallows AttributeError.
        if self.use_te:
            return self.te_linear.weight
        weight = self._parameters.get("weight")
        if weight is None:
            raise AttributeError("weight")
        return weight

    def _local_matmul(self, x):
        if self.use_te:
            # te.Linear operates on the last (channel) dim; for nchw we transpose
            # to channels-last around the GEMM
            if self.input_format == "nchw":
                x = self.te_linear(x.permute(0, 2, 3, 1).contiguous())
                return x.permute(0, 3, 1, 2).contiguous()
            return self.te_linear(x)
        return self.matmul_handle(x, self.weight, bias=None)

    def forward(self, x):
        if self.parallel_mode == "column":
            # replicated input -> sharded output, no output reduction
            x = copy_to_parallel_region(x, self.comm_name)
            x = self._local_matmul(x)
        else:
            # sharded input -> reduced (replicated) output
            x = self._local_matmul(x)
            x = reduce_from_parallel_region(x, self.comm_name)

        if self.bias is not None:
            x = x + self.bias

        return x


# distributed encoder/decoder
class DistributedEncoderDecoder(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, hidden_dim, act_layer, gain=1.0, input_format="nchw", comm_name="matmul", use_te=False):
        super(DistributedEncoderDecoder, self).__init__()

        self.comm_name = comm_name

        # the chain takes a replicated input and must produce a replicated output.
        # we therefore drive the column/row fork-join from the END: the last layer
        # is row-parallel (reduces to a replicated output) and we alternate
        # backwards. for the first layer to be column-parallel (consuming the
        # replicated input) the number of layers must be even when we are actually
        # feature-parallel.
        if (comm.get_size(comm_name) > 1) and (num_layers % 2 != 0):
            raise ValueError(
                f"DistributedEncoderDecoder requires an even number of layers under matmul parallelism, got {num_layers}."
            )
        modes = ["row" if ((num_layers - 1 - i) % 2 == 0) else "column" for i in range(num_layers)]

        # get list of modules
        encoder_modules = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            encoder_modules.append(
                DistributedMatmul(current_dim, hidden_dim, input_format=input_format, comm_name=comm_name, parallel_mode=modes[i], bias=True, use_te=use_te)
            )

            # proper initialization
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
            if encoder_modules[-1].bias is not None:
                nn.init.constant_(encoder_modules[-1].bias, 0.0)

            encoder_modules.append(act_layer())
            current_dim = hidden_dim

        # final layer (row-parallel, replicated output, no bias)
        encoder_modules.append(DistributedMatmul(current_dim, output_dim, input_format=input_format, comm_name=comm_name, parallel_mode=modes[-1], bias=False, use_te=use_te))

        # proper initialization of final layer
        scale = math.sqrt(gain / current_dim)
        nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
        if encoder_modules[-1].bias is not None:
            nn.init.constant_(encoder_modules[-1].bias, 0.0)

        # create fwd sequence
        self.fwd = nn.Sequential(*encoder_modules)

    def forward(self, x):
        return self.fwd(x)


# more complicated layers
class DistributedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        output_bias=True,
        input_format="nchw",
        comm_name="matmul",
        act_layer=nn.GELU,
        drop_rate=0.0,
        drop_type="iid",
        checkpointing=False,
        gain=1.0,
        use_te=False,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # sanity checks:
        if (input_format == "traditional") and (drop_type == "features"):
            raise NotImplementedError(f"Error, traditional input format and feature dropout cannot be selected simultaneously")

        # column-parallel: replicated input -> hidden sharded over the matmul group
        self.fc1 = DistributedMatmul(in_features, hidden_features, input_format=input_format, comm_name=comm_name, parallel_mode="column", bias=True, use_te=use_te)

        # initialize the weights correctly
        scale = math.sqrt(2.0 / in_features)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=scale)
        nn.init.constant_(self.fc1.bias, 0.0)

        # row-parallel: sharded hidden -> all-reduced (replicated) output
        self.fc2 = DistributedMatmul(hidden_features, out_features, input_format=input_format, comm_name=comm_name, parallel_mode="row", bias=output_bias, use_te=use_te)

        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=scale)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0.0)

        self.act = act_layer()

        if drop_rate > 0.0:
            if drop_type == "iid":
                self.drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                self.drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Error, drop_type {drop_type} not supported")
        else:
            self.drop = nn.Identity()

    def fwd(self, x):
        # do the mlp
        # first layer
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        # second layer
        x = self.fc2(x)
        x = self.drop(x)

        return x

    @torch.compiler.disable(recursive=False)
    def _checkpoint_forward(self, x):
        return checkpoint(self.fwd, x, use_reentrant=False)

    def forward(self, x):
        if self.checkpointing:
            return self._checkpoint_forward(x)
        else:
            return self.fwd(x)

# Stochastic MLP needs comm datastructure
class StochasticMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        output_bias=True,
        input_format="nchw",
        drop_rate=0.0,
        drop_type="iid",
        checkpointing=False,
        gain=1.0,
        seed=333,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # generator objects:
        self.set_rng(seed=seed)

        # First fully connected layer
        if input_format == "nchw":
            self.fc1_weight_std = nn.Parameter(torch.zeros(hidden_features, in_features, 1, 1))
            self.fc1_weight_mean = nn.Parameter(torch.zeros(hidden_features, in_features, 1, 1))
            self.fc1_bias = nn.Parameter(torch.zeros(hidden_features))
        else:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        # sharing settings
        self.fc1_weight_std.is_shared_mp = ["spatial"]
        self.fc1_weight_mean.is_shared_mp = ["spatial"]
        self.fc1_bias.is_shared_mp = ["spatial"]

        # initialize the weights correctly
        scale = math.sqrt(1.0 / in_features)
        nn.init.normal_(self.fc1_weight_std, mean=0.0, std=scale)
        nn.init.normal_(self.fc1_weight_mean, mean=0.0, std=scale)

        # activation
        self.act = act_layer()

        # sanity checks
        if (input_format == "traditional") and (drop_type == "features"):
            raise NotImplementedError(f"Error, traditional input format and feature dropout cannot be selected simultaneously")

        # output layer
        if input_format == "nchw":
            self.fc2_weight_std = nn.Parameter(torch.zeros(out_features, hidden_features, 1, 1))
            self.fc2_weight_mean = nn.Parameter(torch.zeros(out_features, hidden_features, 1, 1))
            self.fc2_bias = nn.Parameter(torch.zeros(out_features)) if output_bias else None
        else:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        # sharing settings
        self.fc2_weight_std.is_shared_mp = ["spatial"]
        self.fc2_weight_mean.is_shared_mp = ["spatial"]
        if self.fc2_bias is not None:
            self.fc2_bias.is_shared_mp = ["spatial"]

        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features / 2)
        nn.init.normal_(self.fc2_weight_std, mean=0.0, std=scale)
        nn.init.normal_(self.fc2_weight_mean, mean=0.0, std=scale)
        if self.fc2_bias is not None:
            nn.init.constant_(self.fc2_bias, 0.0)

        if drop_rate > 0.0:
            if drop_type == "iid":
                self.drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                self.drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Error, drop_type {drop_type} not supported")
        else:
            self.drop = nn.Identity()

    @torch.compiler.disable(recursive=False)
    def set_rng(self, seed=333):
        self.rng_cpu = torch.Generator(device=torch.device("cpu"))
        self.rng_cpu.manual_seed(seed)
        if torch.cuda.is_available():
            self.rng_gpu = torch.Generator(device=torch.device(f"cuda:{comm.get_local_rank()}"))
            self.rng_gpu.manual_seed(seed)

    @torch.compiler.disable(recursive=False)
    def checkpoint_forward(self, x):
        return checkpoint(self.fwd, x, use_reentrant=False)

    def fwd(self, x):

        # generate weight1
        weight1 = torch.empty_like(self.fc1_weight_mean)
        weight1.normal_(mean=0.0, std=1.0, generator=self.rng_gpu if weight1.is_cuda else self.rng_cpu)
        weight1 = self.fc1_weight_std * weight1 + self.fc1_weight_mean

        # fully connected 1
        x = nn.functional.conv2d(x, weight1, bias=self.fc1_bias)

        # activation
        x = self.act(x)

        # dropout
        x = self.drop(x)

        # generate weight1
        weight2 = torch.empty_like(self.fc2_weight_mean)
        weight2.normal_(mean=0.0, std=1.0, generator=self.rng_gpu if weight2.is_cuda else self.rng_cpu)
        weight2 = self.fc2_weight_std * weight2 + self.fc2_weight_mean

        # fully connected 2
        x = nn.functional.conv2d(x, weight2, bias=self.fc2_bias)

        # dropout
        x = self.drop(x)

        return x

    def forward(self, x):
        if self.checkpointing:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)

class DistributedPatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, input_is_matmul_parallel=False, output_is_matmul_parallel=True):
        super().__init__()

        # store params
        self.input_parallel = input_is_matmul_parallel
        self.output_parallel = output_is_matmul_parallel

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        spatial_comm_size = comm.get_size("spatial")

        # compute parameters
        if (img_size[1] // patch_size[1]) % spatial_comm_size != 0:
            raise ValueError(f"the spatial comm size ({spatial_comm_size}) must evenly divide patched W ({img_size[1] // patch_size[1]})")
        num_patches = ((img_size[1] // patch_size[1]) // spatial_comm_size) * (img_size[0] // patch_size[0])
        self.img_size = (img_size[0], img_size[1] // spatial_comm_size)
        self.patch_size = patch_size
        self.num_patches = num_patches

        # get effective embedding size:
        if self.output_parallel:
            if embed_dim % matmul_comm_size != 0:
                raise ValueError(f"the embed_dim ({embed_dim}) needs to be divisible by matmul_parallel_size ({matmul_comm_size})")
            out_chans_local = embed_dim // matmul_comm_size
        else:
            out_chans_local = embed_dim

        # the weights  of this layer is shared across spatial parallel ranks
        self.proj = nn.Conv2d(in_chans, out_chans_local, kernel_size=patch_size, stride=patch_size)

        # make sure we reduce them across rank
        self.proj.weight.is_shared_mp = ["spatial"]
        self.proj.bias.is_shared_mp = ["spatial"]

        # gather shapes
        self.gather_shapes = compute_split_shapes(in_chans, comm.get_size("matmul"))

    def forward(self, x):
        if self.input_parallel:
            x = gather_from_parallel_region(x, 1, self.gather_shapes, "matmul")

        if self.output_parallel:
            x = copy_to_parallel_region(x, "matmul")

        B, C, H, W = x.shape
        torch._check(H == self.img_size[0], lambda: f"Input image height {H} doesn't match model {self.img_size[0]}.")
        torch._check(W == self.img_size[1], lambda: f"Input image width {W} doesn't match model {self.img_size[1]}.")
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x


class DistributedAttention(nn.Module):
    """Distributed Attention layer"""

    def __init__(
        self,
        dim,
        input_format="traditional",
        comm_name="matmul",
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        use_te=False,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by num_heads {num_heads}")
        self.num_heads = num_heads

        if num_heads % comm.get_size(comm_name) != 0:
            raise ValueError(f"heads ({num_heads}) are not evenly split across model ranks ({comm.get_size(comm_name)})")
        self.num_heads_local = num_heads // comm.get_size(comm_name)
        self.head_dim = dim // self.num_heads

        self.comm_name = comm_name

        # column-parallel qkv (heads sharded over the matmul group) -> row-parallel proj
        self.qkv = DistributedMatmul(dim, dim * 3, input_format, comm_name=comm_name, parallel_mode="column", bias=qkv_bias, use_te=use_te)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_rate = attn_drop_rate
        self.proj = DistributedMatmul(dim, dim, input_format, comm_name=comm_name, parallel_mode="row", bias=False, use_te=use_te)
        if proj_drop_rate > 0.0:
            self.proj_drop = nn.Dropout(proj_drop_rate)
        else:
            self.proj_drop = nn.Identity()

        # set up weight sharing, depends on norm type
        if isinstance(self.q_norm, nn.LayerNorm):
            if hasattr(self.q_norm, "weight"):
                self.q_norm.weight.is_shared_mp = []
            if hasattr(self.q_norm, "bias"):
                self.q_norm.bias.is_shared_mp = []

        if isinstance(self.k_norm, nn.LayerNorm):
            if hasattr(self.k_norm, "weight"):
                self.k_norm.weight.is_shared_mp = []
            if hasattr(self.k_norm, "bias"):
                self.k_norm.bias.is_shared_mp = []

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads_local, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate)

        # transpose back
        x = x.transpose(1, 2).reshape(B, N, self.num_heads_local * self.head_dim)

        # this is distributed again
        x = self.proj(x)

        # generally we have to be super careful with dropout layers, since
        # those are normalized over the dropouts. That would need to be reduced across nodes
        x = self.proj_drop(x)

        return x


@torch.compile
def compl_mul_add_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bkixys,kiot->stbkoxy", a, b)
    res = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1) + c
    return res


@torch.compile
def compl_mul_add_fwd_c(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    cc = torch.view_as_complex(c)
    tmp = torch.einsum("bkixy,kio->bkoxy", ac, bc)
    res = tmp + cc
    return torch.view_as_real(res)


class DistributedAFNO2Dv2(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
        use_complex_kernels=False,
    ):
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError(f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}")

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        self.spatial_comm_size = comm.get_size("spatial")

        # select fft function handles
        if self.spatial_comm_size > 1:
            self.fft_handle = distributed_rfft2.apply
            self.ifft_handle = distributed_irfft2.apply
        else:
            self.fft_handle = torch.fft.rfft2
            self.ifft_handle = torch.fft.irfft2

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.gather_shapes = compute_split_shapes(self.num_blocks, matmul_comm_size)
        self.num_blocks_local = self.gather_shapes[comm.get_rank("matmul")]
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.mult_handle = compl_mul_add_fwd_c if use_complex_kernels else compl_mul_add_fwd

        # model paralellism
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # new
        # these weights need to be synced across all spatial ranks!
        self.w1 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size, self.block_size * self.hidden_size_factor, 2))
        self.b1 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size * self.hidden_size_factor, 1, 1, 2))
        self.w2 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size * self.hidden_size_factor, self.block_size, 2))
        self.b2 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size, 1, 1, 2))

        # setting correct sharding and sharing
        self.w1.is_shared_mp = ["spatial"]
        self.w1.sharded_dims_mp = ["matmul", None, None, None]

        self.b1.is_shared_mp = ["spatial"]
        self.b1.sharded_dims_mp = ["matmul", None, None, None, None]

        self.w2.is_shared_mp = ["spatial"]
        self.w2.sharded_dims_mp = ["matmul", None, None, None]

        self.b2.is_shared_mp = ["spatial"]
        self.b2.sharded_dims_mp = ["matmul", None, None, None, None]

    def forward(self, x):
        if not self.input_is_matmul_parallel:
            # distribute data
            x = gather_from_parallel_region(x, 1, self.gather_shapes, "matmul")

        # bias
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W_local = x.shape
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        H_local = H // self.spatial_comm_size
        W = W_local * self.spatial_comm_size
        x = self.fft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.view(B, self.num_blocks_local, self.block_size, H_local, W // 2 + 1)

        # new
        x = torch.view_as_real(x)
        o2 = torch.zeros(x.shape, device=x.device)

        o1 = F.relu(self.mult_handle(x[:, :, :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, :], self.w1, self.b1))
        o2[:, :, :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, :] = self.mult_handle(o1, self.w2, self.b2)

        # finalize
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, H_local, W // 2 + 1)
        x = self.ifft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.type(dtype) + bias

        # gather
        if not self.output_is_matmul_parallel:
            x = gather_from_parallel_region(x, 1, self.gather_shapes, "matmul")

        return x
