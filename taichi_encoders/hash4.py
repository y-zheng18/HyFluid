import numpy as np
import taichi as ti
import torch
from taichi.math import uvec3, uvec4
from torch.cuda.amp import custom_bwd, custom_fwd

from torch.func import jacrev, vmap

from .utils import (data_type, ti2torch, ti2torch_grad, ti2torch_grad_vec,
                    ti2torch_vec, torch2ti, torch2ti_grad, torch2ti_grad_vec,
                    torch2ti_vec, torch_type)


@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4
        # data[I] = ti.random() * 0.4 + 0.1


@ti.func
def fast_hash(pos_grid_local):
    result = ti.uint32(0)
    # primes = uvec3(ti.uint32(1), ti.uint32(1958374283), ti.uint32(2654435761)) 805459861u, 3674653429u
    primes = uvec4(ti.uint32(1), ti.uint32(2654435761), ti.uint32(805459861), ti.uint32(3674653429))
    for i in ti.static(range(4)):
        result ^= ti.uint32(pos_grid_local[i]) * primes[i]
    return result


# ravel (i, j, k, t) to i + i_dim * j + (i_dim * j_dim) * k + (i_dim * j_dim * k_dim) * t
@ti.func
def under_hash(pos_grid_local, resolution):
    result = ti.uint32(0)
    stride = ti.uint32(1)
    for i in ti.static(range(4)):
        result += ti.uint32(pos_grid_local[i] * stride)
        stride *= resolution[i] + 1  # note the +1 here, because 256 x 256 grid actually has 257 x 257 entries
    return result


@ti.func
def grid_pos2hash_index(indicator, pos_grid_local, plane_res, map_size):
    hash_result = ti.uint32(0)
    if indicator == 1:
        hash_result = under_hash(pos_grid_local, plane_res)
    else:
        hash_result = fast_hash(pos_grid_local)

    return hash_result % map_size


@ti.func
def smooth_step(t):
    return t * t * (3 - 2 * t)


@ti.func
def d_smooth_step(t):
    return 6 * t * (1 - t)


@ti.func
def linear_step(t):
    return t


@ti.func
def d_linear_step(t):
    return 1


@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)

@ti.kernel
def hash_encode_kernel_smoothstep(
        xyzts: ti.template(), table: ti.template(),
        xyzts_embedding: ti.template(), hash_map_indicator: ti.template(),
        hash_map_sizes_field: ti.template(), hash_map_shapes_field: ti.template(),
        offsets: ti.template(), B: ti.i32, num_scales: ti.i32):
    # # # get hash table embedding
    ti.loop_config(block_dim=16)
    for i, level in ti.ndrange(B, num_scales):
        res_x = hash_map_shapes_field[level, 0]
        res_y = hash_map_shapes_field[level, 1]
        res_z = hash_map_shapes_field[level, 2]
        res_t = hash_map_shapes_field[level, 3]
        plane_res = ti.Vector([res_x, res_y, res_z, res_t])
        pos = ti.Vector([xyzts[i, 0], xyzts[i, 1], xyzts[i, 2], xyzts[i, 3]]) * plane_res

        pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)  # floor
        pos_grid_uint = ti.math.clamp(pos_grid_uint, 0, plane_res - 1)
        pos -= pos_grid_uint  # pos now represents frac
        pos = ti.math.clamp(pos, 0.0, 1.0)

        offset = offsets[level]

        indicator = hash_map_indicator[level]
        map_size = hash_map_sizes_field[level]

        local_feature_0 = 0.0
        local_feature_1 = 0.0

        for idx in ti.static(range(16)):
            w = 1.
            pos_grid_local = uvec4(0)

            for d in ti.static(range(4)):
                t = smooth_step(pos[d])
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d] = pos_grid_uint[d]
                    w *= 1 - t
                else:
                    pos_grid_local[d] = pos_grid_uint[d] + 1
                    w *= t

            index = grid_pos2hash_index(indicator, pos_grid_local, plane_res, map_size)
            index_table = offset + index * 2  # the flat index for the 1st entry
            index_table_int = ti.cast(index_table, ti.int32)
            local_feature_0 += w * table[index_table_int]
            local_feature_1 += w * table[index_table_int + 1]

        xyzts_embedding[i, level * 2] = local_feature_0
        xyzts_embedding[i, level * 2 + 1] = local_feature_1


@ti.kernel
def hash_encode_kernel(
        xyzts: ti.template(), table: ti.template(),
        xyzts_embedding: ti.template(), hash_map_indicator: ti.template(),
        hash_map_sizes_field: ti.template(), hash_map_shapes_field: ti.template(),
        offsets: ti.template(), B: ti.i32, num_scales: ti.i32):
    # # # get hash table embedding
    ti.loop_config(block_dim=16)
    for i, level in ti.ndrange(B, num_scales):
        res_x = hash_map_shapes_field[level, 0]
        res_y = hash_map_shapes_field[level, 1]
        res_z = hash_map_shapes_field[level, 2]
        res_t = hash_map_shapes_field[level, 3]
        plane_res = ti.Vector([res_x, res_y, res_z, res_t])
        pos = ti.Vector([xyzts[i, 0], xyzts[i, 1], xyzts[i, 2], xyzts[i, 3]]) * plane_res

        pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)  # floor
        pos_grid_uint = ti.math.clamp(pos_grid_uint, 0, plane_res - 1)
        pos -= pos_grid_uint  # pos now represents frac
        pos = ti.math.clamp(pos, 0.0, 1.0)

        offset = offsets[level]

        indicator = hash_map_indicator[level]
        map_size = hash_map_sizes_field[level]

        local_feature_0 = 0.0
        local_feature_1 = 0.0

        for idx in ti.static(range(16)):
            w = 1.
            pos_grid_local = uvec4(0)

            for d in ti.static(range(4)):
                t = linear_step(pos[d])
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d] = pos_grid_uint[d]
                    w *= 1 - t
                else:
                    pos_grid_local[d] = pos_grid_uint[d] + 1
                    w *= t

            index = grid_pos2hash_index(indicator, pos_grid_local, plane_res, map_size)
            index_table = offset + index * 2  # the flat index for the 1st entry
            index_table_int = ti.cast(index_table, ti.int32)
            local_feature_0 += w * table[index_table_int]
            local_feature_1 += w * table[index_table_int + 1]

        xyzts_embedding[i, level * 2] = local_feature_0
        xyzts_embedding[i, level * 2 + 1] = local_feature_1


@ti.kernel
def hash_encode_kernel_grad(
        xyzts: ti.template(), table: ti.template(),
        xyzts_embedding: ti.template(), hash_map_indicator: ti.template(),
        hash_map_sizes_field: ti.template(), hash_map_shapes_field: ti.template(),
        offsets: ti.template(), B: ti.i32, num_scales: ti.i32, xyzts_grad: ti.template(), table_grad: ti.template(),
        output_grad: ti.template()):
    # # # get hash table embedding

    ti.loop_config(block_dim=16)
    for i, level in ti.ndrange(B, num_scales):
        res_x = hash_map_shapes_field[level, 0]
        res_y = hash_map_shapes_field[level, 1]
        res_z = hash_map_shapes_field[level, 2]
        res_t = hash_map_shapes_field[level, 3]
        plane_res = ti.Vector([res_x, res_y, res_z, res_t])
        pos = ti.Vector([xyzts[i, 0], xyzts[i, 1], xyzts[i, 2], xyzts[i, 3]]) * plane_res

        pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)  # floor
        pos_grid_uint = ti.math.clamp(pos_grid_uint, 0, plane_res - 1)
        pos -= pos_grid_uint  # pos now represents frac
        pos = ti.math.clamp(pos, 0.0, 1.0)

        offset = offsets[level]

        indicator = hash_map_indicator[level]
        map_size = hash_map_sizes_field[level]

        local_feature_0 = 0.0
        local_feature_1 = 0.0

        for idx in ti.static(range(16)):
            w = 1.
            pos_grid_local = uvec4(0)
            dw = ti.Vector([0., 0., 0., 0.])
            # prods = ti.Vector([0., 0., 0.,0.])
            for d in ti.static(range(4)):
                t = linear_step(pos[d])
                dt = d_linear_step(pos[d])
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d] = pos_grid_uint[d]
                    w *= 1 - t
                    dw[d] = -dt

                else:
                    pos_grid_local[d] = pos_grid_uint[d] + 1
                    w *= t
                    dw[d] = dt

            index = grid_pos2hash_index(indicator, pos_grid_local, plane_res, map_size)
            index_table = offset + index * 2  # the flat index for the 1st entry
            index_table_int = ti.cast(index_table, ti.int32)
            table_grad[index_table_int] += w * output_grad[i, 2 * level]
            table_grad[index_table_int + 1] += w * output_grad[i, 2 * level + 1]
            for d in ti.static(range(4)):
                # eps = 1e-15
                # prod = w / ((linear_step(pos[d]) if idx & (1 << d) > 0 else 1 - linear_step(pos[d])) + eps)
                # prod=1.0
                # for k in range(4):
                #     if k == d:
                #         prod *= dw[k]
                #     else:
                #         prod *= 1- linear_step(pos[k]) if (idx & (1 << k) == 0) else linear_step(pos[k])
                prod = dw[d] * (
                    linear_step(pos[(d + 1) % 4]) if (idx & (1 << ((d + 1) % 4)) > 0) else 1 - linear_step(
                        pos[(d + 1) % 4])
                ) * (
                           linear_step(pos[(d + 2) % 4]) if (idx & (1 << ((d + 2) % 4)) > 0) else 1 - linear_step(
                               pos[(d + 2) % 4])
                       ) * (
                           linear_step(pos[(d + 3) % 4]) if (idx & (1 << ((d + 3) % 4)) > 0) else 1 - linear_step(
                               pos[(d + 3) % 4])
                       )
                xyzts_grad[i, d] += table[index_table_int] * prod * plane_res[d] * output_grad[i, 2 * level]
                xyzts_grad[i, d] += table[index_table_int + 1] * prod * plane_res[d] * output_grad[i, 2 * level + 1]


@ti.kernel
def hash_encode_kernel_smoothstep_grad(
        xyzts: ti.template(), table: ti.template(),
        xyzts_embedding: ti.template(), hash_map_indicator: ti.template(),
        hash_map_sizes_field: ti.template(), hash_map_shapes_field: ti.template(),
        offsets: ti.template(), B: ti.i32, num_scales: ti.i32, xyzts_grad: ti.template(), table_grad: ti.template(),
        output_grad: ti.template()):
    # # # get hash table embedding

    ti.loop_config(block_dim=16)
    for i, level in ti.ndrange(B, num_scales):
        res_x = hash_map_shapes_field[level, 0]
        res_y = hash_map_shapes_field[level, 1]
        res_z = hash_map_shapes_field[level, 2]
        res_t = hash_map_shapes_field[level, 3]
        plane_res = ti.Vector([res_x, res_y, res_z, res_t])
        pos = ti.Vector([xyzts[i, 0], xyzts[i, 1], xyzts[i, 2], xyzts[i, 3]]) * plane_res

        pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)  # floor
        pos_grid_uint = ti.math.clamp(pos_grid_uint, 0, plane_res - 1)
        pos -= pos_grid_uint  # pos now represents frac
        pos = ti.math.clamp(pos, 0.0, 1.0)

        offset = offsets[level]

        indicator = hash_map_indicator[level]
        map_size = hash_map_sizes_field[level]

        local_feature_0 = 0.0
        local_feature_1 = 0.0

        for idx in ti.static(range(16)):
            w = 1.
            pos_grid_local = uvec4(0)
            dw = ti.Vector([0., 0., 0., 0.])
            # prods = ti.Vector([0., 0., 0.,0.])
            for d in ti.static(range(4)):
                t = smooth_step(pos[d])
                dt = d_smooth_step(pos[d])
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d] = pos_grid_uint[d]
                    w *= 1 - t
                    dw[d] = -dt

                else:
                    pos_grid_local[d] = pos_grid_uint[d] + 1
                    w *= t
                    dw[d] = dt

            index = grid_pos2hash_index(indicator, pos_grid_local, plane_res, map_size)
            index_table = offset + index * 2  # the flat index for the 1st entry
            index_table_int = ti.cast(index_table, ti.int32)
            table_grad[index_table_int] += w * output_grad[i, 2 * level]
            table_grad[index_table_int + 1] += w * output_grad[i, 2 * level + 1]
            for d in ti.static(range(4)):
                # eps = 1e-15
                # prod = w / ((smooth_step(pos[d]) if idx & (1 << d) > 0 else 1 - smooth_step(pos[d])) + eps)
                # prod=1.0
                # for k in range(4):
                #     if k == d:
                #         prod *= dw[k]
                #     else:
                #         prod *= 1- smooth_step(pos[k]) if (idx & (1 << k) == 0) else smooth_step(pos[k])
                prod = dw[d] * (
                    smooth_step(pos[(d + 1) % 4]) if (idx & (1 << ((d + 1) % 4)) > 0) else 1 - smooth_step(
                        pos[(d + 1) % 4])
                ) * (
                           smooth_step(pos[(d + 2) % 4]) if (idx & (1 << ((d + 2) % 4)) > 0) else 1 - smooth_step(
                               pos[(d + 2) % 4])
                       ) * (
                           smooth_step(pos[(d + 3) % 4]) if (idx & (1 << ((d + 3) % 4)) > 0) else 1 - smooth_step(
                               pos[(d + 3) % 4])
                       )
                xyzts_grad[i, d] += table[index_table_int] * prod * plane_res[d] * output_grad[i, 2 * level]
                xyzts_grad[i, d] += table[index_table_int + 1] * prod * plane_res[d] * output_grad[i, 2 * level + 1]


class Hash4Encoder(torch.nn.Module):
    def __init__(self,
                 max_res=np.array([512, 512, 512, 512]),
                 min_res=np.array([16, 16, 16, 16]),
                 num_scales=16,
                 max_num_queries=10000000,
                 data_type=data_type,
                 max_params=2 ** 19,
                 interpolation='linear'
                 ):
        super(Hash4Encoder, self).__init__()

        b = np.exp((np.log(max_res) - np.log(min_res)) / (num_scales - 1))

        self.num_scales = num_scales
        self.interpolation = interpolation
        self.offsets = ti.field(ti.i32, shape=(num_scales,))
        self.hash_map_sizes_field = ti.field(ti.uint32, shape=(num_scales,))
        self.hash_map_shapes_field = ti.field(ti.uint32, shape=(num_scales, 4))
        self.hash_map_indicator = ti.field(ti.i32, shape=(num_scales,))

        offset_ = 0
        hash_map_sizes = []
        hash_map_shapes = []
        for i in range(num_scales):  # loop through each level
            res = np.ceil(min_res * np.power(b, i)).astype(int)
            hash_map_shapes.append(res)
            params_in_level_raw = (res[0] + 1) * (res[1] + 1) * (
                    res[2] + 1) * (res[3] + 1)  # number of params required to store everything
            params_in_level = int(params_in_level_raw) if params_in_level_raw % 8 == 0 \
                else int((params_in_level_raw + 8 - 1) / 8) * 8  # make sure is multiple of 8
            # if max_params has enough space, store everything; otherwise store as much as we can
            params_in_level = min(max_params, params_in_level)
            hash_map_sizes.append(params_in_level)
            self.hash_map_indicator[
                i] = 1 if params_in_level_raw <= params_in_level else 0  # i if have stored everything, 0 if collision
            self.offsets[i] = offset_
            offset_ += params_in_level * 2  # multiply by two because we store 2 features per entry
        print("hash map sizes", hash_map_sizes)
        print("hash map shapes", hash_map_shapes)
        print("offsets", self.offsets.to_numpy())
        print("hash map indicator", self.hash_map_indicator.to_numpy())
        size = np.uint32(np.array(hash_map_sizes))
        self.hash_map_sizes_field.from_numpy(size)
        shape = np.uint32(np.array(hash_map_shapes))
        self.hash_map_shapes_field.from_numpy(shape)

        self.total_hash_size = offset_

        # the main storage, pytorch
        self.hash_table = torch.nn.Parameter(torch.zeros(self.total_hash_size,
                                                         dtype=torch_type),
                                             requires_grad=True)
        random_initialize(self.hash_table)  # randomly initialize

        # the taichi counterpart of self.hash_table
        self.parameter_fields = ti.field(data_type,
                                         shape=(self.total_hash_size,),
                                         needs_grad=True)

        # output fields will have num_scales * 2 entries (2 features per scale)
        self.output_fields = ti.field(dtype=data_type,
                                      shape=(max_num_queries, num_scales * 2),
                                      needs_grad=True)
        if interpolation == 'linear':
            self._hash_encode_kernel = hash_encode_kernel
            self._hash_encode_kernel_grad = hash_encode_kernel_grad
        elif interpolation == 'smoothstep':
            self._hash_encode_kernel = hash_encode_kernel_smoothstep
            self._hash_encode_kernel_grad = hash_encode_kernel_smoothstep_grad
        else:
            raise NotImplementedError
        # input assumes a dimension of 4
        self.input_fields = ti.field(dtype=data_type,
                                     shape=(max_num_queries, 4),
                                     needs_grad=True)
        self.input_fields_grad = ti.field(dtype=data_type,
                                          shape=(max_num_queries, 4),
                                          needs_grad=True)
        self.parameter_fields_grad = ti.field(dtype=data_type,
                                              shape=(self.total_hash_size,),
                                              needs_grad=True)
        self.output_grad = ti.field(dtype=data_type,
                                    shape=(max_num_queries, num_scales * 2),
                                    needs_grad=True)

        self.register_buffer(
            'hash_grad',
            torch.zeros(self.total_hash_size, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'hash_grad2',
            torch.zeros(self.total_hash_size, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'input_grad',
            torch.zeros(max_num_queries, 4, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'input_grad2',
            torch.zeros(max_num_queries, 4, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'output_embedding',
            torch.zeros(max_num_queries, num_scales * 2, dtype=torch_type),
            persistent=False
        )

        class _module_function(torch.autograd.Function):
            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, input_pos, params):
                output_embedding = self.output_embedding[:input_pos.
                    shape[0]].contiguous(
                )
                torch2ti(self.input_fields, input_pos.contiguous())
                torch2ti(self.parameter_fields, params.contiguous())

                self._hash_encode_kernel(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.hash_map_indicator,
                    self.hash_map_sizes_field,
                    self.hash_map_shapes_field,
                    self.offsets,
                    input_pos.shape[0],
                    self.num_scales,
                )
                ti2torch(self.output_fields, output_embedding)
                ctx.save_for_backward(input_pos, params)
                return output_embedding

            @staticmethod
            @custom_bwd
            def backward(ctx, doutput):
                self.zero_grad()
                input_pos, params = ctx.saved_tensors
                return self._module_function_grad.apply(input_pos, params, doutput)

        class _module_function_ad(torch.autograd.Function):

            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, input_pos, params):
                output_embedding = self.output_embedding[:input_pos.
                    shape[0]].contiguous(
                )
                torch2ti(self.input_fields, input_pos.contiguous())
                torch2ti(self.parameter_fields, params.contiguous())

                self._hash_encode_kernel(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.hash_map_indicator,
                    self.hash_map_sizes_field,
                    self.hash_map_shapes_field,
                    self.offsets,
                    input_pos.shape[0],
                    self.num_scales,
                )
                ti2torch(self.output_fields, output_embedding)
                ctx.save_for_backward(input_pos, params)
                return output_embedding

            @staticmethod
            @custom_bwd
            def backward(ctx, doutput):
                self.zero_grad()

                torch2ti_grad(self.output_fields, doutput.contiguous())
                self._hash_encode_kernel.grad(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.hash_map_indicator,
                    self.hash_map_sizes_field,
                    self.hash_map_shapes_field,
                    self.offsets,
                    doutput.shape[0],
                    self.num_scales,
                )
                ti2torch_grad(self.parameter_fields,
                              self.hash_grad.contiguous())
                ti2torch_grad(self.input_fields, self.input_grad.contiguous()[:doutput.shape[0]])
                return self.input_grad[:doutput.shape[0]], self.hash_grad

        class _module_function_grad(torch.autograd.Function):
            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, input_pos, params, doutput):
                torch2ti(self.input_fields, input_pos.contiguous())
                torch2ti(self.parameter_fields, params.contiguous())
                torch2ti(self.output_grad, doutput.contiguous())
                self._hash_encode_kernel_grad(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.hash_map_indicator,
                    self.hash_map_sizes_field,
                    self.hash_map_shapes_field,
                    self.offsets,
                    doutput.shape[0],
                    self.num_scales,
                    self.input_fields_grad,
                    self.parameter_fields_grad,
                    self.output_grad
                )

                ti2torch(self.input_fields_grad, self.input_grad.contiguous())
                ti2torch(self.parameter_fields_grad, self.hash_grad.contiguous())
                return self.input_grad[:doutput.shape[0]], self.hash_grad

            @staticmethod
            @custom_bwd
            def backward(ctx, d_input_grad, d_hash_grad):
                self.zero_grad_2()
                torch2ti_grad(self.input_fields_grad, d_input_grad.contiguous())
                torch2ti_grad(self.parameter_fields_grad, d_hash_grad.contiguous())
                self._hash_encode_kernel_grad.grad(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.hash_map_indicator,
                    self.hash_map_sizes_field,
                    self.hash_map_shapes_field,
                    self.offsets,
                    d_input_grad.shape[0],
                    self.num_scales,
                    self.input_fields_grad,
                    self.parameter_fields_grad,
                    self.output_grad
                )
                ti2torch_grad(self.input_fields, self.input_grad2.contiguous()[:d_input_grad.shape[0]])
                ti2torch_grad(self.parameter_fields, self.hash_grad2.contiguous())
                # set_trace(term_size=(120,30))
                return self.input_grad2[:d_input_grad.shape[0]], self.hash_grad2, None

        self._module_function = _module_function
        self._module_function_grad = _module_function_grad

    def zero_grad(self):
        self.parameter_fields.grad.fill(0.)
        self.input_fields.grad.fill(0.)
        self.input_fields_grad.fill(0.)
        self.parameter_fields_grad.fill(0.)

    def zero_grad_2(self):
        self.parameter_fields.grad.fill(0.)
        self.input_fields.grad.fill(0.)
        # self.input_fields_grad.grad.fill(0.)
        # self.parameter_fields_grad.grad.fill(0.)

    def forward(self, positions):
        # positions: (N, 4), normalized to [-1, 1]
        positions = positions * 0.5 + 0.5
        return self._module_function.apply(positions, self.hash_table)

if __name__ == '__main__':
    ti.init(arch=ti.cpu, device_memory_GB=4.0)

    import torch.nn as nn
    import torch.nn.functional as F

    print(torch.__version__)


    class NeRFSmallPotential(nn.Module):
        def __init__(self,
                     num_layers=3,
                     hidden_dim=64,
                     geo_feat_dim=15,
                     num_layers_color=2,
                     hidden_dim_color=16,
                     input_ch=3,
                     use_f=False
                     ):
            super(NeRFSmallPotential, self).__init__()

            self.input_ch = input_ch
            self.rgb = torch.nn.Parameter(torch.tensor([0.0]))

            # sigma network
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            self.geo_feat_dim = geo_feat_dim

            sigma_net = []
            for l in range(num_layers):
                if l == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = hidden_dim

                if l == num_layers - 1:
                    out_dim = hidden_dim  # 1 sigma + 15 SH features for color
                else:
                    out_dim = hidden_dim

                sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
            self.sigma_net = nn.ModuleList(sigma_net)
            self.out = nn.Linear(hidden_dim, 3, bias=True)
            self.use_f = use_f
            if use_f:
                self.out_f = nn.Linear(hidden_dim, hidden_dim, bias=True)
                self.out_f2 = nn.Linear(hidden_dim, 3, bias=True)

        def forward(self, x):
            h = x
            for l in range(self.num_layers):
                h = self.sigma_net[l](h)
                h = F.relu(h, True)

            v = self.out(h)
            if self.use_f:
                f = self.out_f(h)
                f = F.relu(f, True)
                f = self.out_f2(f)
            else:
                f = v * 0
            return v, f



    # embedding = h(x)
    network_vel = NeRFSmallPotential(input_ch=32)
    embed_vel = Hash4Encoder()

    pts = torch.rand(100, 4)
    pts.requires_grad = True
    with torch.enable_grad():
        h = embed_vel(pts)
        vel_output, f_output = network_vel(h)

        print('vel_output', vel_output.shape)
        print('h', h.shape)
        def g(x):
            return network_vel(x)[0]

        jac = vmap(jacrev(g))(h)
        print('jac', jac.shape)
        jac_x = [] #_get_minibatch_jacobian(h, pts)
        for j in range(h.shape[1]):
            dy_j_dx = torch.autograd.grad(
                h[:, j],
                pts,
                torch.ones_like(h[:, j], device='cpu'),
                retain_graph=True,
                create_graph=True,
            )[0].view(pts.shape[0], -1)
            jac_x.append(dy_j_dx.unsqueeze(1))
        jac_x = torch.cat(jac_x, dim=1)
        print(jac_x.shape)
        jac = jac @ jac_x
        assert jac.shape == (pts.shape[0], 3, 4)
        _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]  # (N,1)

    jac = torch.stack([_u_x, _u_y, _u_z], dim=-1)  # [N, 3, 3]
    curl = torch.stack([jac[:, 2, 1] - jac[:, 1, 2],
                        jac[:, 0, 2] - jac[:, 2, 0],
                        jac[:, 1, 0] - jac[:, 0, 1]], dim=-1)  # [N, 3]
    # curl = curl.view(list(pts_shape[:-1]) + [3])  # [..., 3]
    print(curl.shape)
    vorticity_norm = torch.norm(curl, dim=-1, keepdim=True)

    vorticity_norm_grad = []

    print(vorticity_norm.shape)
    for j in range(vorticity_norm.shape[1]):
        # breakpoint()

        dy_j_dx = torch.autograd.grad(
            vorticity_norm[:, j],
            pts,
            torch.ones_like(vorticity_norm[:, j], device='cpu'),
            retain_graph=True,
            create_graph=True,
        )[0]
        vorticity_norm_grad.append(dy_j_dx.unsqueeze(1))
    vorticity_norm_grad = torch.cat(vorticity_norm_grad, dim=1)
    print(vorticity_norm_grad.shape)