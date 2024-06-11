# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%%
import paddle
from paddle import base
from paddle import pir
import numpy as np
from pyparsing import opAssoc

pass_attr_list = [
    # {
    #     'fc_fuse_pass': {}
    # },
    {
        'trt_sub_graph_extract_pass': {}
    },
]

def run_pir_pass(program):
    pm = pir.PassManager(opt_level=4)
    pm.enable_print_statistics()
    pm.enable_ir_printing()
    for pass_item in pass_attr_list:
        for pass_name, pass_attr in pass_item.items():
            pm.add_pass(pass_name, pass_attr)
    pm.run(program)
    return program

def get_program():
    with paddle.pir_utils.IrGuard():
        main_program = paddle.static.Program()
        default_startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, default_startup_program):
            scope = paddle.static.global_scope()
            input = paddle.static.data(
                shape=[-1, 512, 64], dtype='float32', name='input'
            )
            # 第一种创建weight方式
            # weight = paddle.full(
            #     shape=[64, 64], fill_value=0.3, dtype='float32'
            # )
            # 第二种创建weight 的方式
            weight_numpy = np.random.rand(64, 64).astype('float32')
            weight = paddle.create_parameter(
                name="w",
                shape=[64, 64],
                dtype='float32',
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(weight_numpy)
                ),
            )
            bias_numpy = np.random.rand(64).astype('float32')
            bias = paddle.create_parameter(
                name="b",
                shape=[64],
                dtype='float32',
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(bias_numpy)
                ),
            )
            # bias = paddle.full(shape=[64], fill_value=1.0, dtype='float32')
            x = paddle.matmul(input, weight)
            y = paddle.add(x, bias)
            y = paddle.nn.functional.relu(y)
            y = paddle.nn.functional.relu(y)
            # y = paddle.nn.functional.relu(y)
            # y = paddle.nn.functional.relu(y)
            # y = paddle.nn.functional.gelu(y)
            # y = paddle.nn.functional.silu(y)
            # y = paddle.nn.functional.leaky_relu(y)
            # y = paddle.nn.functional.swish(y)
        main_program = run_pir_pass(main_program)
        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        exe.run(default_startup_program)
        
        params = main_program.global_block().all_parameters()
        param_dict = {}
        # save parameters
        for v in params:
            name = v.get_defining_op().attrs()["parameter_name"]
            param_dict.update({name: np.array(scope.var(name).get_tensor())})
        # file_path = "save_my_file.json"
        # pir_version = 1
        # base.core.serialize_pir_program(
        #     main_program, file_path, pir_version
        # )
    return main_program, scope, param_dict
    
program, scope, param_dict = get_program()
for i in range(len(program.global_block().ops)):
    op = program.global_block().ops[i]
    if op.name() == "cinn_op.group":
        print(op)
        sub_blocks = list(op.blocks())[0].ops
        break

#%%
# %%
results_map = {}
for i in range(len(sub_blocks)):
    values = sub_blocks[i].results()
    for j in range(len(values)):
        value_j = values[j]
        results_map.update({value_j.id: value_j})

inputs_map = {}
for i in range(len(sub_blocks)):
    ll = []
    for j in range(len(sub_blocks[i].operands())):
        value = sub_blocks[i].operands()[j].source()
        inputs_map.update({value.id: value})

print(results_map)
print(inputs_map)
# %%
# sub_blocks[0].results()[0].all_used_ops()
# sub_blocks[1].operands()[0].source().all_used_ops()

sub_blocks[0].operands()[0].source().get_defining_op().attrs()["parameter_name"]
# %%
# %%
sub_blocks[1].operands()[0].source().all_used_ops()
# %%

# %%
def find_graph_inputs_and_outputs(operations):
    all_values = {}
    output_values = {}
    graph_output_values = []
    
    def __is_output_value(value):
        for op in value.all_used_ops():
            if op.name() == "cf.yield":
                return True
        return False

    # Collect all output values from all operations
    for op in operations:
        for result in op.results():
            output_values[result.id] = result
            all_values[result.id] = result
            if __is_output_value(result):
                graph_output_values.append(result)
            
        for operand in op.operands():
            source = operand.source()
            all_values[source.id] = source

    # Input values are those that are in all_values but not in output_values
    input_values = [value for value_id, value in all_values.items() if value_id not in output_values]

    return input_values, graph_output_values

inputs, outputs = find_graph_inputs_and_outputs(sub_blocks)
# %%
sub_blocks[-2].results()[0].all_used_ops()
# %%
outputs
# %%

# %%
import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE  )


builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
            
# %%
dir(network)
# %%
