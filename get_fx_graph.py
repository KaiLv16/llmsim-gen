import torch
import math
import warnings
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from nets.resnet50 import Bottleneck, ResNet
import torch.fx as fx
import pandas as pd
from functools import reduce
from operator import mul

from utils_get_fx_graph import *
from hardware_model import *
from flow_definition import *
from collective_communication_lib import *


class LLMSimWarning(UserWarning):
    pass


# 提取GraphModule信息并转换为DataFrame
def graph_module_to_dataframe(graph_module, input_size, batch_size=1):
    # # 前向传播过程中记录每个节点的输出形状
    input_shapes = {}
    output_shapes = {}
    param_shapes = {}
    param_num = {}
    dtype_size = {}
    layer_types = {}
    input_output_shapes = {}

    count_dict = {}
    def update_count(key):
        if key in count_dict:
            count_dict[key] += 1
        else:
            count_dict[key] = 0
        return count_dict[key]

    def record_input_output_shapes(module, input, output):
        module_name = ''
        # print("--------------------------------------------------")
        for name, mod in model.named_modules():
            # print(name, mod)
            
            if mod is module:
                module_name = name
                index = update_count(module_name)
                if index > 0:
                    module_name = name.replace('.', '_') + '_' + str(index)
                else:
                    module_name = name.replace('.', '_')
                # print(module_name)
                break
        
        input_shapes[module_name] = input[0].shape if isinstance(input, tuple) else input.shape
        output_shapes[module_name] = output.shape
        # 记录参数形状
        param_shapes[module_name] = {name: p.shape for name, p in module.named_parameters()}
        param_size = 0
        for _, p in module.named_parameters():
            param_size += torch.prod(torch.tensor(p.shape)).item()
        param_num[module_name] = int(param_size)
        dtype_size[module_name] = [output.element_size()]
        layer_types[module_name] = type(module).__name__

        input_output_shapes[module_name] = {
            'input_shape': input_shapes[module_name],
            'output_shape': output_shapes[module_name],
            'Param #': param_num[module_name],
            'param_shape': param_shapes[module_name],
            'dtype_size': dtype_size[module_name],
            'layer_type': layer_types[module_name]
        }

    # 注册钩子
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Sequential):
            hooks.append(module.register_forward_hook(record_input_output_shapes))

    # 前向传播,  创建一个dummy输入
    x = torch.rand(batch_size, *input_size)
    with torch.no_grad():
        model(x)

    # 移除钩子
    for hook in hooks:
        hook.remove()


    node_info = {
        "name": [],
        "args": [],
        "input_#": [],
        "output_#": [],
        "Param_#": [],
        "dp_index": 0,
        "pp_index": 0,
        "tp_index": 0,  # currently not used

        "opcode": [],
        "target": [],
        "layer_type": [],
        "input_shape": [],
        "output_shape": [],
        "param_shape": [],
        "kwargs": [],
        "dtype_size": []
    }

    for node in graph_module.graph.nodes:
        node_info["name"].append(node.name)
        node_info["opcode"].append(node.op)
        node_info["target"].append(node.target)
        node_info["args"].append(node.args)
        node_info["kwargs"].append(node.kwargs)

        # 获取输入和输出形状以及参数形状
        # print(node.name)
        if node.op == 'call_module' or node.name in input_output_shapes:
            node_info["input_shape"].append(list(input_output_shapes[node.name]['input_shape']))
            node_info["output_shape"].append(list(input_output_shapes[node.name]['output_shape']))
            node_info["param_shape"].append(input_output_shapes[node.name]['param_shape'])
            node_info["dtype_size"].append(input_output_shapes[node.name]['dtype_size'])
            node_info["input_#"].append(reduce(mul, list(input_output_shapes[node.name]['input_shape'])))
            node_info["output_#"].append(reduce(mul, list(input_output_shapes[node.name]['output_shape'])))
            node_info["Param_#"].append(input_output_shapes[node.name]['Param #'])
            node_info["layer_type"].append(input_output_shapes[node.name]['layer_type'])
        else:
            if len(node_info["output_shape"]) > 0:     # 用于处理原本为None的情况：给前后层连起来
                node_info["input_shape"].append(node_info["output_shape"][-1])
                node_info["output_shape"].append(node_info["output_shape"][-1])
                node_info["param_shape"].append(0)
                node_info["dtype_size"].append(node_info["dtype_size"][-1])
                node_info["input_#"].append(reduce(mul, node_info["output_shape"][-1]))
                node_info["output_#"].append(reduce(mul, node_info["output_shape"][-1]))
                node_info["Param_#"].append(0)
                node_info["layer_type"].append(None)    # dont care
            else:
                node_info["input_shape"].append(None)
                node_info["output_shape"].append(None)
                node_info["param_shape"].append(None)
                node_info["dtype_size"].append(None)
                node_info["input_#"].append(0)
                node_info["output_#"].append(0)
                node_info["Param_#"].append(0)
                node_info["layer_type"].append(None)

    df = pd.DataFrame(node_info)
    return df


# 支持的模式： 'avg_mem' 和 'min_comm'
def model_pp_cut(df, stages=8, cut_mode='avg_mem', pp_grp_label='avg_mem_grp_idx', debug_mode=False):
    print("cut_mode =", cut_mode)
    if stages < 1:
        stages = 1
    if debug_mode:
        target_device_num = 2
        debug_update_dataframe(df, 'Param_#', target_device_num, pp_grp_label)
        return
    
    if 'avg_mem' in cut_mode:
        # print("avg_mem!")
        update_dataframe(df, 'Param_#', stages, pp_grp_label)

    elif 'min_comm' in cut_mode:
        # print("min_comm!")
        avg_param_per_host = df['Param_#'].sum() / stages
        # 单核版本有点bug（边界条件）
        # find_feasible_partition_min_comm_single_core(df, stages, avg_param_per_host * 2)
        find_feasible_partition_min_comm_multi_core(df, stages, avg_param_per_host * 2)
    else:
        print("Error: In function 'model_pp_cut()': cut_mode unrecognized! ")


# currently support 'avg_mem'.
def get_compute_density_ops_and_time(df, fwd_df=None, pass_type='FWD'):
    assert pass_type in ['FWD', 'BKWD'], f"Unrecognized pass_type {pass_type}!"
    df['compute_density_ops'] = 0
    for i in range(len(df)):
        if df['layer_type'][i] == None:
            df.at[i, 'compute_density_ops'] = df['input_#'][i]

        else:
            if df['layer_type'][i].lower() == 'conv2d'.lower():
                kernel_shape = list(df['param_shape'][i]['weight'])[-3:]
                df.at[i, 'compute_density_ops'] = reduce(mul, kernel_shape) * df['output_#'][i]

            elif df['layer_type'][i].lower() == 'BatchNorm2d'.lower():
                a = df['input_#'][i]
                b = list(df['param_shape'][i]['weight'])[0]
                c = list(df['param_shape'][i]['bias'])[0]
                calc_avg = a + 4 * b     # a次加法 b次除法。认为除法是加法的4倍
                calc_square_err = a * (1 + 2 + 1) + 4 * b  # a次减法 a次平方 a次加法 b次除法
                calc_norm = a * (1 + 4 + 4 + 1)  # 对每个输入进行归一化需要a次减法、平方根、除法和加法
                calc_result = a * (1 + 2)   # 对每个输入应用缩放和平移需要a次乘法和加法
                df.at[i, 'compute_density_ops'] = calc_avg + calc_square_err + calc_norm + calc_result

            elif df['layer_type'][i].lower() == 'ReLU'.lower():
                df.at[i, 'compute_density_ops'] = df['input_#'][i] * 4

            elif df['layer_type'][i].lower() == 'MaxPool2d'.lower():
                pool_num = df['output_#'][i]
                compare_per_pool = df['input_#'][i] / df['output_#'][i]
                df.at[i, 'compute_density_ops'] = pool_num * compare_per_pool

            elif df['layer_type'][i].lower() == 'AvgPool2d'.lower():
                pool_num = df['output_#'][i]
                compare_per_pool = df['input_#'][i] / df['output_#'][i] + 4
                df.at[i, 'compute_density_ops'] = pool_num * compare_per_pool

            elif df['layer_type'][i].lower() == 'Linear'.lower():
                batch_size = math.sqrt(df['input_#'][i] * df['output_#'][i])
                calc_mul = df['Param_#'][i] * batch_size * (1 + 2)   # 乘加
                calc_offset = df['output_#'][i]
                df.at[i, 'compute_density_ops'] = calc_mul + calc_offset

            else:
                raise Exception("Unrecognized layer type!")
        
            # Conv2D: 约 2 倍
            # BatchNorm2D: 约 2 倍
            # ReLU: 1 倍
            # MaxPool2D: 约 2 倍
            # AvgPool2D: 1 倍
            # Linear: 约 2 倍
            
            if pass_type == 'BKWD':
                fwd_cpt = -1
                for j in range(len(fwd_df)):
                    if fwd_df['name'][j] == df['name'][i]:
                        fwd_cpt = fwd_df['compute_density_ops'][j]
                        break
                if fwd_cpt == -1:
                    raise Exception(f"layer {df['name'][i]} not found in fwd_df!")
                if df['layer_type'][i].lower() in ['conv2d', 'batchnorm2d', 'maxpool2d', 'linear']:
                    df.at[i, 'compute_density_ops'] = fwd_cpt * 2
                else:
                    df.at[i, 'compute_density_ops'] = fwd_cpt


def bkwd_df_gen(df):
    # 反转DataFrame的行顺序
    def generate_bkwd_df(fwd_df):
        # 初始化输出节点的依赖字典
        output_dependencies = defaultdict(list)
        # 填充输出节点的依赖字典
        for idx, row in fwd_df.iterrows():
            print(idx, row['args'])
            for arg in row['args']:
                output_dependencies[str(arg)].append(row['name'])
        # 逆序 fwd_df，并保持其他列不变
        # print("output_dependencies:")
        # print(output_dependencies)

        reversed_df = fwd_df.iloc[::-1].reset_index(drop=True)
        # 更新 reversed_df 中的依赖关系
        for i in range(len(reversed_df)):
            reversed_df.at[i, 'args'] = tuple(output_dependencies[reversed_df['name'][i]])
                
        # print(fwd_df)
        # print(reversed_df)
        return reversed_df
    
    bkwd_df = generate_bkwd_df(df)
    
    # 修改其他性质
    for i in range(len(bkwd_df)):
        bkwd_df.at[i, 'direction'] = 'BKWD'
        bkwd_df.at[i, 'input_#'], bkwd_df.at[i, 'output_#'] = bkwd_df['output_#'][i], bkwd_df['input_#'][i]
        bkwd_df.at[i, 'input_shape'], bkwd_df.at[i, 'output_shape'] = bkwd_df['output_shape'][i], bkwd_df['input_shape'][i]
    return bkwd_df


def sim_node_gen(fwd_df, bkwd_df, batches=3, microbatch_num_per_batch=10, pp_mode='F_then_B', pp_granularity='layer', pp_grp_label='avg_mem_grp_idx', enable_pp_backpressure=True, dp_overlap=False, dp_granularity='device', bypass_first_fwd=False, bypass_last_bkwd=False):
    print(f"\nEnter sim_node_gen(). pp_mode={pp_mode}, bypass_first_fwd={bypass_first_fwd}, bypass_last_bkwd={bypass_last_bkwd}")
    assert pp_mode in ['native', 'F_then_B', '1F1B'], f"Only support ['native', 'F_then_B', '1F1B'] currently while you set {pp_mode}"
    assert pp_granularity in ['device', 'layer'], f"Only support ['device', 'layer'] currently while you set {pp_granularity}"
    assert batches > 0, f'You specified batches = {batches}, which <= 0!'
    assert len(fwd_df) == len(bkwd_df), f"Error: len(fwd_df) ({len(fwd_df)}) != len(bkwd_df) ({len(bkwd_df)})"
    assert ~(batches == 1 and bypass_first_fwd == True and bypass_last_bkwd == True), \
            f"batches = 1, bypass_first_fwd = True and bypass_last_bkwd = True together cannot form a single pass. "
    
    def add_extra_idx(layers_tmp, b_idx, pass_type, mb_idx, char_br='/'):
        layers = layers_tmp.copy(deep=True)
        for j in range(len(layers)):        # for each layer in current model template
            my_args = []
            for s in layers['args'][j]:     # for each dependency of this layer
                if isinstance(s, torch.fx.node.Node):
                    my_args.append(s.name + char_br + str(b_idx) + char_br + str(mb_idx) + char_br + str(pass_type.lower()))
                elif isinstance(s, int):
                    # print(f"{b_idx}-{mb_idx // 2}: Layer {layers['name'][j]} has args {s} that is not of type 'torch.fx.node.Node', but {type(s)}")
                    my_args.append(s)
                elif isinstance(s, str):
                    my_args.append(s + char_br + str(b_idx) + char_br + str(mb_idx) + char_br + str(pass_type.lower()))
                else:
                    raise Exception(f"{b_idx}-{mb_idx}-{pass_type}: Layer {layers['name'][j]} has args {s} that is of type {type(s)}")
            layers.at[j, 'args'] = tuple(my_args)
            layers.at[j, 'name'] = layers['name'][j] + char_br + str(b_idx) + char_br + str(mb_idx) + char_br + str(pass_type.lower())
            # 得到的各个节点的名称前缀，类似于 "layer1_0_conv1/0/FWD/"
        return layers
        
    def get_model_idx_for_pp(seq_id, pass_agg=True):
        '''
        pass_agg 标志了如何排序。
        如果为真，则microbatch之间是类似于“fwd1-fwd2-fwd3-fwd4 - bkwd1-bkwd2-bkwd3-bkwd4”的形式排列
        否则，microbatch之间是类似于“fwd1-bkwd1 - fwd2-bkwd2 - fwd3-bkwd3 - fwd4-bkwd4”的形式排列
        '''
        is_first_pass = False
        is_last_pass = False
        if bypass_first_fwd == True:
            # assert pass_agg == True, f"bypass_first_fwd cannot obtained without pass_agg. "
            if pass_agg == True:
                seq_id += microbatch_num_per_batch
            else:
                seq_id += 1
        b_idx = seq_id // (2 * microbatch_num_per_batch)
        if pass_agg:
            mb_idx = seq_id % microbatch_num_per_batch
            if mb_idx == 0:
                is_first_pass = True
            if mb_idx == microbatch_num_per_batch - 1:
                is_last_pass = True
            pass_type = 'FWD' if seq_id - b_idx * (2 * microbatch_num_per_batch) < microbatch_num_per_batch else 'BKWD'
        else:
            mb_idx = (seq_id - b_idx * (2 * microbatch_num_per_batch)) // 2
            pass_type = 'FWD' if (seq_id - b_idx * (2 * microbatch_num_per_batch)) % 2 == 0 else 'BKWD'
            if mb_idx == 0:
                is_first_pass = True
            if mb_idx == microbatch_num_per_batch - 1:
                is_last_pass = True
            
        return b_idx, mb_idx, pass_type, is_first_pass, is_last_pass


    def get_grp_boundary(layers, pp_grp_label, i):
        '''
        NOTE: the value returns are the indexs of dataframe.
        For FWD, max_idx is the latter layer, while for BKWD, max_index is the former layer.
        '''
        min_idx = i
        max_idx = i
        while layers[pp_grp_label][min_idx] == layers[pp_grp_label][i]:
            min_idx -= 1
            if min_idx < 0:
                break
        while layers[pp_grp_label][max_idx] == layers[pp_grp_label][i]:
            max_idx += 1
            if max_idx >= len(layers):
                break
        return min_idx + 1, max_idx - 1
    

    def get_last_pp_stage_index(df, pp_granularity, columnname, my_index, is_first=False):
        # assert not is_first, "get_last_pp_stage_index() is only designed to ease the calc of pp dependency when mb_idx > 0."
        if is_first:
            return None
        
        index = None
        if pp_granularity == 'layer':
            index == my_index
            return index
        
        if pp_granularity == 'device':
            min_bound_index, max_bound_index = get_grp_boundary(df, columnname, my_index)
            if min_bound_index == my_index:
                index = max_bound_index
            return index
        

    def add_dp_hook_with_priority(sim_df, layers, b_idx, mb_idx, pass_type, last_b_idx, last_pass_type, last_mb_idx, pp_grp_label, dp_granularity, dp_overlap, bucket_size=-1):
        '''
        return: the max dp_prio. 
        Should be applicable to <native> <F_then_B> and <1F1B>?
        If dp_overlap == True, FWD of the next batch will not start until all the parameters are syncronized. 
                                else, PP can overlap with DP, which sharts the next fwd as soon as the dependency of correspond layer (device).
        For simplicity, prio 0 is the LOW priority, followed by prio 1, 2, ...
        '''
        print(f"下一个 batch 的第一个 fwd: {b_idx}, {mb_idx}, {pass_type}")
        # 添加 DP 的 hook
        assert last_b_idx == b_idx - 1 and last_pass_type == 'BKWD' and last_mb_idx == microbatch_num_per_batch - 1
        dp_prio = 0
        if dp_granularity == 'layer':
            for i in range(len(sim_df[(last_b_idx, last_pass_type, last_mb_idx)])):
                size = sim_df[(last_b_idx, last_pass_type, last_mb_idx)]['Param_#'][i]
                if size > 0:
                    size *= sim_df[(last_b_idx, last_pass_type, last_mb_idx)]['dtype_size'][i][0]   # bytes
                dst_idx = (len(layers) - 1) - i if dp_overlap else 0    # fwd and bkwd have different layer sequence
                src_layer = sim_df[(last_b_idx, last_pass_type, last_mb_idx)]['name'][i]
                dst_layer = layers['name'][dst_idx]
                grp_id = layers[pp_grp_label][dst_idx]
                print(dst_idx)
                unique_dp_id = assign_dp_id()
                dp_meta = [unique_dp_id, size, src_layer, dst_layer, grp_id, dp_prio]
                dp_prio += 1
                print(dp_meta)
                dp_src_meta = ['src'] + dp_meta
                dp_dst_meta = ['dst'] + dp_meta
                sim_df[(last_b_idx, last_pass_type, last_mb_idx)].at[i, 'dp_args'] = \
                            sim_df[(last_b_idx, last_pass_type, last_mb_idx)].at[i, 'dp_args'] + [dp_src_meta]
                layers.at[dst_idx, 'dp_args'] = layers.at[dst_idx, 'dp_args'] + [dp_dst_meta]
                
        else:  # device DP. The default behavior, and can be used to test bucket split / aggregation with further modifications of dp_flow_gen module. 
            i = 0
            while i < len(sim_df[(last_b_idx, last_pass_type, last_mb_idx)]) - 1:
                min_idx_bkwd, max_idx_bkwd = get_grp_boundary(sim_df[(last_b_idx, last_pass_type, last_mb_idx)], pp_grp_label, i)
                grp_id = sim_df[(last_b_idx, last_pass_type, last_mb_idx)][pp_grp_label][i]   # bkwd pass, grp_id is in decrease sequence

                # bucket 参数聚合，不单发
                total_size = 0
                for idx in range(min_idx_bkwd, max_idx_bkwd + 1):
                    size = sim_df[(last_b_idx, last_pass_type, last_mb_idx)]['Param_#'][idx]
                    if size > 0:
                        size *= sim_df[(last_b_idx, last_pass_type, last_mb_idx)]['dtype_size'][idx][0]   # bytes
                    total_size += size
                
                dst_idx = 0
                min_idx_fwd = 0
                if dp_overlap:
                    dst_idx = (len(layers) - 1) - max_idx_bkwd
                    min_idx_fwd, max_idx_fwd = get_grp_boundary(layers, pp_grp_label, dst_idx)
                        # fwd and bkwd have different layer sequence
                    print(max_idx_bkwd, min_idx_fwd)
                    assert sim_df[(last_b_idx, last_pass_type, last_mb_idx)][pp_grp_label][max_idx_bkwd] == layers[pp_grp_label][min_idx_fwd]
                    assert sim_df[(last_b_idx, last_pass_type, last_mb_idx)][pp_grp_label][max_idx_bkwd-1] == layers[pp_grp_label][min_idx_fwd+1]
                    if min_idx_fwd != 0 and max_idx_bkwd != len(layers) - 1:
                        assert layers[pp_grp_label][min_idx_fwd] != layers[pp_grp_label][min_idx_fwd-1]
                
                src_layer = sim_df[(last_b_idx, last_pass_type, last_mb_idx)]['name'][max_idx_bkwd]
                dst_layer = layers['name'][min_idx_fwd]
                grp_id = layers[pp_grp_label][min_idx_fwd]
                unique_dp_id = assign_dp_id()
                dp_meta = [unique_dp_id, total_size, src_layer, dst_layer, grp_id, dp_prio]
                dp_prio += 1
                print(dp_meta)
                dp_src_meta = ['src'] + dp_meta
                dp_dst_meta = ['dst'] + dp_meta
                sim_df[(last_b_idx, last_pass_type, last_mb_idx)].at[max_idx_bkwd, 'dp_args'] = \
                            sim_df[(last_b_idx, last_pass_type, last_mb_idx)].at[max_idx_bkwd, 'dp_args'] + [dp_src_meta]
                layers.at[min_idx_fwd, 'dp_args'] = layers.at[min_idx_fwd, 'dp_args'] + [dp_dst_meta]
                i = max_idx_bkwd + 1
        return dp_prio

    pass_enum = {'FWD' : 0, 'BKWD' : 1}

    total_passes =  batches * microbatch_num_per_batch * 2
    if bypass_first_fwd:
        total_passes -= microbatch_num_per_batch
    if bypass_last_bkwd:
        total_passes -= microbatch_num_per_batch

    sim_df = OrderedDict()
    max_dp_prio_idx = 0

    # build the dependency across passes
    if pp_mode == 'native':
        # pp_granularity='layer', enable_pp_backpressure=True  没啥用
        if microbatch_num_per_batch > 1:
            warnings.warn("Typically you should not specify microbatch num > 1 when pp_mode = native, unless you know clearly what you're doing.", LLMSimWarning)
        # 如果 microbatch_num_per_batch > 1，那么就是n次 native PP 后接一次 DP。虽然这个模式不常用，但是也可以用来看看（sigcomm24）
        for seq_id in range(total_passes):
            b_idx, mb_idx, pass_type, _, _ = get_model_idx_for_pp(seq_id, pass_agg=False)
            if pass_type == 'FWD':
                layers = add_extra_idx(fwd_df, b_idx, pass_type, mb_idx, '_')
            else:
                layers = add_extra_idx(bkwd_df, b_idx, pass_type, mb_idx, '_')
            if seq_id > 0:
                last_b_idx, last_mb_idx, last_pass_type, _, _ = get_model_idx_for_pp(seq_id - 1, pass_agg=False)
                layers.at[0, 'args'] = tuple([sim_df[(last_b_idx, last_pass_type, last_mb_idx)].iloc[-1]['name']])

            if b_idx > 0 and mb_idx == 0 and pass_type == 'FWD':        # 下一个 batch 的第一个 fwd
                max_dp_prio_idx = add_dp_hook_with_priority(sim_df, layers, b_idx, mb_idx, pass_type, last_b_idx, last_pass_type, last_mb_idx, pp_grp_label, dp_granularity, dp_overlap)
            
            sim_df[(b_idx, pass_type, mb_idx)] = layers
            
    elif pp_mode == 'F_then_B':
        # 使用 pp_granularity='layer', enable_pp_backpressure=True 
        print(f"Total passes: {total_passes}")
        for seq_id in range(total_passes):
            b_idx, mb_idx, pass_type, is_first, is_last = get_model_idx_for_pp(seq_id, pass_agg=True)
            print(f"pass_info for seq_id {seq_id}\t", b_idx, mb_idx, pass_type, is_first, is_last)
            
            if pass_type == 'FWD':
                layers = add_extra_idx(fwd_df, b_idx, pass_type, mb_idx, '_')
            else:
                layers = add_extra_idx(bkwd_df, b_idx, pass_type, mb_idx, '_')

            if is_first and seq_id > 0:     # 每个batch的第一个 micro-batch。全局第一个bkwd不应该有前向依赖
                last_b_idx, last_mb_idx, last_pass_type, last_is_first, last_is_last = get_model_idx_for_pp(seq_id - 1, pass_agg=True)
                assert last_pass_type != pass_type and last_is_last, f"balabala."
                layers.at[0, 'args'] = tuple([sim_df[(last_b_idx, last_pass_type, last_mb_idx)].iloc[-1]['name']])

            elif not is_first:  # 每个batch的其他micro-batch
                last_b_idx, last_mb_idx, last_pass_type, last_is_first, last_is_last = get_model_idx_for_pp(seq_id - 1, pass_agg=True)
                assert last_pass_type == pass_type and not last_is_last, f"labalaba."
                
                if pp_granularity == 'layer':
                    if enable_pp_backpressure:
                        dependency_shuffle_idx = [[i, i] for i in range(len(layers))]   # 下一个microbatch一个layer开始依赖于上一个microbatch的同一个layer结束
                    else:
                        dependency_shuffle_idx = [[0, 0]]       # 只在最前面触发一下
                if pp_granularity == 'device':
                    dependency_shuffle_idx = []
                    search_coverage = 1
                    if enable_pp_backpressure:
                        search_coverage = len(layers)
                    for i in range(search_coverage):
                        ret = get_last_pp_stage_index(layers, pp_granularity, pp_grp_label, i, is_first)
                        if ret is not None:
                            dependency_shuffle_idx.append([i, ret])

                for depend_pair in dependency_shuffle_idx:
                    depend_layer_pp = sim_df[(last_b_idx, last_pass_type, last_mb_idx)].iloc[depend_pair[1]]['name'] + '/shadow'
                    lst = list(layers['args'][depend_pair[0]])
                    # print("print(depend_pair, lst, depend_layer_pp): ", depend_pair, lst, depend_layer_pp)
                    lst.append(depend_layer_pp)
                    layers.at[depend_pair[0], 'args'] = tuple(lst)
            
            if b_idx > 0 and mb_idx == 0 and pass_type == 'FWD':        # 下一个 batch 的第一个 fwd
                max_dp_prio_idx = add_dp_hook_with_priority(sim_df, layers, b_idx, mb_idx, pass_type, last_b_idx, last_pass_type, last_mb_idx, pp_grp_label, dp_granularity, dp_overlap)
            
            sim_df[(b_idx, pass_type, mb_idx)] = layers
        
    elif pp_mode == '1F1B':
        raise Exception("1F1B not implemented yet.")
    
    else:
        raise Exception(f"Your PP_mode {pp_mode} is not supported yet, but you can define it on your own!")

    result_df = None
    for k, v in sim_df.items():
        # if isinstance(result_df, None):
        if result_df is None:
            result_df = v
        else:
            result_df = pd.concat([result_df, v], axis=0).reset_index(drop=True)

    return result_df, max_dp_prio_idx


def add_dp_label_to_name_args(dp_df, i):
    prefix_str = 'dp' + str(i) + '_'
    for i in range(len(dp_df)):
        dp_df.at[i, 'name'] = prefix_str + dp_df['name'][i]
        args_list = list(dp_df['args'][i])
        new_args_list = []
        for arg in args_list:
            if isinstance(arg, int):
                new_args_list.append(arg)
            else:
                new_args_list.append(prefix_str + arg)
        dp_df.at[i, 'args'] = tuple(new_args_list)


if __name__ == '__main__':
    # 设置 Pandas 显示选项，不折叠输出
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)  # 允许每一列折叠

    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Set parameters for the program.")
    
    parser.add_argument('-d', '--debug_mode', action='store_true', help='Enable debug mode')
    parser.add_argument('-b', '--batches', type=int, default=2, help='Number of batches')
    parser.add_argument('-m', '--microbatch_num_per_batch', type=int, default=2, help='Number of microbatches per batch')
    parser.add_argument('-s', '--data_size_per_microbatch', type=int, default=10, help='Size of data per microbatch')
    parser.add_argument('--bypass_first_fwd', type=bool, default=True, help='Bypass first forward')
    parser.add_argument('--bypass_last_bkwd', type=bool, default=True, help='Bypass last backward')
    parser.add_argument('--PP', type=bool, default=True, help='Enable pipeline parallelism')
    parser.add_argument('-t', '--target_stages', type=int, default=8, help='Number of GPUs for pipeline parallelism')
    parser.add_argument('--DP', type=bool, default=True, help='Enable data parallelism')
    parser.add_argument('--dp_grp_size', type=int, default=2, help='Data parallelism group size')
    parser.add_argument('--dp_overlap', type=bool, default=False, help='DP sync overlaps with PP pass. Default not enable.')
    parser.add_argument('--dp_granularity', type=str, default='device', help='data parallelism granularity. support <device> and <layer>')
    parser.add_argument('--pp_mode', type=str, default='F_then_B', help='Pipeline parallelism mode. support <native> <F_then_B> <1F1B>(not implemented yet)')
    parser.add_argument('--pp_granularity', type=str, default='device', help='Pipeline parallelism granularity. support <device> and <layer>')
    parser.add_argument('--enable_pp_backpressure', type=bool, default=True, help='Enable pipeline parallelism backpressure. default=True')
    parser.add_argument('--pp_grp_label', type=str, default='avg_mem_grp_idx', help='Pipeline parallelism group label. default=avg_mem_grp_idx')
    parser.add_argument('--pp_flow_bucket_size', type=int, default=-1, help='bucket size')
    parser.add_argument('--dp_flow_bucket_size', type=int, default=-1, help='bucket size')
    parser.add_argument('--plot_seed', type=int, default=1, help='plot seed')
    parser.add_argument('--plot_pp_only', action='store_true', help='only plot pp nodes and flows')
    parser.add_argument('--plot_cc_only', action='store_true', help='only plot cc nodes and flows')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    assert args.pp_mode in ['native', 'F_then_B', '1F1B']
    assert args.pp_granularity in ['device', 'layer']

    # 打印参数
    print("You should specify 'batch size' by specifying microbatch_num_per_batch and data_size_per_microbatch seperately.")
    print(f"""
Parameters:
    - Debug Mode:             {args.debug_mode}
    - Batches:                {args.batches}
    - Microbatch Per Batch:   {args.microbatch_num_per_batch}
    - Microbatch Data Size:   {args.data_size_per_microbatch}
    - Bypass First Forward:   {args.bypass_first_fwd}
    - Bypass Last Backward:   {args.bypass_last_bkwd}
    - PP enable:              {args.PP}
    - Target PP Stage Num:    {args.target_stages}
    - DP enable:              {args.DP}
    - DP Grp Size:            {args.dp_grp_size}
    - DP overlap:             {args.dp_overlap}
    - DP Granularity:         {args.dp_granularity}
    - PP Mode:                {args.pp_mode}
    - PP Granularity:         {args.pp_granularity}
    - Enable PP Backpressure: {args.enable_pp_backpressure}
    - PP Grp Label:           {args.pp_grp_label}
    - PP Bucket size:         {args.pp_flow_bucket_size} B
    - DP Bucket size:         {args.dp_flow_bucket_size} B
    - Plot PP Only:           {args.plot_pp_only}
    - Plot CC Only:           {args.plot_cc_only}
    - Plot Seed:              {args.plot_seed}
    """)

    #######################################################
    #                      获取计算图
    #######################################################
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
    # 使用FX跟踪模型
    tracer = fx.Tracer()
    graph = tracer.trace(model)                 # 创建计算图
    # print("print(graph)")
    # print(graph)

    graph_module = fx.GraphModule(model, graph) # 创建GraphModule
    traced = fx.symbolic_trace(model)           # 创建symbolic的计算图
    # print(traced)
    input_size=(1, 28, 28)
    fwd_df = graph_module_to_dataframe(traced, input_size, args.data_size_per_microbatch)

    delete_input_layer = True
    if delete_input_layer:
        # 查找 'name' 列中值为 'x' 的行的索引
        indices_to_remove = fwd_df.index[fwd_df['name'] == 'x'].tolist()
        # print("Rows to be removed:", indices_to_remove)    # 打印找到的索引
        fwd_df = fwd_df.drop(indices_to_remove)            # 删除这些行
        # 重置索引，以避免产生不连续的索引
        fwd_df = fwd_df.reset_index(drop=True)
        
        # 更新依赖：删除元组中的 'x'
        for tmp in range(len(fwd_df)):
            fwd_df.at[tmp, 'args'] = tuple(delete_node_from_tuple(fwd_df['args'][tmp], 'x', False))

    
    # print("\nThe original fwd_df:")
    # print(fwd_df)

    fwd_df['layer_index'] = fwd_df.index
    fwd_df['direction'] = 'FWD'
    # 将 'layer_index' 列移动到第一列
    fwd_df = fwd_df[['layer_index'] + [col for col in fwd_df.columns if col != 'layer_index']]

    #######################################################
    #              模型切分  （给layers分组）
    #######################################################
    # 模型使用pp切分
    model_pp_cut(fwd_df, args.target_stages, cut_mode='avg_mem', debug_mode=args.debug_mode)
    # model_pp_cut(fwd_df, args.target_stages, cut_mode='min_comm')
    col_name = 'avg_mem_grp_idx'
    fwd_df = fwd_df[[col_name] + [col for col in fwd_df.columns if col != col_name]]
    
    # 在 args列之后插入dp_args列
    args_index = fwd_df.columns.get_loc('args')
    empty_tuples = [[]] * len(fwd_df)
    fwd_df.insert(args_index + 1, 'dp_args', empty_tuples)

    #######################################################
    #             使用前向计算图生成反向计算图
    #######################################################
    if args.debug_mode:
        print("Debug Mode is ON. Forcely cut and divide the df into easy-to-analyze format.")
    bkwd_df = bkwd_df_gen(fwd_df)
    # 计算前向和反向的计算魔都（ops），即compute_density_ops列
    get_compute_density_ops_and_time(fwd_df, 'FWD')
    get_compute_density_ops_and_time(bkwd_df, fwd_df, 'BKWD')

    assert len(fwd_df) == len(bkwd_df), "len(fwd_df) != len(bkwd_df)"
    # print(type(fwd_df['args'][2][0]))

    # print(f"\nforward layers:")
    # print(fwd_df)
    # print(f"\nbackward layers:")
    # print(bkwd_df)

    #######################################################
    # 生成仿真节点 （fwd/bkwd展开、流水线展开，得到有向无环图）
    # DP 流的生成，在sim_node_gen() 的 add_dp_hook_with_priority()函数中
    #######################################################
    sim_df, max_dp_prio_idx = sim_node_gen(fwd_df, bkwd_df, 
                          args.batches, args.microbatch_num_per_batch, 
                          args.pp_mode, args.pp_granularity, args.pp_grp_label, args.enable_pp_backpressure, 
                          args.dp_overlap, args.dp_granularity,
                          args.bypass_first_fwd, args.bypass_last_bkwd
                          )
    sim_df['global_index'] = sim_df.index
    sim_df = sim_df[['global_index'] + [col for col in sim_df.columns if col != 'global_index']]

    # print(f"\nsim layers:\n")
    # print(sim_df)

    if sim_df is not None:
        sim_df.to_csv('sim_df.csv', index=False)

    #######################################################
    #              将单个模型的流复制到多个DP
    #    （并使用dpx_作为name的前缀，以区分不同dp的node）
    #######################################################

    DP_sim_grp = {}
    for i in range(args.dp_grp_size):
        dp_df = sim_df.copy(deep=True)
        add_dp_label_to_name_args(dp_df, i)
        DP_sim_grp[i] = dp_df

    # for i in range(args.dp_grp_size):
    #     print(f"\nNodes of DP {i}:")
    #     print(DP_sim_grp[i])

    #######################################################
    #                   把 仿真节点 映射到 硬件
    #######################################################
    # 为每个DP创建硬件资源，存储在hardware_model.py的server_dicts全局变量中
    for i in range(args.dp_grp_size):
        host_num=4
        max_gpu_per_host=4
        max_nic_per_host=4
        build_system(i, host_num, max_gpu_per_host, max_nic_per_host)  

    # build a model manager for the whole system
    model_manager = ModelManager(args.dp_grp_size)
    # 从 DataFrame 创建 Layer 实例
    global_indexs_per_gpuid = {}  # each item represent a DP grp
    for dp_idx in range(args.dp_grp_size):
        global_indexs_per_gpuid[dp_idx] = model_manager.create_model_layers_from_df(dp_idx, DP_sim_grp[dp_idx], args.pp_grp_label)

    # 使用模型映射表将模型的层添加到不同host的不同GPU上
    dp_sim_df_with_hw = {}      # each item is a DP grp
    for dp_idx in range(args.dp_grp_size):
        # 创建{模型组ID : [server_id, GPU_id]}的字典。
        layerGrp_2_physicalId_dict = build_layerGrp_2_physicalId_dict(dp_idx, global_indexs_per_gpuid[dp_idx], host_num, max_gpu_per_host, max_nic_per_host)
        dp_sim_df_with_hw[dp_idx] = model_manager.load_model(dp_idx, layerGrp_2_physicalId_dict)
        print(f"\nNodes with hw info of DP {dp_idx}:")
        print(dp_sim_df_with_hw[dp_idx])


    #######################################################
    #                    生成仿真流量
    #######################################################
    node_list, flow_list, dep_list = sim_node_and_flow_gen(dp_sim_df_with_hw, max_dp_prio_idx, 
                             args.pp_granularity, 
                             args.pp_flow_bucket_size, args.dp_flow_bucket_size, args.pp_grp_label,
                            print_detail=True,
                            plot_pp_only = args.plot_pp_only
                            )  # output_mode 包括 'interdevice', 'intradevice' , 分别对应device级别和layer级别的粒度
    
    print(f"\nnode_list ({len(node_list)}):")
    pprint(node_list)
    print(f"\nflow_list ({len(flow_list)}):")
    pprint(flow_list)
    print(f"\ndep_list ({len(dep_list)}):")
    pprint(dep_list)

    # 把这些节点画出来，以方便地看结果是不是对
    plot_sim_graph(node_list, flow_list, dep_list, args.plot_seed, args.plot_cc_only)

