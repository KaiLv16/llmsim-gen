import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import cv2
from nets.resnet50 import Bottleneck, ResNet
from tqdm import tqdm
import torch.nn as nn
import torch.fx as fx
from torchsummary import summary
from pprint import pprint
from collections import OrderedDict
import pandas as pd
from functools import reduce
from operator import mul

def print_layer_data_transfer(graph_module, model, input_size, batch_size=1):
    def find_node_name(graph_module, node_name):
        for node in graph_module.graph.nodes:
            if node.target == node_name:
                return node.name
        return None
    
    def find_node_op(graph_module, node_name):
        for node in graph_module.graph.nodes:
            if node.target == node_name:
                return node.op
        return None
    
    def find_node_args(graph_module, node_name):
        for node in graph_module.graph.nodes:
            if node.target == node_name:
                return node.args
        return None
    
    def find_node_kwargs(graph_module, node_name):
        for node in graph_module.graph.nodes:
            if node.target == node_name:
                return node.kwargs
        return None
    
    def register_hook(module, name):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[module_idx] = OrderedDict()
            summary[module_idx]["Layer (type)"] = m_key
            summary[module_idx]["opcode"] = find_node_op(graph_module, name)
            summary[module_idx]["name"] = find_node_name(graph_module, name)
            summary[module_idx]["target"] = name
            summary[module_idx]["layer_type"] = class_name
            summary[module_idx]["args"] = find_node_args(graph_module, name)
            summary[module_idx]["kwargs"] = find_node_kwargs(graph_module, name)
            summary[module_idx]["input_shape"] = str(list(input[0].size()) if isinstance(input, tuple) else list(input.size()))
            summary[module_idx]["output_shape"] = str(list(output.size()))

            # 计算每层输出的数据量（以字节为单位）
            output_size = torch.tensor(output.size()).prod().item()
            dtype_size = output.element_size()
            summary[module_idx]["output_size (bytes)"] = output_size * dtype_size
            assert output_size == reduce(mul, list(output.size())), "输出不相等！"

            # 初始化当前模块的参数的形状和每个参数的字节大小
            param_shapes = {name: str(p.size()) for name, p in module.named_parameters()}
            param_dtype_size = [p.element_size() for p in module.parameters()]
            summary[module_idx]["param_shapes"] = str(param_shapes)
            summary[module_idx]["dtype_size"] = str(param_dtype_size)
            print("\n")
            for name, p in module.named_parameters():
                print(name ,p)

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []

    # 注册钩子
    for name, module in model.named_modules():
        register_hook(module, name)

    # 创建一个dummy输入
    x = torch.rand(batch_size, *input_size)
    with torch.no_grad():
        model(x)

    # 移除钩子
    for h in hooks:
        h.remove()

    return summary

# 设置 Pandas 显示选项，不折叠输出
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
traced = fx.symbolic_trace(model)  # 创建symbolic的计算图
# 使用torchsummary来显示每层的计算量和参数传递量
input_size = (1, 28, 28)
layer_data = print_layer_data_transfer(traced, model, input_size, batch_size=1)

# 转换为pandas DataFrame
df_summary = pd.DataFrame.from_dict(layer_data, orient='index')

# print(df_summary)
