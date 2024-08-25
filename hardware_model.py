from pprint import pprint
import pandas as pd
from collections import OrderedDict, defaultdict

from flow_definition import *

server_dicts = defaultdict(lambda: OrderedDict())

# 示例数据
example_data = {
    'global_index': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
    'layer_index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    'name': ['x', 'conv1', 'bn1', 'relu', 'maxpool', 'layer1_0_conv1', 'layer1_0_bn1', 'layer1_0_relu', 'layer1_0_conv2', 'layer1_0_bn2', 'layer1_0_relu_1', 'layer1_0_conv3', 'layer1_0_bn3', 'layer1_0_downsample_0', 'layer1_0_downsample_1', 'add', 'layer1_0_relu_2', 'layer1_1_conv1', 'layer1_1_bn1', 'layer1_1_relu', 'layer1_1_conv2', 'layer1_1_bn2', 'layer1_1_relu_1', 'layer1_1_conv3', 'layer1_1_bn3', 'add_1'],
    'args': [(), ('x',), ('conv1',), ('bn1',), ('relu',), ('maxpool',), ('layer1_0_conv1',), ('layer1_0_bn1',), ('layer1_0_relu',), ('layer1_0_conv2',), ('layer1_0_bn2',), ('layer1_0_relu_1',), ('layer1_0_conv3',), ('layer1_0_downsample_0',), ('layer1_0_downsample_1',), ('layer1_0_bn3', 'layer1_0_downsample_1'), ('add',), ('layer1_0_relu_2',), ('layer1_1_conv1',), ('layer1_1_bn1',), ('layer1_1_relu',), ('layer1_1_conv2',), ('layer1_1_bn2',), ('layer1_1_relu_1',), ('layer1_1_conv3',), ('layer1_1_bn3',), ('layer1_1_bn3', 'layer1_0_relu_2')],
    'input_#': [100, 7840, 576000, 576000, 576000, 144000, 144000, 144000, 144000, 144000, 144000, 144000, 576000, 576000, 576000, 576000, 576000, 144000, 144000, 144000, 144000, 144000, 144000, 576000, 576000, 576000],
    'output_#': [100, 576000, 576000, 576000, 144000, 144000, 144000, 144000, 144000, 144000, 144000, 576000, 576000, 576000, 576000, 576000, 576000, 144000, 144000, 144000, 144000, 144000, 144000, 576000, 576000, 576000],
    'Param_#': [0, 576, 128, 0, 0, 4096, 128, 0, 36864, 128, 0, 16384, 512, 16384, 512, 0, 0, 16384, 128, 0, 36864, 128, 0, 16384, 512, 0],
    'dp_index': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'pp_index': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'tp_index': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'opcode': ['placeholder', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_function', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_module', 'call_function'],
    'target': ['x', 'conv1', 'bn1', 'relu', 'maxpool', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.relu_1', 'layer1.0.conv3', 'layer1.0.bn3', 'layer1.0.downsample.0', 'layer1.0.downsample.1', '<built-in function add>', 'layer1.0.relu', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.relu_1', 'layer1.1.conv3', 'layer1.1.bn3', '<built-in function add>'],
    'layer_type': [None, 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Tensor', 'ReLU', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Tensor'],
    'input_shape': ['torch.Size([32, 3, 224, 224])', 'torch.Size([32, 64, 112, 112])', 'torch.Size([32, 64, 112, 112])', 'torch.Size([32, 64, 112, 112])', 'torch.Size([32, 64, 112, 112])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])'],
    'output_shape': ['torch.Size([32, 64, 112, 112])', 'torch.Size([32, 64, 112, 112])', 'torch.Size([32, 64, 112, 112])', 'torch.Size([32, 64, 112, 112])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 64, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])', 'torch.Size([32, 256, 56, 56])'],
    'param_shape': [None, 'torch.Size([64, 3, 7, 7])', 'torch.Size([64])', None, None, 'torch.Size([64, 64, 1, 1])', 'torch.Size([64])', None, 'torch.Size([64, 64, 3, 3])', 'torch.Size([64])', None, 'torch.Size([256, 64, 1, 1])', 'torch.Size([256])', 'torch.Size([256, 64, 1, 1])', 'torch.Size([256])', None, None, 'torch.Size([64, 64, 1, 1])', 'torch.Size([64])', None, 'torch.Size([64, 64, 3, 3])', 'torch.Size([64])', None, 'torch.Size([256, 64, 1, 1])', 'torch.Size([256])', None],
    'kwargs': [None, "{'stride': (2, 2), 'padding': (3, 3), 'bias': False}", "{'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}", "{'inplace': True}", "{'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1, 'ceil_mode': False}", "{'kernel_size': (1, 1), 'stride': (1, 1), 'padding': (0, 0), 'dilation': (1, 1), 'groups': (1,), 'bias': False}", "{'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}", "{'inplace': True}", "{'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'dilation': (1, 1), 'groups': (1,), 'bias': False}", "{'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}", "{'inplace': True}", "{'kernel_size': (1, 1), 'stride': (1, 1), 'padding': (0, 0), 'dilation': (1, 1), 'groups': (1,), 'bias': False}", "{'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}", "{'kernel_size': (1, 1), 'stride': (2, 2), 'padding': (0, 0), 'dilation': (1, 1), 'groups': (1,), 'bias': False}", "{'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}", None, "{'inplace': True}", "{'kernel_size': (1, 1), 'stride': (1, 1), 'padding': (0, 0), 'dilation': (1, 1), 'groups': (1,), 'bias': False}", "{'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}", "{'inplace': True}", "{'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'dilation': (1, 1), 'groups': (1,), 'bias': False}", "{'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}", "{'inplace': True}", "{'kernel_size': (1, 1), 'stride': (1, 1), 'padding': (0, 0), 'dilation': (1, 1), 'groups': (1,), 'bias': False}", "{'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}", None],
    'dtype_size': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    'direction': ['FWD' for i in range(30)],
    'avg_mem_grp_idx': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7],
    'avg_mem_grp_size': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    'min_comm_grp_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'min_comm_grp_sum': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'min_comm_grp_last_weight': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'compute_density_ops': [i * 1000 + 10000 for i in range(30)]            # 随机赋值，仅供测试
}

# 转换长短不一，所以统一一下
size = 10000
for key, _list in example_data.items():
    if len(_list) < size:
        size = len(_list)

for key in example_data:
    example_data[key] = example_data[key][:size]


class NetworkCard:
    def __init__(self, network_card_id, server_id, pcie_bandwidth, network_bandwidth):
        self.server = None
        self.network_card_id = network_card_id
        self.server_id = server_id
        self.pcie_bandwidth_Bps = pcie_bandwidth
        self.network_bandwidth_bps = network_bandwidth


class Server:
    def __init__(self, server_id, pcie_version, pcie_bandwidth, max_gpus, max_network_cards):
        self.server_id = server_id
        self.pcie_version = pcie_version
        self.pcie_bandwidth_Bps = pcie_bandwidth * (2 ** 30)
        self.max_gpus = max_gpus
        self.max_network_cards = max_network_cards
        self.gpus = OrderedDict()
        self.network_cards = OrderedDict()

        self.model_layers = OrderedDict()      # 重要：维护着host上的所有层。key: layer_index; value: 所在GPU
        self.min_layer_index = 1000000
        self.max_layer_index = -1

    def add_gpu(self, fp32_flops, fp16_flops, fp8_flops, pcie_bandwidth, nvlink_supported, nvlink_bandwidth, memory_size_gb, memory_bandwidth_GB_s):
        if len(self.gpus) < self.max_gpus:
            gpu_id = f"gpu{len(self.gpus) + 1}"
            effective_pcie_bandwidth = min(self.pcie_bandwidth_Bps, pcie_bandwidth * (2 ** 30))  # 转换为字节
            memory_size_B = memory_size_gb * (2 ** 30)  # 将GB转换为B
            memory_bandwidth_Bps = memory_bandwidth_GB_s * (2 ** 30)  # 将GB/s转换为B/s
            nvlink_bandwidth_Bps = nvlink_bandwidth * (2 ** 30)  # 将GB/s转换为B/s
            gpu = GPU(gpu_id, self.server_id, fp32_flops, fp16_flops, fp8_flops, effective_pcie_bandwidth, nvlink_supported, nvlink_bandwidth_Bps, memory_size_B, memory_bandwidth_Bps, None)
            gpu.server = self    # 双向绑定
            self.gpus[gpu_id] = gpu
        else:
            raise Exception("Exceeded the number of GPUs that can be added to this server.")

    def add_network_card(self, pcie_bandwidth_GBps, network_bandwidth_Gbps):
        if len(self.network_cards) < self.max_network_cards:
            network_card_id = f"nic{len(self.network_cards) + 1}"
            effective_pcie_bandwidth = min(self.pcie_bandwidth_Bps, pcie_bandwidth_GBps * (2 ** 30))  # 转换为字节
            network_bandwidth_bps = network_bandwidth_Gbps * (10 ** 9)  # 将Gbps转换为Bps
            network_card = NetworkCard(network_card_id, self.server_id, effective_pcie_bandwidth, network_bandwidth_bps)
            network_card.server = self
            self.network_cards[network_card_id] = network_card
        else:
            raise Exception("Exceeded the number of network cards that can be added to this server.")

    # 可以自定义 GPU 和 NIC 之间的关联方式，这里采用一一对应
    def link_gpu_network_card(self):
        if len(self.gpus) != len(self.network_cards):
            raise Exception("The number of GPUs and network cards must be equal to link them.")
        nic_list = list(self.network_cards.keys())
        for i, (gpu_id, gpu) in enumerate(self.gpus.items()):
            self.gpus[gpu_id].network_card_id = self.network_cards[nic_list[i]].network_card_id

    # should be called by gpu_add_model_layer()
    def server_add_model_layer(self, model):
        model_layer_cnt = len(self.model_layers)
        self.model_layers[model_layer_cnt] = model
        # 更新 host 上的layer统计值，其中最大最小层对应的需要通过网卡和其他host通信
        self.max_layer_index = max(self.max_layer_index, model.layer_index)
        self.min_layer_index = min(self.min_layer_index, model.layer_index)
        return len(self.model_layers) - 1

    # should only be called after all the NN layers are added to GPU. 
    def sanity_check(self):
        # 导出所有 Device 对象的 layer_index 属性到一个列表
        sanity = True
        model_indexs = []
        for _, gpu in self.gpus.items():
            # sanity = sanity and gpu.sanity_check()
            model_indexs.append([layer.layer_index for layer in gpu.model_layers.values()])
        model_indexs = sorted(model_indexs)
        for i in range(1, len(model_indexs)):
            if model_indexs[i] - model_indexs[i - 1] != 1:
                sanity = False

        if not sanity:
            raise Exception(f"Sanity is not OK in {self.server_id}!")
        return sanity

class GPU:
    def __init__(self, gpu_id, server_id, fp32_flops, fp16_flops, fp8_flops, pcie_bandwidth, nvlink_supported, nvlink_bandwidth, memory_size, memory_bandwidth, network_card_id):
        self.server = None
        self.gpu_id = gpu_id
        self.server_id = server_id
        self.fp32_flops = fp32_flops
        self.fp16_flops = fp16_flops
        self.fp8_flops = fp8_flops
        self.pcie_bandwidth_Bps = pcie_bandwidth
        self.nvlink_supported = nvlink_supported
        self.nvlink_bandwidth_Bps = nvlink_bandwidth
        self.memory_size_B = memory_size
        self.memory_bandwidth_Bps = memory_bandwidth
        self.network_card_id = network_card_id

        self.model_layers = OrderedDict()
        self.model_size = 0

        self.min_layer_index = 1000000
        self.max_layer_index = -1
    
    # should only be called after all the NN layers are added to GPU. 
    def sanity_check(self):
        # 导出所有 Device 对象的 layer_index 属性到一个列表
        model_indexs = sorted([layer.layer_index for layer in self.model_layers.values()])
        sanity_ok = True
        for i in range(1, len(model_indexs)):
            if model_indexs[i] - model_indexs[i - 1] != 1:
                sanity_ok = False
        
        if not sanity_ok:
            raise Exception(f"Sanity is not OK in {self.server_id}-{self.gpu_id}!")
        return sanity_ok
     
    # TODO: 理论上正向、反向的输入、参数、输出都得能放下才对。需要和切分算法联调
    # 返回这个layer在该GPU上的编号，用于确认该层是否需要跨GPU/主机通信
    def gpu_add_model_layer(self, model):
        # print(model.param_num, model.dtype_size)
        target_mem_usage = self.model_size + model.param_num * model.dtype_size
        if target_mem_usage > self.memory_size_B:
            raise Exception(f"model size exceeds HBM capacity! HBM cap = {self.memory_size_B / (2**30)} GB, but we are trying to put {target_mem_usage / (2**30)} GB param on it! ")
        model_idx_gpu = len(self.model_layers)
        self.model_layers[model_idx_gpu] = model
        self.model_size = target_mem_usage
        # 通知host
        assert isinstance(self.server, Server)
        model_idx_host = self.server.server_add_model_layer(model)
        # 更新统计值
        self.max_layer_index = max(self.max_layer_index, model.layer_index)
        self.min_layer_index = min(self.min_layer_index, model.layer_index)
        return model_idx_gpu, model_idx_host

class Layer:
    def __init__(self, manager, global_index, layer_index, name, args, dp_args, input_num, output_num, param_num, dp_index, pp_index, tp_index, 
                opcode, target, layer_type, input_shape, output_shape, param_shape, kwargs, dtype_size, direction, pp_grp_label, 
                avg_mem_grp_idx=None, avg_mem_grp_size=None, min_comm_grp_id=None, min_comm_grp_sum=None, min_comm_grp_last_weight=None,
                # add more grp_labels here ...
                compute_density_ops=0):
        # 普通参数
        self.args = args
        self.dp_args = dp_args
        self.input_num = input_num
        self.output_num = output_num
        self.param_num = param_num
        self.dp_index = dp_index
        self.pp_index = pp_index
        self.tp_index = tp_index
        self.opcode = opcode
        self.target = target
        self.layer_type = layer_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.param_shape = param_shape
        self.kwargs = kwargs
        self.dtype_size = dtype_size
        self.direction = direction
        self.pp_grp_label = pp_grp_label

        self.avg_mem_grp_idx = avg_mem_grp_idx
        self.avg_mem_grp_size = avg_mem_grp_size

        self.min_comm_grp_id = min_comm_grp_id
        self.min_comm_grp_sum = min_comm_grp_sum
        self.min_comm_grp_last_weight = min_comm_grp_last_weight
        # add more grp_labels here ...
        self.compute_density_ops = compute_density_ops

        # 重要参数
        self.global_index = global_index
        self.layer_index = layer_index
        self.name = name
        self.model_manager = manager
        
        self.server_id = ''     # 我所在的server的ID √
        self.gpu_id = ''        # 我所在的GPU的ID    √
        self.nic_id = ''        # √

        self.index_on_host = -1     # 本层在host上面的执行顺序（index） √
        self.index_on_GPU = -1      # 本层在GPU上面的执行顺序（index）  √

        # 每个节点的特性：读数据时间、写数据时间、计算时间。通过结合GPU的capacity计算得出
        self.read_time = 0      # √
        self.exec_time = 0      # √
        self.write_time = 0     # √

        # 每个节点应该有：依赖流ID列表、触发流ID列表。
        self.pp_flows_depend = []
        self.pp_flows_invoke = []
        self.dp_flows_depend = []
        self.dp_flows_invoke = []

        # 该节点的上下层都在哪。这个参数仅用于 PP
        self.prev_server = None     # √
        self.next_server = None     # √
        self.prev_gpu = None        # √
        self.next_gpu = None        # √
        self.prev_nic = None          # √
        self.next_nic = None          # √

        self.input_source = None    # 'network', 'host', 'peerGPU', 'HBM'. 'host' 意味着是从网卡来的，涉及到跨主机通信
        self.output_dest = None     # 'HBM', 'peerGPU', 'host', 'network'. 'host' 意味着是去网卡的，涉及到跨主机通信
    
    def get_value(self, var_name):
        '''
        给定一个pp_grp_label，返回其对应的组ID
        '''
        if var_name == 'avg_mem_grp_idx':
            return self.avg_mem_grp_idx
        elif var_name == 'min_comm_grp_id':
            return self.min_comm_grp_id
        # # register your own group id here:
        # elif var_name == 'my_grp_label':
        #     return self.my_grp_label
        else:
            raise ValueError(f"column name for grouping {var_name} not registered.")
        

    # this function is only used to add server_id and GPU_id for each layer in advance, to make it easy for load_to_gpu(). 
    def load_to_gpu_warmup(self, dp_idx, server_id, gpu_id, nic_id):
        if server_dicts[dp_idx][server_id].gpus[gpu_id] == None:
            raise Exception(f"target device cannot find! <{server_id}, {gpu_id}>")

        # set my param
        self.server_id = server_id
        self.gpu_id = gpu_id
        self.nic_id = nic_id

    def load_to_gpu(self, dp_idx, server_id, gpu_id, nic_id, layerGrp_to_host_GPU_dict):       # 无用参数：
        # 这里实现将模型层加载到GPU上的方法
        if server_dicts[dp_idx][server_id].gpus[gpu_id] == None:
            raise Exception(f"target device cannot find! <{server_id}, {gpu_id}>")

        # set my param
        self.server_id = server_id
        self.gpu_id = gpu_id
        self.nic_id = nic_id
        # notify GPU and host
        self.index_on_GPU, self.index_on_host = server_dicts[dp_idx][server_id].gpus[gpu_id].gpu_add_model_layer(self)
        
        # print(f"Loading layer {self.name} to {server_id}[{self.index_on_host}], {gpu_id}[{self.index_on_GPU}]")

        # 更新该节点的上下层以及存储介质类型
        # self.model_manager.layers       # {layer_index, layer对象}
        # self.model_manager.GPU_grps     # {grp_id, [layer_index]}
        # layerGrp_to_host_GPU_dict       # {grp_id : [server_id, GPU_id]}

        # self.layers[dp_idx][layer.global_index] = layer
        layer_cnt = len(self.model_manager.layers[dp_idx])      # 可以这样使用，因为model_manager.layers已经填充完毕

        # 计算本级输入存放位置
        if self.global_index > 0:
            self.prev_server = self.model_manager.layers[dp_idx][self.global_index - 1].server_id
            self.prev_gpu = self.model_manager.layers[dp_idx][self.global_index - 1].gpu_id
            self.prev_nic = self.model_manager.layers[dp_idx][self.global_index - 1].nic_id
        if self.prev_server is None:  # 一般用于本地输入
            self.input_source = 'host'
        elif self.prev_server != self.server_id:  # 跨机器通信
            self.input_source = 'network'
        elif self.prev_server == self.server_id and self.prev_gpu != self.gpu_id:     # 单机多卡通信
            self.input_source = 'peerGPU'
        elif self.prev_server == self.server_id and self.prev_gpu == self.gpu_id:     # 卡内通信
            self.input_source = 'HBM'
        else:
            raise Exception(f"Global_idx {self.global_index} has undefined prev_server relation <{self.prev_server}> !")           
            
        # 计算上一级输出存放位置
        
        if self.global_index < layer_cnt - 1:
            self.next_server = self.model_manager.layers[dp_idx][self.global_index + 1].server_id
            self.next_gpu = self.model_manager.layers[dp_idx][self.global_index + 1].gpu_id
            self.next_nic = self.model_manager.layers[dp_idx][self.global_index + 1].nic_id
            # print("1111", self.next_server, self.next_gpu)
        if self.next_server is None:              # 一般用于本地输出loss
            self.output_dest = 'HBM'
        elif self.next_server != self.server_id:  # 跨机器通信
            self.output_dest = 'network'
        elif self.next_server == self.server_id and self.next_gpu != self.gpu_id:     # 单机多卡通信
            self.output_dest = 'peerGPU'
            # self.model_manager.add_pp_flow(self.layer_index, self.layer_index + 1, self.output_num * self.dtype_size)
        elif self.next_server == self.server_id and self.next_gpu == self.gpu_id:     # 卡内通信
            self.output_dest = 'HBM'
        else:
            raise Exception(f"Global_idx {self.global_index} has undefined next_server relation <{self.prev_server}> !")   

        # 计算时间
        if self.dtype_size == 0:
            self.exec_time = 0
        elif self.dtype_size == 1:
            self.exec_time = self.compute_density_ops / server_dicts[dp_idx][server_id].gpus[gpu_id].fp8_flops
        elif self.dtype_size == 2:
            self.exec_time = self.compute_density_ops / server_dicts[dp_idx][server_id].gpus[gpu_id].fp16_flops
        elif self.dtype_size == 4:
            self.exec_time = self.compute_density_ops / server_dicts[dp_idx][server_id].gpus[gpu_id].fp32_flops
        else:
            raise Exception(f"Error: dtype_size {self.dtype_size} is not currently supported!")
        
        # 读数据时间（in host）
        self.input_data_byte = self.input_num * self.dtype_size
        if self.input_source in ['host', 'network']:
            self.read_time = self.input_data_byte / server_dicts[dp_idx][server_id].gpus[gpu_id].pcie_bandwidth_Bps
        elif self.input_source == 'peerGPU':
            if server_dicts[dp_idx][server_id].gpus[gpu_id].nvlink_supported:
                self.read_time = self.input_data_byte / server_dicts[dp_idx][server_id].gpus[gpu_id].nvlink_bandwidth_Bps
            else:
                self.read_time = self.input_data_byte / server_dicts[dp_idx][server_id].gpus[gpu_id].pcie_bandwidth_Bps
        elif self.input_source =='HBM':
            self.read_time = self.input_data_byte / server_dicts[dp_idx][server_id].gpus[gpu_id].memory_bandwidth_Bps
        else:
            raise Exception(f"Error: layer.input_source {self.input_source} unknown!")
        
        # 写数据时间（in host）
        self.output_data_byte = self.output_num * self.dtype_size
        if self.output_dest  in ['host', 'network']:
            self.write_time = self.output_data_byte / server_dicts[dp_idx][server_id].gpus[gpu_id].pcie_bandwidth_Bps
        elif self.output_dest == 'peerGPU':
            if server_dicts[dp_idx][server_id].gpus[gpu_id].nvlink_supported:
                self.write_time = self.output_data_byte / server_dicts[dp_idx][server_id].gpus[gpu_id].nvlink_bandwidth_Bps
            else:
                self.write_time = self.output_data_byte / server_dicts[dp_idx][server_id].gpus[gpu_id].pcie_bandwidth_Bps
        elif self.output_dest =='HBM':
            self.write_time = self.output_data_byte / server_dicts[dp_idx][server_id].gpus[gpu_id].memory_bandwidth_Bps
        else:
            raise Exception(f"Error: layer.output_source {self.input_source} unknown!")

        
class ModelManager:
    def __init__(self, dps):
        self.called_create_model_layers_from_df = [False for i in range(dps)]
        self.called_load_model = [False for i in range(dps)]
        self.layers = defaultdict(lambda: OrderedDict())
        # 创建一个 defaultdict，其中每个键的默认值是一个空列表
        self.GPU_grps = defaultdict(lambda: defaultdict(list))
        self.host_flows = defaultdict(lambda: defaultdict(list))
        self.pp_flows = defaultdict(lambda: defaultdict(list))      # cross_host pipeline paralism flow 
        self.cross_host_pp_flow = defaultdict(list)
        self.cc_flows = defaultdict(list)                           # cross_host collective communication flow



        
    # must be called before load_model().
    def create_model_layers_from_df(self, dp_idx, df, pp_grp_label):
        '''
        每个GPU上有哪些global_index
        '''
        for _, row in df.iterrows():
            layer = Layer(
                manager = self,
                global_index = row['global_index'],
                layer_index=row['layer_index'],
                name=row['name'],
                args=row['args'],
                dp_args=row['dp_args']  if row['dp_args'] is not None else None,
                input_num=row['input_#'],
                output_num=row['output_#'],
                param_num=row['Param_#'],
                dp_index=row['dp_index'],
                pp_index=row['pp_index'],
                tp_index=row['tp_index'],
                opcode=row['opcode'],
                target=row['target'],
                layer_type=row['layer_type'],
                input_shape=row['input_shape'],
                output_shape=row['output_shape'],
                param_shape=row['param_shape'],
                kwargs=row['kwargs'],
                dtype_size=row['dtype_size'][0] if row['dtype_size'] is not None else 0,
                direction=row['direction'],
                # 需要指定一个pp_grp_label
                pp_grp_label=pp_grp_label,
                # 按照内存相等划分的列标
                avg_mem_grp_idx=row['avg_mem_grp_idx'] if 'avg_mem_grp_idx' in df.columns else None,
                avg_mem_grp_size=row['avg_mem_grp_size'] if 'avg_mem_grp_size' in df.columns else None,
                # 按照通信最小划分的列标
                min_comm_grp_id=row['min_comm_grp_id'] if 'min_comm_grp_id' in df.columns else None,
                min_comm_grp_sum=row['min_comm_grp_sum'] if 'min_comm_grp_sum' in df.columns else None,
                min_comm_grp_last_weight=row['min_comm_grp_last_weight'] if 'min_comm_grp_last_weight' in df.columns else None,
                # 可以类似上面的格式自定义其他的列标。不要忘记在 Layer.get_value() 中注册！
                compute_density_ops=row['compute_density_ops']
            )
            self.layers[dp_idx][layer.global_index] = layer
            layer_grp_idx = layer.get_value(layer.pp_grp_label)
            self.GPU_grps[dp_idx][layer_grp_idx].append(layer.global_index)     # 被分配到特定组的sim_layer index
            self.called_create_model_layers_from_df[dp_idx] = True
        return self.GPU_grps[dp_idx]


    def convert_layers_to_df(self, dp_idx):
        data = {
            'global_index': [],
            'avg_mem_grp_idx': [],
            'layer_index': [],
            
            'server_id': [],
            'gpu_id': [],
            'nic_id': [],
            'prev_server': [],
            'prev_gpu': [],
            'prev_nic': [],
            'next_server': [],
            'next_gpu': [],
            'next_nic': [],

            'name': [],
            'args': [],
            'dp_args': [],
            'input_#': [],
            'output_#': [],
            'Param_#': [],
            'dp_index': [],
            'pp_index': [],
            'tp_index': [],
            'opcode': [],
            'target': [],
            'layer_type': [],
            'input_shape': [],
            'output_shape': [],
            'param_shape': [],
            'kwargs': [],
            'dtype_size': [],
            'direction': [],

            'pp_grp_label': [],

            # 'avg_mem_grp_idx': [],    # 放到前面了，与输入的 sim_df 一致
            'avg_mem_grp_size': [],

            'min_comm_grp_id': [],
            'min_comm_grp_sum': [],
            'min_comm_grp_last_weight': [],
            # add your grp_label here...
            'compute_density_ops': [],
            # 新加的
            'index_on_host': [],
            'index_on_GPU': [],

            'input_source': [],
            'output_dest': [],

            'read_time': [],
            'exec_time': [],
            'write_time': []

        }

        for layer in self.layers[dp_idx].values():
            data['global_index'].append(layer.global_index)
            data['avg_mem_grp_idx'].append(layer.avg_mem_grp_idx)
            data['layer_index'].append(layer.layer_index)
            data['name'].append(layer.name)
            data['args'].append(layer.args)
            data['dp_args'].append(layer.dp_args)
            data['input_#'].append(layer.input_num)
            data['output_#'].append(layer.output_num)
            data['Param_#'].append(layer.param_num)
            data['dp_index'].append(layer.dp_index)
            data['pp_index'].append(layer.pp_index)
            data['tp_index'].append(layer.tp_index)
            data['opcode'].append(layer.opcode)
            data['target'].append(layer.target)
            data['layer_type'].append(layer.layer_type)
            data['input_shape'].append(layer.input_shape)
            data['output_shape'].append(layer.output_shape)
            data['param_shape'].append(layer.param_shape)
            data['kwargs'].append(layer.kwargs)
            data['dtype_size'].append(layer.dtype_size)
            data['direction'].append(layer.direction)

            data['pp_grp_label'].append(layer.pp_grp_label)

            # 这里的avg_mem_grp_idx放到前面了，和输入的sim_df一致
            data['avg_mem_grp_size'].append(layer.avg_mem_grp_size)

            data['min_comm_grp_id'].append(layer.min_comm_grp_id)
            data['min_comm_grp_sum'].append(layer.min_comm_grp_sum)
            data['min_comm_grp_last_weight'].append(layer.min_comm_grp_last_weight)
            # add your columns here...
            data['compute_density_ops'].append(layer.compute_density_ops)

            data['server_id'].append(layer.server_id)
            data['gpu_id'].append(layer.gpu_id)
            data['nic_id'].append(layer.nic_id)
            data['index_on_host'].append(layer.index_on_host)
            data['index_on_GPU'].append(layer.index_on_GPU)

            data['input_source'].append(layer.input_source)
            data['output_dest'].append(layer.output_dest)
            
            data['prev_server'].append(layer.prev_server)
            data['next_server'].append(layer.next_server)
            data['prev_gpu'].append(layer.prev_gpu)
            data['next_gpu'].append(layer.next_gpu)
            data['prev_nic'].append(layer.prev_nic)
            data['next_nic'].append(layer.next_nic)
            

            data['read_time'].append(layer.read_time)
            data['exec_time'].append(layer.exec_time)
            data['write_time'].append(layer.write_time)
        return pd.DataFrame(data)
    

    # must be called after create_model_layers_from_df().
    def load_model(self, dp_idx, layerGrp_to_host_GPU_dict):
        if self.called_create_model_layers_from_df[dp_idx] == False:
            raise Exception(f"ModelManager.load_model() must be called after ModelManager.create_model_layers_from_df()")
        self.called_load_model[dp_idx] = True
        # Note: layer n 执行 load_to_cpu之后，layer n+1 还未执行，因此不知其 server_id 和 gpu_id。因此需要warmup
        for global_index, layer in self.layers[dp_idx].items():
            # pprint(vars(layer))
            grp_idx = layer.get_value(layer.pp_grp_label)
            server_ID = layerGrp_to_host_GPU_dict[grp_idx][0]
            gpu_ID = layerGrp_to_host_GPU_dict[grp_idx][1]
            nic_ID = layerGrp_to_host_GPU_dict[grp_idx][2]

            layer.load_to_gpu_warmup(dp_idx, server_ID, gpu_ID, nic_ID)
        
        # layerGrp_to_host_GPU_dict: { grp_id : [server_id, GPU_id] }
        for global_index, layer in self.layers[dp_idx].items():
            # pprint(vars(layer))
            grp_idx = layer.get_value(layer.pp_grp_label)
            server_ID = layerGrp_to_host_GPU_dict[grp_idx][0]
            gpu_ID = layerGrp_to_host_GPU_dict[grp_idx][1]
            nic_ID = layerGrp_to_host_GPU_dict[grp_idx][2]

            layer.load_to_gpu(dp_idx, server_ID, gpu_ID, nic_ID, layerGrp_to_host_GPU_dict)
        df_with_hw = self.convert_layers_to_df(dp_idx)
        return df_with_hw
    
    # TODO and is not used
    def add_pp_flow(self, dp_idx, src_layer_idx, dst_layer_idx, size_bytes):
        if self.called_load_model[dp_idx] == False:
            raise Exception(f"ModelManager.add_pp_flow() must be called after ModelManager.load_model()")
        
        src_host = self.layers[src_layer_idx].server_id
        src_gpu =  self.layers[src_layer_idx].gpu_id
        src_nic =  server_dicts[dp_idx][src_host].gpus[src_gpu].network_card_id
        srcEp = FlowEndpointInfo(src_host, src_gpu, src_nic)
        dst_host = self.layers[dst_layer_idx].server_id
        dst_gpu =  self.layers[dst_layer_idx].gpu_id
        dst_nic =  server_dicts[dp_idx][dst_host].gpus[dst_gpu].network_card_id
        dstEp = FlowEndpointInfo(dst_host, dst_gpu, dst_nic)
        self.pp_flows[src_layer_idx] = [dst_layer_idx, Flow(flowid_assigner(), size_bytes, srcEp, dstEp, opcode='pp')]



# could be defined by user
# model_grps 的类型是 defaultdict(list)    key: layer_grp_id; value: [layer_id]
def build_layerGrp_2_physicalId_dict(dp_idx, model_grps, host_num, max_gpu_per_host, max_nic_per_host):
    total_GPUs = host_num * max_gpu_per_host
    if len(model_grps) > total_GPUs:
        raise Exception(f"Error: Model is divided into {len(model_grps)} groups, but we only have {total_GPUs} GPU(s). ")
    layerGrp_2_physicalId_dict = {}        #每个item应该是一个 [server_id, GPU_id, nic_id] tuple
    # print(model_grps)
    for grp_id, _ in model_grps.items():
        host_id = grp_id // max_gpu_per_host + 1
        GPU_id = grp_id % max_gpu_per_host + 1
        target_host_id = f"dp{dp_idx}_server{host_id}"
        target_GPU_id = 'gpu' + str(GPU_id)
        target_nic_id = server_dicts[dp_idx][target_host_id].gpus[target_GPU_id].network_card_id
        layerGrp_2_physicalId_dict[grp_id] = [target_host_id, target_GPU_id, target_nic_id]

    return layerGrp_2_physicalId_dict


def build_system(dp_idx, host_num=4, max_gpu_per_host=4, max_nic_per_host=4):
    # NVIDIA A100 GPU specs
    fp32_flops = 19.5e12  # 19.5 TFLOPS
    fp16_flops = 156e12   # 156 TFLOPS
    fp8_flops = 312e12    # 312 TFLOPS
    gpu_pcie_bandwidth_GB = 64   # 64 GB/s
    nvlink_supported = True
    nvlink_bandwidth_GB = 600  # 600 GB/s 单向带宽
    memory_size_GB = 40      # 40 GB
    memory_bandwidth_gb_s = 1555 # 1555 GB/s 单向带宽
    
    # CX7 NIC specs
    network_card_pcie_bandwidth_gb_s = 64  # PCIe 5.0 bandwidth
    network_bandwidth_gbps = 400  # 400 Gbps

    # build cluster for single model
    for i in range(host_num):
        server_id = f"dp{dp_idx}_server{i + 1}"
        server_dicts[dp_idx][server_id] = Server(server_id=server_id, pcie_version="5.0", pcie_bandwidth=128, max_gpus=max_gpu_per_host, max_network_cards=max_nic_per_host)
        gpu_per_host = max_gpu_per_host
        nic_per_host = max_nic_per_host
        
        # 添加 GPU                           
        for _ in range(gpu_per_host):
            server_dicts[dp_idx][server_id].add_gpu(fp32_flops, fp16_flops, fp8_flops, gpu_pcie_bandwidth_GB, nvlink_supported, nvlink_bandwidth_GB, memory_size_GB, memory_bandwidth_gb_s)
        # 添加 NIC
        for _ in range(nic_per_host):
            server_dicts[dp_idx][server_id].add_network_card(pcie_bandwidth_GBps=network_card_pcie_bandwidth_gb_s, network_bandwidth_Gbps=network_bandwidth_gbps)
        # 为每个GPU指定它走的NIC
        server_dicts[dp_idx][server_id].link_gpu_network_card()
    

if __name__ == '__main__':
    # 设置 Pandas 显示选项，不折叠输出
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    DPs = 2

    # 为每个DP创建硬件资源
    for i in range(DPs):
        host_num=4
        max_gpu_per_host=4
        max_nic_per_host=4
        build_system(i, host_num, max_gpu_per_host, max_nic_per_host)  

    # 创建 DataFrame
    df = pd.DataFrame(example_data)
    
    # build a model
    model_manager = ModelManager(DPs)
    # 从 DataFrame 创建 Layer 实例
    global_index_per_gpuid = {}  # each item is a DP grp
    for dp_idx in range(DPs):
        global_index_per_gpuid[dp_idx] = model_manager.create_model_layers_from_df(dp_idx, df)
    
    # # 创建{模型组ID : [server_id, GPU_id]}的字典。
    # layerGrp_2_physicalId_dict = build_layerGrp_2_physicalId_dict(model_grps, host_num, max_gpu_per_host, max_nic_per_host)

    # 使用模型映射表将模型的层添加到不同host的不同GPU上
    df_with_hw = {}      # each item is a DP grp
    for dp_idx in range(DPs):
        # 创建{模型组ID : [server_id, GPU_id]}的字典。
        layerGrp_2_physicalId_dict = build_layerGrp_2_physicalId_dict(dp_idx, global_index_per_gpuid[dp_idx], host_num, max_gpu_per_host, max_nic_per_host)
        df_with_hw[dp_idx] = model_manager.load_model(dp_idx, layerGrp_2_physicalId_dict)
        print(f"\nNodes with hw info of DP {dp_idx}:")
        print(df_with_hw[dp_idx])


    # OrderdDict 正确的迭代方式
    for dp_idx, server_dict in server_dicts.items():
        print(f"DP group {dp_idx}")
        for server_id, server in server_dict.items():
            print(server.__dict__)
            for gpu_id, gpu in server.gpus.items():
                print(gpu.__dict__)
            for nic_id, nic in server.network_cards.items():
                print(nic.__dict__)
    
