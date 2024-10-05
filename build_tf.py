# -*- coding: utf-8 -*-

import pandas as pd
import argparse
from build_llm_exec_graph import *
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Transformers Layer Configuration")
    parser.add_argument('--num_of_layers', type=int, default=3, help='Number of layers in the transformer model (default: 3)')
    parser.add_argument('--global_batch', type=int, default=8192, help='Global batch size (default: 8192)')
    parser.add_argument('--micro_batch', type=int, default=1, help='Micro batch size (default: 1)')
    parser.add_argument('--seq_length', type=int, default=4096, help='Sequence length (default: 4096)')
    parser.add_argument('--pp_cut', type=int, default=-1, help='Degree of compression, usually 0 (minimal) or -1 (no compression) (default: -1)')
    parser.add_argument('--passes', type=int, default=2, help='Number of passes')
    parser.add_argument('--pp', type=int, default=-1, help='Number of stages')
    parser.add_argument('--bypass_first_fwd', type=bool, default=True, help='Bypass first forward pass')
    parser.add_argument('--bypass_last_bkwd', type=bool, default=True, help='Bypass last backward pass')
    return parser.parse_args()


# 全局 node ID 分发器
global_lid_register = -1
def get_node_id():
    global global_lid_register
    global_lid_register += 1
    return global_lid_register


# 全局 flow ID 分发器
global_fid_register = -1
def get_flow_id():
    global global_fid_register
    global_fid_register += 1
    return global_fid_register


# 修改函数来处理可能的空格
def get_parameters_by_name(df, name):
    # 打印列名检查
    # 如果有空格或隐藏字符，可以尝试清理列名
    df.columns = df.columns.str.strip()
    # print(df.columns)
    
    # 去掉 name 两边的空格，避免匹配错误
    row = df[df['Name'].str.strip() == name.strip()]
    
    if not row.empty:
        parameters = row.iloc[0].drop('Name').to_dict()
        Parameter_size = parameters['Parameter_size']
        Hidden_size = parameters['Hidden_size']
        Num_of_layers = parameters['Num_of_layers']
        Attention_heads = parameters['Attention_heads']
        Sequence_length = parameters['Sequence_length']
        FFN_hidden_size = parameters['FFN_hidden_size']
        World_size = parameters['World_size']
        TP = parameters['TP']
        PP = parameters['PP']
        DP = int(World_size/(PP*TP))
            # 打印返回的值
        print(f"Name: {name}")
        print(f"Parameter_size: {Parameter_size}")
        print(f"Hidden_size: {Hidden_size}")
        print(f"Num_of_layers: {Num_of_layers}")
        print(f"Attention_heads: {Attention_heads}")
        print(f"Sequence_length: {Sequence_length}")
        print(f"FFN_hidden_size: {FFN_hidden_size}")
        print(f"World_size: {World_size}")
        print(f"TP: {TP}")
        print(f"PP: {PP}")
        print(f"DP: {DP}")

        return Parameter_size, Hidden_size, Num_of_layers, Attention_heads, Sequence_length, FFN_hidden_size, World_size, TP, PP, DP
    else:
        return None, None, None, None, None, None, None, None, None, None   # 如果没有找到，返回 None
    

# Usage: 
# dep = Dep(src=, dst=, depend_node=[], invoke_node=[])
# print(dep)
class Dep:
    def __init__(self, src=-1, dst=-1, depend_node=[], invoke_node=[]):
        self.src = src
        self.dst = dst
        self.depend_node = depend_node
        self.invoke_node = invoke_node

    def __repr__(self) -> str:
        repr_str = f"""src={self.src}, dst={self.dst}, depend_node={self.depend_node}, invoke_node={self.invoke_node}"""
        return repr_str

class Node:
    def __init__(self, layer_start_id, layer_end_id, name='node'):
        self.class_name = name
        self.layer_start_id = layer_start_id
        self.layer_end_id = layer_end_id
        self.flow_dep_id = []
        self.flow_invoke_id = []

    def print_node(self, file_name=None, end='\n'):
        if file_name is None:
            print(f"{self.class_name}_id=<{self.layer_start_id}, {self.layer_end_id}>{end} ")
            print(f"dep={self.flow_dep_id}{end} ")
            print(f"invoke={self.flow_invoke_id}")
        else:
            with open(file_name, 'w') as f:
                f.write(f"{self.class_name}_id=<{self.layer_start_id}, {self.layer_end_id}>{end} ")
                f.write(f"dep={self.flow_dep_id}{end} ")
                f.write(f"invoke={self.flow_invoke_id}")

class VirtNode(Node):
    def __init__(self, layer_start_id, layer_end_id):
        super().__init__(layer_start_id, layer_end_id, name='virt_node')


def RingAllReduce(src_list, dst_list):
    virt_nodes = []
    cc_flows = []

    return virt_nodes, cc_flows

def ring_all_reduce(N, total_data_size):
    assert N > 1, f"Exception when generating AllReduce flows: DP grp must bigger than 1!"
    flows = defaultdict(list)               # key: data chunk   value:  cc_flow

    v_node = defaultdict(list)       # key: dp_id        value: cc_vnode 
    v_nodename_start = []       # key: dp_id        value: cc_vnode 
    v_nodename_end = []       # key: dp_id        value: cc_vnode 
    
    flow_data_size = total_data_size / N

    data_chunk_start_idx = [i for i in range(N)]
    data_chunk_idx = [i for i in range(N)]
    
    for chunk in range(N):
        chunk_flow_id = 0
        
        for round in range(2 * N - 2):
            src = data_chunk_idx[chunk]             # 对应 源dp_id
            dst = (data_chunk_idx[chunk] + 1) % N   # 对应 目的dp_id
            data_chunk_idx[chunk] = dst             # 更新该chunk的指向的dp_id的指针，供下一次for循环使用
            if round < N - 1 - 1:
                src_op = '_rs'      # reduce-scatter
                dst_op = '_rs'      # all gather
            elif round == N - 1 - 1:
                src_op = '_rs'
                dst_op = '_ag'
            else:
                src_op = '_ag' 
                dst_op = '_ag'

            src_node_name = f"virt_dp{src}_chunk{chunk}" + src_op
            if src_node_name not in v_node:
                v_node[src_node_name] = cc_vnode(src_node_name, src)
                if round == 0:
                    v_nodename_start.append(src_node_name)

            dst_node_name = f"virt_dp{dst}_chunk{chunk}" + dst_op
            if dst_node_name not in v_node:
                v_node[dst_node_name] = cc_vnode(dst_node_name, dst)
                if round >= N - 1 - 1:
                    v_nodename_end.append(dst_node_name)

            flow = cc_Flow(f"chunk{chunk}_flow{chunk_flow_id}", src, dst, flow_data_size, 'reduce-scatter', round, v_node[src_node_name], v_node[dst_node_name])
            
            v_node[src_node_name].flow_invoke.append(flow)
            v_node[dst_node_name].flow_depend.append(flow)
            flows[chunk].append(flow)

            chunk_flow_id += 1

    # 提取 nodename 中的dp组信息
    def extract_dp_number(s):
        match = re.search(r'dp(\d+)', s)
        if match:
            return int(match.group(1))
        else:
            return None
        
    # 把vnode转换成按照dp分组的形式
    vnode_grp_by_dpid = defaultdict(list)
    for nodename, node in v_node.items():
        dp_id = extract_dp_number(nodename)
        vnode_grp_by_dpid[(dp_id, nodename)] = node
    
    assert len(v_nodename_start) == N       # 每个DP只主动触发自己的1/N的数据的发送，剩下的数据需要等待其他DP发过来
    assert len(vnode_grp_by_dpid) == len(v_node)      # N组
    assert len(v_nodename_end) == N * N     # 每个node上面的数据分为N份，N个node

    return flows, vnode_grp_by_dpid, v_nodename_start, v_nodename_end



# 返回值1是class VirtNode对象的列表, 返回值2是 class Flow的列表
def AllReduce(src_list, dst_list, size, method='RingAllReduce'):
    assert len(src_list) > 0
    assert len(dst_list) > 0
    assert len(src_list) == len(dst_list)
    assert isinstance(src_list[0], Node)   # Node类或者其子类的对象
    assert isinstance(dst_list[0], Node)

    virt_nodes = []
    cc_flows = []
    if method == 'RingAllReduce':
        virt_nodes, cc_flows = RingAllReduce(src_list, dst_list, size)
    else:
        assert False, f'Method {method} has not implemented yet!'

    return virt_nodes, cc_flows

# Usage: 
# flow = Flow(src=, dst=, size=, lat=, depend_flow=[], invoke_flow=[])
# print(flow)
class Flow(Dep):
    def __init__(self, prio=3, src=-1, dst=-1, size=10000, lat=0, depend_node=[], invoke_node=[], depend_flow=[], invoke_flow=[]):
        super().__init__(src=src, dst=dst, depend_node=depend_node, invoke_node=invoke_node)
        self.prio = prio
        self.size = size
        self.lat = lat
        self.depend_flow = depend_flow
        self.invoke_flow = invoke_flow

    def __repr__(self) -> str:
        repr_str = f"""{self.prio}, src={self.src}, dst={self.dst}, size={self.size}, pre_lat={self.lat}, depend_node={self.depend_node}, invoke_node={self.invoke_node}, depend_flow={self.depend_flow}, invoke_flow={self.invoke_flow}"""
        return repr_str


class ShadowNode(Node):
    def __init__(self, node_start_id, node_end_id, origin_node_start_id, origin_node_end_id):
        super().__init__(origin_node_start_id, origin_node_end_id)
        self.shadow_layer_start_id = node_start_id
        self.shadow_layer_end_id = node_end_id

    def __repr__(self):
        repr_str = 'WARNING: printing method of class ShadowNode is not implemented yet.'
        return repr_str

    def print_node(self, file_name=None, end='\n'):
        if file_name is None:
            print(f"node_id=<{self.shadow_layer_start_id}, {self.shadow_layer_end_id}>{end} ")
            print(f"origin_node_id=<{self.layer_start_id}, {self.layer_end_id}>{end} ")
            print(f"dep={self.flow_dep_id}{end} ")
            print(f"invoke={self.flow_invoke_id}")
        else:
            with open(file_name, 'w') as f:
                f.write(f"node_id=<{self.shadow_layer_start_id}, {self.shadow_layer_end_id}>{end} ")
                print(f"origin_node_id=<{self.layer_start_id}, {self.layer_end_id}>{end} ")
                f.write(f"dep={self.flow_dep_id}{end} ")
                f.write(f"invoke={self.flow_invoke_id}")


class Layer(Node):
    def __init__(self, layer_start_id, layer_end_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_start_id, layer_end_id)
        self.inherent_id = inherent_id
        self.input_size = input_size
        self.output_size = output_size
        self.calc_time = calc_time
        self.bkwd_calc_time = 2 * calc_time
        self.param_size = param_size
        self.former_layer_id = former_layer_id
        self.next_layer_id = next_layer_id

        self.tp_grp_start = []
        self.tp_grp_end = []
        self.tp_fwd_type = None
        self.tp_bkwd_type = None

        self.dp_src_grp = []
        self.dp_dst_grp = []
        self.dp_type = None

        self.pp_dep = []
        self.pp_invoke = []

    def __repr__(self):
        return (f"Layer <{self.layer_start_id}, {self.layer_end_id}>: "
                # f"Input_Size = {self.input_size}, Output_Size = {self.output_size}, "
                # f"Calc_Time = {self.calc_time}s, Param_Size = {self.param_size}, "
                f"Former_Layer = {self.former_layer_id}, Next_Layer = {self.next_layer_id}, "
                f"TP_Grp_start = {self.tp_grp_start}, TP_Grp_end = {self.tp_grp_end}, TP_fwd_Type = {self.tp_fwd_type}, TP_bkwd_Type = {self.tp_bkwd_type}, "
                f"DP_src = {self.dp_src_grp}, DP_dst = {self.dp_dst_grp}, DP_Type = {self.dp_type}, "
                f"pp_dep={self.pp_dep}, pp_invoke={self.pp_invoke}")
    
    def set_tp_grp(self, tp_grp_start, tp_grp_end, tp_fwd_type=None, tp_bkwd_type=None):
        self.tp_grp_start = tp_grp_start
        self.tp_grp_end = tp_grp_end
        self.tp_fwd_type = tp_fwd_type if tp_fwd_type else None
        self.tp_bkwd_type = tp_bkwd_type if tp_bkwd_type else None

    def set_dp_grp(self, dp_grp, dp_type):
        self.dp_grp = dp_grp
        self.dp_type = dp_type if dp_grp else None


# 包括QKV层 和输出层：4 * N^2
class Attention(Layer):
    def __init__(self, tf_object, layer_start_id, layer_end_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_start_id, layer_end_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id, next_layer_id)
        self.layer_type = "Attn"
        self.tf_layer = tf_object

    def __repr__(self):
        repo_str = self.layer_type + ' '
        repo_str += super().__repr__()
        return repo_str


# 包括两个线性层： N -> 4N  &&  4N -> N, N为隐藏层大小。参数量 = (N * 4 * N + 4 * N) + (4 * N * N + N)
class MLP(Layer):
    def __init__(self, tf_object, layer_start_id, layer_end_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_start_id, layer_end_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id, next_layer_id)
        self.layer_type = "MLP"
        self.tf_layer = tf_object

    def __repr__(self):
        repo_str = self.layer_type + ' '
        repo_str += super().__repr__()
        return repo_str


class TransformerLayer:
    '''
    inherent_id: 用于确定这一层的物理位置
    '''
    def __init__(self, inherent_id, step, pass_type, layer_start_id_attn, layer_end_id_attn, layer_start_id_mlp, layer_end_id_mlp, input_size, output_size, \
                 mlp_calc_time, attn_calc_time, mlp_param_size, attn_param_size, former_layer_id=None, next_layer_id=None, did=0, mbs=0, Num_of_layers=1, TP=1):
        output_input_size = input_size
        self.inherent_id = inherent_id
        self.step = step
        self.pass_type = pass_type
        self.did = did
        self.mbs = mbs
        self.Num_of_layers = Num_of_layers  # 设置适当的层数
        self.TP = TP             # 设置适当的TP值

        self.physical_layer_id = self.extract_ids()[4]
        self.physical_tp_id = self.extract_ids()[5]
        
        
        if self.pass_type == 'FWD':
            self.attention_layer = Attention(self, layer_start_id=layer_start_id_attn, layer_end_id=layer_end_id_attn, inherent_id=inherent_id, input_size=input_size, output_size=output_input_size, 
                                         calc_time=attn_calc_time, param_size=attn_param_size, 
                                         former_layer_id=former_layer_id, next_layer_id=layer_start_id_mlp)
        
            self.mlp_layer = MLP(self, layer_start_id=layer_start_id_mlp, layer_end_id=layer_end_id_mlp, inherent_id=inherent_id, input_size=output_input_size, output_size=input_size, 
                             calc_time=mlp_calc_time, param_size=mlp_param_size, 
                             former_layer_id=layer_end_id_attn, next_layer_id=next_layer_id)
        else:
            self.mlp_layer = MLP(self, layer_start_id=layer_start_id_mlp, layer_end_id=layer_end_id_mlp, inherent_id=inherent_id, input_size=output_input_size, output_size=input_size, 
                             calc_time=mlp_calc_time, param_size=mlp_param_size, 
                             former_layer_id=former_layer_id, next_layer_id=layer_start_id_attn)
            
            self.attention_layer = Attention(self, layer_start_id=layer_start_id_attn, layer_end_id=layer_end_id_attn, inherent_id=inherent_id, input_size=input_size, output_size=output_input_size, 
                                         calc_time=attn_calc_time, param_size=attn_param_size, 
                                         former_layer_id=layer_end_id_mlp, next_layer_id=next_layer_id)
    
    def __repr__(self):
        result = self.extract_ids()
        # return f"Transformer Layer {self.inherent_id} (step={result[0]} pass_type={result[1]} did={result[2]} mbid={result[3]} lid={result[4]} tid={result[5]}):\n{self.attention_layer}\n{self.mlp_layer}"
        
        if self.pass_type == 'FWD':
            return f"FWD Transformer Layer {self.inherent_id} (step, type, did, mbid, lid, tid = {result}):\n{self.attention_layer}\n{self.mlp_layer}"
        else:
            return f"BKWD Transformer Layer {self.inherent_id} (step, type, did, mbid, lid, tid = {result}):\n{self.mlp_layer}\n{self.attention_layer}"
       
    def extract_ids(self):
        # 计算 did
        did = self.inherent_id // (self.mbs * self.Num_of_layers * self.TP)
        # 计算 mbid
        remaining_after_did = self.inherent_id % (self.mbs * self.Num_of_layers * self.TP)
        mbid = remaining_after_did // (self.Num_of_layers * self.TP)
        # 计算 lid
        remaining_after_mbid = remaining_after_did % (self.Num_of_layers * self.TP)
        lid = remaining_after_mbid // self.TP
        # 计算 tid
        tid = remaining_after_mbid % self.TP
        return self.step, self.pass_type, did, mbid, lid, tid

    def set_tp_groups(self, tp_grp_attn_start, tp_grp_attn_end, tp_grp_mlp_start, tp_grp_mlp_end, tp_fwd_type=None, tp_bkwd_type=None):
        self.attention_layer.set_tp_grp(tp_grp_attn_start, tp_grp_attn_end, tp_fwd_type, tp_bkwd_type)
        self.mlp_layer.set_tp_grp(tp_grp_mlp_start, tp_grp_mlp_end, tp_fwd_type, tp_bkwd_type)
        
    def set_dp_groups(self, dp_grp, dp_type):
        self.mlp_layer.set_dp_grp(dp_grp, dp_type)
        self.attention_layer.set_dp_grp(dp_grp, dp_type)



#######################################
###            M A I N              ###
#######################################

if __name__ == '__main__':
    args = parse_arguments()
    # 读取 CSV 文件到 DataFrame
    workload_df = pd.read_csv('workload.csv', sep='\t')
    Parameter_size, Hidden_size, Num_of_layers, Attention_heads, Seq_len, FFN_hidden_size, World_size, TP, PP, DP \
        = get_parameters_by_name(workload_df, 'GPT_7B')
    # 开始构造流
    flow_list = []
    flow_file = "llm_flow.txt"
    if args.pp != -1:
        PP = args.pp

    # 用transformer或者mlp的全局唯一的ID，找Transformer对象的五个index :)
    lid_2_idx_dict = {}

    # Num_of_layers = 3
    # global_batch = 8192
    # micro_batch = 1
    # seq_length = 4096
    # pp_cut = -1    # 压缩的程度。一般来讲应该是 0(最小压缩) 或者 -1 （不压缩）
    passes = args.passes
    bypass_first_fwd = args.bypass_first_fwd
    bypass_last_bkwd = args.bypass_last_bkwd

    # 部分参数更新, for ease of simulation
    Num_of_layers = args.num_of_layers
    global_batch = args.global_batch
    micro_batch = args.micro_batch
    seq_length = args.seq_length
    pp_cut = args.pp_cut    # 压缩的程度。一般来讲应该是 0(最小压缩) 或者 -1 （不压缩）
    
    mbs = global_batch // micro_batch
    assert global_batch == mbs * micro_batch
    
    if pp_cut >= 0:
        mbs = min(global_batch // micro_batch, PP + pp_cut)

    mb_input_size = Seq_len * micro_batch * Hidden_size
    assert FFN_hidden_size == Hidden_size * 4

    mlp_param_size_tp = (Hidden_size * FFN_hidden_size + FFN_hidden_size + FFN_hidden_size * Hidden_size + Hidden_size) / TP
    Head_size = Hidden_size / Attention_heads
    attn_param_size_tp = (3 * (Hidden_size * Head_size * Attention_heads) + Hidden_size * Hidden_size) / TP   #  = 4 * Hidden_size^2 / TP

    steps = passes * 2
    first_step = 1 if bypass_first_fwd else 0
    last_step = 1 if bypass_last_bkwd else 0
    tf_layers = [[[[[] for _ in range(Num_of_layers)] for _ in range(mbs)] for _ in range(DP)] for _ in range(steps)]
    total_layer_cnt = 0

    dp_grp_fwd = [[[[] for _ in ['attn', 'mlp']] for _ in range(TP)] for _ in range(Num_of_layers)]  # 对最后一个 mbid 的每一个 Transformer layer，都维护一个 DP 组
    dp_grp_bkwd = [[[[] for _ in ['attn', 'mlp']] for _ in range(TP)] for _ in range(Num_of_layers)]  # 对最后一个 mbid 的每一个 Transformer layer，都维护一个 DP 组

    for step in range(first_step, steps - last_step):
        pass_type = 'FWD' if step % 2 == 0 else 'BKWD'

        if pass_type == 'FWD':
            dp_grp_fwd = [[[[] for _ in ['attn', 'mlp']] for _ in range(TP)] for _ in range(Num_of_layers)]
            for did in range(DP):
                for mbid in range(mbs):
                    # 为每一层构建多个 TP 组
                    for lid in range(Num_of_layers):
                        # 构建并设置 TP 组
                        tp_grp_attn_start = []
                        tp_grp_attn_end = []
                        tp_grp_mlp_start = []
                        tp_grp_mlp_end = []
                        last_layer = -1
                        next_layer = -1
                        for tid in range(TP):
                            inherent_id = did * (mbs * Num_of_layers * TP) + mbid * (Num_of_layers * TP) + lid * TP + tid
                            lid_start_attn = get_node_id()
                            lid_end_attn = get_node_id()
                            lid_start_mlp = get_node_id()
                            lid_end_mlp = get_node_id()
                            tp_grp_attn_start.append(lid_start_attn)
                            tp_grp_attn_end.append(lid_end_attn)
                            tp_grp_mlp_start.append(lid_start_mlp)
                            tp_grp_mlp_end.append(lid_end_mlp)

                            # 添加DP依赖
                            if mbid == mbs - 1:
                                dp_grp_fwd[lid][tid][0].append(lid_start_attn)
                                dp_grp_fwd[lid][tid][1].append(lid_start_mlp)

                            tf_layers[step][did][mbid][lid].append(
                                TransformerLayer(inherent_id=inherent_id, step=step, pass_type=pass_type,
                                            layer_start_id_attn=lid_start_attn, layer_end_id_attn=lid_end_attn, 
                                            layer_start_id_mlp=lid_start_mlp, layer_end_id_mlp=lid_end_mlp, 
                                            input_size=mb_input_size, output_size=mb_input_size, 
                                            mlp_calc_time=0.005, attn_calc_time=0.010, 
                                            mlp_param_size=mlp_param_size_tp, attn_param_size=attn_param_size_tp,
                                            did=did, mbs=mbs, Num_of_layers=Num_of_layers, TP=TP)
                                )
                            total_layer_cnt += 1

                            lid_2_idx_dict[lid_start_attn] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_end_attn] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_start_mlp] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_end_mlp] = [step, did, mbid, lid, tid]

                        # TODO
                        for tid in range(TP):
                            tf_layers[step][did][mbid][lid][tid].set_tp_groups(tp_grp_attn_start=tp_grp_attn_start, tp_grp_attn_end=tp_grp_attn_end, \
                                                                               tp_grp_mlp_start=tp_grp_mlp_start, tp_grp_mlp_end=tp_grp_mlp_end, \
                                                                               tp_fwd_type='ALLREDUCE', tp_bkwd_type='ALLREDUCE')

                    
                    # 连接多层Transformer网络
                    for tid in range(TP):
                        for lid in range(1, Num_of_layers):
                            tf_layers[step][did][mbid][lid][tid].attention_layer.former_layer_id = tf_layers[step][did][mbid][lid - 1][tid].mlp_layer.layer_end_id
                            tf_layers[step][did][mbid][lid - 1][tid].mlp_layer.next_layer_id = tf_layers[step][did][mbid][lid][tid].attention_layer.layer_start_id
                            
                            # 添加前后层连接的Flow
                            src_id = tf_layers[step][did][mbid][lid - 1][tid].mlp_layer.layer_end_id
                            dst_id = tf_layers[step][did][mbid][lid][tid].attention_layer.layer_start_id
                            size = tf_layers[step][did][mbid][lid - 1][tid].mlp_layer.output_size
                            lat = tf_layers[step][did][mbid][lid - 1][tid].mlp_layer.calc_time
                            flow_list.append(Flow(src=src_id, dst=dst_id, size=size, lat=lat))
                    
                # 添加 PP 依赖
                for mbid in range(1, mbs):     # 每个mb
                    for lid in range(Num_of_layers):  # 每层
                        for tid in range(TP):       # 每个 TP 组
                            for tgrp in range(TP):
                                tf_layers[step][did][mbid-1][lid][tid].mlp_layer.pp_invoke.append(tf_layers[step][did][mbid][lid][tgrp].attention_layer.layer_start_id)
                                tf_layers[step][did][mbid][lid][tid].attention_layer.pp_dep.append(tf_layers[step][did][mbid-1][lid][tgrp].mlp_layer.layer_end_id)
                            # 添加跨不同的mbid的前后层依赖
                            src_id = tf_layers[step][did][mbid-1][lid][tid].mlp_layer.layer_end_id
                            dst_id = tf_layers[step][did][mbid][lid][tid].attention_layer.layer_start_id
                            flow_list.append(Dep(src=src_id, dst=dst_id))
                            
            # 最后：TP并行下，每个 TP 部分分别做 DP 的 all-reduce
            # TODO: 添加 ALLREDUCE 的 flow
            for did in range(DP):
                for lid in range(Num_of_layers):            
                    # if len(dp_grp_bkwd[0]) != 0 and len(dp_grp_fwd[0]) != 0:
                    if step > first_step:
                        for tid in range(TP):       # 每个 TP 部分分别做 DP 的 all-reduce
                            for dgrp in range(DP):
                                tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][tid].attention_layer.dp_dst_grp.append(tf_layers[step][dgrp][0][lid][tid].attention_layer.layer_start_id)
                                tf_layers[step][did][0][lid][tid].attention_layer.dp_src_grp.append(tf_layers[step - 1][dgrp][mbs - 1][Num_of_layers - 1 - lid][tid].attention_layer.layer_end_id)

                                tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][tid].mlp_layer.dp_dst_grp.append(tf_layers[step][dgrp][0][lid][tid].mlp_layer.layer_start_id)
                                tf_layers[step][did][0][lid][tid].mlp_layer.dp_src_grp.append(tf_layers[step - 1][dgrp][mbs - 1][Num_of_layers - 1 - lid][tid].mlp_layer.layer_end_id)

                            tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][tid].attention_layer.dp_src_grp = tf_layers[step][did][0][lid][tid].attention_layer.dp_src_grp
                            tf_layers[step][did][0][lid][tid].attention_layer.dp_dst_grp = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][tid].attention_layer.dp_dst_grp
                            tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][tid].mlp_layer.dp_src_grp = tf_layers[step][did][0][lid][tid].mlp_layer.dp_src_grp
                            tf_layers[step][did][0][lid][tid].mlp_layer.dp_dst_grp = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][tid].mlp_layer.dp_dst_grp

                            tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][tid].attention_layer.dp_type = 'ALLREDUCE'
                            tf_layers[step][did][0][lid][tid].attention_layer.dp_type = 'ALLREDUCE'
                            tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][tid].mlp_layer.dp_type = 'ALLREDUCE'
                            tf_layers[step][did][0][lid][tid].mlp_layer.dp_type = 'ALLREDUCE'
                    else:
                        pass



        else:       # BKWD
            dp_grp_bkwd = [[[[] for _ in ['attn', 'mlp']] for _ in range(TP)] for _ in range(Num_of_layers)]
            for did in range(DP):
                for mbid in range(mbs):
                    # 为每一层构建多个 TP 组
                    for lid in range(Num_of_layers):
                        # 构建并设置 TP 组
                        tp_grp_attn_start = []
                        tp_grp_attn_end = []
                        tp_grp_mlp_start = []
                        tp_grp_mlp_end = []
                        last_layer = -1
                        next_layer = -1
                        for tid in range(TP):
                            inherent_id = did * (mbs * Num_of_layers * TP) + mbid * (Num_of_layers * TP) + lid * TP + tid
                            lid_start_mlp = get_node_id()
                            lid_end_mlp = get_node_id()
                            lid_start_attn = get_node_id()
                            lid_end_attn = get_node_id()
                            
                            tp_grp_attn_start.append(lid_start_attn)
                            tp_grp_attn_end.append(lid_end_attn)
                            tp_grp_mlp_start.append(lid_start_mlp)
                            tp_grp_mlp_end.append(lid_end_mlp)

                            # 添加DP依赖
                            if mbid == 0:
                                dp_grp_bkwd[lid][tid][0].append(lid_end_attn)
                                dp_grp_bkwd[lid][tid][1].append(lid_end_mlp)

                            tf_layers[step][did][mbid][lid].append(
                                TransformerLayer(inherent_id=inherent_id, step=step, pass_type=pass_type,
                                            layer_start_id_attn=lid_start_attn, layer_end_id_attn=lid_end_attn, 
                                            layer_start_id_mlp=lid_start_mlp, layer_end_id_mlp=lid_end_mlp, 
                                            input_size=mb_input_size, output_size=mb_input_size, 
                                            mlp_calc_time=0.005, attn_calc_time=0.010, 
                                            mlp_param_size=mlp_param_size_tp, attn_param_size=attn_param_size_tp,
                                            did=did, mbs=mbs, Num_of_layers=Num_of_layers, TP=TP)
                                )
                            total_layer_cnt += 1
                            lid_2_idx_dict[lid_start_attn] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_end_attn] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_start_mlp] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_end_mlp] = [step, did, mbid, lid, tid]
                            
                        for tid in range(TP):
                            tf_layers[step][did][mbid][lid][tid].set_tp_groups(tp_grp_attn_start=tp_grp_attn_start, tp_grp_attn_end=tp_grp_attn_end, \
                                                                               tp_grp_mlp_start=tp_grp_mlp_start, tp_grp_mlp_end=tp_grp_mlp_end, \
                                                                               tp_fwd_type='ALLREDUCE', tp_bkwd_type='ALLREDUCE')

                    # 连接多层Transformer网络 (反向传播)
                    for tid in range(TP):
                        for lid in range(1, Num_of_layers):
                            tf_layers[step][did][mbid][lid][tid].mlp_layer.former_layer_id = tf_layers[step][did][mbid][lid - 1][tid].attention_layer.layer_end_id
                            tf_layers[step][did][mbid][lid - 1][tid].attention_layer.next_layer_id = tf_layers[step][did][mbid][lid][tid].mlp_layer.layer_start_id
                            
                            # 添加前后层连接的Flow
                            src_id = tf_layers[step][did][mbid][lid - 1][tid].attention_layer.layer_end_id
                            dst_id = tf_layers[step][did][mbid][lid][tid].mlp_layer.layer_start_id
                            size = tf_layers[step][did][mbid][lid - 1][tid].attention_layer.output_size
                            lat = tf_layers[step][did][mbid][lid - 1][tid].attention_layer.calc_time
                            flow_list.append(Flow(src=src_id, dst=dst_id, size=size, lat=lat, depend_flow=[], invoke_flow=[]))

                            
                    # 连接前向和反向传播的代码。这里使用PP进行连接。note: 这里是dependency而非flow
                    if mbid == 0:
                        if step - 1 >= first_step:
                            for tid in range(TP):
                                # X 是复制的，所以不需要group操作
                                tf_layers[step][did][mbid][0][tid].mlp_layer.pp_dep.append(tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.layer_end_id)
                                tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.pp_invoke.append(tf_layers[step][did][mbid][0][tid].mlp_layer.layer_start_id)
                                # for tgrp in range(TP):
                                #     tf_layers[step][did][mbid][0][tid].mlp_layer.pp_dep.append(tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tgrp].mlp_layer.layer_end_id)
                                #     tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.pp_invoke.append(tf_layers[step][did][mbid][0][tgrp].mlp_layer.layer_start_id)
  
                                # 添加前后层连接的Flow
                                src_id = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.layer_end_id
                                dst_id = tf_layers[step][did][mbid][0][tid].mlp_layer.layer_start_id
                                size = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.output_size
                                lat = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.calc_time
                                flow_list.append(Flow(src=src_id, dst=dst_id, size=size, lat=lat, depend_flow=[], invoke_flow=[]))

                # 添加 PP 依赖
                for mbid in range(1, mbs):     # 每个mb
                    for lid in range(Num_of_layers):  # 每层
                        for tid in range(TP):       # 每个 TP 组
                            for tgrp in range(TP):
                                tf_layers[step][did][mbid-1][lid][tid].attention_layer.pp_invoke.append(tf_layers[step][did][mbid][lid][tgrp].mlp_layer.layer_start_id)
                                tf_layers[step][did][mbid][lid][tid].mlp_layer.pp_dep.append(tf_layers[step][did][mbid-1][lid][tgrp].attention_layer.layer_end_id)
                            # 添加跨不同的mbid的前后层依赖
                            src_id = tf_layers[step][did][mbid-1][lid][tid].attention_layer.layer_end_id
                            dst_id = tf_layers[step][did][mbid][lid][tid].mlp_layer.layer_start_id
                            flow_list.append(Dep(src=src_id, dst=dst_id))


    for step in range(first_step, steps - last_step):
        for dp in range(len(tf_layers[step])):
            for mb in range(len(tf_layers[step][dp])):
                for layer in range(len(tf_layers[step][dp][mb])):
                    for tid in range(len(tf_layers[step][dp][mb][layer])):
                        print(f"\n{tf_layers[step][dp][mb][layer][tid]}")

    
    # 将实例的信息写入文件
    with open('llm_flow.txt', 'w') as f:
        for flow in flow_list:
            f.write(repr(flow) + '\n')



    print(f'\npasses: {list(range(first_step, steps - last_step))}')                
    print(f'new_DP: {DP}')
    print(f'new_mbs: {mbs}')
    print(f'new_Num_of_layers: {Num_of_layers}')
    print(f'new_TP: {TP}')
    print(f'total_layers: {total_layer_cnt}')
