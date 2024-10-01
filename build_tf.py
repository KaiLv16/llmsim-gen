# -*- coding: utf-8 -*-

import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Transformers Layer Configuration")
    parser.add_argument('--num_of_layers', type=int, default=3, help='Number of layers in the transformer model (default: 3)')
    parser.add_argument('--global_batch', type=int, default=8192, help='Global batch size (default: 8192)')
    parser.add_argument('--micro_batch', type=int, default=1, help='Micro batch size (default: 1)')
    parser.add_argument('--seq_length', type=int, default=4096, help='Sequence length (default: 4096)')
    parser.add_argument('--pp_cut', type=int, default=-1, help='Degree of compression, usually 0 (minimal) or -1 (no compression) (default: -1)')
    return parser.parse_args()


# 全局 flow ID 分发器
global_lid_register = -1
def get_id():
    global global_lid_register
    global_lid_register += 1
    return global_lid_register


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
    

class Layer:
    def __init__(self, layer_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        self.layer_id = layer_id
        self.inherent_id = inherent_id
        self.input_size = input_size
        self.output_size = output_size
        self.calc_time = calc_time
        self.bkwd_calc_time = 2 * calc_time
        self.param_size = param_size
        self.former_layer_id = former_layer_id
        self.next_layer_id = next_layer_id
        self.tp_grp = []
        self.tp_fwd_type = None
        self.tp_bkwd_type = None
        self.dp_grp = []
        self.dp_type = None

    def set_tp_grp(self, tp_grp, tp_fwd_type=None, tp_bkwd_type=None):
        self.tp_grp = tp_grp
        self.tp_fwd_type = tp_fwd_type if tp_fwd_type else None
        self.tp_bkwd_type = tp_bkwd_type if tp_bkwd_type else None

    def set_dp_grp(self, dp_grp, dp_type):
        self.dp_grp = dp_grp
        self.dp_type = dp_type if dp_grp else None

    def __repr__(self):
        return (f"Layer {self.layer_id}: Input_Size = {self.input_size}, Output_Size = {self.output_size}, "
                f"Calc_Time = {self.calc_time}s, Param_Size = {self.param_size}, "
                f"Former_Layer = {self.former_layer_id}, Next_Layer = {self.next_layer_id}, "
                f"TP_Group = {self.tp_grp}, TP_fwd_Type = {self.tp_fwd_type}, TP_bkwd_Type = {self.tp_bkwd_type}, DP_Group = {self.dp_grp}, DP_Type = {self.dp_type}")


# 包括QKV层 和输出层：4 * N^2
class Attention(Layer):
    def __init__(self, tf_object, layer_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id, next_layer_id)
        self.layer_type = "Attention"
        self.tf_layer = tf_object

    def __repr__(self):
        repo_str = self.layer_type + ' '
        repo_str += super().__repr__()
        return repo_str


# 包括两个线性层： N -> 4N  &&  4N -> N, N为隐藏层大小。参数量 = (N * 4 * N + 4 * N) + (4 * N * N + N)
class MLP(Layer):
    def __init__(self, tf_object, layer_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id, next_layer_id)
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
    def __init__(self, inherent_id, step, pass_type, layer_id_attn, layer_id_mlp, input_size, output_size, mlp_calc_time, attn_calc_time, mlp_param_size, attn_param_size, former_layer_id=None, next_layer_id=None, mbs=0, Num_of_layers=1, TP=1):
        output_input_size = input_size
        self.inherent_id = inherent_id
        self.step = step
        self.pass_type = pass_type
        self.mbs = mbs
        self.Num_of_layers = Num_of_layers  # 设置适当的层数
        self.TP = TP             # 设置适当的TP值
        self.attention_layer = Attention(self, layer_id=layer_id_attn, inherent_id=inherent_id, input_size=input_size, output_size=output_input_size, 
                                         calc_time=attn_calc_time, param_size=attn_param_size, 
                                         former_layer_id=former_layer_id, next_layer_id=layer_id_mlp)
        
        self.mlp_layer = MLP(self, layer_id=layer_id_mlp, inherent_id=inherent_id, input_size=output_input_size, output_size=input_size, 
                             calc_time=mlp_calc_time, param_size=mlp_param_size, 
                             former_layer_id=layer_id_attn, next_layer_id=next_layer_id)
    
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
        return did, mbid, lid, tid


    def set_tp_groups(self, tp_grp_mlp, tp_grp_attn, tp_fwd_type=None, tp_bkwd_type=None):
        self.attention_layer.set_tp_grp(tp_grp_attn, tp_fwd_type, tp_bkwd_type)
        self.mlp_layer.set_tp_grp(tp_grp_mlp, tp_fwd_type, tp_bkwd_type)
        

    def set_dp_groups(self, dp_grp, dp_type):
        self.mlp_layer.set_dp_grp(dp_grp, dp_type)
        self.attention_layer.set_dp_grp(dp_grp, dp_type)

    def __repr__(self):
        return f"Transformer Layer {self.inherent_id} ({self.extract_ids()}):\n{self.attention_layer}\n{self.mlp_layer}"

if __name__ == '__main__':
    args = parse_arguments()
    # 读取 CSV 文件到 DataFrame
    workload_df = pd.read_csv('workload.csv', sep='\t')
    Parameter_size, Hidden_size, Num_of_layers, Attention_heads, Seq_len, FFN_hidden_size, World_size, TP, PP, DP \
        = get_parameters_by_name(workload_df, 'GPT_7B')

    # Num_of_layers = 3
    # global_batch = 8192
    # micro_batch = 1
    # seq_length = 4096
    # pp_cut = -1    # 压缩的程度。一般来讲应该是 0(最小压缩) 或者 -1 （不压缩）

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

    tf_layers = [[[[] for _ in range(Num_of_layers)] for _ in range(mbs)] for _ in range(DP)]

    passes = 2
    bypass_first_fwd = True
    bypass_last_bkwd = True

    steps = passes * 2
    first_step = 1 if bypass_first_fwd else 0
    last_step = 1 if bypass_last_bkwd else 0
    for step in range(first_step, steps - last_step):
        pass_type = 'FWD' if step//2 == 0 else 'BKWD'
        for did in range(DP):
            dp_grp = [[] for _ in range(Num_of_layers)]  # 对最后一个 mbid 的每一个 Transformer layer，都维护一个 DP 组
            for mbid in range(mbs):
                # 为每一层构建多个 TP 组
                for lid in range(Num_of_layers):
                    # 构建并设置 TP 组
                    tp_grp_attn = []
                    tp_grp_mlp = []
                    last_layer = -1
                    next_layer = -1
                    for tid in range(TP):
                        inherent_id = did * (mbs * Num_of_layers * TP) + mbid * (Num_of_layers * TP) + lid * TP + tid
                        lid1 = get_id()
                        lid2 = get_id()
                        tp_grp_attn.append(lid1)
                        tp_grp_mlp.append(lid2)
                        tf_layers[did][mbid][lid].append(
                            TransformerLayer(inherent_id=inherent_id, step=step, pass_type=pass_type,
                                        layer_id_attn=lid1, layer_id_mlp=lid2, 
                                        input_size=mb_input_size, output_size=mb_input_size, 
                                        mlp_calc_time=0.005, attn_calc_time=0.010, 
                                        mlp_param_size=mlp_param_size_tp, attn_param_size=attn_param_size_tp,
                                        mbs=mbs, Num_of_layers=Num_of_layers, TP=TP)
                            )
                    for tid in range(TP):
                        tf_layers[did][mbid][lid][tid].set_tp_groups(tp_grp_attn=tp_grp_attn, tp_grp_mlp=tp_grp_mlp, tp_fwd_type='ALLREDUCE', tp_bkwd_type='ALLREDUCE')

                
                # 连接多层Transformer网络
                for tid in range(TP):
                    for lid in range(1, Num_of_layers):
                        tf_layers[did][mbid][lid][tid].attention_layer.former_layer_id = tf_layers[did][mbid][lid - 1][tid].mlp_layer.layer_id
                        tf_layers[did][mbid][lid - 1][tid].mlp_layer.next_layer_id = tf_layers[did][mbid][lid][tid].attention_layer.layer_id

                    for lid in range(Num_of_layers):
                        print(f"\n{tf_layers[did][mbid][lid][tid]}")
                
                if pass_type == 'BKWD' and mbid == mbs - 1:
                    # 为最后一个 mbid 添加 DP
                    pass

                if pass_type == 'FWD' and mbid == 0:
                    # 为第一个 mbid 添加 DP
                    pass



    print(f'\nnew_DP: {DP}')
    print(f'new_mbs: {mbs}')
    print(f'new_Num_of_layers: {Num_of_layers}')
    print(f'new_TP: {TP}')