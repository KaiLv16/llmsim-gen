# -*- coding: utf-8 -*-

import pandas as pd

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
    def __init__(self, layer_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id, next_layer_id)
        self.layer_type = "Attention"

    def __repr__(self):
        repo_str = self.layer_type + ' '
        repo_str += super().__repr__()
        return repo_str


# 包括两个线性层： N -> 4N  &&  4N -> N, N为隐藏层大小。参数量 = (N * 4 * N + 4 * N) + (4 * N * N + N)
class MLP(Layer):
    def __init__(self, layer_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id, next_layer_id)
        self.layer_type = "MLP"

    def __repr__(self):
        repo_str = self.layer_type + ' '
        repo_str += super().__repr__()
        return repo_str


class TransformerLayer:
    '''
    inherent_id: 用于确定这一层的物理位置
    '''
    def __init__(self, inherent_id, layer_id_attn, layer_id_mlp, input_size, output_size, mlp_calc_time, attn_calc_time, mlp_param_size, attn_param_size, former_layer_id=None, next_layer_id=None):
        output_input_size = input_size
        self.inherent_id = inherent_id
        self.attention_layer = Attention(layer_id=layer_id_attn, inherent_id=inherent_id, input_size=input_size, output_size=output_input_size, 
                                         calc_time=attn_calc_time, param_size=attn_param_size, 
                                         former_layer_id=former_layer_id, next_layer_id=layer_id_mlp)
        
        self.mlp_layer = MLP(layer_id=layer_id_mlp, inherent_id=inherent_id, input_size=output_input_size, output_size=input_size, 
                             calc_time=mlp_calc_time, param_size=mlp_param_size, 
                             former_layer_id=layer_id_attn, next_layer_id=next_layer_id)
    
    def get_grp_id(self, Num_of_layers, TP):
        did_lid_part = self.inherent_id // TP
        tid = self.inherent_id % TP
        did = did_lid_part // Num_of_layers
        lid = did_lid_part % Num_of_layers
        return did, lid, tid

    def set_tp_groups(self, tp_grp_mlp, tp_grp_attn, tp_fwd_type=None, tp_bkwd_type=None):
        self.attention_layer.set_tp_grp(tp_grp_attn, tp_fwd_type, tp_bkwd_type)
        self.mlp_layer.set_tp_grp(tp_grp_mlp, tp_fwd_type, tp_bkwd_type)
        

    def set_dp_groups(self, dp_grp, dp_type):
        self.mlp_layer.set_dp_grp(dp_grp, dp_type)
        self.attention_layer.set_dp_grp(dp_grp, dp_type)

    def __repr__(self):
        Num_of_layers = 3  # 设置适当的层数
        TP = 4             # 设置适当的TP值
        return f"Transformer Layer {self.inherent_id} ({self.get_grp_id(Num_of_layers, TP)}):\n{self.attention_layer}\n{self.mlp_layer}"

if __name__ == '__main__':
    # 读取 CSV 文件到 DataFrame
    workload_df = pd.read_csv('workload.csv', sep='\t')
    Parameter_size, Hidden_size, Num_of_layers, Attention_heads, Seq_len, FFN_hidden_size, World_size, TP, PP, DP \
        = get_parameters_by_name(workload_df, 'GPT_7B')

    Num_of_layers = 3
    global_batch = 8192
    micro_batch = 1
    seq_length = 4096

    mb_input_size = Seq_len * micro_batch * Hidden_size
    assert FFN_hidden_size == Hidden_size * 4
    mlp_param_size_tp = (Hidden_size * FFN_hidden_size + FFN_hidden_size + FFN_hidden_size * Hidden_size + Hidden_size) / TP
    Head_size = Hidden_size / Attention_heads
    attn_param_size_tp = (3 * (Hidden_size * Head_size * Attention_heads) + Hidden_size * Hidden_size) / TP   #  = 4 * Hidden_size^2 / TP

    tf_layers = [[[] for _ in range(Num_of_layers)] for _ in range(DP)]
    for did in range(DP):
        dp_grp = []
        for lid in range(Num_of_layers):
            tp_grp_attn = []
            tp_grp_mlp = []
            last_layer = -1
            next_layer = -1
            for tid in range(TP):
                inherent_id = did * (Num_of_layers * TP) + lid * TP + tid
                lid1 = get_id()
                lid2 = get_id()
                tp_grp_attn.append(lid1)
                tp_grp_mlp.append(lid2)
                tf_layers[did][lid].append(TransformerLayer(inherent_id=inherent_id, 
                                layer_id_attn=lid1, layer_id_mlp=lid2, 
                                input_size=mb_input_size, output_size=mb_input_size, 
                                mlp_calc_time=0.005, attn_calc_time=0.010, 
                                mlp_param_size=mlp_param_size_tp, attn_param_size=attn_param_size_tp)
                                )
            for tid in range(TP):
                # 设置tp_grp 和 dp_grp (if exist)
                tf_layers[did][lid][tid].set_tp_groups(tp_grp_attn=tp_grp_attn, tp_grp_mlp=tp_grp_mlp, tp_fwd_type='ALLREDUCE', tp_bkwd_type='ALLREDUCE')

                tf_layers[did][lid][tid].set_dp_groups(dp_grp=[3], dp_type='ALLREDUCE')

                # 打印Transformer层信息
                print(f"\n{tf_layers[did][lid][tid]}")
