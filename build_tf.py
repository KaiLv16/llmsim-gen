# 全局 flow ID 分发器
global_lid_register = -1
def assign_layer_id():
    global global_lid_register
    global_lid_register += 1
    return global_lid_register


class Layer:
    def __init__(self, layer_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        self.layer_id = layer_id
        self.input_size = input_size
        self.output_size = output_size
        self.calc_time = calc_time
        self.param_size = param_size
        self.former_layer_id = former_layer_id
        self.next_layer_id = next_layer_id
        self.tp_grp = []
        self.tp_type = None
        self.dp_grp = []
        self.dp_type = None

    def set_tp_grp(self, tp_grp, tp_type):
        self.tp_grp = tp_grp
        self.tp_type = tp_type if tp_grp else None

    def set_dp_grp(self, dp_grp, dp_type):
        self.dp_grp = dp_grp
        self.dp_type = dp_type if dp_grp else None

    def __repr__(self):
        return (f"Layer {self.layer_id}: Input Size = {self.input_size}, Output Size = {self.output_size}, "
                f"Calc Time = {self.calc_time}s, Param Size = {self.param_size}, "
                f"Former Layer = {self.former_layer_id}, Next Layer = {self.next_layer_id}, "
                f"TP Group = {self.tp_grp}, TP Type = {self.tp_type}, DP Group = {self.dp_grp}, DP Type = {self.dp_type}")


# 包括两个线性层： N -> 4N  &&  4N -> N, N为隐藏层大小。参数量 = (N * 4 * N + 4 * N) + (4 * N * N + N)
class MLP(Layer):
    def __init__(self, layer_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_id, input_size, output_size, calc_time, param_size, former_layer_id, next_layer_id)
        self.layer_type = "MLP"

    def __repr__(self):
        repo_str = self.layer_type + '\n'
        repo_str += super().__repr__()
        return repo_str

class Attention(Layer):
    def __init__(self, layer_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_id, input_size, output_size, calc_time, param_size, former_layer_id, next_layer_id)
        self.layer_type = "Attention"

    def __repr__(self):
        repo_str = self.layer_type + '\n'
        repo_str += super().__repr__()
        return repo_str

class TransformerLayer:
    def __init__(self, layer_id1, layer_id2, input_size, output_size, mlp_calc_time, attn_calc_time, mlp_param_size, attn_param_size, former_layer_id=None, next_layer_id=None):
        output_input_size = mlp_param_size / input_size
        self.attention_layer = Attention(layer_id=layer_id1, input_size=input_size, output_size=output_input_size, 
                                         calc_time=attn_calc_time, param_size=attn_param_size, 
                                         former_layer_id=former_layer_id, next_layer_id=layer_id2)
        
        self.mlp_layer = MLP(layer_id=layer_id2, input_size=output_input_size, output_size=input_size, 
                             calc_time=mlp_calc_time, param_size=mlp_param_size, 
                             former_layer_id=layer_id1, next_layer_id=next_layer_id)
        

    def set_tp_dp_groups(self, tp_grp, tp_type, dp_grp, dp_type):
        self.mlp_layer.set_tp_dp_groups(tp_grp, tp_type, dp_grp, dp_type)
        self.attention_layer.set_tp_dp_groups(tp_grp, tp_type, dp_grp, dp_type)

    def __repr__(self):
        return f"Transformer Layer {self.mlp_layer.layer_id}:\n{self.mlp_layer}\n{self.attention_layer}"

# 示例：创建一个Transformer层
transformer_layer = TransformerLayer(layer_id=1, input_size=512, output_size=512, mlp_calc_time=0.005, attn_calc_time=0.010, 
                                     mlp_param_size=1048576, attn_param_size=262144)

# 设置tp_grp和dp_grp
transformer_layer.set_tp_dp_groups(tp_grp=[1, 2], tp_type='Type_A', dp_grp=[3], dp_type='Type_B')

# 打印Transformer层信息
print(transformer_layer)
