from collections import defaultdict
from hardware_model import *
from node_flow_def import * 


class cc_vnode:
    '''
    This is the virtual node used for generating CCL flows.
    '''
    def __init__(self, name, dp_id, exec_time=0):
        self.name = name
        self.dp_id = dp_id
        self.exec_time = exec_time
        
        self.flow_depend = []
        self.flow_invoke = []
        # 用来处理有反压的情况
        self.dep_depend = []
        self.dep_invoke = []

    def __repr__(self):
        return \
f"""cc_vnode (name={self.name}, dp_id={self.dp_id}, exec_time={self.exec_time},
        flow_depend: {self.flow_depend}, 
        flow_invoke: {self.flow_invoke}, 
        dep_depend: {self.dep_depend}, 
        dep_invoke: {self.dep_invoke}
    )"""

    def easyprint(self):
        return f"name={self.name}"
    
    def convert_to_NodeInfo(self, name, host, gpu, nic):
        node = NodeInfo(name, -1, -1, self.dp_id, host, gpu, nic, 0, 0, 0, True)
        return node


class cc_Flow: 
    def __init__(self, flow_id, src, dst, size, opcode, round, src_node, dst_node):
        self.flow_id = flow_id
        self.src = src
        self.dst = dst
        self.size = size
        self.opcode = opcode
        self.round = round
        self.src_node = src_node
        self.dst_node = dst_node

    def __repr__(self):
        return f"cc_Flow (flow_id={self.flow_id}, src={self.src}, dst={self.dst}, size={self.size}, op={self.opcode}, round={self.round}, src_node={self.src_node.easyprint()}, dst_node={self.dst_node.easyprint()})"

    def easyprint(self):
        return f"cc_Flow (flow_id={self.flow_id}, src={self.src}, dst={self.dst}, size={self.size}, op={self.opcode}"
    
    def convert_to_Flow(self, flowid, srcEp, dstEp, ftype, prio):
        flow = Flow(flowid, self.size, srcEp, dstEp, ftype, opcode='dp', prio=prio, round=-1, lat=-1)
        return flow


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



def tree_all_reduce(self, tensor_list):

    pass

def hierarchical_all_reduce(self, tensor_list):

    pass
