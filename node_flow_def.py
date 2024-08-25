import networkx as nx
import numpy as np
import re
import random
import matplotlib.pyplot as plt
from pyvis import network as pvnet
from collective_communication_lib import *

# 全局 flow ID 分发器
global_flowid_register = -1

def assign_flow_id():
    global global_flowid_register
    global_flowid_register += 1
    return global_flowid_register

global_dependency_register = -1

def assign_dep_id():
    global global_dependency_register
    global_dependency_register += 1
    return global_dependency_register


# 全局 DP ID 分发器
global_DP_ID = -1

def assign_dp_id():
    global global_DP_ID
    global_DP_ID += 1
    return global_DP_ID



# 当前只支持后向的bucketing，即把输出拆分出来。
class NodeInfo:
    def __init__(self, name, gid, lid, dp_id, host, gpu, nic, read_time, exec_time, write_time, virtual=False):
        self.name = name
        self.virt = virtual
        self.grp_id = gid
        self.layer_id = lid
        self.dp_id = dp_id
        self.host = host
        self.gpu = gpu
        self.nic = nic
        self.read_time = read_time
        self.exec_time = exec_time
        self.write_time = write_time
        self.read_exec_delay = read_time + exec_time + write_time
        
        self.flow_depend = []   # TODO: 想想怎么定义？需要包含哪些要素？需要包含PP和DP，后期需要扩展TP
        self.flow_invoke = []
        # 用来处理有反压的情况
        self.dep_depend = []
        self.dep_invoke = []

    def __repr__(self):
        return f"""NodeInfo(name={self.name}, gid={self.grp_id}, host={self.host}, gpu={self.gpu}, nic={self.nic}, read_exec_delay={self.read_exec_delay},
            flow_depend: {self.flow_depend}, 
            flow_invoke: {self.flow_invoke}, 
            dep_depend: {self.dep_depend}, 
            dep_invoke: {self.dep_invoke}
    )"""

    def easyprint(self):
        return f"NodeInfo(name={self.name}, gid={self.grp_id}, host={self.host}, gpu={self.gpu}, nic={self.nic}, read_exec_delay={self.read_exec_delay})"
    
    def add_flow(self, flow_id, op):
        assert op in ['depend', 'invoke']
        if op == 'depend':
            self.flow_depend.append(flow_id)
        else:
            self.flow_invoke.append(flow_id)

    def add_dependency(self, dep_id, op):
        assert op in ['depend', 'invoke']
        if op == 'depend':
            self.flow_depend.append(dep_id)
        else:
            self.flow_invoke.append(dep_id)



class Flow:
    def __init__(self, id, size, srcEp, dstEp, ftype='intradevice', opcode='pp',prio=-1, round=-1, lat=-1):
        '''
        For now, ftype supports <intradevice> / <intrahost> / <interhost>
        opcode supports <dp> / <pp>
        '''
        self.id = id        # 全局唯一的flowid
        self.size = size
        self.ftype = ftype
        assert isinstance(srcEp, NodeInfo)
        assert isinstance(dstEp, NodeInfo)
        self.srcEp = srcEp
        self.dstEp = dstEp
        self.opcode = opcode
        self.prio = prio    # only used in DP
        self.round = round
        self.lat = lat

    
    def __repr__(self):
        return f"Flow (id={self.id}, size={self.size}, flow_type={self.ftype}, srcEp={self.srcEp.easyprint()}, dstEp={self.dstEp.easyprint()}, opcode={self.opcode}, round={self.round}, lat={self.lat}"


class Dep:
    def __init__(self, id, srcEp, dstEp):
        self.id = id        # 全局唯一的flowid
        assert isinstance(srcEp, NodeInfo)
        assert isinstance(dstEp, NodeInfo)
        self.srcEp = srcEp
        self.dstEp = dstEp
    
    def __repr__(self):
        return f"Dep (id={self.id}, srcEp={self.srcEp.easyprint()}, dstEp={self.dstEp.easyprint()}"




'''
给定一个node_list 列表和一个flow_list 列表：
node_list中的每个元素是一个Node类的对象，包括nodeid、grp_id、calc_time（节点的执行时间）、dependflow(自己所依赖的流ID列表、invokeflow（自己触发传输的流ID列表）
我们定义，node对象需要等待dependflow为空，才开始执行。执行结束后，会开始触发invokeflow列表中的所有流。
flow_list中的每个元素是一个Flow类的对象，包括flowid、transmittime、src_node（起始于哪个node对象）、dst_node（结束于哪个node对象）


我希望实现一个功能，将属于同一个grp_id的所有node融合成一个新的node。在同一个grp——id的node之间的通信可以使用flow的transmittime掩盖掉，利用其计算新的节点的calc_time。

例如：

node_list包括：
[
    Node(nodeid='1', grp_id=0, calc_time=1, dependflow=[], invokeflow=[1])
    Node(nodeid='2', grp_id=0, calc_time=3, dependflow=[1], invokeflow=[2])
    Node(nodeid='3', grp_id=0, calc_time=2, dependflow=[2], invokeflow=[4, 6])
    Node(nodeid='4', grp_id=1, calc_time=3, dependflow=[4], invokeflow=[5])
    Node(nodeid='5', grp_id=1, calc_time=2, dependflow=[5, 6], invokeflow=[7])
    Node(nodeid='8', grp_id=2, calc_time=4, dependflow=[7], invokeflow=[])
]
flow_list包括：
[
    Flow(flowid=1, transmittime=4, src_node='1', dst_node='2')
    Flow(flowid=2, transmittime=5, src_node='2', dst_node='3')
    Flow(flowid=4, transmittime=7, src_node='3', dst_node='4')
    Flow(flowid=5, transmittime=5, src_node='4', dst_node='5')
    Flow(flowid=6, transmittime=2, src_node='3', dst_node='5')
    Flow(flowid=7, transmittime=0, src_node='5', dst_node='8')

]

输出一个新的node列表：
[
    MergedNode(id='1', grp_id=0, calc_time=15, dependflow=[], invokeflow=[4, 6])
    MergedNode(id='2', grp_id=0, calc_time=3, dependflow=[4, 6], invokeflow=[7])
    MergedNode(id='2', grp_id=0, calc_time=3, dependflow=[3, 4, 6], invokeflow=[7])
]

'''