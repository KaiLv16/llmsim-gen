import networkx as nx
import matplotlib.pyplot as plt
from pyvis import network as pvnet
from collective_communication_lib import *
from node_flow_def import * 


def cc_wrapper(node_list, src_node_list, dst_node_list, data_size, prio, cc_mode='ring-all-reduce'):
    '''
    这个函数的作用是：
    '''
    N = len(src_node_list)
    assert len(src_node_list) == len(dst_node_list)
    # 新增的点和边
    flow_dict = {}
    node_dict = {}
    dep_dict = {}
    # main logic. new CC algo can be inserted.
    if cc_mode == 'ring-all-reduce':
        flows, vnode_grp_by_dpid, v_nodename_start, v_nodename_end = ring_all_reduce(N, data_size)
        #  绑定node和flow 到 相应的 dict
        for dpid_name_tuple, node in vnode_grp_by_dpid.items():
            dp_id, nodename = dpid_name_tuple
            print(dp_id, nodename)  # 打印对象的类型
            src_node = node_list[src_node_list[dp_id]]
            new_node_name = src_node.name + node.name
            new_nodeinfo = node.convert_to_NodeInfo(new_node_name, src_node.host, src_node.gpu, src_node.nic)
            node_dict[new_node_name] = new_nodeinfo
            
        
        for chunk, flow_list in flows.items():
            for flow in flow_list:
                flow_id = assign_flow_id()
                for node_name, node in node_dict.items():
                    if flow.src_node.name in node_name:
                        srcEp = node
                    if flow.dst_node.name in node_name:
                        dstEp = node
                assert isinstance(srcEp, NodeInfo) and isinstance(dstEp, NodeInfo)
                if srcEp.host == dstEp.host:
                    if srcEp.gpu == dstEp.gpu:
                        ftype = 'intradevice' 
                    else:
                        ftype = 'intrahost'
                else:
                    ftype = 'interhost'
                flow_dict[flow_id] = flow.convert_to_Flow(flow_id, srcEp, dstEp, ftype, prio)

        # 连接 dp 到 src layers
        for node_name, node in node_dict.items():
            is_start = False
            is_end = False
            for start_name in  v_nodename_start:
                if start_name in node_name:
                    is_start = True
            for end_name in  v_nodename_end:
                if end_name in node_name:
                    is_end = True
            assert not (is_start and is_end), "a node cannot be both start and end!"

            if is_start:
                depid = assign_dep_id()
                node_list[src_node_list[node.dp_id]].dep_invoke.append([depid, 0])
                node.dep_depend.append([depid, 0])
                dep_dict[depid] = Dep(depid, node_list[src_node_list[node.dp_id]], node)

            if is_end:
                depid = assign_dep_id()
                node_list[dst_node_list[node.dp_id]].dep_depend.append([depid, 0])
                node.dep_invoke.append([depid, 0])
                dep_dict[depid] = Dep(depid, node, node_list[dst_node_list[node.dp_id]])

    else:
        raise Exception(f"'{cc_mode}' not implemented yet.")
    
    return node_dict, flow_dict, dep_dict
    
    

def sim_node_and_flow_gen(dp_sim_grps, max_dp_prio_idx, pp_granularity, pp_bucket_size=-1, dp_bucket_size=-1, pp_grp_label='avg_mem_grp_idx', print_detail=False, plot_pp_only=False):
    '''
    output_mode 包括 'interdevice', 'intradevice' 
    '''
    assert pp_granularity in ['layer', 'device']

    node_list = {}
    flow_list = {}
    dep_list = {}
    # step 2: generate (and filter out useless) PP flow (same dp group)
    for dp_idx, dpg in dp_sim_grps.items():
        if print_detail:
            print(f"generating pp flows for dp {dp_idx} (node num = {len(dpg)})")
        assert len(dpg['args'][0]) == 0, "The first node of a dp group should not rely on other nodes!"

        # 从1开始，因为0行是输入
        for i in range(1, len(dpg)):
            assert dpg['next_server'][i-1] == dpg['server_id'][i], f"dp {dp_idx}, idx {i}: {dpg['next_server'][i-1]} != {dpg['server_id'][i]}"
            assert dpg['next_gpu'][i-1] == dpg['gpu_id'][i]
            assert dpg['next_nic'][i-1] == dpg['nic_id'][i], f"dpg['next_nic'][i-1]: {dpg['next_nic'][i-1]}, dpg['nic_id'][i]: {dpg['nic_id'][i]}, i-1: {i-1}, i : {i}, dp_idx: {dp_idx}"
            assert dpg['server_id'][i-1] == dpg['prev_server'][i]
            assert dpg['gpu_id'][i-1] == dpg['prev_gpu'][i]
            assert dpg['nic_id'][i-1] == dpg['prev_nic'][i], f"dpg['nic_id'][i-1]: {dpg['nic_id'][i-1]}, dpg['prev_nic'][i]: {dpg['prev_nic'][i]}, i-1: {i-1}, i : {i}, dp_idx: {dp_idx}"

            # 检查流的属性。注意！这里并不是按照node的顺序来的，而是按照依赖图来的。
            dependent_nodes = dpg['args'][i]
            for prev_str in dependent_nodes:
                assert isinstance(prev_str, int) or isinstance(prev_str, str), f"unknown type for dependency node {prev_str}: {type(prev_str)}"
                if isinstance(prev_str, str): #   and not prev_node.endswith('shadow'):
                    prev_node = prev_str
                    if prev_str.endswith('/shadow'):
                        prev_node = prev_str[:-7]

                    prev_layer_idx = dpg.loc[dpg['name'] == prev_node].index
                    assert len(prev_layer_idx) == 1,  f"{dpg['name'][i]}: found {len(prev_layer_idx)} dependency node(s) for {prev_node} arg of {dpg['name'][i]}!"
                    prev_layer_idx = prev_layer_idx[0]

                    # print(dpg['name'][i], prev_node, prev_layer_idx)
                    # print(dpg['output_#'][prev_layer_idx], ": ", type(dpg['output_#'][prev_layer_idx]), dpg['input_#'][i], ": ", type(dpg['input_#'][i]))
                    if not prev_str.endswith('/shadow'):
                        assert dpg['output_#'][prev_layer_idx] == dpg['input_#'][i], \
                                f"The output element cnt of prev layer ({dpg['nic_id'][prev_layer_idx]}) and the input element num of my layer ({dpg['nic_id'][i]}) is not equal!  prev_layer_idx: {prev_layer_idx}, i: {i}, dp_idx: {dp_idx}" 
                    size = dpg['input_#'][i] * dpg['dtype_size'][i]    # in Bytes
                    
                    src_ep = NodeInfo(dpg['name'][prev_layer_idx], dpg[pp_grp_label][prev_layer_idx], dpg['global_index'][prev_layer_idx], dp_idx,\
                                        dpg['prev_server'][i], dpg['prev_gpu'][i], dpg['prev_nic'][i], \
                                        dpg['read_time'][prev_layer_idx], dpg['exec_time'][prev_layer_idx], dpg['write_time'][prev_layer_idx])
                    
                    dst_ep = NodeInfo(dpg['name'][i], dpg[pp_grp_label][i], dpg['global_index'][i], dp_idx, \
                                      dpg['server_id'][i], dpg['gpu_id'][i], dpg['nic_id'][i], \
                                      dpg['read_time'][i], dpg['exec_time'][i], dpg['write_time'][i])
                    
                    if dpg['name'][prev_layer_idx] not in node_list:
                        node_list[dpg['name'][prev_layer_idx]] = src_ep
                    
                    if dpg['name'][i] not in node_list:
                        node_list[dpg['name'][i]] = dst_ep
                    
                    if not prev_str.endswith('/shadow'):    # 正常数据流
                        # ftype supports <intradevice> / <intrahost> / <interhost>
                        if dpg['server_id'][i-1] == dpg['server_id'][i]:
                            if dpg['gpu_id'][i-1] == dpg['gpu_id'][i]:
                                ftype = 'intradevice' 
                            else:
                                ftype = 'intrahost'
                        else:
                            ftype = 'interhost'
                        # opcode supports <dp> / <pp>
                        opcode = 'pp'
                        round = -1    # 这个参数是用来干啥的来着？？忘了啊，笑死
                        lat = -1      # 用于压缩用.  lat=-1，则为待填充状态（需要交给NS3）. 如果lat == 0，则意味着是同一个GPU的流，上一个node写完数据，下一个node读数据就可以了
                        
                        if pp_granularity == 'device':
                            # 前后层在同一个GPU上，则压缩该流：上一个node写完数据，下一个node读数据就可以了
                            if src_ep.host == dst_ep.host and src_ep.gpu == dst_ep.gpu:
                                lat = 0
                            # 前后层在同一个host的不同GPU上：走nvlink
                            if src_ep.host == dst_ep.host and src_ep.gpu != dst_ep.gpu:
                                lat = 0

                        if pp_bucket_size == -1:
                            flow_id = assign_flow_id()
                            flow = Flow(flow_id, size, src_ep, dst_ep, ftype, opcode, -1, round, lat)
                            flow_list[flow_id] = flow
                            if print_detail:
                                print("Add flow:", flow)
                            node_list[src_ep.name].flow_invoke.append([flow_id, src_ep.write_time])
                            node_list[dst_ep.name].flow_depend.append([flow_id, 0])
                        else:
                            Size = size
                            while size > 0:
                                flow_id = assign_flow_id()
                                flow = Flow(flow_id, pp_bucket_size if size > pp_bucket_size else size, src_ep, dst_ep, ftype, opcode, -1, round, lat)
                                flow_list[flow_id] = flow
                                if print_detail:
                                    print("Add flow with bucket:", flow)
                                node_list[src_ep.name].flow_invoke.append([flow_id, (size / Size) * src_ep.write_time])
                                node_list[dst_ep.name].flow_depend.append([flow_id, 0])
                                size -= pp_bucket_size
                    
                    else:    # 单纯添加依赖
                        dep_id = assign_dep_id()
                        dep = Dep(dep_id, src_ep, dst_ep)
                        dep_list[dep_id] = dep
                        if print_detail:
                            print("Add dep:", dep)
                        node_list[src_ep.name].dep_invoke.append(dep_id)
                        node_list[dst_ep.name].dep_depend.append(dep_id)

    if not plot_pp_only:
        # step 2: generate DP flow (with collective communication primitives)
        for dp_idx, dpg in dp_sim_grps.items():
            for i in range(0, len(dpg)):
                dp_jobs = dpg['dp_args'][i]
                for dp_job in dp_jobs:
                    # dp_job 的定义：['src/dst', unique_dp_id, total_size, src_layer, dst_layer, grp_id, dp_prio]
                    src_name = dp_job[3]
                    dst_name = dp_job[4]
                    data_size = dp_job[2]
                    prio = dp_job[6]
                    assert isinstance(src_name, str) and isinstance(dst_name, str)
                    if dp_job[0] == 'src':
                        src_node_name_list = []
                        dst_node_name_list = []
                        # 暴力扫描一遍
                        for dp_idx, dpg in dp_sim_grps.items():
                            # 对于每个dp
                            for i in range(0, len(dpg)):
                                if src_name in dpg['name'][i]:
                                    src_node_name_list.append(dpg['name'][i])
                                if dst_name in dpg['name'][i]:
                                    dst_node_name_list.append(dpg['name'][i])
                    else:
                        # TODO: maybe can use this for further check? maybe, but useless for now.
                        pass
                    assert len(src_node_name_list) == len(dp_sim_grps)
                    print(dst_node_name_list)
                    assert len(dst_node_name_list) == len(dp_sim_grps)

                    dp_node_dict, dp_flow_dict, dp_dep_dict = cc_wrapper(node_list, src_node_name_list, dst_node_name_list, data_size, prio, cc_mode='ring-all-reduce')

                    for nodename, node in dp_node_dict.items():
                        node_list[nodename] = node

                    for flowid, flow in dp_flow_dict.items():
                        flow_list[flowid] = flow
                        
                    for depid, dep in dp_dep_dict.items():
                        dep_list[depid] = dep

    return node_list, flow_list, dep_list

'''
# 创建节点和流
node_list = [
    Node(nodeid='1', grp_id=0, calc_time=1, dependflow=[], invokeflow=[1]),
    Node(nodeid='2', grp_id=0, calc_time=3, dependflow=[1], invokeflow=[2]),
    Node(nodeid='3', grp_id=0, calc_time=2, dependflow=[2], invokeflow=[4, 6]),
    Node(nodeid='4', grp_id=1, calc_time=3, dependflow=[4], invokeflow=[5]),
    Node(nodeid='5', grp_id=1, calc_time=2, dependflow=[5, 6], invokeflow=[7]),
    Node(nodeid='8', grp_id=2, calc_time=4, dependflow=[7], invokeflow=[])
]

flow_list = [
    Flow(flowid=1, transmittime=4, src_node='1', dst_node='2'),
    Flow(flowid=2, transmittime=5, src_node='2', dst_node='3'),
    Flow(flowid=4, transmittime=7, src_node='3', dst_node='4'),
    Flow(flowid=5, transmittime=5, src_node='4', dst_node='5'),
    Flow(flowid=6, transmittime=2, src_node='3', dst_node='5'),
    Flow(flowid=7, transmittime=0, src_node='5', dst_node='8')
]
'''

def plot_sim_graph(node_list, flow_list, dep_list, draw_seed=1, plot_cc_only=False):

    # 创建一个空的有向图
    # G = nx.DiGraph()
    G = nx.MultiDiGraph()
    pos = {}
    # 添加节点到图中，并附加计算时间作为节点属性
    for nodeid, node in node_list.items():
        # pos[node.name] = (node.dp_id, node.layer_id)            # 指定每个点的位置
        if not plot_cc_only:
            G.add_node(node.name, grp_id=node.grp_id, calc_time=node.read_exec_delay)
        else:
            if 'chunk' in node.name:
                print('122365326356127981')
                G.add_node(node.name, grp_id=node.grp_id, calc_time=node.read_exec_delay)

    # 添加边到图中，并附加传输时间作为边属性
    for flowid, flow in flow_list.items():
        G.add_edge(flow.srcEp.name, flow.dstEp.name, key=flow.id, flowid=flow.id, size=flow.size, op=flow.opcode, txtime=flow.lat)
    
    # 添加依赖到图中，并附加传输时间作为边属性
    for depid, dep in dep_list.items():
        G.add_edge(dep.srcEp.name, dep.dstEp.name, key=dep.id, flowid=dep.id)


    # 使用布局算法
    # pos=nx.circular_layout(G)          # 生成圆形节点布局
    # pos=nx.random_layout(G)            # 生成随机节点布局
    # pos=nx.shell_layout(G)             # 生成同心圆节点布局
    # pos=nx.spring_layout(G)            # 利用Fruchterman-Reingold force-directed算法生成节点布局
    pos = nx.spring_layout(G, seed=draw_seed)  # 设置随机种子以确保布局可重复
    # pos=nx.spectral_layout(G)          # 利用图拉普拉斯特征向量生成节点布局
    # pos=nx.kamada_kawai_layout(G)      #使用Kamada-Kawai路径长度代价函数生成布局

    # 手动设置节点位置
    # 

    # pos = {
    # '1': (0, 0),
    # '2': (1, 1),
    # '3': (2, 0),
    # '4': (3, 1),
    # '5': (4, 0),
    # '8': (5, 1)
    # }
    # 定义不同 grp_id 的颜色
    color_map = {
        0: 'lightblue',
        1: 'lightgreen',
        2: 'lightcoral'
    }

    # node_color = 'lightblue'  # 所有节点的颜色为浅蓝色
    if not plot_cc_only:
        node_colors = [
            'gray' if G.nodes[node]['grp_id'] == -1 
            else color_map[G.nodes[node]['grp_id'] % 3]
            for node in G.nodes()
        ]

        # node_sizes = [500 for node in G.nodes()]  # 根据计算时间调整节点大小

        # 创建节点标签，显示nodeid和calc_time
        node_labels = {node: f"{node}\nTime: {G.nodes[node]['calc_time']}" for node in G.nodes()}

        # 获取边的标签，显示flowid和transmittime
        edge_labels = {(flow.srcEp.name, flow.dstEp.name): f"Flow {flow.id}\nSize: {flow.size}" for fid, flow in flow_list.items()}
        for depid, dep in dep_list.items():
            edge_labels[(dep.srcEp.name, dep.dstEp.name)] = f"Dep {dep.id}"

        # 绘制节点
        # nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='black')

        # # 绘制有向边，确保箭头不会被节点遮挡
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray', connectionstyle=f'arc3,rad=0.2')
        
    # ax = plt.gca()
    # for e in G.edges:
    #     ax.annotate("",
    #                 xy=pos[e[0]], xycoords='data',
    #                 xytext=pos[e[1]], textcoords='data',
    #                 arrowprops=dict(arrowstyle="-|>", color="0.5",
    #                                 shrinkA=10, shrinkB=10,
    #                                 patchA=None, patchB=None,
    #                                 connectionstyle="arc3,rad=rrr".replace('rrr',str(random.uniform(0.1, 0.6))
    #                                 ),
    #                                 ),
    #                 )

        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold', verticalalignment='center')

        # 绘制边标签
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')
    
    
    else:
        nx.draw_networkx_nodes(G, pos)

        # # 绘制有向边，确保箭头不会被节点遮挡
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray', connectionstyle=f'arc3,rad=0.2')

    # # 自定义边标签的位置和旋转
    # for (src, dst), label in edge_labels.items():
    #     # 获取边的中点位置
    #     x_start, y_start = pos[src]
    #     x_end, y_end = pos[dst]
    #     x_label = (x_start + x_end) / 2
    #     y_label = (y_start + y_end) / 2
        
    #     # 计算边的角度（旋转角度）
    #     angle = (np.arctan2(y_end - y_start, x_end - x_start) * 180 / np.pi) - 90
        
    #     # 绘制边标签，旋转90度
    #     plt.text(x_label, y_label, label, fontsize=10, color='red', ha='center', va='center', rotation=angle, bbox=dict(facecolor='white', edgecolor='none', pad=0.2))

    # 显示图
    plt.show()





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