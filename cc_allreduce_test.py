from collective_communication_lib import * 


if __name__ == '__main__':
    # Example usage
    N = 2
    total_data_size = 60
    flows, v_nodes, node_start, node_end = ring_all_reduce(N, total_data_size)

    flow_num = 0
    for chunk, flow_list in flows.items():
        flow_num += len(flow_list)
        for flow in flow_list:
            print(flow)
    print(f"All-Reduce operation with DpGrpNum = {N} has {flow_num} flows. ")

    node_num = 0
    for node_name, nodelist in v_nodes.items():
        node_num += len(nodelist)
        print("DP", node_name, nodelist)
    print(f"All-Reduce operation with DpGrpNum = {N} has {node_num} virtual nodes. ")
    
    print("node_start: ")
    print(node_start)
    
    print("node_end: ")
    print(node_end)
    
    
