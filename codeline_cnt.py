import os

def count_lines_in_file(file_path):
    """统计单个文件的代码行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            return len(lines)
    except Exception as e:
        print(f"无法读取文件 {file_path}。错误: {e}")
        return 0

def count_total_lines(file_list):
    """统计文件列表中的代码行数总和"""
    total_lines = 0
    for file_path in file_list:
        if os.path.isfile(file_path):
            total_lines += count_lines_in_file(file_path)
        else:
            print(f"文件 {file_path} 不存在或不是一个文件。")
    return total_lines

# 示例文件列表
file_list = [
    'bkwd_test.py', 
    'cc_allreduce_test.py', 
    'collective_communication_lib.py',
    'flow_definition.py',
    'get_fx_graph.py',
    'hardware_model.py',
    'layer_definition.py',
    'node_flow_def.py',
    'test.py',
    'utils_get_fx_graph.py'
    ]

# 统计总行数
total_lines = count_total_lines(file_list)
print(f"文件列表中的代码行数总和为: {total_lines}")
