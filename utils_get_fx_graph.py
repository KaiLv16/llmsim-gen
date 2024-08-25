import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import OrderedDict
import torch
import torchvision
import torch.nn as nn

def delete_node_from_tuple(my_tuple, node_to_delete, change_to_str=True):
    assert isinstance(node_to_delete, str), "The input 'node_to_delete' must be Str!"
    my_args = []
    for s in my_tuple:
        if isinstance(s, torch.fx.node.Node):
            if node_to_delete != s.name:
                my_args.append(s.name if change_to_str else s)
        elif isinstance(s, int):
            my_args.append(s)
        elif isinstance(s, str):
            if node_to_delete != s:
                my_args.append(s)
        else:
            raise Exception(f"arg {s} is of type {type(s)}")
    return my_args


def str_to_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. Got: {value}')
    

# 获取每层输出形状和计算数据量
def print_layer_data_transfer(model, input_size, batch_size=1):
    def register_hook(module, name):
        def hook(name, module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["name"] = name
            summary[m_key]["output_shape"] = list(output.size())
        
            # 计算每层输出的数据量（以字节为单位）
            output_size = torch.tensor(output.size()).prod().item()
            dtype_size = output.element_size()
            summary[m_key]["output_size (bytes)"] = output_size * dtype_size

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []

# 注册钩子
    for name, module in model.named_modules():
        register_hook(module, name)

    # 创建一个dummy输入
    x = torch.rand(batch_size, *input_size)
    with torch.no_grad():
        model(x)

    # 移除钩子
    for h in hooks:
        h.remove()

    return summary

# usage:

    # 使用torchsummary来显示每层的计算量和参数传递量
    # print("\nsummary()")
    # summary(model=model, input_size=(1, 28, 28), batch_size=1, device='cpu')
    # print("\nsummary end")

    # input_size = (1, 28, 28)
    # layer_data = print_layer_data_transfer(model, input_size, batch_size=1)


    # print("\nprint(df)")
    # df = pd.DataFrame(layer_data).T
    # print(df)



def can_partition(nums, n, max_sum):
    current_sum = 0
    count = 1
    for num in nums:
        if current_sum + num > max_sum:
            count += 1
            current_sum = num
            if count > n:
                return False
        else:
            current_sum += num
    return True



def partition_list(nums, n):
    if n == 0:
        return None
    if len(nums) <= n:
        return [[num] for num in nums] + [[] for _ in range(n - len(nums))]

    low, high = max(nums), sum(nums)
    while low < high:
        mid = (low + high) // 2
        if can_partition(nums, n, mid):
            high = mid
        else:
            low = mid + 1

    partitions = []
    current_sum = 0
    current_partition = []
    for num in nums:
        if current_sum + num > low:
            partitions.append(current_partition)
            current_partition = [num]
            current_sum = num
        else:
            current_partition.append(num)
            current_sum += num

    partitions.append(current_partition)
    
    while len(partitions) < n:
        partitions.append([])

    return partitions


def debug_update_dataframe(df, df_title='Param_#', layers_per_device=4, pp_grp_label='avg_mem_grp_idx'):
    df.drop(df.index[layers_per_device * layers_per_device : ], inplace=True)  # 原地删除行
    df[pp_grp_label] = 0
    df['avg_mem_grp_size'] = 0

    for i in range(layers_per_device * layers_per_device):
        df.at[i, pp_grp_label] = i // layers_per_device
        df.at[i, 'avg_mem_grp_size'] = 1000000 + i



def update_dataframe(df, df_title='Param_#', n=8, pp_grp_label='avg_mem_grp_idx'):
    nums = df[df_title].tolist()  # Updated column name here
    partitions = partition_list(nums, n)

    # Initialize columns
    df[pp_grp_label] = 0
    df['avg_mem_grp_size'] = 0
    
    # Fill the DataFrame with group assignments and sums
    group_sums = [sum(part) for part in partitions]
    current_group = 0
    current_sum = 0

    for i, num in enumerate(nums):
        if current_sum + num > group_sums[current_group]:
            current_group += 1
            current_sum = 0
        df.at[i, pp_grp_label] = current_group + 1
        df.at[i, 'avg_mem_grp_size'] = group_sums[current_group]
        current_sum += num

    return df



def find_feasible_partition_min_comm_single_core(df, n, M):
    # Step 1: Create a list of tuples (output_weight, index)
    weight_idx = list(df[['output_#', 'layer_index']].itertuples(index=False, name=None))
    weight_idx = [t for t in weight_idx if t[1] != 0]
    # Step 2: Sort the list by output_weight
    weight_idx.sort()
    # print(weight_idx)
    print("weight_idx.len =", len(weight_idx) + 1)

    k = n
    
    while True:
        print("searching in", k+1, "intervals.")
        # Step 3: Take the first k tuples where index is not 0
        k_tuples = weight_idx[:k]
        
        # Step 4: Calculate the number of elements to remove
        remove_count = k - n
        
        if remove_count <= 0:
            # If k <= n, we don't need to remove any elements
            combinations = [k_tuples]
        else:
            # Otherwise, generate all combinations of removing remove_count elements
            combinations = list(itertools.combinations(k_tuples, n))
        
        for comb in combinations:
            indices = [t[1] for t in comb]
            
            # Create partitions
            partitions = sorted([-1] + indices + [len(df)-1])
            # if k <= n + 1:
            #     print(partitions)

            feasible = True
            
            # Check sums of partitions
            for i in range(len(partitions) - 1):
                segment_sum = df.loc[partitions[i] + 1 : partitions[i+1], 'Param_#'].sum()
                if segment_sum > M:
                    feasible = False
                    break
            
            if feasible:
                # Update the original DataFrame with group information
                df['min_comm_grp_id'] = -1
                df['min_comm_grp_sum'] = 0
                df['min_comm_grp_last_weight'] = 0
                
                for i in range(len(partitions) - 1):
                    print("[", partitions[i]+1, ", ", partitions[i+1], "]")
                    df.loc[partitions[i]+1:partitions[i+1], 'min_comm_grp_id'] = i
                    grp_sum = df.loc[partitions[i]+1:partitions[i+1], 'Param_#'].sum()
                    df.loc[partitions[i]+1:partitions[i+1], 'min_comm_grp_sum'] = grp_sum
                    grp_last_weight = df.loc[partitions[i+1], 'output_#']
                    df.loc[partitions[i]+1:partitions[i+1], 'min_comm_grp_last_weight'] = grp_last_weight
                
                print(f"Feasible partition found with k={k}: {partitions}")
                # print(df)
                return df
        
        # Increment k and try again
        k += 1



# 多线程版本：
def check_feasible_partition(df, k_tuples, k, n, M, result, stop_flag):
    print("searching in", k + 1, "intervals.")
    assert len(k_tuples) == k, "len(k_tuples) != k"

    if not stop_flag.is_set():
        # Step 4: Calculate the number of elements to remove
        remove_count = k - n

        if remove_count <= 0:
            combinations = [k_tuples]
        else:
            combinations = list(itertools.combinations(k_tuples, n))

        for comb in combinations:
            indices = [t[1] for t in comb]
            partitions = sorted([-1] + indices + [len(df) - 1])

            feasible = True
            for i in range(len(partitions) - 1):
                segment_sum = df.loc[partitions[i] + 1:partitions[i + 1], 'Param_#'].sum()
                if segment_sum > M:
                    feasible = False
                    break

            if feasible:
                stop_flag.set()
                if len(result) > 0 and k < result[0][0]:
                    result.insert(0, (k, partitions))
                else:
                    result.append((k, partitions))

                print("\nget a result at k =", k)
                print(partitions)
                print("\n")
                return



def find_feasible_partition_min_comm_multi_core(df, n, M):
    start_index = 0
    end_index = 0
    while df['name'][start_index] == 'x':
        start_index += 1
    start_index += 0  # 希望往第一层参数的后面插，不要把输入和第一层参数切分开来
    
    while df['name'][end_index] != 'output':
        end_index += 1
    end_index -= 1    # 希望往最后一层参数的前面插，不要把输出和最后一层参数切分开来


    print("start_index =", start_index, "end_index =", end_index)
    
    manager = mp.Manager()
    result = manager.list()
    exist_result = list()
    stop_flag = manager.Event()
    
    # Step 1: Create a list of tuples (output_weight, layer_index)
    weight_idx = list(df[['output_#', 'layer_index']][start_index: end_index].itertuples(index=False, name=None))

    weight_idx = [t for t in weight_idx if t[1] != 0]
    print(weight_idx)
    print("\n")
    # Step 2: Sort the list by output_weight
    weight_idx.sort()
    print(weight_idx)
    max_interval = len(weight_idx)
    print("max interval =", max_interval)

    num_processes = mp.cpu_count()

    n = n - 1 # 找n-1个插空
    k = n

    k_file = 'the_promising_k.txt'
    having_k_file = False
    if os.path.exists(k_file):
        having_k_file = True
        with open(k_file, 'r') as file:
            number_str = file.read()
            print("using recorded k...(" + number_str + ')')
            exist_k = int(number_str)
        print("searching in", exist_k + 1, "intervals.")
        k_tuples = weight_idx[:exist_k]
        assert len(k_tuples) == exist_k, "len(k_tuples) != exist_k"

        # Step 4: Calculate the number of elements to remove
        remove_count = exist_k - n

        if remove_count <= 0:
            combinations = [k_tuples]
        else:
            combinations = list(itertools.combinations(k_tuples, n))

        for comb in combinations:
            indices = [t[1] for t in comb]
            partitions = sorted([-1] + indices + [len(df) - 1])

            feasible = True
            for i in range(len(partitions) - 1):
                segment_sum = df.loc[partitions[i] + 1:partitions[i + 1], 'Param_#'].sum()
                if segment_sum > M:
                    feasible = False
                    break

            if feasible:
                exist_result.append((exist_k, partitions))

                print("\nget a result at k =", exist_k)
                print(partitions)
                print("\n")
                break


    else:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_k = {}
            futures = []

            for i in range(len(weight_idx) - n):
                # Step 3: Take the first k tuples where index is not 0
                k_tuples = weight_idx[:k]
                future = executor.submit(check_feasible_partition, df, k_tuples, k, n, M, result, stop_flag)
                future_to_k[future] = k
                futures.append(future)
                k += 1

                # 检查并启动新的任务
                if len(future_to_k) >= num_processes:
                    for future in as_completed(future_to_k):
                        if future.result() is not None:
                            # 取消所有未完成的任务
                            for f in futures:
                                f.cancel()
                            # 终止所有运行中的进程
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                    future_to_k.clear()
                    futures.clear()

            # 等待所有进程完成
            for future in as_completed(future_to_k):
                future.result()

    if result or exist_result:
        if having_k_file:
            k, partitions = exist_result[0]
        else:
            k, partitions = result[0]
            with open(k_file, 'w') as f:
                f.write(str(k))
        df['min_comm_grp_id'] = -1
        df['min_comm_grp_sum'] = 0
        df['min_comm_grp_last_weight'] = 0

        for i in range(len(partitions) - 1):
            print("[", partitions[i] + 1, ", ", partitions[i + 1], "]")
            df.loc[partitions[i] + 1:partitions[i + 1], 'min_comm_grp_id'] = i
            grp_sum = df.loc[partitions[i] + 1:partitions[i + 1], 'Param_#'].sum()
            df.loc[partitions[i] + 1:partitions[i + 1], 'min_comm_grp_sum'] = grp_sum
            grp_last_weight = df.loc[partitions[i + 1], 'output_#']
            df.loc[partitions[i] + 1:partitions[i + 1], 'min_comm_grp_last_weight'] = grp_last_weight

        print(f"Feasible partition found with k={k}: {partitions}")
    else:
        print("No feasible partition found.")

    return df

