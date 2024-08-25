import pandas as pd
from collections import defaultdict

def generate_bkwd_df(fwd_df):
    # 初始化输出节点的依赖字典
    output_dependencies = defaultdict(list)
    
    # 填充输出节点的依赖字典
    for idx, row in fwd_df.iterrows():
        for arg in row['args']:
            output_dependencies[arg].append(row['name'])
    
    # 逆序 fwd_df，并保持其他列不变
    reversed_df = fwd_df.iloc[::-1].reset_index(drop=True)
    
    # 更新 reversed_df 中的依赖关系
    reversed_df['args'] = reversed_df['name'].map(output_dependencies).apply(tuple)
    
    return reversed_df


if __name__ == '__main__':
    # 设置显示选项以显示完整的 DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    # 示例数据
    data = {
        'index': list(range(25)),
        'name': ['x', 'conv1', 'bn1', 'relu', 'maxpool', 
                'layer1_0_conv1', 'layer1_0_bn1', 'layer1_0_relu', 'layer1_0_conv2', 'layer1_0_bn2', 
                'layer1_0_relu_1', 'layer1_0_conv3', 'layer1_0_bn3', 'layer1_0_downsample_0', 'layer1_0_downsample_1', 
                'add', 'layer1_0_relu_2', 'layer1_1_conv1', 'layer1_1_bn1', 'layer1_1_relu', 
                'layer1_1_conv2', 'layer1_1_bn2', 'layer1_1_relu_1', 'layer1_1_conv3', 'layer1_1_bn3'],
        'args': [(), ('x',), ('conv1',), ('bn1',), ('relu',), 
                ('maxpool',), ('layer1_0_conv1',), ('layer1_0_bn1',), ('layer1_0_relu',), ('layer1_0_conv2',), 
                ('layer1_0_bn2',), ('layer1_0_relu_1',), ('layer1_0_conv3',), ('maxpool',), ('layer1_0_downsample_0',), 
                ('layer1_0_bn3', 'layer1_0_downsample_1'), ('add',), ('layer1_0_relu_2',), ('layer1_1_conv1',), ('layer1_1_bn1',), 
                ('layer1_1_relu',), ('layer1_1_conv2',), ('layer1_1_bn2',), ('layer1_1_relu_1',), ('layer1_1_conv3',)],
        'other_col1': range(25),  # 示例其他列1
        'other_col2': range(100, 125)  # 示例其他列2
    }

    fwd_df = pd.DataFrame(data)
    print(fwd_df)
    bkwd_df = generate_bkwd_df(fwd_df)
    print(bkwd_df)
