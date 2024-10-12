# llmsim

## 基本情况

这个repo的代码实现了给定一个模型，生成NS3仿真用的依赖文件 （节点和边），需要搭配 llmsim-sim 一起使用

当前调试OK的神经网络是resnet。你可以需要根据自己的网路添加一些内容，如：
- 不同的算子的计算复杂度
- 不同的host/GPU/NIC的capability

## 运行

```
python3 get_fx_graph.py --pp_mode F_then_B --pp_granularity device --dp_granularity device --pp_flow_bucket_size 10000000 -d --plot_seed 11 > graph.txt
```

## 对于 LLM, 我们没有使用运算图，而是重新对LLM训练过程中的DP PP TP 进行了建模。运行方法为：
```
python3 build_tf.py --pp_cut 0 --global_batch 2 --micro_batch 1 --num_of_layers 2 --enable_ar true
```

最终生成的输出在 `mix` ，供 `llmsim-sim` 使用。用法请见对应的repo。

对于 `llm_flow.txt` 中的每一条流的 `src` 和 `dst`，你可以通过控制下面的参数来决定打印的是vnode还是node.  
```
print_vnode = False
```

# Reference
Resnet的代码来自这里：
[https://github.com/wangyunjeff/ResNet50-MNIST-pytorch](https://github.com/wangyunjeff/ResNet50-MNIST-pytorch)
