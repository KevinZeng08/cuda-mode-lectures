## Notes for Lecture 1: Profiling CUDA kernels in PyTorch

### Key Points

- Pytorch Profiler

- Triton Profiler

- CUDA Profiling with ncu

### Takeaway

- 在pytorch，CUDA调用计时使用`torch.cuda.Event()`和`torch.cuda.synchronize()`，因为CUDA调用是async（异步）的

- torch.cpp_extensions的`load_inline()`可以方便地将cpp cuda代码自动编译和执行，内部创建了main函数和cuda文件，使用pybind绑定。输出见`build_directory`所指定的目录

- TODO: Triton的PTX code，输入输出都默认8个register（为什么？）

- TODO: What is "Long Scoreboard Stalls" and "Tail Effect"?
    
    - Tail Effect: Try padding

    - Long scoreboard stalls: always caused by memory bandwidth or latency with GPU. Try to use shared memory

- Triton and CUDA

![Triton versus CUDA](imgs/triton_vs_cuda.png)


- Triton Kernel示例 copilot生成注释：`torch.compile(torch.square(torch.square(a)))`. 可看出Triton是block-level的编程，先拿到block_id再算thread_id

```python
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # 定义一个Triton kernel函数，使用@triton.jit装饰器进行JIT编译。
    # 参数：
    # in_ptr0: 输入数据的指针
    # out_ptr0: 输出数据的指针
    # xnumel: 数据元素的数量
    # XBLOCK: 每个线程块处理的元素数量（编译时常量）

    xnumel = 100000000
    # 将xnumel设置为100000000（覆盖传入的参数值）

    xoffset = tl.program_id(0) * XBLOCK
    # 计算当前线程块的起始索引
    # tl.program_id(0) 返回当前线程块的ID

    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    # 计算当前线程块中每个线程处理的全局索引
    # tl.arange(0, XBLOCK) 生成一个从0到XBLOCK-1的数组

    xmask = xindex < xnumel
    # 创建一个掩码，标记哪些索引在有效范围内（小于xnumel）

    x0 = xindex
    # 将xindex赋值给x0

    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    # 从输入指针加载数据，使用掩码确保只加载有效范围内的数据

    tmp1 = tmp0 * tmp0
    # 计算tmp0的平方

    tmp2 = tmp1 * tmp1
    # 计算tmp1的平方（即tmp0的四次方）

    tl.store(out_ptr0 + (x0), tmp2, xmask)
    # 将结果存储到输出指针，使用掩码确保只存储有效范围内的数据
```