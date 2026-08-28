# phxloader — Phoenix DMA Loader for GPU Direct Storage

正式发行包，提供基于 Phoenix GDS（GPU Direct Storage）的 safetensors 权重高速加载能力，通过 DMA 直接从 NVMe 存储传输到 GPU 显存，绕过 CPU 内存中转。

基于特定xPU厂商后端，环境变量`PHXFS_VENDOR`应设置为NVIDIA（默认值）或METAX...

**当前版本：V2.2**

## 版本演进

### V1 — 连续 DMA 同步加载

- 单次连续 DMA 读取整个 safetensors data section（`read_data_section`）
- 每个文件独立分配 GPU buffer，4K 对齐
- 同步阻塞式 I/O，读完后才 yield
- 实现位置：`phxloader_v1/`（历史版本，代码保留在 `tencent-backup` 分支）
- vllm load_format：`phxsafetensors_v1`

### V2 — Batch DMA + 共享 Buffer

- 将每个文件按 read group 拆分（相邻 tensor gap < 64K 合并），4K 对齐
- 单一共享 GPU buffer 跨所有文件复用，通过 `regmem`/`deregmem` 注册
- 每个 read group 一次 `phxfs_read`（批量 DMA），取代 V1 的整段读取
- 方法名：`read_into_registered`
- 实现位置：`phxloader_v2/`（历史版本，代码保留在 `tencent-backup` 分支）
- vllm load_format：`phxsafetensors_v2`

### V2.1 — 双缓冲 + 异步 DMA

- 双 GPU buffer（bufA/bufB）交替使用
- `read_into_registered_async` 在 C++ 后台线程执行 DMA，`wait_dma` 等待完成
- `yield(copy_)` 与下一文件的后台 DMA 重叠，隐藏 yield 延迟（约 2.5s）
- 新增 DMA 计时器（`reset_dma_timer` / `get_dma_seconds`），纯 DMA 耗时可观测
- 实现位置：`phxloader_v2/`（历史版本，与 V2 共用 C++ 实现，异步路径由 vllm 侧迭代器驱动；代码保留在 `tencent-backup` 分支）
- vllm load_format：`phxsafetensors_v2_1`

### V2.2（当前版本）— 正式发行 + API 重命名

- **功能与 V2.1 完全一致**，无性能差异
- API 重命名，提升可读性：
  - `read_into_registered` → `load_tensors_into_buffer`
  - `read_into_registered_async` → `load_tensors_into_buffer_async`
  - C++ 类名 `PhxLoaderV2` → `PhxLoader`
- safetensors header 解析加固：8 字节长度字段校验 + `header_len` 上限保护（防止超大值导致 OOM）
- 作为正式发行包 `phxloader` 发布（去除版本号后缀）
- vllm load_format：`phxsafetensors`

## 目录结构

```
phxloader/
├── phxloader/                  # Python 包
│   ├── __init__.py             # 导出 PhxLoader, parse_safetensor_header, ...
│   ├── read_group.py           # ReadGroup / FilePlan / build_read_groups / build_file_plan
│   └── safetensors_parser.py   # parse_safetensor_header（含边界校验）
├── src/
│   ├── phx_loader.h            # PhxLoader 类声明
│   ├── phx_loader.cpp          # PhxLoader 实现（DMA, regmem, async）
│   └── bindings.cpp            # pybind11 绑定
├── setup.py                    # 构建脚本（pybind11 extension）
├── install.sh                  # 便捷安装脚本
└── README.md                   # 本文件
```

## 安装

```bash
# 前置：conda activate <phoenix_env>
# 前置：Phoenix libphoenix 已编译（Phoenix/build/libphoenix.so）

cd Phoenix/adapters/vLLM/phxloader
bash install.sh
```

## 使用

在 vllm 中通过 load_format 指定：

```bash
# vllm 启动参数
--load-format phxsafetensors
```

或通过 vllm Python API：

```python
from vllm import LLM
llm = LLM(model="...", load_format="phxsafetensors")
```

## API

### PhxLoader

```python
from phxloader import PhxLoader

loader = PhxLoader(cuda_device_id=0)

# 注册 GPU buffer 用于 DMA
loader.regmem(gpu_ptr, size)

# 同步批量读取：将多个 (buf_offset, file_offset, nbytes) 条目从文件 DMA 到 GPU buffer
loader.load_tensors_into_buffer(path, gpu_ptr, batch)

# 异步批量读取：在 C++ 后台线程执行 DMA，立即返回
loader.load_tensors_into_buffer_async(path, gpu_ptr, batch)

# 等待最近的异步 DMA 完成
loader.wait_dma()

# DMA 计时
loader.reset_dma_timer()
loader.get_dma_seconds()  # → float (秒)

# 注销 GPU buffer
loader.deregmem(gpu_ptr, size)

# 关闭设备（析构时自动调用）
loader.close()
```

### 辅助函数

```python
from phxloader import (
    parse_safetensor_header,
    build_read_groups,
    build_file_plan,
)
```

## 依赖

- **libphoenix**：Phoenix C API（`phxfs_open`/`phxfs_read`/`phxfs_regmem`/`phxfs_deregmem`）
- **liburing**：Linux io_uring
- **CUDA**：GPU 运行时
- **pybind11**：Python C++ 扩展绑定
- **PyTorch**：tensor 操作（Python 侧）
