# RT-DETR CUDA使用现状与优化方案

## 📊 当前CUDA使用情况分析

### ✅ 已使用CUDA的部分

#### 1. **推理部分** - ✅ 已使用CUDA

**代码位置**: `inferenceTool/predict_batch_torchscript.py`

```python
def resolve_device(device_str: str) -> torch.device:
    """自动检测并返回设备"""
    if device_str == "auto":
        if torch.cuda.is_available():
            print("⚡ 检测到可用 GPU，使用 CUDA 进行推理")
            return torch.device("cuda")
    return torch.device("cpu")

# 模型加载到GPU
model = torch.jit.load(model_path, map_location=device)  # device = "cuda"

# 图像数据传输到GPU
image_tensor = transforms(image_pil).unsqueeze(0).to(device)
orig_sizes = torch.tensor([[orig_w, orig_h]], device=device)
```

**配置**:
```yaml
model:
  device: "auto"  # 自动检测CUDA，优先使用GPU
```

**使用情况**:
- ✅ TorchScript模型运行在GPU上
- ✅ 输入数据自动传输到GPU
- ✅ 推理计算在GPU上进行
- ✅ 结果自动传回CPU

---

### ❌ 未使用CUDA的部分

#### 1. **切片部分** - ❌ 纯CPU实现

**代码位置**: `auto_process_package/auto_process_monitor.py` → `slice_image()`

```python
def slice_image(self, image_path: Path, output_dir: Path):
    """切片图像 - 使用PIL（CPU）"""
    from PIL import Image, ImageFile
    
    pil_img = Image.open(image_path)  # CPU加载
    
    for y_idx in range(num_patches_y):
        for x_idx in range(num_patches_x):
            patch_pil = pil_img.crop(patch_box)  # CPU裁剪
            patch_pil.save(save_path, 'PNG')  # CPU保存
```

**性能瓶颈**:
- 🐌 大图像（如40X全扫）加载到内存慢
- 🐌 逐个patch裁剪和保存慢
- 🐌 I/O操作占用主要时间

#### 2. **过滤部分** - ❌ 纯CPU实现

**代码位置**: `auto_process_package/auto_process_monitor.py` → `filter_patches()`

```python
def filter_patches(self, src_dir: Path, keep_dir: Path, trash_dir: Path):
    """过滤空白patch - 使用NumPy（CPU）"""
    def is_blank(img: Image.Image):
        arr = np.asarray(img)  # CPU数组
        
        # CPU计算
        diff = np.abs(arr_rgb.astype(np.int16) - bg).sum(axis=2)
        ratio = bg_mask.mean()
        std = float(gray.std())
        
        return (ratio >= self.bg_ratio) and (std <= self.std_thresh)
    
    # 多线程CPU并行（不是GPU）
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_one, p) for p in files]
```

**性能瓶颈**:
- 🐌 图像加载和像素计算在CPU
- 🐌 8个线程并行，但无法利用GPU的数千个核心
- 🐌 大量小文件I/O操作

---

## 📈 性能瓶颈分析

### 完整处理流程的时间分布

以一个 **50GB, 100000x80000像素** 的40X全扫图像为例：

| 步骤 | 当前实现 | 耗时 | CPU/GPU | 瓶颈 |
|------|----------|------|---------|------|
| **1. 切片** | PIL (CPU) | ~180秒 (3分钟) | CPU | ❌ I/O + CPU裁剪 |
| **2. 过滤** | NumPy + 8线程 (CPU) | ~120秒 (2分钟) | CPU | ❌ CPU计算 + I/O |
| **3. 推理** | PyTorch (GPU) | ~60秒 (1分钟) | GPU | ✅ 已优化 |
| **总计** | | ~360秒 (6分钟) | | |

### 性能占比

```
切片: 50% (180/360秒)  ← 最大瓶颈！
过滤: 33% (120/360秒)  ← 第二瓶颈
推理: 17% (60/360秒)   ← 已优化
```

---

## 🚀 优化方案

### 方案1: 切片加速（推荐）⭐⭐⭐⭐⭐

#### 优化策略
使用 **OpenSlide + 分块加载** 替代PIL全图加载

**当前问题**:
```python
pil_img = Image.open(image_path)  # 一次性加载50GB到内存！
```

**优化方案**:
```python
import openslide

def slice_image_optimized(self, image_path: Path, output_dir: Path):
    """使用OpenSlide优化的切片方法"""
    # OpenSlide支持按需读取，不加载全图
    slide = openslide.OpenSlide(str(image_path))
    
    for y_idx in range(num_patches_y):
        for x_idx in range(num_patches_x):
            # 只读取需要的区域（零拷贝）
            patch = slide.read_region(
                (x, y),           # 位置
                0,                # 层级（0=最高分辨率）
                (patch_size, patch_size)  # 尺寸
            )
            patch.save(save_path, 'PNG')
```

**预期提升**:
- 内存使用: 50GB → 640×640×3 = 1.2MB
- 切片速度: **180秒 → 60秒** (提升3倍)
- 支持格式: `.svs`, `.ndpi`, `.tif` 等医学图像格式

---

### 方案2: 过滤加速（推荐）⭐⭐⭐⭐

#### 2.1 使用CuPy替代NumPy

**当前实现** (CPU):
```python
import numpy as np

arr = np.asarray(img)  # CPU
diff = np.abs(arr.astype(np.int16) - bg).sum(axis=2)  # CPU计算
ratio = bg_mask.mean()  # CPU
```

**优化实现** (GPU):
```python
import cupy as cp

arr = cp.asarray(np.array(img))  # 传输到GPU
diff = cp.abs(arr.astype(cp.int16) - bg).sum(axis=2)  # GPU计算
ratio = float(bg_mask.mean())  # GPU计算，结果传回CPU
```

**预期提升**:
- 计算速度: **120秒 → 20秒** (提升6倍)
- 并行度: 8线程 → GPU数千核心

#### 2.2 批量加载到GPU

```python
def filter_patches_gpu(self, src_dir: Path, keep_dir: Path, trash_dir: Path):
    """GPU加速的批量过滤"""
    import cupy as cp
    
    # 批量加载（如32张图）
    batch_size = 32
    for batch in tqdm(batched(files, batch_size)):
        # 一次性加载多张图到GPU
        images_gpu = []
        for img_path in batch:
            img = Image.open(img_path).convert('RGB')
            img_gpu = cp.asarray(np.array(img))
            images_gpu.append(img_gpu)
        
        # GPU批量计算
        results = self._batch_is_blank_gpu(images_gpu)
        
        # 批量移动文件
        for img_path, is_blank in zip(batch, results):
            dest = trash_dir if is_blank else keep_dir
            shutil.copy2(img_path, dest / img_path.name)
```

---

### 方案3: 推理进一步加速⭐⭐⭐

#### 3.1 批量推理（当前是单张）

**当前实现**:
```python
# 逐张处理
for patch_path in patch_files:
    image_tensor = prepare_image(patch_path, device)  # (1, 3, 640, 640)
    outputs = model(image_tensor, orig_sizes)  # 单张推理
```

**优化实现**:
```python
# 批量处理
batch_size = 16
for batch_paths in batched(patch_files, batch_size):
    # 批量加载
    images = []
    for path in batch_paths:
        img_tensor = prepare_image(path, device)
        images.append(img_tensor)
    
    # 批量推理（一次处理16张）
    batch_tensor = torch.cat(images, dim=0)  # (16, 3, 640, 640)
    outputs = model(batch_tensor, batch_orig_sizes)  # 批量推理
```

**预期提升**:
- 推理速度: **60秒 → 20秒** (提升3倍)
- GPU利用率: 30% → 80%

#### 3.2 混合精度推理（FP16）

```python
# 使用半精度（FP16）加速
model = model.half()  # 转换为FP16
image_tensor = image_tensor.half()  # 输入也转FP16

# 推理速度提升约40%，显存占用减半
```

#### 3.3 TensorRT优化

将TorchScript模型转换为TensorRT：
```bash
# 转换为TensorRT（需要NVIDIA GPU）
trtexec --onnx=rtdetr.onnx --saveEngine=rtdetr.trt --fp16
```

**预期提升**:
- 推理速度: **60秒 → 15秒** (提升4倍)

---

### 方案4: 并行流水线⭐⭐⭐⭐⭐

#### 4.1 三阶段流水线

当前是串行：
```
切片 → 过滤 → 推理
```

优化为流水线：
```
Image1: [切片] → [过滤] → [推理]
Image2:          [切片] → [过滤] → [推理]
Image3:                   [切片] → [过滤] → [推理]
```

**实现**:
```python
import queue
import threading

# 创建三个队列
slice_queue = queue.Queue()
filter_queue = queue.Queue()
inference_queue = queue.Queue()

# 三个独立线程
def slice_worker():
    while True:
        img_path = slice_queue.get()
        patches = slice_image(img_path)
        filter_queue.put(patches)

def filter_worker():
    while True:
        patches = filter_queue.get()
        kept_patches = filter_patches(patches)
        inference_queue.put(kept_patches)

def inference_worker():
    while True:
        patches = inference_queue.get()
        results = batch_predict(patches)
        save_results(results)
```

**预期提升**:
- 3个图像总时间: 1080秒 → 360秒 (提升3倍)

---

## 💰 成本效益分析

### 优化方案对比

| 方案 | 实现难度 | 性能提升 | 需要依赖 | 推荐度 |
|------|----------|----------|----------|--------|
| **方案1: OpenSlide切片** | ⭐⭐ 简单 | **3倍** | `pip install openslide-python` | ⭐⭐⭐⭐⭐ |
| **方案2.1: CuPy过滤** | ⭐⭐⭐ 中等 | **6倍** | `pip install cupy-cuda11x` | ⭐⭐⭐⭐ |
| **方案2.2: 批量GPU过滤** | ⭐⭐⭐⭐ 较难 | **8倍** | CuPy | ⭐⭐⭐ |
| **方案3.1: 批量推理** | ⭐⭐ 简单 | **3倍** | 无（修改代码） | ⭐⭐⭐⭐⭐ |
| **方案3.2: FP16推理** | ⭐ 很简单 | **1.4倍** | 无（一行代码） | ⭐⭐⭐⭐ |
| **方案3.3: TensorRT** | ⭐⭐⭐⭐⭐ 很难 | **4倍** | TensorRT | ⭐⭐ |
| **方案4: 流水线** | ⭐⭐⭐⭐ 较难 | **3倍** | 无（架构改造） | ⭐⭐⭐ |

---

## 🎯 推荐实施方案（分阶段）

### 阶段1: 快速优化（1-2小时）⭐⭐⭐⭐⭐

**目标**: 提升50%性能，最小代码改动

**实施内容**:
1. ✅ OpenSlide替代PIL切片（30分钟）
2. ✅ 批量推理替代单张推理（30分钟）
3. ✅ FP16混合精度（5分钟）

**预期效果**:
```
优化前: 360秒
优化后: 180秒
提升: 100% (快一倍)
```

**代码改动量**: 约50行

---

### 阶段2: 深度优化（1-2天）⭐⭐⭐⭐

**目标**: 提升200%性能

**实施内容**:
1. ✅ CuPy替代NumPy过滤（4小时）
2. ✅ 批量GPU过滤（4小时）
3. ✅ 三阶段流水线（8小时）

**预期效果**:
```
优化前: 360秒
优化后: 90秒
提升: 300% (快4倍)
```

**代码改动量**: 约200行

---

### 阶段3: 极限优化（1-2周）⭐⭐⭐

**目标**: 提升500%性能（适合生产环境）

**实施内容**:
1. ✅ TensorRT模型转换
2. ✅ CUDA C++自定义算子
3. ✅ 异步I/O + GPU流水线
4. ✅ 分布式多GPU推理

**预期效果**:
```
优化前: 360秒
优化后: 60秒
提升: 600% (快6倍)
```

**代码改动量**: 约1000行 + 部分C++代码

---

## 📝 快速实施指南（阶段1）

### 1. 安装依赖

```bash
# OpenSlide（切片加速）
pip install openslide-python

# 验证CUDA可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 修改切片函数

**文件**: `auto_process_monitor.py`

```python
def slice_image_optimized(self, image_path: Path, output_dir: Path):
    """使用OpenSlide优化的切片方法"""
    import openslide
    from tqdm import tqdm
    import csv
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # OpenSlide打开（支持.svs, .ndpi, .tif等）
    try:
        slide = openslide.OpenSlide(str(image_path))
    except:
        # 降级到PIL
        return self.slice_image(image_path, output_dir)
    
    width, height = slide.dimensions
    self._log(f"图像尺寸: {width} x {height} 像素")
    
    num_patches_x = width // self.patch_size
    num_patches_y = height // self.patch_size
    total_patches = num_patches_x * num_patches_y
    self._log(f"将切分为 {num_patches_x} x {num_patches_y} = {total_patches} 个 patch")
    
    x_r, y_r = [], []
    patch_count = 0
    
    self._log("开始切分图像...")
    for y_idx in tqdm(range(num_patches_y), desc="处理行", unit="行"):
        if self.stop_event.is_set():
            break
            
        for x_idx in range(num_patches_x):
            if self.stop_event.is_set():
                break
                
            x = x_idx * self.patch_size
            y = y_idx * self.patch_size
            
            x_r.append(x)
            y_r.append(y)
            
            # 零拷贝读取patch（只读取需要的区域）
            patch_pil = slide.read_region(
                (x, y), 0, (self.patch_size, self.patch_size)
            ).convert('RGB')
            
            # 保存
            save_path = output_dir / f'patch_{patch_count}.png'
            patch_pil.save(save_path, 'PNG')
            patch_count += 1
    
    # 保存坐标信息
    csv_path = output_dir / 'patch_coordinates.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'x_start', 'y_start', 'x_end', 'y_end'])
        for i in range(patch_count):
            filename = f'patch_{i}.png'
            writer.writerow([filename, x_r[i], y_r[i], 
                           x_r[i] + self.patch_size, y_r[i] + self.patch_size])
    
    self._log(f"\n✓ 完成！共保存 {patch_count} 个 patch")
    self._log(f"✓ 坐标信息已保存到: {csv_path}")
```

### 3. 修改推理为批量模式

**文件**: `inferenceTool/predict_batch_torchscript.py`

在 `main()` 函数中添加批量处理：

```python
def main():
    args = parse_args()
    
    # ... 现有代码 ...
    
    # 批量推理参数
    BATCH_SIZE = 16  # 根据GPU显存调整（8/16/32）
    
    # 批量处理
    for i in range(0, len(patch_files), BATCH_SIZE):
        batch_files = patch_files[i:i+BATCH_SIZE]
        
        # 批量加载
        images = []
        orig_sizes_list = []
        for patch_path in batch_files:
            _, img_tensor, orig_sz = prepare_image(str(patch_path), device)
            images.append(img_tensor)
            orig_sizes_list.append(orig_sz)
        
        # 拼接为批量
        batch_images = torch.cat(images, dim=0)
        batch_orig_sizes = torch.cat(orig_sizes_list, dim=0)
        
        # 批量推理
        with torch.no_grad():
            outputs = model(batch_images, batch_orig_sizes)
        
        # 处理批量结果
        # ... 后处理代码 ...
```

### 4. 启用FP16

在模型加载后添加：

```python
# 启用混合精度
if device.type == 'cuda':
    model = model.half()  # 转FP16
    print("✓ 已启用 FP16 混合精度")
```

---

## 🧪 性能测试

### 测试脚本

```bash
# 测试优化前
time python service_main.py

# 测试优化后
time python service_main.py
```

### 监控GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用
pip install gpustat
gpustat -i 1
```

---

## 📊 预期性能提升总结

| 阶段 | 优化内容 | 耗时 | 总提升 |
|------|----------|------|--------|
| **当前** | 无优化 | 360秒 | - |
| **阶段1** | OpenSlide + 批量推理 + FP16 | 180秒 | **2倍** |
| **阶段2** | + CuPy过滤 + 流水线 | 90秒 | **4倍** |
| **阶段3** | + TensorRT + 多GPU | 60秒 | **6倍** |

---

## 💡 注意事项

1. **GPU显存**: 批量推理需要更多显存
   - 16GB显存: batch_size=16-32
   - 8GB显存: batch_size=8-16
   - 4GB显存: batch_size=4-8

2. **OpenSlide支持格式**:
   - ✅ .svs, .tif, .ndpi, .vms, .vmu, .scn, .mrxs, .bif
   - ❌ 普通.jpeg/.png 需要降级到PIL

3. **CuPy安装**: 
   ```bash
   # CUDA 11.x
   pip install cupy-cuda11x
   
   # CUDA 12.x
   pip install cupy-cuda12x
   ```

---

**文档创建日期**: 2025年12月10日  
**优化建议**: 先实施阶段1（2小时，提升100%），效果明显再考虑后续阶段

