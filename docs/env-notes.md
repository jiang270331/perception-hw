# 环境管理学习记录

> 任务要求了解：为什么不同项目要独立环境、版本不一致会有什么问题、conda / uv / venv 的区别。
> 下面是查资料 + 自己动手后的整理，以及在本机上的实测记录。

---

## 一、为什么不同项目要用独立环境

**一句话：依赖是会「打架」的。**

### 1. 同一个包的不同版本不能共存

```
项目 A 需要 numpy 1.24（旧代码用了已废弃的 np.float）
项目 B 需要 numpy 2.0（新代码用了新的类型系统）
    ↓
全局环境只能装一个版本 → 另一个必然报错
```

**本机实测**：

```
D:\world-model-lab\env 里装的是 numpy 2.5.4
如果这个项目要用某个只支持 numpy 1.x 的库（比如旧版 opencv），
直接装会把 numpy 降级 → world-model-lab 的代码可能立刻报错
```

### 2. 传递依赖（依赖的依赖）也会冲突

**不只是一个包的问题。** 比如：

```
你装 ultralytics        → 它要 torch >= 2.0
你装某个旧版本库         → 它要 torch == 1.13
    ↓
pip 会「尽力而为」，最后装成一个谁都不满意的版本
```

### 3. 环境能被完整复现

```
项目目录里放 environment.yml 或 requirements.txt
    ↓
别人 clone 下来执行一行命令 → 得到和你一模一样的环境
    ↓
★ 这是「别人能跑起来」的前提
```

### 4. 出问题了能一键重置

```
环境坏了（装错包、版本乱、依赖树坏了）
    ↓
rm -rf .venv && 重新建一个    ← 30 秒
    ↓
★ 全局环境坏了 → 可能要重装整个 Python
```

### 5. 系统 Python 不能乱动

**Linux 上尤其重要：** 系统自带的 Python 是给系统工具（apt、桌面环境）用的，装了包可能把系统搞崩。

**Windows 上 Anaconda base 环境也是同理** —— base 里塞满项目包，以后很难维护。

---

## 二、版本不一致会带来什么问题

**按「报错方式」从明显到隐蔽排：**

### 1. 直接 ImportError / AttributeError（最好处理）

```python
# numpy 1.x 时代可以这样
import numpy as np
x = np.float(3.14)      # numpy 2.0 里已经删掉了

# numpy 2.0 下：
AttributeError: module 'numpy' has no attribute 'float'
```

**★ 这种一跑就报错，最好排查。**

### 2. 行为变了但不报错（最麻烦）

```python
# Python 3.12 之前
assert 0.1 + 0.2 == 0.3      # False

# 某个库的内部实现从「返回 list」变成「返回 generator」
for x in func():  ...        # 仍然能跑
len(func())                  # TypeError: object of type 'generator' has no len()
```

**★★ 这种「部分功能悄悄坏了」最难查。**

### 3. 精度/性能变化（几乎查不出来）

```
torch 2.0 → 2.1：同样的代码，训练结果可能不同（kernel 实现变了）
CUDA 版本不匹配：能在 CPU 上跑，GPU 上静默回退，速度慢 10 倍
```

**本机实测**：`torch 2.14.1+cu130` —— 如果某个库要求 `torch<2.0`，它会强制降级，world-model-lab 的 CUDA 训练就跑不了了。

### 4. 二进制不兼容（最硬）

```
numpy 用 C 扩展写的，ABI 会变
    ↓
用 numpy 1.x 编译的包，在 numpy 2.x 下 import 直接崩（Segmentation fault）
    ↓
★ 这种连 Python 异常都没有，进程直接退出
```

---

## 三、Conda / uv / venv 的特点和取舍

| | **venv** | **conda** | **uv** |
| --- | --- | --- | --- |
| 装什么 | 只隔离 Python 包 | 包 **+ 非 Python 依赖**（CUDA、MKL、编译器） | 只隔离 Python 包 |
| Python 版本 | 用系统已有的 | **可以装不同 Python 版本** | 可以装不同 Python 版本 |
| 速度 | 慢（pip 解析依赖） | 慢（尤其首次解析） | **极快**（Rust 写的） |
| 磁盘 | 小 | **大**（每个环境复制一份 Python + 科学计算库） | 小（硬链接复用） |
| 跨平台 | 好 | 好（但有平台差异） | 好 |
| 生态 | PyPI | conda-forge + PyPI | PyPI |
| 适合场景 | 简单项目、CI | **要 CUDA / 非 Python 依赖、要切 Python 版本** | 日常开发、依赖重的项目 |

### 具体点

**venv（Python 自带）**

```
优点：零依赖，Python 自带，最轻
缺点：
  · 只能装 PyPI 上的包，装不了「非 Python 的东西」
  · 不能换 Python 版本（用系统的）
  · 依赖解析慢
适合：小脚本、CI、不想装额外工具
```

**conda**

```
优点：
  · 能装 CUDA toolkit、MKL、ffmpeg 这些「不是 Python 包」的东西
  · 能装任意 Python 版本（conda create -n x python=3.9）
  · 科学计算库有优化过的二进制（MKL 加速的 numpy）
缺点：
  · 慢（求解依赖很慢，尤其第一次）
  · 占磁盘（一个环境 2-5 GB）
  · 包版本有时比 PyPI 旧
  · base 环境容易被污染
适合：要用 GPU、要切 Python 版本、科学计算
★ 本项目的做法：用 conda 建环境（任务要求），但包从 PyPI 装（更新更快）
```

**uv（Astral 出的，用 Rust 写的）**

```
优点：
  · 快 10-100 倍（依赖解析 + 下载 + 安装）
  · 能用硬链接复用包，磁盘占用小
  · 兼容 pip / requirements.txt / pyproject.toml
  · 能管 Python 版本
缺点：
  · 比较新（2024 才有），生态还在成熟
  · 不能装「非 Python 依赖」（CUDA 这些还是得 conda）
适合：日常开发；如果项目不涉及 CUDA，uv 是现在最舒服的选择
```

### 一句话总结

```
简单脚本          → venv
要 GPU / 非 Python 依赖  → conda
日常开发、追求速度      → uv
★ 三者不互斥：可以用 conda 管 Python 版本和 CUDA，用 uv 装 Python 包
```

---

## 四、本机的实际操作记录

### 4.1 检查 conda

```powershell
C:\Users\Jiang\anaconda3\Scripts\conda.exe --version
# conda 26.1.1
```

**注意**：`conda` 不在 PATH 里，要用全路径调用（或者在 Anaconda Prompt 里用）。

**要加到 PATH 的话**：

```
系统属性 → 环境变量 → Path → 新建：
  C:\Users\Jiang\anaconda3
  C:\Users\Jiang\anaconda3\Scripts
  C:\Users\Jiang\anaconda3\condabin
```

### 4.2 现有环境

```powershell
conda env list

# conda environments:
# base                     C:\Users\Jiang\anaconda3
#                          D:\world-model-lab\env
```

**★ 发现**：`D:\world-model-lab\env` 是一个**非标准位置**的 conda 环境（不是 `envs/` 目录下），是之前建世界模型项目时用 `conda create -p` 建的。

### 4.3 本次任务用的环境

**决定：复用 `D:\world-model-lab\env`，只补装缺的两个包。**

**理由**：

```
这个环境已有 torch 2.14.1+cu130（2.5 GB）
新建环境要重装 torch → 下载 2.5 GB，纯浪费时间
    ↓
任务原文也说「已有适用环境的同学可继续使用」
    ↓
★ 但要注意：这违反「一个项目一个环境」的原则
★★ 正确的做法见 README 第 3 节（新建 perception 环境）
```

**实际做的**：

```powershell
D:\world-model-lab\env\python.exe -m pip install ultralytics opencv-python
# 装了 ultralytics 8.4.197、opencv-python 5.0.0
# 下载时 polars_runtime 反复超时（53.9 MB），重试 5 次后成功
```

### 4.4 环境常用命令速查

```powershell
# 创建
conda create -n <名字> python=3.12

# 激活 / 退出
conda activate <名字>
conda deactivate

# 查看
conda env list                    # 所有环境
conda list                        # 当前环境所有包
conda list -n <名字>              # 指定环境

# 安装
conda install numpy               # 从 conda-forge 装（慢，但能装非 Python 包）
pip install numpy                 # 从 PyPI 装（快，但只能装 Python 包）

# 导出（三种粒度）
conda env export > environment.yml          # 完整（含 build 号，跨平台可能装不上）
conda env export --no-builds > environment.yml   # 去掉 build 号，跨平台好些
conda list --export > requirements.txt      # 只列包名+版本

# 从文件重建
conda env create -f environment.yml
pip install -r requirements.txt

# 删除
conda remove -n <名字> --all
conda env remove -n <名字>
```

---

## 五、我踩过的坑

### 坑 1：conda 不在 PATH

```
直接敲 conda → 「不是内部或外部命令」
    ↓
解决：用全路径，或者开 Anaconda Prompt，或者手动加 PATH
```

### 坑 2：pip 和 conda 混用同一个环境

```
conda install 装了一个包
再用 pip install 装了同名包的另一个版本
    ↓
★ 两个包管理器互相不知道对方干了什么 → 环境状态混乱
    ↓
原则：一个环境里尽量只用一种方式装包
（本项目全用 pip，conda 只用来管 Python 版本）
```

### 坑 3：下载依赖时反复超时

```
pip 下 polars_runtime（53.9 MB）时连续 5 次 ReadTimeout
    ↓
pip 会自动断点续传（"Attempting to resume incomplete download"）
    ↓
最终成功 —— 所以遇到超时先别急着重装，先让它重试
```

### 坑 4：环境建在项目目录里，被 git 追踪

```
venv/ 或者 env/ 目录如果没写进 .gitignore
    ↓
git add . 会把几万个文件加进去 → 仓库爆炸
    ↓
解决：.gitignore 里写 env/ 和 .venv/（本项目的 .gitignore 已经写了）
```

---

## 六、还没搞清楚的

1. **conda 的 libmamba 求解器**：听说比默认快很多，但没实测对比
2. **uv 能不能完全替代 conda**：如果只是装 PyPI 包、不需要 CUDA，应该可以；但要装 CUDA toolkit 就不行
3. **environment.yml 跨平台可移植性**：`--no-builds` 之后在 Linux 上能不能直接 create 没试过
