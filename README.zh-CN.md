# FiRE（公开整理版）

这是基于你提供的原始脚本整理出来的一版**更适合开源发布**的 FiRE 代码骨架。论文对应工作是 **FiRE: Enhancing MLLMs with Fine-Grained Context Learning for Complex Image Retrieval**，已发表于 SIGIR 2025。论文公开摘要里强调了两个方向：面向复杂图像检索的细粒度上下文建模，以及分阶段的细粒度微调策略。

这份整理版的目标不是“逐行保留作者本地环境”，而是把你给的代码改成下面这种更适合开源的状态：
- **无硬编码绝对路径**
- **超参数全部配置化**
- **训练 / 评测入口分离**
- **默认只保留公平评测路径**
- **尽量保留原方法里的核心损失与多模态编码思路**

---

## 目录结构

```text
fire_opensource_clean/
├── README.md
├── README.zh-CN.md
├── requirements.txt
├── configs/
│   ├── train_stage2.example.yaml
│   └── eval.example.yaml
├── docs/
│   └── cleanup_notes.md
├── scripts/
│   ├── train.py
│   └── eval.py
└── src/fire_open/
    ├── __init__.py
    ├── config.py
    ├── datasets.py
    ├── losses.py
    ├── modeling.py
    └── trainer.py
```

---

## 这版做了什么

### 1) 路径全部改成配置项
原始脚本里有大量类似 `/home/share/...` 的本地路径。现在统一由 YAML 控制，例如：

```yaml
data:
  image_root: ./data/images
  train_metadata: ./data/annotations/fire_train.jsonl
```

### 2) 超参数全部改成可配置
例如：
- LoRA 的 `r / alpha / dropout`
- `learning_rate`
- `batch_size`
- `num_train_epochs`
- `warmup_steps`
- loss 权重

都在 `configs/*.yaml` 中显式给出。

### 3) 默认只保留公平评测
原始代码中存在一些只适合内部实验、不适合作为公开默认评测入口的分支，例如：
- 私有中间 json
- 私有 caption 补充文件
- 私有预计算 vision token 缓存
- 多种含义不清的内部 mode

公开版默认评测只走：
- **query** = reference image + modification text
- **gallery** = candidate image
- **metric** = Recall@K

这样更容易复现，也更不容易引入测试阶段额外信息。

---

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 训练数据格式（推荐 jsonl）

训练集推荐使用 `jsonl`，每行一个样本：

```json
{
  "sample_id": "000001",
  "reference_image": "train/ref/0001.jpg",
  "target_image": "train/tgt/0001.jpg",
  "reference_id": "ref_0001",
  "target_id": "tgt_0001",
  "modification": "change the red shirt into a blue striped shirt",
  "reference_caption": "a person wearing a plain red shirt",
  "target_caption": "a person wearing a blue striped shirt"
}
```

其中：
- `reference_image` / `target_image` / `modification` 为必需字段
- `reference_caption` / `target_caption` 为可选字段
- 如果你没有 caption，可以先保留为空字符串，但训练 prompt 的信息量会下降

---

## 公平评测数据格式（自定义）

### query metadata（jsonl）

```json
{
  "query_id": "q1",
  "reference_image": "eval/ref/001.jpg",
  "reference_id": "img_001",
  "modification": "make the bag black and remove the logo",
  "target_id": "img_128",
  "exclude_ids": ["img_001"]
}
```

### gallery metadata（jsonl）

```json
{
  "image_id": "img_128",
  "image_path": "eval/gallery/128.jpg"
}
```

默认评测**不会**读取 `target_caption` 之类的额外字段。

---

## 运行训练

先修改：`configs/train_stage2.example.yaml`

然后执行：

```bash
python scripts/train.py --config configs/train_stage2.example.yaml
```

---

## 运行评测

先修改：`configs/eval.example.yaml`

然后执行：

```bash
python scripts/eval.py --config configs/eval.example.yaml
```

输出示例：

```json
{
  "Recall@1": 0.23,
  "Recall@5": 0.51,
  "Recall@10": 0.64,
  "Recall@50": 0.88
}
```

---

## 对 FashionIQ / CIRR 的支持

这版代码带了两个公开 benchmark 读取器：
- `FashionIQEvalDataset`
- `CIRREvalDataset`

### FashionIQ
把 `data.image_root` 指到 `fashion_iq_data` 根目录，并设置：

```yaml
data:
  task: fashioniq
  image_root: ./data/fashion_iq_data
  split: val
  dress_type: dress
```

### CIRR
把 `data.image_root` 指到 `CIRR` 根目录，并设置：

```yaml
data:
  task: cirr
  image_root: ./data/CIRR
  split: val
```

---

## 和原始代码相比，哪些东西没有保留为默认公开实现

这部分是有意为之：

1. **不默认加载作者私有 checkpoint 路径**
2. **不默认依赖私有 `hbh_*` 标注文件**
3. **不默认依赖本地缓存的 vision token `.pt`**
4. **不把 `case / pre_vision / classic` 之类内部模式直接暴露成公开默认评测入口**
5. **不把测试阶段额外文本信息作为默认输入**

这些调整的目标只有一个：让别人拿到仓库时，不需要复刻作者机器目录，也不会无意间跑到“内部实验分支”。

---

## 已知说明

1. 这版以**公开复现友好**为优先，不是对原始私有工程的 1:1 镜像。
2. 原始工程里有不少和私有中间数据耦合的逻辑；现在统一改成了“显式 metadata + 显式 config”。
3. 如果你后面想继续补：
   - 多卡/DDP
   - DeepSpeed
   - 预计算 vision token 加速
   - 更多 benchmark reader
   都可以在这个结构上继续加。
