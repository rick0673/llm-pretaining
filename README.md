# LLM Pretraining

这是一个从零实现的小型语言模型预训练项目，目标是完整跑通从文本语料、BPE 分词器、数据预处理、Transformer 语言模型训练到交互式推理的基础流程。项目重点不依赖高级封装，而是手写核心模块，便于理解大语言模型预训练中的关键组件和训练细节。

## 项目亮点

- 实现 Byte-level BPE 分词器，支持特殊 token 保留、编码、解码和流式编码。
- 实现 Decoder-only Transformer 语言模型，包括 Embedding、RMSNorm、RoPE、Causal Self-Attention、SwiGLU 和 LM Head。
- 手写 AdamW 优化器、梯度裁剪、交叉熵损失和 warmup + cosine decay 学习率调度。
- 支持使用 `np.memmap` 读取大规模 token 数据，降低训练时的数据加载内存压力。
- 支持 checkpoint 保存与恢复，便于中断后继续训练。
- 提供训练脚本和交互式推理脚本，支持 temperature、top-p 采样。
- 预留消融实验参数，可对 RMSNorm、RoPE、Pre/Post Norm、FFN 类型进行对比。

## 目录结构

```text
.
|-- transformer/
|   |-- train_bpe.py       # 训练 Byte-level BPE 分词器
|   |-- tokenizer.py       # BPE 分词器实现
|   |-- preprocess.py      # 将文本语料编码为 token id 二进制文件
|   |-- nn.py              # Transformer 核心网络模块
|   |-- optimizer.py       # AdamW 与梯度裁剪
|   |-- scheduler.py       # warmup + cosine decay 学习率调度
|   |-- data.py            # 语言模型训练 batch 采样
|   |-- losses.py          # 交叉熵损失
|   |-- checkpointing.py   # checkpoint 保存和加载
|   |-- main_train.py      # 训练入口
|   |-- inference.py       # 交互式文本生成
|   `-- sgd.py             # SGD 优化器实验实现
```

## 环境依赖

建议使用 Python 3.10+，并安装以下依赖：

```bash
pip install torch numpy regex einops wandb jaxtyping
```

如果直接运行脚本，需要确保源码目录已经加入 Python 的包搜索路径。也可以根据自己的工程习惯，将源码目录整理为标准 Python package 后再运行。

## 使用流程

### 1. 训练 BPE 分词器

在 `transformer/train_bpe.py` 中配置训练语料路径、词表大小和输出目录，然后运行：

```bash
python -m transformer.train_bpe
```

运行后会生成：

```text
vocab.json
merges.txt
```

### 2. 预处理训练语料

在 `transformer/preprocess.py` 中配置文本输入路径、分词器路径和输出路径，然后运行：

```bash
python -m transformer.preprocess
```

输出文件为 `uint16` token id 二进制文件，训练阶段通过 `np.memmap` 读取。

### 3. 启动模型训练

```bash
python -m transformer.main_train \
  --train_data_path data/TinyStoriesV2-GPT4-train.bin \
  --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
  --out_dir out \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 8 \
  --d_ff 2048 \
  --batch_size 32 \
  --max_iters 10000
```

可选消融参数：

```bash
--no_rms_norm
--norm_mode post
--no_rope
--ffn_type silu
```

训练过程会记录训练损失、验证损失和学习率，并定期保存 checkpoint。

### 4. 交互式推理

```bash
python -m transformer.inference \
  --checkpoint_path out/ckpt_final.pt \
  --tokenizer_dir data/TinyStoriesV2-GPT4-train \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --temperature 0.7 \
  --top_p 0.9
```

启动后可以在命令行输入 prompt，模型会基于当前 checkpoint 生成文本。

## 核心实现说明

### 分词器

`tokenizer.py` 实现 Byte-level BPE 编码和解码流程。训练阶段从初始 256 个 byte token 开始，根据语料中相邻 token pair 的频次迭代合并，得到最终词表和 merges 规则。编码阶段使用 GPT-2 风格的预分词正则，并按 merges 的优先级逐步合并。

### Transformer

`nn.py` 中实现了 Decoder-only 语言模型的主要结构：

- `Linear`：自定义线性层，使用截断正态初始化。
- `RMSNorm`：均方根归一化。
- `RotaryPositionalEmbedding`：RoPE 旋转位置编码。
- `CausalSelfAttention`：带 causal mask 的多头自注意力。
- `SwiGLU`：门控前馈网络。
- `TransformerLM`：多层 Transformer Block 堆叠后的语言模型。

### 训练

`main_train.py` 负责完整训练流程，包括：

- 加载 `uint16` token 数据。
- 构建 TransformerLM。
- 使用 AdamW、梯度裁剪和 cosine schedule 更新参数。
- 周期性验证并保存 checkpoint。
- 支持从已有 checkpoint 恢复训练。

## 版本管理说明

仓库只保留源码和说明文档，不提交大规模语料、生成的 `.bin` 文件、模型权重、checkpoint 和实验日志。这些文件可根据本地环境重新生成。
