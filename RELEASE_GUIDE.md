# PoultryCaduceus GitHub 发布指南

## 📋 发布前检查清单

### 1. 代码准备

- [ ] 确保所有代码可以正常运行
- [ ] 运行测试: `pytest tests/ -v`
- [ ] 代码格式化: `black poultry_caduceus/` 和 `isort poultry_caduceus/`
- [ ] 类型检查: `mypy poultry_caduceus/`
- [ ] 更新版本号: `poultry_caduceus/__version__.py`

### 2. 文档准备

- [ ] 更新 README.md 中的占位符 (YOUR_USERNAME, 性能数据等)
- [ ] 添加模型架构图到 `assets/architecture.png`
- [ ] 添加 logo 到 `assets/logo.png` (可选)
- [ ] 检查所有链接是否正确

### 3. 模型权重准备

- [ ] 导出预训练模型权重
- [ ] 导出 MPRA 微调模型权重
- [ ] 准备上传到 HuggingFace Hub

---

## 🚀 发布步骤

### Step 1: 创建 GitHub 仓库

```bash
# 1. 在 GitHub 上创建新仓库
# 访问 https://github.com/new
# 仓库名: PoultryCaduceus
# 描述: A Bidirectional DNA Language Model for Chicken Genome
# 选择: Public, 不初始化 README

# 2. 本地初始化
cd PoultryCaduceus
git init
git add .
git commit -m "Initial commit: PoultryCaduceus v1.0.0"

# 3. 连接远程仓库
git remote add origin https://github.com/YOUR_USERNAME/PoultryCaduceus.git
git branch -M main
git push -u origin main
```

### Step 2: 创建 Release

```bash
# 1. 创建版本标签
git tag -a v1.0.0 -m "PoultryCaduceus v1.0.0 - Initial Release"
git push origin v1.0.0

# 2. 在 GitHub 上创建 Release
# 访问: https://github.com/YOUR_USERNAME/PoultryCaduceus/releases/new
# 选择标签: v1.0.0
# Release 标题: PoultryCaduceus v1.0.0
# 描述: 见下方模板
```

### Step 3: 上传模型到 HuggingFace Hub

```bash
# 1. 安装 huggingface_hub
pip install huggingface_hub

# 2. 登录
huggingface-cli login

# 3. 创建模型仓库
huggingface-cli repo create poultry-caduceus-base --type model

# 4. 上传模型
cd checkpoints/pretrain/final_model
huggingface-cli upload YOUR_USERNAME/poultry-caduceus-base .

# 5. 上传 MPRA 模型
huggingface-cli repo create poultry-caduceus-mpra --type model
cd checkpoints/mpra/final_model
huggingface-cli upload YOUR_USERNAME/poultry-caduceus-mpra .
```

### Step 4: 发布到 PyPI (可选)

```bash
# 1. 安装构建工具
pip install build twine

# 2. 构建包
python -m build

# 3. 上传到 TestPyPI (测试)
twine upload --repository testpypi dist/*

# 4. 测试安装
pip install --index-url https://test.pypi.org/simple/ poultry-caduceus

# 5. 上传到 PyPI (正式)
twine upload dist/*
```

---

## 📝 Release Notes 模板

```markdown
# PoultryCaduceus v1.0.0

## 🎉 Initial Release

We are excited to announce the first release of **PoultryCaduceus**, a bidirectional DNA language model specifically pre-trained on the chicken (*Gallus gallus*) genome.

### ✨ Features

- **Chicken-specific pre-training** on GRCg7b reference genome (~1.1 Gb)
- **Bidirectional Mamba architecture** with reverse complement equivariance
- **Long-range modeling** up to 65,536 bp context
- **MPRA fine-tuning** for experimentally-validated regulatory predictions
- **Multi-task support** for eQTL prediction and GWAS fine-mapping

### 📦 Available Models

| Model | Description | HuggingFace |
|-------|-------------|-------------|
| `poultry-caduceus-base` | Base pre-trained model | [Link](https://huggingface.co/YOUR_USERNAME/poultry-caduceus-base) |
| `poultry-caduceus-mpra` | MPRA fine-tuned model | [Link](https://huggingface.co/YOUR_USERNAME/poultry-caduceus-mpra) |

### 🚀 Quick Start

```python
from poultry_caduceus import PoultryCaduceus

model = PoultryCaduceus.from_pretrained("YOUR_USERNAME/poultry-caduceus-base")
embeddings = model.get_embeddings("ATGCGATCGATCG")
```

### 📊 Performance

| Task | Metric | Score |
|------|--------|-------|
| MPRA Prediction | Pearson r | X.XX |
| eQTL Classification | AUROC | X.XX |
| Fine-mapping | Recall@10 | X.XX |

### 📄 Citation

If you use PoultryCaduceus, please cite:

```bibtex
@article{poultrycaduceus2024,
  title={PoultryCaduceus: A Bidirectional DNA Language Model for Chicken Genome},
  author={Your Name},
  year={2024}
}
```

### 🙏 Acknowledgments

- [Caduceus](https://github.com/kuleshov-group/caduceus) for the base architecture
- NCBI for the chicken reference genome

---

**Full Changelog**: https://github.com/YOUR_USERNAME/PoultryCaduceus/commits/v1.0.0
```

---

## 📁 最终仓库结构

```
PoultryCaduceus/
├── README.md                    # 主文档
├── LICENSE                      # MIT 许可证
├── setup.py                     # 安装脚本
├── pyproject.toml              # 现代 Python 打包配置
├── requirements.txt            # 依赖列表
├── environment.yml             # Conda 环境
├── .gitignore                  # Git 忽略文件
│
├── poultry_caduceus/           # 主包
│   ├── __init__.py
│   ├── __version__.py
│   ├── config.py               # 配置类
│   ├── model.py                # 模型实现
│   ├── tokenizer.py            # DNA 分词器
│   └── utils.py                # 工具函数
│
├── scripts/                    # 训练脚本
│   ├── pretrain.py
│   ├── finetune_mpra.py
│   ├── finetune_eqtl.py
│   └── evaluate.py
│
├── configs/                    # 配置文件
│   ├── pretrain.yaml
│   ├── finetune_mpra.yaml
│   └── finetune_eqtl.yaml
│
├── tests/                      # 单元测试
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_config.py
│
├── examples/                   # 示例 notebooks
│   ├── 01_quick_start.ipynb
│   ├── 02_mpra_prediction.ipynb
│   └── 03_variant_effects.ipynb
│
├── docs/                       # 文档
│   ├── architecture.md
│   ├── pretraining.md
│   ├── finetuning.md
│   └── api.md
│
└── assets/                     # 图片资源
    ├── logo.png
    └── architecture.png
```

---

## 🔧 HuggingFace Model Card 模板

创建 `README.md` 文件放在 HuggingFace 模型仓库中:

```markdown
---
language:
- en
license: mit
tags:
- genomics
- dna
- chicken
- biology
- caduceus
datasets:
- custom
metrics:
- pearson_r
library_name: poultry-caduceus
pipeline_tag: feature-extraction
---

# PoultryCaduceus Base

A bidirectional DNA language model pre-trained on the chicken (Gallus gallus) genome.

## Model Description

PoultryCaduceus is based on the Caduceus architecture with:
- Bidirectional Mamba layers
- Reverse complement equivariance
- 65,536 bp context length

## Training Data

Pre-trained on GRCg7b chicken reference genome (~1.1 Gb).

## Usage

```python
from poultry_caduceus import PoultryCaduceus

model = PoultryCaduceus.from_pretrained("YOUR_USERNAME/poultry-caduceus-base")
embeddings = model.get_embeddings("ATGCGATCGATCG")
```

## Citation

```bibtex
@article{poultrycaduceus2024,
  title={PoultryCaduceus: A Bidirectional DNA Language Model for Chicken Genome},
  author={Your Name},
  year={2024}
}
```
```

---

## ⚠️ 注意事项

1. **替换占位符**: 搜索并替换所有 `YOUR_USERNAME`、`your.email@example.com` 等占位符

2. **更新性能数据**: 用实际实验结果替换 `X.XX` 占位符

3. **添加图片**: 
   - 创建模型架构图 (`assets/architecture.png`)
   - 可选: 创建项目 logo (`assets/logo.png`)

4. **测试安装**: 在发布前测试 `pip install -e .` 是否正常工作

5. **检查许可证**: 确保 LICENSE 文件中的年份和姓名正确

6. **敏感信息**: 确保没有提交任何敏感信息 (API keys, 密码等)

---

## 📞 需要帮助?

如果在发布过程中遇到问题，可以:
1. 查看 GitHub 文档: https://docs.github.com
2. 查看 HuggingFace 文档: https://huggingface.co/docs
3. 查看 PyPI 文档: https://packaging.python.org

祝发布顺利! 🎉
