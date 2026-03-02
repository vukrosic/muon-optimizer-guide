# LLM Research Kit & Muon Optimizer Guide

A high-performance codebase for LLM research, pretraining, and optimization.

---

## 📚 The Muon Optimizer Guide

The Muon optimizer is replacing AdamW across the industry (DeepSeek, OpenAI, Meta, Moonshot AI). This guide explains why and how to use it to speed up training by up to 2x.

[**👉 Click here to read the Muon Optimizer Guide**](course/muon_optimizer_guide.md)

<img src="https://pbs.twimg.com/media/GZoW2CLbsAAeXVK.jpg" alt="NanoGPT wall-clock time curves" width="600">


---

## 🚀 Other than this guide, this repo contains LLM that you can train and do research on

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download the Dataset

### Option A: 1B tokens
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 1B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain')
os.makedirs('processed_data/pretrain_1B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_1B')
print('✅ Full Data Ready!')
"
```

### Option B: 2B tokens
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 2B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-2B-pretrain')
os.makedirs('processed_data/pretrain_2B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_2B')
print('✅ Full Data Ready!')
"
```

### Option C: Quick Start (40M Tokens)
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 40M Token Subset...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain', split='train[:20000]')
os.makedirs('processed_data/speedrun_40M', exist_ok=True)
ds.save_to_disk('processed_data/speedrun_40M')
print('✅ Speedrun Data Ready!')
"
```



