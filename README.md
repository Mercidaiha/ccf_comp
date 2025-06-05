# ccf_comp

ccf_leihuo_competition_2025

### 提交文件

所有提交文件都放在`./competition_data/submit_data/`文件夹下:

```
cd ./competition_data/submit_data/
```

### prepare question embedding via all-MiniLM-L6-v2 (可选)

```
python gen_emb.py
```

### get historical best model in each task (可选)

```
python get_historical_best.py
```

### run embedllm （直接运行）

```
bash run.sh
```
