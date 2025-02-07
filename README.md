# CENET: Contrastive Event Network

This is the official code base of the paper

[Temporal Knowledge Graph Reasoning with Historical Contrastive Learning](https://arxiv.org/abs/2211.10904)

![architecture](architecture.png)

## Statistics of Datasets
![datasets](data/datasets.png)


## Preprocessing
```bash
cd data/YAGO
python get_history_graph.py
```

## Training and Testing

```bash
# 用原始数据集，训练
python main.py -d YAGO --description yago_hard --max-epochs 30 --oracle-epochs 20 --valid-epochs 5 --alpha 0.2 --lambdax 2 --batch-size 1024 --lr 0.001 --oracle_lr 0.001 --oracle_mode hard --save_dir SAVE --eva_dir SAVE

# 用算网数据集，训练
cd data/OURS
python get_history_graph.py
python main.py -d OURS --description ours_hard --max-epochs 30 --oracle-epochs 20 --valid-epochs 5 --alpha 0.2 --lambdax 2 --batch-size 1024 --lr 0.001 --oracle_lr 0.001 --oracle_mode hard --save_dir SAVE --eva_dir SAVE
```

Note that we use hard mode for YAGO and WIKI, soft mode for event-based TKGs. The model performance fluctuates by less than 1% under different seed settings. For example, you will get better performance than the paper results under the setting of Seed 987.


You can use function *load_all_answers_for_time_filter* and *split_by_time* in [script](https://github.com/Lee-zix/RE-GCN/blob/master/rgcn/utils.py) implemented by RE-GCN to get the time-aware filtered results.

## Citation ##

If you find this project useful in your research, please cite the following paper:

```bibtex
@inproceedings{xu-etal-2023-cenet,
  title = {Temporal Knowledge Graph Reasoning with Historical Contrastive Learning},
  author = "Xu, Yi and Ou, Junjie and Xu, Hui and Fu, Luoyi",
  booktitle = "AAAI",
  year = "2023"
}
```
