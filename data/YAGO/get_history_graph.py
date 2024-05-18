import numpy as np
import os
import pickle
import dgl
import torch
import tqdm
from scipy.sparse import csc_matrix

# 由于同名，被后面的函数所覆盖，所以这里的函数没有被调用
def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), "r") as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), "r") as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


# 获取实体和关系的数量
def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), "r") as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


# 打开文件，读取数据，返回数据和时间
def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), "r") as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])  # 头实体
            tail = int(line_split[2])  # 尾实体
            rel = int(line_split[1])  # 关系
            time = int(line_split[3])  # 时间
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), "r") as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.array(quadrupleList), np.asarray(times)


def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel_o = np.concatenate((rel + num_rels, rel))
    rel_s = np.concatenate((rel, rel + num_rels))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update(
        {"id": torch.from_numpy(uniq_v).long().view(-1, 1), "norm": norm.view(-1, 1)}
    )
    g.edata["type_s"] = torch.LongTensor(rel_s)
    g.edata["type_o"] = torch.LongTensor(rel_o)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g


# 获取数据集中每个实体的历史事件，并计算目标标签，以及生成稀疏矩阵表示的历史关系频率
def get_history_target(quadruples, s_history_event_o, o_history_event_s):
    s_history_oid = []
    o_history_sid = []
    ss = quadruples[:, 0]
    rr = quadruples[:, 1]
    oo = quadruples[:, 2]
    s_history_related = np.zeros((quadruples.shape[0], num_e), dtype=float)
    o_history_related = np.zeros((quadruples.shape[0], num_e), dtype=float)
    for ix in tqdm.tqdm(range(quadruples.shape[0])):
        s_history_oid.append([])
        o_history_sid.append([])
        for con_events in s_history_event_o[ix]:
            idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
            cur_events = con_events[idxx, 1].tolist()
            s_history_oid[-1] += con_events[:, 1].tolist()
            s_history_related[ix][cur_events] += 1
        for con_events in o_history_event_s[ix]:
            idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
            cur_events = con_events[idxx, 1].tolist()
            o_history_sid[-1] += con_events[:, 1].tolist()
            o_history_related[ix][cur_events] += 1

    s_history_label_true = np.zeros((quadruples.shape[0], 1))
    o_history_label_true = np.zeros((quadruples.shape[0], 1))
    s_cnt = 0
    o_cnt = 0
    for ix in tqdm.tqdm(range(quadruples.shape[0])):
        # print('----------------------------------')
        if oo[ix] in s_history_oid[ix]:
            s_history_label_true[ix] = 1
        else:
            s_cnt += 1
            # print(oo[ix])
            # print('sssssssssssssssssss')
            # print(s_history_oid[ix])
        if ss[ix] in o_history_sid[ix]:
            o_history_label_true[ix] = 1
        else:
            o_cnt += 1
            # print(ss[ix])
            # print('oooooooooooo')
            # print(o_history_sid[ix])
    print(
        "sss",
        s_cnt,
        "oooo",
        o_cnt,
        s_cnt / quadruples.shape[0],
        o_cnt / quadruples.shape[0],
    )
    s_history_related = csc_matrix(s_history_related)
    o_history_related = csc_matrix(o_history_related)
    return (
        s_history_label_true,
        o_history_label_true,
        s_history_related,
        o_history_related,
    )


# step 1 读取训练数据
graph_dict_train = {}

train_data, train_times = load_quadruples("", "train.txt")
test_data, test_times = load_quadruples("", "test.txt")
dev_data, dev_times = load_quadruples("", "valid.txt")
# total_data, _ = load_quadruples('', 'train.txt', 'test.txt')

# 获取实体和关系的数量
num_e, num_r = get_total_number("", "stat.txt")

s_his = [[] for _ in range(num_e)]  # 主语的历史关系
o_his = [[] for _ in range(num_e)]  # 宾语的历史关系
s_his_t = [[] for _ in range(num_e)]  # 主语的历史关系时间
o_his_t = [[] for _ in range(num_e)]  # 宾语的历史关系时间
s_history_data = [[] for _ in range(len(train_data))]  # 主语的历史关系数据
o_history_data = [[] for _ in range(len(train_data))]  # 宾语的历史关系数据
s_history_data_t = [[] for _ in range(len(train_data))]  # 主语的历史关系时间数据
o_history_data_t = [[] for _ in range(len(train_data))]  # 宾语的历史关系时间数据
e = []
r = []
latest_t = 0
s_his_cache = [[] for _ in range(num_e)]  # 主语的历史关系缓存
o_his_cache = [[] for _ in range(num_e)]  # 宾语的历史关系缓存
s_his_cache_t = [None for _ in range(num_e)]  # 主语的历史关系时间缓存
o_his_cache_t = [None for _ in range(num_e)]  # 宾语的历史关系时间缓存

# 遍历训练数据中的时间，获取每个时间的图
for tim in train_times:
    print(str(tim) + "\t" + str(max(train_times)))
    data = get_data_with_t(train_data, tim)
    graph_dict_train[tim] = get_big_graph(data, num_r)

# 遍历训练数据，获取每个实体在不同时刻的历史关系
for i, train in enumerate(train_data):
    # 每隔1w条数据打印一次，显示进度
    if i % 10000 == 0:
        print("train", i, len(train_data))

    # 时间不同时，说明已经进入到了下一个时间点，需要更新历史信息
    # 将缓存的数据加入到历史数据中，并将缓存清空
    t = train[3]
    if latest_t != t:
        for ee in range(num_e):
            # 主语
            if len(s_his_cache[ee]) != 0:
                s_his[ee].append(s_his_cache[ee].copy())
                s_his_t[ee].append(s_his_cache_t[ee])
                s_his_cache[ee] = []
                s_his_cache_t[ee] = None
            # 宾语
            if len(o_his_cache[ee]) != 0:
                o_his[ee].append(o_his_cache[ee].copy())
                o_his_t[ee].append(o_his_cache_t[ee])
                o_his_cache[ee] = []
                o_his_cache_t[ee] = None
        latest_t = t

    # 获取当前数据的主语、关系、宾语
    s = train[0]
    r = train[1]
    o = train[2]

    # 将当前数据的历史关系和时间加入到历史数据中，包括主语和宾语
    s_history_data[i] = s_his[s].copy()
    o_history_data[i] = o_his[o].copy()
    s_history_data_t[i] = s_his_t[s].copy()
    o_history_data_t[i] = o_his_t[o].copy()

    # 将当前数据加入到缓存中，一个缓存对应一个时间
    # 主语
    if len(s_his_cache[s]) == 0:
        s_his_cache[s] = np.array([[r, o]])
    else:
        s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
    s_his_cache_t[s] = t

    # 宾语
    if len(o_his_cache[o]) == 0:
        o_his_cache[o] = np.array([[r, s]])
    else:
        o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
    o_his_cache_t[o] = t

# 获取实体标签和历史关系
s_label_train, o_label_train, s_history_related_train, o_history_related_train = (
    get_history_target(train_data, s_history_data, o_history_data)
)

# 保存数据
with open("train_graphs.txt", "wb") as fp:
    pickle.dump(graph_dict_train, fp)
with open("train_history_sub.txt", "wb") as fp:
    pickle.dump([s_history_data, s_history_data_t], fp)
with open("train_history_ob.txt", "wb") as fp:
    pickle.dump([o_history_data, o_history_data_t], fp)
with open("train_s_label.txt", "wb") as fp:
    pickle.dump(s_label_train, fp)
with open("train_o_label.txt", "wb") as fp:
    pickle.dump(o_label_train, fp)
with open("train_s_frequency.txt", "wb") as fp:
    pickle.dump(s_history_related_train, fp)
with open("train_o_frequency.txt", "wb") as fp:
    pickle.dump(o_history_related_train, fp)

# step 2 读取验证数据
s_history_data_dev = [[] for _ in range(len(dev_data))]
o_history_data_dev = [[] for _ in range(len(dev_data))]
s_history_data_dev_t = [[] for _ in range(len(dev_data))]
o_history_data_dev_t = [[] for _ in range(len(dev_data))]

for i, dev in enumerate(dev_data):
    if i % 10000 == 0:
        print("valid", i, len(dev_data))
    t = dev[3]
    if latest_t != t:
        for ee in range(num_e):
            if len(s_his_cache[ee]) != 0:
                s_his_t[ee].append(s_his_cache_t[ee])
                s_his[ee].append(s_his_cache[ee].copy())
                s_his_cache[ee] = []
                s_his_cache_t[ee] = None
            if len(o_his_cache[ee]) != 0:

                o_his_t[ee].append(o_his_cache_t[ee])
                o_his[ee].append(o_his_cache[ee].copy())

                o_his_cache[ee] = []
                o_his_cache_t[ee] = None
        latest_t = t
    s = dev[0]
    r = dev[1]
    o = dev[2]
    s_history_data_dev[i] = s_his[s].copy()
    o_history_data_dev[i] = o_his[o].copy()
    s_history_data_dev_t[i] = s_his_t[s].copy()
    o_history_data_dev_t[i] = o_his_t[o].copy()
    if len(s_his_cache[s]) == 0:
        s_his_cache[s] = np.array([[r, o]])
    else:
        s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
    s_his_cache_t[s] = t

    if len(o_his_cache[o]) == 0:
        o_his_cache[o] = np.array([[r, s]])
    else:
        o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
    o_his_cache_t[o] = t

    # print(o_his_cache[o])

s_label_dev, o_label_dev, s_history_related_dev, o_history_related_dev = (
    get_history_target(dev_data, s_history_data_dev, o_history_data_dev)
)
with open("dev_history_sub.txt", "wb") as fp:
    pickle.dump([s_history_data_dev, s_history_data_dev_t], fp)
with open("dev_history_ob.txt", "wb") as fp:
    pickle.dump([o_history_data_dev, o_history_data_dev_t], fp)
with open("dev_s_label.txt", "wb") as fp:
    pickle.dump(s_label_dev, fp)
with open("dev_o_label.txt", "wb") as fp:
    pickle.dump(o_label_dev, fp)
with open("dev_s_frequency.txt", "wb") as fp:
    pickle.dump(s_history_related_dev, fp)
with open("dev_o_frequency.txt", "wb") as fp:
    pickle.dump(o_history_related_dev, fp)


# step 3 读取测试数据
s_history_data_test = [[] for _ in range(len(test_data))]
o_history_data_test = [[] for _ in range(len(test_data))]

s_history_data_test_t = [[] for _ in range(len(test_data))]
o_history_data_test_t = [[] for _ in range(len(test_data))]

for i, test in enumerate(test_data):
    if i % 10000 == 0:
        print("test", i, len(test_data))
    t = test[3]
    if latest_t != t:
        for ee in range(num_e):
            if len(s_his_cache[ee]) != 0:
                s_his_t[ee].append(s_his_cache_t[ee])

                s_his[ee].append(s_his_cache[ee].copy())
                s_his_cache[ee] = []
                s_his_cache_t[ee] = None
            if len(o_his_cache[ee]) != 0:

                o_his_t[ee].append(o_his_cache_t[ee])

                o_his[ee].append(o_his_cache[ee].copy())
                o_his_cache[ee] = []
                o_his_cache_t[ee] = None
        latest_t = t
    s = test[0]
    r = test[1]
    o = test[2]
    s_history_data_test[i] = s_his[s].copy()
    o_history_data_test[i] = o_his[o].copy()
    s_history_data_test_t[i] = s_his_t[s].copy()
    o_history_data_test_t[i] = o_his_t[o].copy()
    if len(s_his_cache[s]) == 0:
        # s_his_cache[s] = np.array([[r, o]])
        pass
    else:
        # s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
        pass
    s_his_cache_t[s] = t

    if len(o_his_cache[o]) == 0:
        # o_his_cache[o] = np.array([[r, s]])
        pass
    else:
        # o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
        pass
    o_his_cache_t[o] = t
    # print(o_his_cache[o])

s_label_test, o_label_test, s_history_related_test, o_history_related_test = (
    get_history_target(test_data, s_history_data_test, o_history_data_test)
)
with open("test_history_sub.txt", "wb") as fp:
    pickle.dump([s_history_data_test, s_history_data_test_t], fp)
with open("test_history_ob.txt", "wb") as fp:
    pickle.dump([o_history_data_test, o_history_data_test_t], fp)
    # print(train)
with open("test_s_label.txt", "wb") as fp:
    pickle.dump(s_label_test, fp)
with open("test_o_label.txt", "wb") as fp:
    pickle.dump(o_label_test, fp)
with open("test_s_frequency.txt", "wb") as fp:
    pickle.dump(s_history_related_test, fp)
with open("test_o_frequency.txt", "wb") as fp:
    pickle.dump(o_history_related_test, fp)
