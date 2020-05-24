# coding: utf-8
# date: 2020/5/24
# author: zonnoz
# description: 生成用户特征数据
#              i, 数据路径不同，可能需要再修改,切换调试模式
#             ii, 获取某 T 特征前需确认 HotTime[T] 已由 popular_time_analysis 生成
#            iii, 同T的 train & test 数据合并后需去重


import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.precision", 16)

train_path = "../data/underexpose_train"
train_path_feat = train_path + "/underexpose_user_feat.csv"
train_path_click = train_path + "/underexpose_train_click-{0}.csv"

test_path = "../data/underexpose_test"
test_path_click = test_path + "/underexpose_test_click/underexpose_test_click-{0}.csv"
test_path_qtime = test_path + "/underexpose_test_qtime/underexpose_test_qtime-{0}.csv"

Debug = True
nrows = 10000 if Debug else None

HotTime = {
    0: [0.983755, 0.983800, 0.983810, 0.983855, 0.983865, 0.983907, 0.983920, 0.983957],
    1: [0.983811, 0.983854, 0.983865, 0.983907, 0.983920, 0.983962, 0.983974, 0.984012],
    2: [0.983866, 0.983907, 0.983920, 0.983962, 0.983974, 0.984016, 0.984029, 0.984067],
    3: [0.983921, 0.983961, 0.983974, 0.984016, 0.984029, 0.984072, 0.984083, 0.984121],
    4: [0.983975, 0.984016, 0.984030, 0.984073, 0.984083, 0.984125, 0.984139, 0.984175],
    5: [0.984030, 0.984072, 0.984083, 0.984125, 0.984138, 0.984179, 0.984192, 0.984230],
    # 6: [],
    # 7: [],
    # 8: [],
    # 9: [],
}


def get_train_click(list_train_click_file):
    """ 获取训练集的点击数据
    :param list_train_click_file:
    :return:
    """
    list_underexpose_click = []
    columns = ["user_id", "item_id", "time", "phase"]
    for phase, file in tqdm(list_train_click_file):
        underexpose_click = pd.read_csv(file,
                                        names=["user_id", "item_id", "time"],
                                        nrows=nrows,
                                        # dtype={"user_id": int, "item_id": int, "time": float},
                                        sep=","
                                        )
        underexpose_click.drop_duplicates(inplace=True)
        underexpose_click["phase"] = phase
        underexpose_click = underexpose_click[columns]
        list_underexpose_click.append(underexpose_click)

    all_underexpose_click = pd.concat(list_underexpose_click)
    all_underexpose_click["category"] = "train"
    return all_underexpose_click


def get_test_click(list_test_phase_file, list_test_phase_query_file):
    """ 测试集合并（click + qtime）
    :param list_test_phase_file:
    :param list_test_phase_query_file:
    :return:
    """
    list_underexpose_click = []
    columns = ["user_id", "item_id", "time", "phase", "category"]
    for phase, file in tqdm(list_test_phase_file):  # 测试集
        underexpose_click = pd.read_csv(file,
                                        names=["user_id", "item_id", "time"],
                                        nrows=nrows,
                                        sep=","
                                        )
        underexpose_click.drop_duplicates(inplace=True)
        underexpose_click["phase"] = phase
        underexpose_click["category"] = "test"
        underexpose_click = underexpose_click[columns]
        list_underexpose_click.append(underexpose_click)

    for phase, file in tqdm(list_test_phase_query_file):
        underexpose_qtime = pd.read_csv(file
                                        , names=["user_id", "query_time"]
                                        , nrows=nrows
                                        , sep=","
                                        )
        underexpose_qtime.drop_duplicates(inplace=True)
        underexpose_qtime.columns = ["user_id", "time"]
        underexpose_qtime["phase"] = phase
        underexpose_qtime["item_id"] = -999  # 预测集标记
        underexpose_qtime["category"] = "predict" # 预测集标记
        underexpose_qtime = underexpose_qtime[columns]
        list_underexpose_click.append(underexpose_qtime)

    all_underexpose_click = pd.concat(list_underexpose_click)

    return all_underexpose_click[columns]


def get_unique_users(*args):
    """ 合并所有关于用户的数据集，拿到所有 user
    :param args: user_feat_data, underexpose_train, underexpose_test
    :return:
    """
    list_users = []
    columns = ["user_id", "phase"]
    for ddf in tqdm(args):
        if len(set(columns) & set(ddf.columns)) != len(columns): # 只传["user_id", "phase"]两列数据
            raise Exception("dataframe index no match, check columns. ")
        list_users.append(ddf[columns])
    all_users = pd.concat(list_users)
    all_users.drop_duplicates(inplace=True)
    all_users.sort_values(by=columns, ascending=True)
    return all_users


def add_status_feat(all_user_phase_feat, underexpose_user_feat):
    underexpose_user_feat = underexpose_user_feat.drop(["phase", "category"], axis=1, inplace=False)
    return pd.merge(all_user_phase_feat, underexpose_user_feat, how="left", on="user_id")


# TODO: 每个周期的特征抽取, 允许phase数据是多 T 合并去重后数据 ！！！
def deal_user_phase_time_series(underexpose_click: pd.DataFrame, phases: list, category="train", merged=False):
    """
    :param underexpose_click: dataframe, index(['user_id', 'item_id', 'time', 'phase', 'category'])
    :param phases: 不同阶段 T
    :param category: 共有 train、test、predict
    :return:
    """
    res = {}

    underexpose_click = underexpose_click[(underexpose_click["phase"].isin(phases)) & (underexpose_click["category"] == category)]

    # add ["user_id", "phase", "item_id"]
    feat_user_item_id = underexpose_click[["user_id", "phase", "item_id"]]

    # add ["user_id", "phase", "start_time", "end_time", "duration"]
    feat_duration_ct = add_user_duration_counts(underexpose_click[["user_id", "phase", "time"]])

    # add ["user_id", "phase", "item_counts", "item_ucounts"]
    feat_user_items_ct = add_grouped_feat_f(underexpose_click[["user_id", "phase", "item_id"]],
                                            by_cols=["user_id", "phase"],
                                            aggregations={"item_id": ["count", "nunique"]},
                                            res_cols=["user_id", "phase", "item_counts", "item_ucounts"],
                                            )

    # ["user_id", "phase", "item_id", "counts"]，每阶段数据的各个时段点击查看的 top k 商品
    k = 1
    feat_top_items = underexpose_click[["user_id", "phase", "item_id"]] \
        .groupby(["user_id", "phase", "item_id"], as_index=False).size().reset_index(name='top_counts')
    feat_top_items.groupby(["user_id", "phase"]).head(k)

    res.update({
        "feat_user_item_id": feat_user_item_id,
        "feat_duration_ct": feat_duration_ct,
        "feat_user_items_ct": feat_user_items_ct,
        "feat_top_items": feat_top_items,
    })

    # 非合并数据，对当前 T 内热门时段做聚合特征
    if not merged:
        underexpose_click["popular_time"] = underexpose_click.apply(cal_popular_time, axis=1)

        underexpose_click["is_popular_time"] = underexpose_click.apply(check_popular_time, axis=1)

        # add ["user_id", "phase", "popular_time", "item_counts_pt", "item_ucounts_pt"]，各个时段的点击物品数、物品个数
        feat_duration_ct_popular = add_grouped_feat_f(
            underexpose_click[["user_id", "phase", "popular_time", "item_id"]],
            by_cols=["user_id", "phase", "popular_time"],
            aggregations={"item_id": ["count", "nunique"]},
            res_cols=["user_id", "phase", "popular_time", "item_counts_pt", "item_ucounts_pt"],
        )

        # 注意：one-hot编码时，这里 not_popular_time 就是上面 popular_time = 0
        # add ["user_id", "phase", "is_popular_time", "item_counts_ipt", "item_ucounts_ipt"]，热门/冷门时段的点击物品数、物品个数
        feat_duration_ct_if_popular = add_grouped_feat_f(
            underexpose_click[["user_id", "phase", "is_popular_time", "item_id"]],
            by_cols=["user_id", "phase", "is_popular_time"],
            aggregations={"item_id": ["count", "nunique"]},
            res_cols=["user_id", "phase", "is_popular_time", "item_counts_ipt", "item_ucounts_ipt"],
        )

        # ["user_id", "phase", "popular_time", "item_id", "counts"]，每阶段数据的各个时段点击查看的 top k 商品
        k = 1
        feat_popular_top_items = underexpose_click[["user_id", "phase", "popular_time", "item_id"]]\
            .groupby(["user_id", "phase", "popular_time", "item_id"], as_index=False).size().reset_index(name='top_counts_pt')
        feat_popular_top_items.groupby(["user_id", "phase", "popular_time"]).head(k)

        feat_popular_top_items_if_popular = underexpose_click[["user_id", "phase", "is_popular_time", "item_id"]]\
            .groupby(["user_id", "phase", "is_popular_time", "item_id"], as_index=False).size().reset_index(name='top_counts_ipt')
        feat_popular_top_items_if_popular.groupby(["user_id", "phase", "is_popular_time"]).head(k)

        res.update({
            "feat_duration_ct_popular": feat_duration_ct_popular,
            "feat_duration_ct_if_popular": feat_duration_ct_if_popular,
            "feat_popular_time_items": feat_popular_top_items,
            "feat_popular_time_items_if_popular": feat_popular_top_items_if_popular,
        })

    return res


def add_grouped_feat_f(df: pd.DataFrame, by_cols: list, aggregations: dict, res_cols: list):
    df = df.groupby(by_cols, as_index=False).agg(aggregations)
    df.columns = res_cols
    return df


def add_user_duration_counts(underexpose_click: pd.DataFrame):
    """
    :param underexpose_click: df, index(["user_id", "phase", "time"])
    :return: df with columns ["user_id", "phase", "start_time", "end_time", "duration"]
    """
    underexpose_click = add_grouped_feat_f(underexpose_click,
                                           by_cols=["user_id", "phase"],
                                           aggregations={"time": ["min", "max"]},
                                           res_cols=["user_id", "phase", "start_time", "end_time"],
                                           )
    # underexpose_click = underexpose_click.groupby(["user_id", "phase"], as_index=False).agg({"time": ["min", "max"]})
    # underexpose_click.columns = ["user_id", "phase", "start_time", "end_time"]
    underexpose_click["duration"] = underexpose_click["end_time"] - underexpose_click["start_time"]
    return underexpose_click


# TODO: merge clustered item features
def add_user_items_feat(underexpose_click: pd.DataFrame):
    """
    :param underexpose_click:
    :return: 返回去重/不去重条件下用户浏览的总商品数、总类数统计
    """
    underexpose_click = underexpose_click.groupby(["user_id", "phase"], as_index=False).agg({"item_id": ["count", "nunique"]})
    underexpose_click.columns = ["user_id", "phase", "item_counts", "item_ucounts"]
    return underexpose_click


def cal_popular_time(row):
    """
    :param row:
    :return: 在某 T 数据范围内，1~4 热门时段，0 为冷门时段
    """
    if row["phase"] not in HotTime:
        raise Exception("need to execute popular_time_analysis first.")
    hot_time = HotTime[row["phase"]]
    if row["time"] < hot_time[0] or row["time"] > hot_time[-1]:
        return 0 # 冷门时间
    for i in range(1, len(hot_time), 2):
        if hot_time[i-1] <= row["time"] <= hot_time[i]:
            return i // 2 + 1 # 细化热门时段


def check_popular_time(row):
    """
    :param row:
    :return: 在某 T 数据范围内，1 热门时段，0 为冷门时段
    """
    if row["phase"] not in HotTime:
        raise Exception("need to execute popular_time_analysis first.")
    hot_time = HotTime[row["phase"]]
    if row["time"] < hot_time[0] or row["time"] > hot_time[-1]:
        return 0 # 冷门时间
    return 1     # 热门时段


if __name__ == '__main__':
    """
    train: user_feat + click_train   
    test: click_test
    predict: qtime 
    underexpose_user_feat: ['user_id', 'user_age_level', 'user_gender', 'user_city_level', 'phase', 'category']
    underexpose_train_click & underexpose_test_click: ['user_id', 'item_id', 'time', 'phase', 'category']
    category -> {train, test, predict}
    """
    phase = [0, 1, 2, 3, 4, 5]

    underexpose_train_click = get_train_click([(x, train_path_click.format(x)) for x in phase])
    underexpose_test_click = get_test_click([(x, test_path_click.format(x)) for x in phase], [(x, test_path_qtime.format(x)) for x in phase])
    underexpose_user_feat = pd.read_csv(train_path_feat,
                                        names=["user_id", "user_age_level", "user_gender", "user_city_level"],
                                        nrows=nrows,
                                        sep=",")
    underexpose_user_feat.drop_duplicates(subset="user_id", keep="first", inplace=True) # 有三个 user_id 重复但其它特征不同，这里默认保留第一条
    underexpose_user_feat.fillna(-1)    # 有少量缺失，填充
    underexpose_user_feat["phase"] = 0
    underexpose_user_feat["category"] = "train"

    # 所有用户加上基本属性特征
    all_users = get_unique_users(underexpose_user_feat, underexpose_train_click, underexpose_test_click)
    all_user_phase_feat = add_status_feat(all_users, underexpose_user_feat)

    # 为各个 phase 的用户加上人工特征
    res = deal_user_phase_time_series(underexpose_train_click, phases=phase)

    for k, v in res.items():
        print(k + ": \n", v.head())

    # TODO: unresolved: a, item类 + time需groupby ; b, merge: 用户基本属性 underexpose_user_feat + 各阶段的人为特征








