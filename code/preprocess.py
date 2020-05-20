import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np


#获取train 数据


# 特征集合

user_feat_list = []
item_feat_list = []
user_feat_cross = []
numberic_feat_list = []
cat_feat_list = []
embeding_feat_list = []


#特征文件路径
feat_columns_path="../data/features_colums/"
train_path = '../data/underexpose_train'
test_path = '../data/underexpose_test'
feat_path = '../user_data/model_data'
train_path_click = train_path+'/underexpose_train_click-{}.csv'
test_path_click = test_path+'/underexpose_test_click-{0}/underexpose_test_click-{0}.csv'
test_path_click_qtime = test_path+'/underexpose_test_click-{0}/underexpose_test_qtime-{0}.csv'
train_path_item = train_path+'/underexpose_item_feat.csv'
train_path_user = train_path+'/underexpose_user_feat.csv'


# item 和user的特征名称
item_feat_cols = ["item_id"]+['text_vd' + str(i) for i in range(128)]+['img_vd' + str(i) for i in range(128)]
user_feat_cols = ["user_id","user_age_level","user_gender","user_city_level"]



# print("开始txt聚类")
# #设置要进行聚类的字段
# loan = np.array(train_item_feat_df[['text_vd' + str(i) for i in range(128)]])
# #设置类别为3
# clf=KMeans(n_clusters=100)
# #将数据代入到聚类模型中
# clf=clf.fit(loan)
# train_item_feat_df['text_cat']=clf.labels_
#
#
# print("开始img聚类")
# #设置要进行聚类的字段
# data2 = np.array(train_item_feat_df[['img_vd' + str(i) for i in range(128)]])
# #设置类别为3
# clf=KMeans(n_clusters=100)
# #将数据代入到聚类模型中
# clf=clf.fit(data2)
# train_item_feat_df['img_cat']=clf.labels_
# print(clf.cluster_centers_)


def user_item_base_feat(base_files, columns=[], isitem=True,  list_feat=[]):
    """
    获取用户或者物品的基础特征
    :param base_files:用户/物品的基础路径
    :param columns: 特征重命名
    :param isitem:  是否是物品 True表示是物品，False表示是用户

    :param list_feat:   需要提取特征的list   [item1,item2]
    :return:
    """
    try:

        if (len(columns) <= 0):
            print("特征列明不可为空")
            return
        feat_columns = []

        if len(list_feat) > 0:
            feat_columns = list_feat
        if base_files:
            dt = pd.read_csv(base_files, header=None,names=columns)
            dt = dt.drop_duplicates()  # 去重
            if isitem == True:
                dt[item_feat_cols[1]] = dt[item_feat_cols[1]].apply(lambda x: x[1:])
                dt[item_feat_cols[128]] = dt[item_feat_cols[128]].apply(lambda x: x[:-1])
                dt[item_feat_cols[129]] = dt[item_feat_cols[129]].apply(lambda x: x[1:])
                dt[item_feat_cols[256]] = dt[item_feat_cols[256]].apply(lambda x: x[:-1])
                # 对这些向量进行归一化
                dt[[item_feat_cols[1][:-1] + str(i) for i in range(128)]] = MinMaxScaler().fit_transform(dt[[item_feat_cols[1][:-1] + str(i) for i in range(128)]])
                dt[[item_feat_cols[129][:-1] + str(i) for i in range(128)]] = MinMaxScaler().fit_transform(dt[[item_feat_cols[129][:-1] + str(i) for i in range(128)]])

            if len(feat_columns) > 0:
                dt = dt[feat_columns]
            print("读取item特征结束")
            return dt
        else:
            print("请输入文件路径")
    except Exception as e:
        print(e)


def get_train_test_click_data(files, names=['user_id', 'item_id', 'time'], num=1,  list_feat=[]):
    """
    获取train/test中click数据
    :param files: 基础数据文件地址
    :param names: 列名
    :param num: 文件个数
    :param list_feat:  需要获取的特征列名list [item1,item2]
    :return:
    """
    try:
        if len(names) <= 0:
            print("特征列名不可为空")
            return
        all_train = pd.DataFrame(columns=names)
        for p in range(num):
            print(p)
            dt = pd.read_csv(files.format(p), header=None,  names=names)
            all_train = pd.concat([all_train, dt])
            del dt
        all_train = all_train.drop_duplicates()  # 去重
        feat_columns = []
        if len(list_feat) > 0:
            feat_columns = list_feat
        if len(feat_columns) > 0:
            all_train = all_train[feat_columns]
        print("读取click特征结束")
        return all_train
    except Exception as e:
        print(e)


def get_user_feat(files, names=['user_id', 'item_id', 'time'], num=1,  list_feat=[]):
    """
    产生用户累计特征
    :param files:
    :param names:
    :param num:
    :param list_feat:
    :return:
    """
    train_click_data = get_train_test_click_data(files, names=names, num=num, list_feat=list_feat)
    # 用户出现次数
    train_click_data["user_id_showup_count"] = train_click_data['user_id'].map(train_click_data.groupby('user_id').size())
    # 用户点击的不同item不重复的个数
    train_click_data["user_id_item_unique"] = train_click_data['user_id'].map(train_click_data.groupby('user_id')["item_id"].nunique())
    # 用户对商品的点击次数
    user_item_count = train_click_data.groupby(['user_id', "item_id"]).size().reset_index()
    user_item_count.rename(columns={0: "user_id_item_count"}, inplace=True)
    train_click_data = pd.merge(train_click_data, user_item_count, how='left', on=['user_id', "item_id"])

    # 获取不同时间长度的时间序列
    for i in range(5, 14):
        col = "time_top_{0}".format(i)
        train_click_data[col] = train_click_data['time'].apply(lambda x: str(x)[:i])
    #不同时间序列下用户点击的次数
    for i in range(5, 14):
        col = "time_top_{0}".format(i)
        test_click_n = train_click_data.groupby(['user_id', col]).size().reset_index()
        col_count = col + "_user_clickcount"
        test_click_n.rename(columns={0: col_count}, inplace=True)
        train_click_data = pd.merge(train_click_data, test_click_n, how='left', on=['user_id', col])
    for i in range(5, 14):
        col = "time_top_{0}".format(i)
        test_click_n = train_click_data.groupby(['item_id', col]).size().reset_index()
        col_count = col + "_item_clickcount"

        test_click_n.rename(columns={0: col_count}, inplace=True)
        train_click_data = pd.merge(train_click_data, test_click_n, how='left', on=['item_id', col])
    return train_click_data

def get_item_feat(files, names=['user_id', 'item_id', 'time'], num=1,  list_feat=[]):
    """
    获取物品累计特征
    :param files:
    :param names:
    :param num:
    :param list_feat:
    :return:
    """
    #获取基础数据
    item_click_data = get_train_test_click_data(train_path_click, names=['user_id', 'item_id', 'time'], num=7,list_feat=[])
    # 获取不同时间长度的时间序列
    feat_col=names
    for i in range(5, 14):
        col = "time_top_{0}".format(i)
        item_click_data[col] = item_click_data['time'].apply(lambda x: str(x)[:i])
    # 不同时间序列下物品被点击的次数
    for i in range(5, 14):
        col = "time_top_{0}".format(i)
        test_click_n = item_click_data.groupby(['item_id', col]).size().reset_index()
        col_count = col + "_item_clickcount"
        feat_col.append(col_count)
        test_click_n.rename(columns={0: col_count}, inplace=True)
        item_click_data = pd.merge(item_click_data, test_click_n, how='left', on=['item_id', col])
    item_click_data=item_click_data[feat_col]
    return item_click_data



def  merage_feat(files, names=['user_id', 'item_id', 'time'], num=1,  list_feat=[]):
    """
     获取用户的合并特征
    :param files:
    :param names:
    :param num:
    :param list_feat:
    :return:
    """
    #获取用户的累计特征
    train_click_data = get_user_feat(files, names=names, num=num,  list_feat=list_feat)
    #获取物品基础特征
    item_feat = user_item_base_feat(train_path_item, item_feat_cols)
    ##找出click中存在于基础物品特征中的数据
    train_click_data = train_click_data[train_click_data["item_id"].isin(item_feat["item_id"])]
    train_click_data = pd.merge(train_click_data, item_feat, on="item_id", how="left")

    return train_click_data









# # 对进行特征工程的特征 进行归一化
# #train_click_merge_item[user_feat_list] = MinMaxScaler().fit_transform(train_click_merge_item[user_feat_list])
# numberic_feat_list+=user_feat_list
# cat_feat_list+=["item_id","user_id",'text_cat','img_cat']
# embeding_feat_list+=['text_vd' + str(i) for i in range(128)]+['img_vd' + str(i) for i in range(128)]
# pd.DataFrame(embeding_feat_list,columns = ["embeding_feature"]).to_csv(feat_columns_path+"embeding_cols_selected.csv")
# pd.DataFrame(cat_feat_list,columns = ["cat_feature"]).to_csv(feat_columns_path+"cat_cols_selected.csv")
# pd.DataFrame(numberic_feat_list,columns = ["num_feature"]).to_csv(feat_columns_path+"num_cols_selected.csv")
# pd.DataFrame(user_feat_list,columns = ["user_feature"]).to_csv(feat_columns_path+"user_cols_selected.csv")
# pd.DataFrame(item_feat_list,columns = ["item_feature"]).to_csv(feat_columns_path+"item_cols_selected.csv")
#
#
# pd.DataFrame(train_click_merge_item).to_csv(feat_columns_path+"train.csv")
#
