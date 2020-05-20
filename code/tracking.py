from model import DeepFM
import pandas as pd
from deepctr.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder
feat_columns_path = "../data/features_colums/"
train_path = '../data/underexpose_train'
train_path_user=train_path+'/underexpose_user_feat.csv'
train_path_item=train_path+'/underexpose_item_feat.csv'
item_feat_cols = ["item_id"]+['text_vd' + str(i) for i in range(128)]+['img_vd' + str(i) for i in range(128)]




def get_input(use_img=True, use_text=True, target='isclick'):

    sequence_feature_list = []
    sparse_feature_df=pd.read_csv(feat_columns_path+"cat_cols_selected.csv")
    dense_feature_df=pd.read_csv(feat_columns_path+"num_cols_selected.csv")
    emb_feat_list=pd.read_csv(feat_columns_path+"embeding_cols_selected.csv")["embeding_feature"].values.tolist()
    user_cols_list=pd.read_csv(feat_columns_path+"user_cols_selected.csv")["user_feature"].values.tolist()
    item_cols_list=pd.read_csv(feat_columns_path+"item_cols_selected.csv")["item_feature"].values.tolist()

    # 用户信息表
    train_user_feat_df = pd.read_csv(train_path_user)
    # item 信息表
    train_item_feat_df = pd.read_csv(train_path_item, names=item_feat_cols)

    cat_feature_list=sparse_feature_df["cat_feature"].values.tolist()
    num_feature_list=list(set(dense_feature_df["num_feature"].values.tolist()))
    #类别聚类还没有跑 这里是手动写的，之后这个删掉
    cat_feature_list=["user_id","item_id"]



    data=pd.read_csv(feat_columns_path+"train.csv").iloc[:-1000]
    data["isclick"]=1

    for missing_col in data.columns.tolist():
        if missing_col in num_feature_list:
            data[missing_col].fillna(data[missing_col].median(), inplace=True)
        elif missing_col in ['text_vd' + str(i) for i in range(128)]+['text_vd' + str(i) for i in range(128)]:
            data[missing_col].fillna(0,inplace=True)
            data[missing_col]= data[missing_col].apply(lambda x : 0 if x=="nan" or x=="null" else x)
    data[cat_feature_list] = data[cat_feature_list].apply(LabelEncoder().fit_transform)
    # data[sparse_feature_list].fillna(-1)
    feature_columns=[]
    sparse_feature_list=[SparseFeat(cat_col, data[cat_col].nunique(), embedding_dim=10) for cat_col in cat_feature_list ]
    dense_feature_list = [DenseFeat(colname , 1) for colname in num_feature_list]
    sequence_feature_list=[]
    feature_columns=sparse_feature_list+dense_feature_list+sequence_feature_list

    test=data.iloc[-1000:]
    train=data.iloc[:-1000]
    train_size=len(train)


    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
        [train[feat.name].values for feat in dense_feature_list]

    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list]

    if use_img:
        ad_cols = ['img_vd' + str(i) for i in range(128)]
        img_input = data[ad_cols].values
        train_model_input += [img_input[:train_size]]
        test_model_input += [img_input[train_size:]]
    if use_text:
        vd_cols = ['text_vd' + str(i) for i in range(128)]
        text_input = data[vd_cols].values
        train_model_input += [text_input[:train_size]]
        test_model_input += [text_input[train_size:]]





    train_labels, test_labels = train[target].values, test[target].values
    feature_dim_dict = {"sparse": sparse_feature_list, "dense": dense_feature_list, "sequence": sequence_feature_list}


    return feature_columns, train_model_input, train_labels, test_model_input, test_labels

def main():
    input_params = {

        'use_text': True,  # 是否使用text
        'use_img': True,  # 是否使用img



        'target': 'isclick'  # 预测目标
    }
    feature_columns, train_model_input, train_labels, test_model_input, test_labels = get_input(**input_params)
    iterations = 10  # 跑多次取平均
    for i in range(iterations):
        print(f'iteration {i + 1}/{iterations}')

        model = DeepFM(feature_columns,feature_columns, use_image=input_params["use_img"],use_text=input_params["use_text"]
                       ,embedding_size=10)
        model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy"])

        history = model.fit(train_model_input, train_labels, batch_size=4096, epochs=1, verbose=1,
                            validation_data=(test_model_input, test_labels))

if __name__ == '__main__':
    main()