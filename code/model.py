from itertools import chain
import tensorflow as tf

from deepctr.inputs import input_from_feature_columns, get_linear_logit, build_input_features, combined_dnn_input, DEFAULT_GROUP_NAME
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.layers.utils import concat_func, add_func
from tensorflow.python.keras.regularizers import l2

def DeepFM(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary',use_image=False,use_text=False,embedding_size=128 ):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    train_path = '../data/underexpose_train'
    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        init_std, seed, support_group=True)

    if use_image:
        video_input = tf.keras.layers.Input(shape=(128,), name='image')
        video_emb = tf.keras.layers.Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg_embedding))(
            video_input)
        video_emb = tf.keras.layers.Reshape((1, embedding_size), input_shape=(embedding_size,))(video_emb)
        group_embedding_dict[DEFAULT_GROUP_NAME].append(video_emb)
        inputs_list.append(video_input)

    if use_text:
        audio_input = tf.keras.layers.Input(shape=(128,), name='text')
        audio_emb = tf.keras.layers.Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg_embedding))(
            audio_input)
        audio_emb = tf.keras.layers.Reshape((1, embedding_size), input_shape=(embedding_size,))(audio_emb)
        group_embedding_dict[DEFAULT_GROUP_NAME].append(audio_emb)
        inputs_list.append(audio_input)

    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)
    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model