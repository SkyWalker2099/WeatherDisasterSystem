import tensorflow as tf
import os
# 用于削减bert预训练模型中的隐藏层层数
config_path = 'chinese_L-12_H-768_A-12\\bert_config2.json'
checkpoint_path = 'chinese_L-12_H-768_A-12\\bert_model.ckpt'
vocab_path = 'chinese_L-12_H-768_A-12\\vocab.txt'

sess = tf.Session()
last_name = 'bert_model.ckpt'
model_path = 'chinese_L-12_H-768_A-12'
imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
imported_meta.restore(sess, os.path.join(model_path, last_name))
init_op = tf.local_variables_initializer()
sess.run(init_op)

bert_dict = {}


v2v_dict = {'bert/encoder/layer_0/': 'bert/encoder/layer_0/',
            'bert/encoder/layer_1/': 'bert/encoder/layer_1/',
            'bert/encoder/layer_2/': 'bert/encoder/layer_3/',
            'bert/encoder/layer_3/': 'bert/encoder/layer_5/',
            'bert/encoder/layer_4/': 'bert/encoder/layer_7/',
            'bert/encoder/layer_5/': 'bert/encoder/layer_8/',
            'bert/encoder/layer_6/': 'bert/encoder/layer_9/',
            'bert/encoder/layer_7/': 'bert/encoder/layer_10/',
            'bert/encoder/layer_8/': 'bert/encoder/layer_11/'}



print(v2v_dict)

for var in tf.global_variables():
    bert_dict[var.name] = sess.run(var).tolist()


# for var in tf.global_variables():
#     if var.name.startswith('bert/encoder/layer_') and not var.name.startswith(
#             'bert/encoder/layer_0/') and not var.name.startswith('bert/encoder/layer_1/'):
#         pass
#     elif var.name.startswith('bert/encoder/layer_1/'):
#         # 寻找11层的var name，将11层的参数给第一层使用
#         new_name = var.name.replace("bert/encoder/layer_1", "bert/encoder/layer_11")
#         op = tf.assign(var, bert_dict[new_name])
#         sess.run(op)
#         need_vars.append(var)
#         print(var)
#     else:
#         need_vars.append(var)
#         print('####',var)


need_vars = []
for var in tf.global_variables():
    if var.name.startswith('bert/encoder/layer_'):
        for n in v2v_dict.keys():
            if var.name.startswith(n):
                new_name = var.name.replace(n, v2v_dict[n])
                op = tf.assign(var,bert_dict[new_name])
                sess.run(op)
                need_vars.append(var)
                print(var,new_name)
                break
    else:
        need_vars.append(var)
        print("###",var)

saver = tf.train.Saver(need_vars)
saver.save(sess, os.path.join('chinese_L-12_H-768_A-12_pruned', 'bert_pruning_9_layer.ckpt'))


