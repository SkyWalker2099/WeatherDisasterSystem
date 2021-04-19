import kashgari
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiGRU_CRF_Model
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
kashgari.config.use_cudnn_cell = True

def load_data(csv_path):
    all_data = pd.read_csv(csv_path)

    X = all_data["line"]
    Y = all_data["tags"]

    X = [[i for i in x] for x in X]
    Y = [[j for j in y] for y in Y]

    return X,Y

x_train,y_train  = load_data("DATA\\train_data.csv")

print(len(x_train),len(y_train))

print("loading bert embedding")
bert_embedding = BERTEmbedding('chinese_L-12_H-768_A-12',task=kashgari.LABELING,
                               sequence_length=200,trainable=False)

print("loading BiGRU crf model")
bigru_crf_model = BiGRU_CRF_Model(bert_embedding)

bigru_crf_model.fit(x_train=x_train,y_train=y_train,batch_size=16,epochs=5)
#
# bigru_crf_model.save("bert_bigru_crf2.h5")



