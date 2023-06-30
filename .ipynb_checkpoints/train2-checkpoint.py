import numpy as np
import csv
from layers.recurrent import BatchLSTM
from layers.pooling import MeanPooling
from layers.core import Softmax,Flatten
from optimizers import Adam
from model import Model
import re


#数据集处理类
class KaggleDataset:
    
    def __init__(self):
        pass
        
    def onehot(array):
        # Determine number of unique values in array
        num_classes = np.unique(array).size
        # Create an empty one-hot matrix of shape (num_classes, arr.size)
        one_hot = np.zeros((num_classes, array.size))
        # Set the corresponding element in each column to 1
        one_hot[array, np.arange(array.size)] = 1
        # Transpose the result to get a matrix of shape (arr.size, num_classes)
        one_hot = one_hot.T
        return one_hot
    
    #比例0.8划分训练集，验证集
    def ratio_spite(ratio,X,y):
        size = X.shape[0]

        # Determine the index at which to split the arrays
        split_idx = int(size * ratio)

        # Split the arrays
        x_train, x_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return x_train, y_train,x_val,y_val
    
    #加载数据集方法
    def load_dataset(dataset_path):
        data=[]
        with open(dataset_path, 'r') as f:
            data=list(csv.reader(f)) # 使用csv.reader将文件读取进来，并转为list形式，方便后续处理
        f.close()
        data =np.array(data)
        X=data[:,0]
        y=list(data[:,1])
        # y=data[:,1]
        return X,y
    
    def load_dataset_predict(dataset_path):
        data=[]
        with open(dataset_path, 'r') as f:
            data=list(csv.reader(f)) # 使用csv.reader将文件读取进来，并转为list形式，方便后续处理
        f.close()
        data =np.array(data)
        X=data[:,1]
        
        # y=data[:,1]
        return X
    
    
    #分词方法
    def tokenlize(sentences_array):
        #加载停用词表
        with open('data/enshort_stopwords.txt', 'r') as f:
            stopwords_list = f.read().splitlines()
        stopwords_arr = np.array(stopwords_list)

        
        # Define regular expression pattern to match punctuation and special characters
        pattern = r'[^\w\s]'

        # Define function to preprocess text
        
            # Tokenize the text into words
        words = sentences_array.lower().split()
        # Remove stopwords, punctuation and special characters
        words = [word for word in words if word not in stopwords_arr]
        words = [re.sub(pattern, '', word) for word in words]
        words = [word for word in words if word]

        # Join the words back into a single string
        processed_text = ' '.join(words)
        processed_text=processed_text.split()
        return processed_text
    
    
    
    
    def load_embedding(embedding_file):
        embeddings_index = {}
        f = open(embedding_file, 'r', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index


    def sentence_embedding(sentence, embeddings_index, embedding_dim, max_len):
        # 自动填充数值0到max_len
        embedding_matrix = np.zeros((max_len, embedding_dim))
        for i, word in enumerate(sentence):
            if i >= max_len:
                break
            
            #如果单词在embeddings_index中，则使用嵌入词向量，否则词向量全0
            if word in embeddings_index:
                embedding_matrix[i] = embeddings_index[word]
        return embedding_matrix
        
    def all_embedding(X, embeddings_index, embedding_dim, max_len):
        embedding_X=[]
        for sentence in X:
            embedding_X.append(KaggleDataset.sentence_embedding(sentence, embeddings_index, embedding_dim, max_len))
        
        return np.array(embedding_X)





#分词方法
def tokenlize(sentences_array,stopwords_arr):
    #加载停用词表
    
    # Define regular expression pattern to match punctuation and special characters
    pattern = r'[^\w\s]'

    # Define function to preprocess text
    
        # Tokenize the text into words
    words = sentences_array.lower().split()
    # Remove stopwords, punctuation and special characters
    words = [word for word in words if word not in stopwords_arr]
    words = [re.sub(pattern, '', word) for word in words]
    words = [word for word in words if word]

    # Join the words back into a single string
    processed_text = ' '.join(words)
    processed_text=processed_text.split()
    return processed_text

#数据预处理和embedding
def prepare_data(X,y):
    #1 分词
    X = [KaggleDataset.tokenlize(text) for text in X]
    # print(X[1:3])
    #2 embedding
    #加载词向量文件
    embedding_file = 'data/glove.6B.200d.txt' # GloVe预训练词向量模型文件路径
    #建立词库
    embeddings_index = KaggleDataset.load_embedding(embedding_file)

    #超参数词向量维度
    embedding_dim = 200 # 词向量维度
    max_len = 20 # 句子最大长度

    embedding_X= KaggleDataset.all_embedding(X, embeddings_index, embedding_dim, max_len)
    #(156060, 20, 50) 156060样本数  10句子长度(10个token),100嵌入词向量维度

    #3 y转onehot
    y=np.array(y,int)  #(156060,)
    y=KaggleDataset.onehot(y)   #(156060, 5)  

    return embedding_X,y

#训练
def train(max_iter,net,embed_X,y_hot):
    nb_batch = 30
    nb_seq = 20    #序列长度

    net = Model()
    net.add(BatchLSTM(n_out=50,n_in=embed_X.shape[2], nb_batch=nb_batch, nb_seq=nb_seq,return_sequence=True)) 
    #(30, 20, 25)
    net.add(BatchLSTM(n_out=25, return_sequence=True))
    net.add(MeanPooling((20, 1)))     #向量维度从50->20  #(30, 1, 25)
    net.add(Flatten())  #(30, 25)
    net.add(Softmax(n_out=5)) #(30, 5)
    net.compile(loss='scce', optimizer=Adam())     #学习率0.001
    net.fit(embed_X, y_hot, batch_size=nb_batch, validation_split=0.2, max_iter=max_iter)
    
    return net

#预测函数
def predict(net):#模型预测
    X=KaggleDataset.load_dataset_predict('data/test.csv')
    # print(X[1:3])
    #step2 分词
    X = [KaggleDataset.tokenlize(text) for text in X]

    #embedding
    #加载词向量文件
    embedding_file = 'data/glove.6B.200d.txt' # GloVe预训练词向量模型文件路径
    #建立词库
    embeddings_index = KaggleDataset.load_embedding(embedding_file)

    #超参数词向量维度
    embedding_dim = 200 # 词向量维度
    max_len = 20 # 句子最大长度

    embedding_X= KaggleDataset.all_embedding(X, embeddings_index, embedding_dim, max_len)
    # print(embedding_X[1:3])
    
    # 使用模型进行预测
    pred = net.predict(embedding_X)
    
    result=[]
    for i in range(pred.shape[0]):
        # print(f"pred[i]-{pred[i]}")
        result.append(np.argmax(pred[i]))
    
    return result


if __name__ == '__main__':
    
    #step1,加载数据集
    X,y=KaggleDataset.load_dataset('data/train.csv')
    print(X.size)  # X是ndarray
    print(len(y))  #y是列表
    #数据预处理
    embed_X,y=prepare_data(X,y)
    #构建模型
    net = Model()
    #模型训练
    net=train(4,net,embed_X,y)
    result=predict(net)
    # 读取CSV文件
    with open('data/sampleSubmission.csv', mode='r') as csv_file:
        reader = csv.reader(csv_file)
        rows = [row for row in reader]

    # 在第二列的第二行及以后的行写入数字
    
    
    for i in range(1, len(result)):
        rows[i][1] = result[i-1]

    # 将修改后的数据写入CSV文件
    with open('data/result.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in rows:
            writer.writerow(row)
            
    
    print('done')
    
    
    
# 第13行至第17行：对IFOG的前3个门应用sigmoid非线性变换，将其限制在0到1之间，表示门的开关状态。
# 对最后一个门应用tanh非线性变换，将其映射到-1到1之间，表示候选单元状态的取值范围。对应的letex公式代码