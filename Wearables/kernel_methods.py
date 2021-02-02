import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from gensim import utils
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import List



class W2VKernel:

    targets = ['A','B','C','M','O','R']

    def __init__(self,n_clusters=30):
        self.n_clusters = n_clusters

    def add_cluster_features(self,model, cluster,cluster_to_word):

        word = cluster_to_word[int(cluster)]
        return model.wv[word]

    def classify_feats_kernel(self,features,b,xp,xn, K):
        """kernel version"""

        mp = np.shape(xp)[0]
        mn = np.shape(xn)[0]

        decisions = []
        for x in features:
            g = b
            for xi in xp:
                g += 1 / mp * K(x, xi)
            for xi in xn:
                g -= 1 / mn * K(x, xi)
            decisions.append(int(np.sign(g) * 0.5 + 0.5))
        return np.array(decisions)

    def convert_binary_class(self,response, target):

        binary = np.zeros(len(response))
        binary[np.where(np.isin(response, target))] = 1
        return binary

    def convert_cluster_word(self,sentences: List,cluster_to_word:dict):
        new_sentences = []
        for s in sentences:
            sn = ''
            for w in s.split():
                # print(int(w))
                sn += cluster_to_word[int(w)] + ' '
            new_sentences.append(sn.strip())
        return new_sentences

    def create_word_sentences(self,data):
        """
        Create a dictionary of "word: vector"
        :return:
        """
        dostoy = 'Above all dont lie to yourself. The man who lies to himself and listens to his own lie comes to a point that he cannot distinguish the truth within him, or around him, and so loses all respect for himself and for others. And having no respect he ceases to love'

        train_text = self.pd_to_sentence(data, self.targets)

        sentences = list(train_text['text'])

        text = train_text['text'].str.cat(sep=' ')

        vocab = list(set(text.split()))

        def clean_word(w):
            return w.lower().strip().replace('.', '').replace(',', '')

        dostoy_vocab = list(set([clean_word(w) for w in dostoy.split()]))

        cluster_to_word = {int(c): dostoy_vocab[int(c)] for c in vocab}

        word_sentences = self.convert_cluster_word(sentences,cluster_to_word)

        word_sentences = [s.split() for s in word_sentences]

        return word_sentences, cluster_to_word

    def mean_score(self,pred, labels):
        n = pred.shape[0]
        score = 0.0
        for i in range(n):
            if pred[i] == labels[i]:
                score += 1

        return score / n

    def pd_to_sentence(self,df, target):
        sentences = []
        activities = list(set(df['activity']))
        for a in activities:
            sub = df[df['activity'] == a]
            sentences.append([' '.join(list(sub['cluster'].apply(lambda x: str(x)))), a, 1 if (a in target) else 0])
        return pd.DataFrame(sentences, columns=['text', 'activity', 'target'])

    def run_kmeans(self,X):

        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(X)
        return kmeans

    def run_w2v(self,sentences):

        model = Word2Vec(sentences=sentences, size=100, window=5, min_count=1, workers=4)

        return model

    def run(self,X,y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=X['subject'])

        print("Train and test shape")
        print(X_train.shape)
        print(X_test.shape)

        #TRAIN K means

        kmeans = self.run_kmeans(X_train[['x','y','z']].to_numpy())

        X_train['cluster'] = kmeans.labels_

        # Create Sentences from clusters
        sentences, cluster_to_word = self.create_word_sentences(X_train)

        # run w2v
        w2v_model = self.run_w2v(sentences)

        # extract features
        X_train['c_feature'] = X_train['cluster'].apply(lambda x: self.add_cluster_features(w2v_model, x,cluster_to_word))

        #X_train['target'] = y_train

        # LR base model
        subjects = set(X['subject'].tolist())
        lr_models = {f'model_{id}':None for id in subjects}
        vornoi_models = {f'model_{id}':() for id in subjects}

        lr_scores = []
        vornoi_scores = []
        for id in subjects:

            print(f'Training for subject = {id}')

            X_id, y_id = X_train[X_train['subject'] == id], y_train[y_train['subject'] == id]['activity']

            y_id = self.convert_binary_class(y_id, self.targets)

            features = X_id['c_feature'].to_numpy()

            X_features = np.array([f for f in features])

            feature_lr = LogisticRegression()

            feature_lr.fit(X_features,y_id)

            print('Logistic in sample score')
            score = feature_lr.score(X_features,y_id)
            print(score)
            lr_scores.append(score)
            lr_models[f'model_{id}'] = feature_lr

            K = lambda x, xi: np.dot(x, xi)

            b,xp,xn = self.train_volnoi(X_features,y_id,K)

            train_predict = self.classify_feats_kernel(X_features,b,xp,xn,K)

            train_score = self.mean_score(train_predict,y_id)
            print('Volnoi in sample score')
            print(train_score)
            vornoi_scores.append(train_score)

            vornoi_models[f'model_{id}'] = (b,xp,xn,K)

        print("Mean Train scores")
        print("LR")
        print(sum(lr_scores) / len(lr_scores))
        print("Vornoi")
        print(sum(vornoi_scores) / len(vornoi_scores))

        #TEST


        X_test['cluster'] = kmeans.predict(X_test[['x', 'y', 'x']].to_numpy())

        sentences, cluster_to_word = self.create_word_sentences(X_test)

        w2v_model = self.run_w2v(sentences)

        X_test['c_feature'] = X_test['cluster'].apply(
            lambda x: self.add_cluster_features(w2v_model, x, cluster_to_word))

        #X_test['target'] = y_test

        lr_test_scores = []
        vornoi_test_scores = []

        for id in subjects:

            X_id, y_id = X_test[X_test['subject'] == id], y_test[y_test['subject'] == id]['activity']

            y_id = self.convert_binary_class(y_id, self.targets)

            features = X_id['c_feature'].to_numpy()

            X_features = np.array([f for f in features])

            feature_lr = lr_models[f'model_{id}']
            print('Logistic in sample score')
            lr_score = feature_lr.score(X_features, y_id)
            lr_test_scores.append(lr_score)
            print(lr_score)

            b,xp,xn,K = vornoi_models[f'model_{id}']

            test_predict = self.classify_feats_kernel(X_features, b, xp, xn, K)

            test_score = self.mean_score(test_predict, y_id)

            print('Volnoi in sample score')
            print(test_score)
            vornoi_test_scores.append(test_score)

        print("Mean Test scores")
        print("LR")
        print(sum(lr_test_scores) / len(lr_test_scores))
        print("Vornoi")
        print(sum(vornoi_test_scores) / len(vornoi_test_scores))

    def train_volnoi(self,X,y,K):

        X, Y = X.copy(), y.copy()
        Y[Y == 0] = -1
        xp = X[Y == 1, :]
        xn = X[Y == -1, :]

        # lengths for positive and negative training data
        mp = np.shape(xp)[0]
        mn = np.shape(xn)[0]

        # combute b
        b = 0
        total_calcs = np.shape(xp)[0] ** 2 + np.shape(xn)[0] ** 2
        for xi in xp:
            for xj in xp:
                b -= 1 / (mp ** 2) * K(xi, xj)
        for xi in xn:
            for xj in xn:
                b += 1 / (mn ** 2) * K(xi, xj)
        b *= 0.5
        b = b

        return b,xp,xn






