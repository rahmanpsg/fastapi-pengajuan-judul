import json
from config.database import db
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2csc
from gensim.models.tfidfmodel import TfidfModel
from gensim import similarities
from utils.cleaner import text_cleaner
import re


class KNN():
    def __init__(self):
        self.loadFromFirestore()
        self.cleanDocument()
        self.createTFIDFModel()
        # self.calculateTFIDF()

    def loadFromFirestore(self):
        print("load data from firestore")
        juduls = list(db.collection(u'juduls').get())

        juduls_dict = list(map(lambda x: x.to_dict(), juduls))

        # save juduls_dict to json
        # with open('juduls_dict.json', 'w') as f:
        #     json.dump(juduls_dict, f)

        self.df = pd.DataFrame(juduls_dict, columns=['judul'])

        print(self.df.judul.values)

        print("total data loaded :", self.df.shape)

    def loadFromJson(self):
        print("load data from json")

        with open('juduls_dict.json', 'r') as f:
            juduls_dict = json.load(f)

        self.df = pd.DataFrame(juduls_dict)

        print("total data loaded :", self.df.shape)

    def cleanDocument(self):
        self.X = pd.DataFrame(self.df.judul.values, columns=['text'])        

        self.document_cleaned = self.X.text.dropna().reset_index(drop=True)
        self.document_cleaned = self.document_cleaned.apply(
            lambda x: text_cleaner(x).split())        

        print("document cleaned")

    def createTFIDFModel(self):
        self.dictionary = Dictionary(self.document_cleaned)
        self.num_docs = self.dictionary.num_docs
        self.num_terms = len(self.dictionary.keys())

        corpus_bow = [self.dictionary.doc2bow(
            doc) for doc in self.document_cleaned]

        self.tfidf = TfidfModel(corpus_bow)
        self.corpus_tfidf = self.tfidf[corpus_bow]

        self.corpus_tfidf_sparse = corpus2csc(
            self.corpus_tfidf, self.num_terms, num_docs=self.num_docs).T

        print("TFIDF is created")

        print(self.tfidf.num_docs)

        print(self.corpus_tfidf_sparse.shape)

        print(self.corpus_tfidf_sparse[0])
    
    def proses(self, teks, k):
        self.model = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        self.model.fit(X=self.corpus_tfidf_sparse)

        test_dokumen = pd.DataFrame({"dokumen": [teks]})
        test_dokumen = test_dokumen.dokumen.dropna().reset_index(drop=True)
        test_dokumen = test_dokumen.apply(lambda x: text_cleaner(x).split())

        print(test_dokumen)

        # test corpus from created dictionary
        test_corpus_bow = [self.dictionary.doc2bow(
            doc) for doc in test_dokumen]

        # test tfidf values from created tfidf model
        test_corpus_tfidf = self.tfidf[test_corpus_bow]        

        # test sparse matrix
        test_corpus_tfidf_sparse = corpus2csc(
            test_corpus_tfidf, self.num_terms).T

        print(test_corpus_tfidf_sparse)

        distances, indices = self.model.kneighbors(test_corpus_tfidf_sparse)

        index = similarities.SparseMatrixSimilarity(self.corpus_tfidf, num_terms=self.num_terms)

        simi = index[test_corpus_tfidf]

        df_hasil = self.df.loc[indices[0]]
        df_hasil['jarak'] = distances[0]
        df_hasil['similarity'] = simi[0][[indices[0]]]

        hasil_tfidf = []

        for doc in self.document_cleaned.loc[indices[0]]:

            text = ' '.join(doc)

            # Memecah setiap kata
            keywords = re.findall(r'[a-zA-Z]\w+', text)

            df = pd.DataFrame(list(set(keywords)),
                              columns=['keyword'])            

            df['count'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[0])
            df['tf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[1])
            df['idf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[2])
            df['tf_idf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[3])

            # remove index where keyword not in list
            df = df.drop(
                df[df.keyword.isin(test_dokumen.values[0]) == False].index)

            # add keyword if not in list
            for keyword in test_dokumen.values[0]:
                if keyword not in df.keyword.values:
                    df = df.append(
                        {'keyword': keyword, 'count': 0, 'tf': 0, 'idf': 0, 'tf_idf': 0}, ignore_index=True)

            # print(df)

            hasil_tfidf.append(df.to_dict(orient='records'))           

        # print(hasil_tfidf)
        df_hasil['hasil_tfidf'] = hasil_tfidf

        return df_hasil.to_dict(orient='records')

    def weightage(self, word, text, number_of_documents=1):
        word_list = re.findall(word, text)
        number_of_times_word_appeared = len(word_list)
        tf = number_of_times_word_appeared/float(len(text))
        idf = np.log((number_of_documents) /
                     float(number_of_times_word_appeared))
        tf_idf = tf*idf
        return number_of_times_word_appeared, tf, idf, tf_idf

    def calculateTFIDF(self):       
        response = []
        for doc in self.document_cleaned:
            # text = ' '.join(self.document_cleaned[0])
            text = ' '.join(doc)

            # Memecah setiap kata
            keywords = re.findall(r'[a-zA-Z]\w+', text)

            df = pd.DataFrame(list(set(keywords)),
                              columns=['keyword'])

            df['count'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[0])
            df['tf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[1])
            df['idf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[2])
            df['tf_idf'] = df['keyword'].apply(
                lambda x: self.weightage(x, text, self.tfidf.num_docs)[3])

            response.append(df.to_dict('records'))
            print('oke', len(response))

        return response