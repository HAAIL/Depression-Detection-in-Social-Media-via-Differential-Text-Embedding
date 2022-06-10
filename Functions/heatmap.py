import pandas as pd
import numpy as np
from datas import Dataset
import re
import scipy
import logging
from typing import  List, AnyStr
from sentence_transformers import SentenceTransformer
import time
from pandas import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SentenceSimilarity2():
    def __init__(self, dataset: Dataset, model: SentenceTransformer = None, n_docs: int = -1):
        self.dataset = dataset
        self.model = model if model else SentenceTransformer("bert-base-nli-stsb-mean-tokens")
        print("embedder loaded...")
        self.sentences = []
        self.doc_id_to_sentence_ids = {}

        self.sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        for d in dataset.get_documents(n=n_docs):
            doc_id = d.get('id')
            text = d.get('text', None)

            sentence_ids = []
            if text:
                sentences = re.split(self.sentence_pattern, text)
                for s in sentences:
                    sentence_ids.append(len(self.sentences))
                    self.sentences.append(s)

            # Map from document to all its sentences (One-to-Many)
            self.doc_id_to_sentence_ids[doc_id] = sentence_ids
        # end for

        logger.debug(f"doc_to_sentence_ids: {self.doc_id_to_sentence_ids}")

        # Map from sentence to the document it came from (Many-to-One)
        self.sentence_id_to_doc_id = {}
        for doc_id, sentence_ids in self.doc_id_to_sentence_ids.items():
            for s_id in sentence_ids:
                self.sentence_id_to_doc_id[s_id] = doc_id

        logger.debug(f"sentence_id_to_doc_id: {self.sentence_id_to_doc_id}")
        # Embedd extracted sentences using SentenceTransformer model.
        start = time.time()

        self.embedded_sentences = self.model.encode(self.sentences)
        logger.info(f"It took {round(time.time() - start, 3)} s to embedd {len(self.sentences)} sentences.")

        # temp  = []
        # for doc_id, sentence_ids in self.doc_id_to_sentence_ids.items():
        #     logger.debug((doc_id, sentence_ids))
        #     temp.append(f"document: {self.dataset.get_documents_by_id([doc_id])} - sentences: {[self.sentences[sid] for sid in sentence_ids]}")
        # logger.debug("\n\n".join(temp))

    # end def

    def get_most_similar(self, query: AnyStr, threshold: float = 1, limit: int = 63) -> List[int]:
        query_sentences = re.split(self.sentence_pattern, query)
        query_embeddings = self.model.encode(query_sentences)

        # logger.info(f"Extracted {len(query_sentences)} sentences from query")
        logger.debug(f"Sentences: {' -- '.join(query_sentences)}")

        # Calculate cosine distance between requested sentences and all sentences
        cosine_dist = scipy.spatial.distance.cdist(query_embeddings, self.embedded_sentences, "cosine")
        # Extract column values where distance is below threshold
        below_threshold = cosine_dist < threshold
        doc_ids, matched_column_ids = np.where(below_threshold)

        # Extract x (input sentence id), y (dataset sentence id) and distance between these.
        x_y_dist = []
        for x, y in zip(doc_ids, matched_column_ids):
            x_y_dist.append([x, y, cosine_dist[x][y]])

        # Sort list based on distance and remove duplicates, keeping the one with lowest distance.
        sorted_x_y_dist = sorted(x_y_dist, key=lambda x: x[2])
        sorted_sentence_ids = [doc[1] for doc in sorted_x_y_dist]
        sorted_doc_ids = [self.sentence_id_to_doc_id[sent_id] for sent_id in sorted_sentence_ids]
        top_similar_dict = {}
        top_similar = []
        top_dist = []
        # logger.info(f"Distance for top documents: {[round(x[2],3) for x in sorted_x_y_dist[:limit]]}")

        top_dist              = [round(x[2], 3) for x in sorted_x_y_dist[:limit]]
        top_similar           = self.dataset.get_documents_by_id(list(dict.fromkeys(sorted_doc_ids).keys())[:limit])
        # top_similar_sentences = self.dataset.get_documents_list(list(dict.fromkeys(sorted_doc_ids).keys())[:limit])
        # print(sorted_sentence_ids)
        if (len(top_dist) != len(top_similar)):
            print(len(top_dist))
            print(len(top_similar))
        anchors_ids = []
        for i in range(len(top_dist)):
            top_similar_dict[top_similar[i]] = top_dist[i]

        return top_similar_dict, self.embedded_sentences, query_embeddings, sorted_doc_ids


def compute_perfomance(num_cat,num_ex,dysfunc_dataset):
    title  = str(num_cat) + str(num_ex)

    sub_df          = dysfunc_dataset[(dysfunc_dataset['category id']<=num_cat) & (dysfunc_dataset['example subset'] <= num_ex)]
    sub_df = sub_df[['Category', 'Example', 'ID']]
    sub_df_examples = sub_df['Example']

    sub_df.to_excel(title + '-thoughts_categories.xlsx',  index=False)
    sub_df.to_csv(title + '-thoughts_categories.csv',  index=False)

    textfile = open("%s-dysfunctional_thoughts_examples.txt" % title, "w", encoding='utf8')
    for element in sub_df_examples:
        textfile.write(element + "\n")
    textfile.close()
    data_ex = Dataset('%s-dysfunctional_thoughts_examples.txt' % title)
    sentence_sim = SentenceSimilarity2(data_ex)
    emb_examples    = sentence_sim.embedded_sentences
    emb_examples_df = pd.DataFrame.from_records(emb_examples)
    emb_examples_df.to_csv(title + '-examples-emb.csv', index=False)
    df1 = pd.read_csv(title + '-examples-emb.csv')

    thoughts_dict = {}
    xls = ExcelFile(title + '-thoughts_categories.xlsx')
    df = xls.parse(xls.sheet_names[0])
    df.columns = ['Category', 'Example', 'ID']
    columns = list(df)
    for r in range(0, df.shape[0]):
        for i in columns:
            if i == 'Category':
                if df[i][r] not in thoughts_dict.keys():
                    thoughts_dict[df[i][r]] = [df['Example'][r]]
                    break
                else:
                    thoughts_dict[df[i][r]].append(df['Example'][r])
                    break
            else:
                break



dt = pd.read_csv('thoughts_categories.csv')
compute_perfomance(2,3,dt)