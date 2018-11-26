import re
import os
import json
import spacy
import unicodedata
import numpy as np
import argparse
import collections
import multiprocessing
import logging
import random
import tqdm
import time
import pickle
from functools import partial
from collections import Counter
from my_utils.tokenizer import Vocabulary, reform_text, normalize_text, normal_query, END, build_vocab
from my_utils.word2vec_utils import load_emb_vocab, build_embedding
from my_utils.utils import set_environment
from my_utils.data_utils import build_data, gen_name
from config import set_args
from my_utils.log_wrapper import create_logger
"""
This script is to preproces SQuAD dataset.
"""

#Turn off
DEBUG_ON = False
DEBUG_SIZE = 2000

NLP = spacy.load('en', disable=['vectors', 'textcat', 'parser'])


def load_data2(path, is_train=True, v2_on=True):    
    '''The diffrence between this model is that we include even the plausable answers as the answer, so our system have more training examples'''
    with open(path , encoding="utf8") as f:
        data = json.load(f)['data']
    rows=[]
    i=0

    if v2_on is False or is_train is False:
        rows = load_data(path, is_train, v2_on)
        return(rows)

    else:
        for article in tqdm.tqdm(data, total=len(data)):
            for paragraph in article['paragraphs']:
                context=paragraph['context']
                context = '{} {}'.format(context, END)

                for qa in paragraph['qas']:
                    uid = qa['id']
                    question = qa['question']

                    if qa['is_impossible']: #Meaning if impossible(True) then we add the plausable answer as the answer.
                        answer= qa['plausible_answers']
                        label = 1 #When answer is False
                    else:
                        answer= qa['answers']
                        label = 0 #When the answer is Correct
        #             print(lable,answer)
                    if len(answer) != 0:
                        answer_text=answer[0]['text']
                        answer_start = answer[0]['answer_start']
                        answer_end = answer[0]['answer_start']+len(answer_text)
                    else: # They are all the plausable answers that sometimes doesn't have any answers.
                        answer_text=[]
                        answer_start=0
                        answer_end=0
                        i=i+1
                        
                    sample = {'uid':uid, 'context':context, 'question':question, 'answer':answer_text , 'answer_start':answer_start ,'answer_end':answer_end ,'label':label}
                    rows.append(sample)
    return(rows)


def load_data(path, is_train=True, v2_on=False):
    rows = []
    with open(path, encoding="utf8") as f:
        data = json.load(f)['data']
    for article in tqdm.tqdm(data, total=len(data)):
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if v2_on:
                context = '{} {}'.format(context, END)
            for qa in paragraph['qas']:
                uid, question = qa['id'], qa['question']
                answers = qa.get('answers', [])
                # used for v2.0
                is_impossible = qa.get('is_impossible', False)
                label = 1 if is_impossible else 0
                if is_train:
                    if (v2_on and label < 1 and len(answers) < 1) or ((not v2_on) and len(answers) < 1): continue
                    if len(answers) > 0:
                        answer = answers[0]['text']
                        answer_start = answers[0]['answer_start']
                        answer_end = answer_start + len(answer)
                        if v2_on:
                            sample = {'uid': uid, 'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'answer_end':answer_end, 'label': label}
                        else:
                            sample = {'uid': uid, 'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'answer_end':answer_end}
                    else:
                        answer = END
                        answer_start = len(context) - len(END)
                        answer_end = len(context)
                        sample = {'uid': uid, 'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'answer_end':answer_end, 'label': label}
                else:
                    sample = {'uid': uid, 'context': context, 'question': question, 'answer': answers, 'answer_start': -1, 'answer_end':-1}
                rows.append(sample)
                if DEBUG_ON and (not is_train) and len(rows) == DEBUG_SIZE:
                    return rows
    return rows

def main():
    args = set_args()
    global logger
    start_time = time.time()
    logger = create_logger(__name__, to_disk=True, log_file=args.log_file)
    v2_on = args.v2_on
    version = 'v1'
    if v2_on:
        msg = '~Processing SQuAD v2.0 dataset~'
        train_path = 'train-v2.0.json'
        dev_path = 'dev-v2.0.json'
        version = 'v2'
    else:
        msg = '~Processing SQuAD dataset~'
        train_path = 'train-v1.1.json'
        dev_path = 'dev-v1.1.json'

    logger.warning(msg)
    if DEBUG_ON:
        logger.error('***DEBUGING MODE***')
    train_path = os.path.join(args.data_dir, train_path)
    valid_path = os.path.join(args.data_dir, dev_path)

    logger.info('The path of training data: {}'.format(train_path))
    logger.info('The path of validation data: {}'.format(valid_path))
    logger.info('{}-dim word vector path: {}'.format(args.embedding_dim, args.glove))
    # could be fasttext embedding
    emb_path = args.glove
    embedding_dim = args.embedding_dim
    set_environment(args.seed)
    if args.fasttext_on:
        logger.info('Loading fasttext vocab.')
    else:
        logger.info('Loading glove vocab.')
    # load data
    train_data = load_data2(train_path, v2_on=v2_on)
    dev_data = load_data2(valid_path, False, v2_on=v2_on)

    wemb_vocab = load_emb_vocab(emb_path, embedding_dim, fast_vec_format=args.fasttext_on)
    logger.info('Build vocabulary')
    vocab, _, _ = build_vocab(train_data + dev_data, wemb_vocab, sort_all=args.sort_all, clean_on=True, cl_on=False)
    logger.info('Done with vocabulary collection')

    # loading ner/pos tagging vocab
    resource_path = 'resource'
    logger.info('Loading resource')

    with open(os.path.join(resource_path, 'vocab_tag.pick'),'rb') as f:
        vocab_tag = pickle.load(f)
    with open(os.path.join(resource_path,'vocab_ner.pick'),'rb') as f:
        vocab_ner = pickle.load(f)

    meta_path = gen_name('data2/', args.meta, version, suffix='pick')
    logger.info('building embedding')
    embedding = build_embedding(emb_path, vocab, embedding_dim, fast_vec_format=args.fasttext_on)
    meta = {'vocab': vocab, 'vocab_tag': vocab_tag, 'vocab_ner': vocab_ner, 'embedding': embedding}
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    logger.info('building training data')
    train_fout = gen_name('data2/', args.train_data, version)
    build_data(train_data, vocab, vocab_tag, vocab_ner, train_fout, True, NLP=NLP, v2_on=v2_on)

    logger.info('building dev data')
    dev_fout = gen_name('data2/', args.dev_data, version)
    build_data(dev_data, vocab, vocab_tag, vocab_ner, dev_fout, False, NLP=NLP, v2_on=v2_on)
    end_time = time.time()
    logger.warning('It totally took {} minutes to processe the data!!'.format((end_time - start_time) / 60.))

if __name__ == '__main__':
    main()
