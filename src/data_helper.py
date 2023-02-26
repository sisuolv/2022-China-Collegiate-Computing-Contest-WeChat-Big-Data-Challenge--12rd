import json
import random
import zipfile
from io import BytesIO
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from category_id_map import category_id_to_lv2id
from util import *
from tqdm import tqdm as tqdm
import pandas as pd

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm as tqdm
import jieba

import joblib

def create_dataloaders(args):
    if args.enable_tfidf:
        print('+++++++++++正在tfidf抽取')
        data = json.load(open(args.train_annotation, 'r', encoding='utf8'))
        # data_test = json.load(open(args.test_annotation, 'r', encoding='utf8'))
        data_unlabel = json.load(open(args.unlabel_annotations, 'r', encoding='utf8'))

        ss = []
        for item in tqdm(data):
            title= item['title']
            asr = item['asr'].replace('嗯', '')
            ocr = ' '.join([v['text'] for v in item['ocr']])
            tmp_sent = ' '.join([title, asr, ocr])
            tmp_sent = ' '.join([x for x in jieba.lcut(tmp_sent) if x not in '，。？！;'])
            ss.append(tmp_sent)

        for item in tqdm(data_unlabel):
            title= item['title']
            asr = item['asr'].replace('嗯', '')
            ocr = ' '.join([v['text'] for v in item['ocr']])
            tmp_sent = ' '.join([title, asr, ocr])
            tmp_sent = ' '.join([x for x in jieba.lcut(tmp_sent) if x not in '，。？！;'])
            ss.append(tmp_sent)


        tdidf_model = TfidfVectorizer().fit(ss)
        svd = TruncatedSVD(n_components=64, n_iter=40, random_state=2022)
        svd_tfidf_data = svd.fit_transform(tdidf_model.transform(ss))
        print(svd_tfidf_data.shape)
        
        joblib.dump(tdidf_model, 'tfidfmodel.pkl')
        joblib.dump(svd, 'svdfmodel.pkl')
        
        np.save('data/annotations/tfidf_fea_train.npy', svd_tfidf_data[:100000])
        np.save('data/annotations/tfidf_fea_zong.npy', svd_tfidf_data)
    if args.pre_enable_tfidf:
        
        data = json.load(open(args.train_annotation, 'r', encoding='utf8'))
        # data_test = json.load(open(args.test_annotation, 'r', encoding='utf8'))
        data_unlabel = json.load(open(args.unlabel_annotations, 'r', encoding='utf8'))

        ss = []
        for item in tqdm(data):
            title= item['title']
            asr = item['asr'].replace('嗯', '')
            ocr = ' '.join([v['text'] for v in item['ocr']])
            tmp_sent = ' '.join([title, asr, ocr])
            tmp_sent = ' '.join([x for x in jieba.lcut(tmp_sent) if x not in '，。？！;'])
            ss.append(tmp_sent)

        for item in tqdm(data_unlabel):
            title= item['title']
            asr = item['asr'].replace('嗯', '')
            ocr = ' '.join([v['text'] for v in item['ocr']])
            tmp_sent = ' '.join([title, asr, ocr])
            tmp_sent = ' '.join([x for x in jieba.lcut(tmp_sent) if x not in '，。？！;'])
            ss.append(tmp_sent)
            
        tdidf_model = joblib.load('tfidfmodel.pkl')
        svd  = joblib.load('svdfmodel.pkl') 
        svd_tfidf_data = svd.transform(tdidf_model.transform(ss))
        print('jiszai:',svd_tfidf_data)
        
        np.save('data/annotations/tfidf_fea_train.npy', svd_tfidf_data[:100000])
        np.save('data/annotations/tfidf_fea_zong.npy', svd_tfidf_data)


    with open(args.train_annotation, 'r', encoding='utf8') as f:
            textanns = json.load(f)
    
    if args.enable_acti:
        print('Enable active')
        activa = pd.read_csv('./activa.csv')
        for i in (range(len(activa))):
            activa.iloc[i, 0] =str(activa.iloc[i,0])
        activa = set(activa.iloc[:, 0])
        # print(activa)
        textanns_acti = pd.DataFrame()
        step = 0
        for i in tqdm(range(len(textanns))):
            # print(int(textanns[i]['id']))
            
            if str(textanns[i]['id']) in activa:  
                textanns[step] = textanns[i]
                step += 1
        textanns =  textanns[:step]
    
    for i in tqdm(range(len(textanns))):
        textann = textanns[i]
        text_title, text_ocr, text_asr = textann['title'], ' '.join([t['text'] for t in textann['ocr']]), textann['asr']
 
        textanns[i]['title'] = text_title
        textanns[i]['ocr'] = text_ocr
        textanns[i]['asr'] = text_asr
        
    task = set(args.task)
    if 'mlm' in task or 'itm' in task:
            
        # with open(args.test_annotation, 'r', encoding='utf8') as f:
        #         test_textanns = json.load(f)
        with open(args.unlabel_annotations, 'r', encoding='utf8') as f:
                unlabel_textanns = json.load(f)

        for i in tqdm(range(len(unlabel_textanns))):
            textann = unlabel_textanns[i]
            text_title, text_ocr, text_asr = textann['title'], ' '.join([t['text'] for t in textann['ocr']]), textann['asr']
            unlabel_textanns[i]['title'] = text_title
            unlabel_textanns[i]['ocr'] = text_ocr
            unlabel_textanns[i]['asr'] = text_asr
        # textanns.extend(test_textanns)
        textanns.extend(unlabel_textanns)
        print('预训练个数：',len(textanns))
        dataset = MultiModalDataset(args, textanns, args.train_zip_feats, args.tfidf_feapath_zong)
    else:
        dataset = MultiModalDataset(args, textanns, args.train_zip_feats, args.tfidf_feapath_train)
    size = len(dataset)
    print('总shape:', size)
    val_size = int(size * args.val_ratio)
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=False, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader



class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path,
                 zip_feats: str,
                 tfidf_feapath: str,
                 test_mode: bool = False):
        self.args = args
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode
        self.tfidf_fea = np.load(tfidf_feapath)
                              
        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        self.task = set(self.args.task)
        
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
            if 'mlm' in self.task or 'itm' in self.task:
                # self.handles_test = [None for _ in range(args.num_workers)]
                self.handles_unlable = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
            if 'mlm' in self.task or 'itm' in self.task:    
                # self.handles_test = zipfile.ZipFile(self.args.test_zip_feats, 'r')
                self.handles_unlable = zipfile.ZipFile(self.args.unlabeled_zip_feats, 'r')
        # load annotations
        # with open(ann_path, 'r', encoding='utf8') as f:
        #     self.anns = json.load(f)
        self.anns = ann_path
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']

        if 'mlm' in self.task or 'itm' in self.task:
            
            if self.num_workers > 0:
                worker_id = torch.utils.data.get_worker_info().id
                if self.handles[worker_id] is None:
                    self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
                handle = self.handles[worker_id]
                if self.handles_unlable[worker_id] is None:
                    self.handles_unlable[worker_id] = zipfile.ZipFile(self.args.unlabeled_zip_feats, 'r')
                handle_unlable = self.handles_unlable[worker_id]
            else:
                handle = self.handles
                handle_test = self.handles_test
                handle_unlable = self.handles_unlable
            try:
                raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
            except:
                raw_feats = np.load(BytesIO(handle_unlable.read(name=f'{vid}.npy')), allow_pickle=True)
        else:
            if self.num_workers > 0:
                worker_id = torch.utils.data.get_worker_info().id
                if self.handles[worker_id] is None:
                    self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
                handle = self.handles[worker_id]
            else:
                handle = self.handles
            raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
            
            
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask


    def pad_clip_text(self, ann):
        # text_title, text_ocr, text_asr = ann['title'], ' '.join([t['text'] for t in ann['ocr']]), ann['asr']
        text_title, text_ocr, text_asr = ann['title'], ann['ocr'], ann['asr']
        # print( text_asr)

        if len(text_title)<=len(text_ocr)<=len(text_asr):
            return f"[CLS][SEP]{text_title}[SEP]{text_ocr}[SEP]{text_asr}[SEP]"
        elif len(text_title)<=len(text_asr)<=len(text_ocr):
            return f"[CLS][SEP]{text_title}[SEP]{text_asr}[SEP]{text_ocr}[SEP]"
        
        elif len(text_asr)<=len(text_title)<=len(text_ocr):
            return f"[CLS][SEP]{text_asr}[SEP]{text_title}[SEP]{text_ocr}[SEP]"
        elif len(text_asr)<=len(text_ocr)<=len(text_title):
            return f"[CLS][SEP]{text_asr}[SEP]{text_ocr}[SEP]{text_title}[SEP]"
        
        elif len(text_ocr)<=len(text_title)<=len(text_asr):
            return f"[CLS][SEP]{text_ocr}[SEP]{text_title}[SEP]{text_asr}[SEP]"
        elif len(text_ocr)<=len(text_asr)<=len(text_title):
            return f"[CLS][SEP]{text_ocr}[SEP]{text_asr}[SEP]{text_title}[SEP]"

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.

        if 'mlm' in self.task or 'mfm' in self.task or 'clip' in self.task:
            
                        
            frame_input, frame_mask = self.get_visual_feats(idx)

            # Step 2, load title tokens

            text_plus = self.pad_clip_text(self.anns[idx])

            # print(len(text_plus))
            title_input, title_mask = self.tokenize_text(text_plus)
            data = dict(
                frame_input=frame_input,
                frame_mask=frame_mask,
                title_input=title_input,
                title_mask=title_mask
            )
            data['id'] = self.anns[idx]['id']
            data['tfidf'] = self.tfidf_fea[idx].astype(np.float32)
            return data
            
        else:
            
            frame_input, frame_mask = self.get_visual_feats(idx)

            # Step 2, load title tokens

            text_plus = self.pad_clip_text(self.anns[idx])

            # print(len(text_plus))
            title_input, title_mask = self.tokenize_text(text_plus)

            # Step 3, summarize into a dictionary
            data = dict(
                frame_input=frame_input,
                frame_mask=frame_mask,
                title_input=title_input,
                title_mask=title_mask
            )


            data['id'] = self.anns[idx]['id']
            data['tfidf'] = self.tfidf_fea[idx].astype(np.float32)
            # Step 4, load label if not test mode
            if not self.test_mode:
                label = category_id_to_lv2id(self.anns[idx]['category_id'])
                data['label'] = torch.LongTensor([label])

            return data
