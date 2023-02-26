import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal
import torch.nn as nn
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import torch.nn.functional as F
from util import *
import numpy as np
import json
from tqdm import tqdm as tqdm
import math
import pandas as pd

def inference(args ):
    # 1. load data
    
    with open(args.train_annotation, 'r', encoding='utf8') as f:
            textanns = json.load(f)
            
    for i in tqdm(range(len(textanns))):
        textann = textanns[i]
        text_title, text_ocr, text_asr = textann['title'], ' '.join([t['text'] for t in textann['ocr']]), textann['asr']
        # print( text_asr)
        cleaner = TextCleaner(remove_sentiment_character=True)
        # text_ocr = cleaner.clean_text(text_ocr)
        # text_asr = cleaner.clean_text(text_asr)
        # text_title = cleaner.clean_text(text_title)
        

        textanns[i]['title'] = text_title
        textanns[i]['ocr'] = text_ocr
        textanns[i]['asr'] = text_asr
        
    dataset = MultiModalDataset(args, textanns, args.train_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=False,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    # 2. load model
    modellists_name = ['./save/v1/model_epoch_0_mean_f1_0.591.bin',
                       './save/v1/model_epoch_1_mean_f1_0.6517.bin',
                       './save/v1/model_epoch_2_mean_f1_0.6686.bin',
                       './save/v1/model_epoch_3_mean_f1_0.6802.bin',
                       './save/v1/model_epoch_4_mean_f1_0.6836.bin', 
                      ]
    modellists = []
    for t in range(len(modellists_name)):
        
        model = MultiModal(args)
        checkpoint = torch.load(modellists_name[t], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    modellists.append(model)
    
    
    
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.train()

    # 3. inference
    softmax = nn.Softmax(dim=1)
    idset = []
    step = 0
    with torch.no_grad():
        for batch in dataloader:
            for ttt in range(10):
                for tt in range(len(modellists)):
                    model = modellists[tt]
                    model.train()
                    embeddings = model.acti(batch)
                    prediction = args.S*F.linear(F.normalize(embeddings), F.normalize(model.fc.weight))
                    if ttt==0 and tt == 0:
                        predictions = prediction.unsqueeze(1)
                    else:
                        predictions = torch.cat((predictions, prediction.unsqueeze(1)), dim=1)
            # print('kokokoko:', predictions)

            candiate_scores, candidate_indices = get_bald_batch(predictions.double(), int(predictions.shape[0]*0.9))
            candidate_indices = np.array(candidate_indices)
            # print(batch)
            for cc in candidate_indices:
                cc =int(cc)
                idset.append(str(batch['id'][cc])) 

            # idset = pd.DataFrame(idset)
            # idset = idset.iloc[:, 0]
            # print(idset)
    print(len(idset)) 
    idset = pd.DataFrame(idset)
    idset.to_csv('./activa.csv', index=None)
if __name__ == '__main__':
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)
    inference(args )
