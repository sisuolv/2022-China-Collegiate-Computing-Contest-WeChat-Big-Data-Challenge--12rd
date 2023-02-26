import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
# from model import MultiModal
import torch.nn as nn
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import torch.nn.functional as F
from util import *
from tqdm import tqdm as tqdm
import json
import jieba

import joblib
def inference(args ):
    # 1. load data
    data_test_b = json.load(open(args.test_b_annotation, 'r', encoding='utf8'))
    ss = []

    for item in tqdm(data_test_b):
        title = item['title']
        asr = item['asr'].replace('嗯', '')
        ocr = ' '.join([v['text'] for v in item['ocr']])
        tmp_sent = ' '.join([title, asr, ocr])
        tmp_sent = ' '.join([x for x in jieba.lcut(tmp_sent) if x not in '，。？！;'])
        ss.append(tmp_sent)

    tdidf_model = joblib.load('tfidfmodel.pkl')
    svd = joblib.load('svdfmodel.pkl')
    svd_tfidf_data = svd.transform(tdidf_model.transform(ss))
    print('jiszai:',svd_tfidf_data)


    np.save('data/annotations/tfidf_fea_test_b_test.npy', svd_tfidf_data)
    args.tfidf_feapath_test_b = 'data/annotations/tfidf_fea_test_b_test.npy'

    with open(args.test_b_annotation, 'r', encoding='utf8') as f:
            textanns = json.load(f)
    for i in tqdm(range(len(textanns))):
        textann = textanns[i]
        text_title, text_ocr, text_asr = textann['title'], ' '.join([t['text'] for t in textann['ocr']]), textann['asr']
        textanns[i]['title'] = text_title
        textanns[i]['ocr'] = text_ocr
        textanns[i]['asr'] = text_asr        
            
    dataset = MultiModalDataset(args, textanns, args.test_b_zip_feats, args.tfidf_feapath_test_b,test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=False,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    
    weight_all = [
                  0.6,0.6,0.6,0.6,0.6,
                  0.6,0.6,0.6,0.6,0.6,
                  0.6,0.6,0.6,0.6,0.6,
                 ]
    model_lists = [
    
                    '/hy-tmp/save/v1/model_hunliu_large_best_2022.bin',
                    '/hy-tmp/save/v1/model_hunliu_large_best_2021.bin',
                    '/hy-tmp/save/v1/model_hunliu_large_best_2020.bin',
                    '/hy-tmp/save/v1/model_hunliu_large_best_2019.bin',
                    '/hy-tmp/save/v1/model_hunliu_large_best_2018.bin',

        
                    '/hy-tmp/save/v1/model_hunliu_small_best_2022.bin',
                    '/hy-tmp/save/v1/model_hunliu_small_best_2021.bin',
                    '/hy-tmp/save/v1/model_hunliu_small_best_2020.bin',
                    '/hy-tmp/save/v1/model_hunliu_small_best_2019.bin',
                    '/hy-tmp/save/v1/model_hunliu_small_best_2018.bin',

                    '/hy-tmp/save/v1/model_double_best_2022.bin',
                    '/hy-tmp/save/v1/model_double_best_2021.bin',
                    '/hy-tmp/save/v1/model_double_best_2020.bin',
                    '/hy-tmp/save/v1/model_double_best_2019.bin',
                    '/hy-tmp/save/v1/model_double_best_2018.bin',

                   ]
    assert len(weight_all)== len(model_lists)
    # 3. inference
    softmax = nn.Softmax(dim=1)
    predictions_all=[]
    for model_list in model_lists:
        # 2. load model
        print('++++++++++测试的模型：',model_list)
        if 'double' in model_list:
            from model_double import MultiModal
        elif 'hunliu_Xlarge' in model_list:
            from model_hunliu_Xlarge import MultiModal
        elif 'hunliu_large' in model_list:
            from model_hunliu_large import MultiModal
        elif 'hunliu_small' in model_list:
            from model_hunliu_small import MultiModal
        else:
            print('请填入模型的case：：：：：：：：：：：：：：：')
        model = MultiModal(args).to(args.device)
        checkpoint = torch.load(model_list)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                embedding = model.extract(batch)
                prediction = softmax(args.S*F.linear(F.normalize(embedding), F.normalize(model.fc.weight)))
                predictions.extend(prediction.detach().cpu().numpy())
        predictions_all.append(predictions)
    
    predictions_all = np.array(predictions_all)
    print(predictions_all.shape)  
    predictions = 0
    for t in range(len(weight_all)):
        predictions += (weight_all[t]*predictions_all[t])
    predictions = predictions.argmax(1) 

    
    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)
    inference(args )
