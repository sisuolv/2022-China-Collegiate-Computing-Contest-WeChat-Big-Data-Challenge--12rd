import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
    

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='data/annotations/test_a.json')
    parser.add_argument('--test_b_annotation', type=str, default='data/annotations/test_b.json')

    parser.add_argument('--unlabel_annotations', type=str, default='data/annotations/unlabeled.json')
    parser.add_argument('--tfidf_feapath_train', type=str, default='data/annotations/tfidf_fea_train.npy')
    parser.add_argument('--tfidf_feapath_test', type=str, default='data/annotations/tfidf_fea_test.npy')
    parser.add_argument('--tfidf_feapath_test_b', type=str, default='data/annotations/tfidf_fea_test_b.npy')

    parser.add_argument('--tfidf_feapath_zong', type=str, default='data/annotations/tfidf_fea_zong.npy')

    parser.add_argument('--train_zip_feats', type=str, default='data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='data/zip_feats/test_a.zip')
    parser.add_argument('--test_b_zip_feats', type=str, default='data/zip_feats/test_b.zip')
    parser.add_argument('--unlabeled_zip_feats', type=str, default='data/zip_feats/unlabeled.zip')
    parser.add_argument('--test_output_csv', type=str, default='data/result.csv')
    
    
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='save/v1')
    parser.add_argument('--ckpt_file', type=str, default='save/v1/model_double_be.bin')
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-macbert-base')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=196)
    
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    
    
    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=768, help="linear size before final linear")

    # ========================== Arcface =============================
    parser.add_argument('--S', type=float, default=30)
    parser.add_argument('--M', type=float, default=0.00)
    parser.add_argument('--EASY_MERGING', type=bool, default=False)
    parser.add_argument('--LS_EPS', type=float, default=0.0)
    
    # ========================== mixup =============================
    parser.add_argument('--mix_up', type=bool, default=False)
    parser.add_argument('--mixup_prob', type=float, default=0.25)
    parser.add_argument('--alpha', type=float, default=1)
    
    # ========================== Awp =============================
    parser.add_argument('--attack_func', type=str, default='awp')
    
    #
    parser.add_argument('--ema', type=bool, default=True)
    
    
    
    parser.add_argument('--enable_tfidf', type=bool, default=False)
    parser.add_argument('--pre_enable_tfidf', type=bool, default=False)
    
    parser.add_argument('--modele_case',  type=str, default='d')
    
    parser.add_argument('--task', type=list, default=['fine']) #parser.add_argument('--task', type=list, default=['mlm','itm'])
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--print_steps', type=int, default=500, help="Number of steps to log training metrics.")
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')



    parser.add_argument('--pre', type=bool, default=True)
    parser.add_argument('--pre_ckpt_file', type=str, default='save/v1/model_epoch_large_19_loss_1.553844690322876.bin')
    
    parser.add_argument('--enable_acti', type=bool, default=False)
    
    

    return parser.parse_args()
