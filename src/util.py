import logging
import random

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from category_id_map import lv2id_to_lv1id
import re
from opencc import OpenCC
import math

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(en in n for en, ep in model.bert_text.named_parameters())],
         'weight_decay': args.weight_decay, 'lr':args.bert_learning_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(en in n for en, ep in model.bert_text.named_parameters())], 'weight_decay': 0.0, 'lr':args.bert_learning_rate},
        

        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(en in n for en, ep in model.bert_text.named_parameters())],
         'weight_decay': args.weight_decay, 'lr':args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(en in n for en, ep in model.bert_text.named_parameters())], 'weight_decay': 0.0, 'lr':args.learning_rate}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    # optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate, weight_decay=1e-3)
    
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=args.max_steps)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5,
                                                                         last_epoch=-1)
    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
class PGD(object):
    """
    基于PGD算法的攻击机制
    Args:
        module (:obj:`torch.nn.Module`): 模型
    Examples::
        >>> pgd = PGD(module)
        >>> K = 3
        >>> for batch_input, batch_label in data:
        >>>     # 正常训练
        >>>     loss = module(batch_input, batch_label)
        >>>     loss.backward() # 反向传播，得到正常的grad
        >>>     pgd.backup_grad()
        >>>     # 对抗训练
        >>>     for t in range(K):
        >>>         pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        >>>         if t != K-1:
        >>>             optimizer.zero_grad()
        >>>         else:
        >>>             pgd.restore_grad()
        >>>         loss_adv = module(batch_input, batch_label)
        >>>         loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        >>>     pgd.restore() # 恢复embedding参数
        >>>     # 梯度下降，更新参数
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()
    Reference:
        [1]  https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, module):
        self.module = module
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(
        self,
        epsilon=1.,
        alpha=0.3,
        emb_name='word_embeddings',
        is_first_attack=False
    ):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

class AWP(object):
    """ [Adversarial weight perturbation helps robust generalization](https://arxiv.org/abs/2004.05884)
    """
    def __init__(
        self,
        model,
        emb_name="weight",
        epsilon=0.001,
        alpha=1.0,
    ):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.param_backup = {}
        self.param_backup_eps = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        if self.alpha == 0: return
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                # save
                if is_first_attack:
                    self.param_backup[name] = param.data.clone()
                    grad_eps = self.epsilon * param.abs().detach()
                    self.param_backup_eps[name] = (
                        self.param_backup[name] - grad_eps,
                        self.param_backup[name] + grad_eps,
                    )
                # attack
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.alpha * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data,
                            self.param_backup_eps[name][0]
                        ),
                        self.param_backup_eps[name][1]
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.param_backup:
                param.data = self.param_backup[name]
        self.param_backup = {}
        self.param_backup_eps = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if name in self.grad_backup:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}


class AWP_fast:
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.001, adv_eps=0.001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
                
    
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
class TextCleaner:
    def __init__(self,
                 remove_space=False, # 去除空格
                 remove_suspension=False, # 转换省略号
                 only_zh=False, # 只保留汉子
                 remove_sentiment_character=False, # 去除表情符号
                 to_simple=False, # 转化为简体中文
                 remove_stop_words=False,
                 stop_words_dir="./",
                 with_space=False,
                 batch_size=256):
        self._remove_space = remove_space
        self._remove_suspension = remove_suspension
        self._remove_sentiment_character = remove_sentiment_character

        self._only_zh = only_zh
        self._to_simple = to_simple

        # self._remove_html_label = remove_html_label
        self._remove_stop_words = remove_stop_words
        self._stop_words_dir = stop_words_dir

        self._with_space = with_space
        self._batch_size = batch_size

    def clean_single_text(self, text):
        if self._remove_space:
            text = self.remove_space(text)
        if self._remove_suspension:
            text = self.remove_suspension(text)
        if self._remove_sentiment_character:
            text = self.remove_sentiment_character(text)
        if self._to_simple:
            text = self.to_simple(text)
        if self._only_zh:
            text = self.get_zh_only(text)
        return text

    def clean_text(self, text_list):
        text_list = [self.clean_single_text(text) for text in text_list]
        tokenized_words_list = text_list
        if self._remove_stop_words:
            text_list = [self.remove_stop_words(words_list, self._stop_words_dir, self._with_space) for words_list in tokenized_words_list]
        return ''.join(text_list)

    def remove_space(self, text):     #定义函数
        return text.replace(' ','')   # 去掉文本中的空格

    def remove_suspension(self, text):
        # text = text.replace('...', '。')
        text = text.replace('—', '-')
        text = text.replace('"', "'")
        text = text.replace('“', "'")
        text = text.replace('”', "'")
        text = text.replace('【', '')
        text = text.replace('】', '')
#         text = text.replace('嗯', '') 
#         text = text.replace('哦', '') 
        
        return text

    def get_zh_only(self, text):
        def is_chinese(uchar):
            if uchar >= u'\u4e00' and uchar <= u'\u9fa5':  # 判断一个uchar是否是汉字
                return True
            else:
                return False

        content = ''
        for i in text:
            if is_chinese(i):
                content = content+i
        return content

    def remove_sentiment_character(self, sentence):    
        pattern = re.compile("[^\u4e00-\u9fa5^,^.^!^，^。^?^？^！^a-z^A-Z^0-9]")  #只保留中英文、数字和符号，去掉其他东西
        #若只保留中英文和数字，则替换为[^\u4e00-\u9fa5^a-z^A-Z^0-9]
        line = re.sub(pattern,'',sentence)  #把文本中匹配到的字符替换成空字符
        new_sentence=''.join(line.split())    #去除空白
        return new_sentence

    def to_simple(self, sentence):
        new_sentence = OpenCC('t2s').convert(sentence)   # 繁体转为简体
        return new_sentence

    def to_tradition(self, sentence):
        new_sentence = OpenCC('s2t').convert(sentence)   # 简体转为繁体
        return new_sentence

    # def remove_html(self, text):
    #     return BeautifulSoup(text, 'html.parser').get_text() #去掉html标签

    def remove_stop_words(self, words_list, stop_words_dir, with_space=False):
        """
        中文数据清洗  stopwords_chineses.txt存放在博客园文件中
        :param text:
        :return:
        """
        # stop_word_filepath_list = [[嗯]]#glob(stop_words_dir + "/*.txt")
        # for stop_word_filepath in stop_word_filepath_list:
        #     with open(stop_word_filepath) as fp:
        #         stopwords = {}.fromkeys([line.rstrip() for line in fp]) #加载停用词(中文)
        eng_stopwords = set(['嗯','哦']) #去掉重复的词
        words = [w for w in words_list if w not in eng_stopwords] #去除文本中的停用词
        if with_space:
            return ' '.join(words)
        else:
            return ''.join(words)
        
        
def compute_conditional_entropy(log_probs_N_K_C):
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)
    
    for i in range(N):
        log_probs_n_K_C = log_probs_N_K_C[i:i+1,:,:]
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)
        entropies_N[i:i+1].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
    return entropies_N


def compute_entropy(log_probs_N_K_C):
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)
    for i in range(N):
        log_probs_n_K_C = log_probs_N_K_C[i:i+1,:,:]
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)
        entropies_N[i:i+1].copy_(-torch.sum(nats_n_C, dim=1))
    return entropies_N


def get_bald_batch(log_probs_N_K_C, batch_size: int, dtype=None, device=None):
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)

    candiate_scores, candidate_indices = torch.topk(-scores_N, batch_size)

    return candiate_scores.tolist(), candidate_indices.tolist()


