import logging
import os
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from util import *#import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import torch.nn as nn
import tqdm as tqdm
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np

def validate(model, val_dataloader, args):
    model.eval()
    predictions = []
    labels = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in val_dataloader:
            # loss, _, pred_label_id, label = model(batch)
            
            label = batch['label'].squeeze(dim=1).to(args.device)
            embeddings = model.module.extract(batch)
            prediction = softmax(args.S*F.linear(F.normalize(embeddings), F.normalize(model.module.fc.weight)))
            # prediction = model.module.extract(batch)
            # prediction = model(batch)
            pred_label_id = torch.argmax(prediction, dim=1)
            
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())

    results = evaluate(predictions, labels)

    model.train()
    return results


def cal_accuracy(prediction, label):
    with torch.no_grad():
        pred_label_id = torch.argmax(prediction, dim=1)
        accuracy = (label == pred_label_id).float().sum() / label.shape[0]
    return accuracy, pred_label_id, label

def mixup_data(y, alpha=1.0, use_cuda=True):
    # 对数据的mixup 操作 x = lambda*x_i+(1-lamdda)*x_j
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = y.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    y_a, y_b = y, y[index]    # 记录下y_i 和y_j
    return  index, y_a, y_b, lam    # 返回y_i 和y_j 以及lambda

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    # 对loss函数进行混合，criterion是crossEntropy函数
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def multi_forward_hook(module, inputs, outputs):
    mix_input = outputs[0] * lam + outputs[0][index] * (1 - lam)
    return tuple([mix_input])



def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)
    
    if args.modele_case=='double':
        from model_double import MultiModal
    elif args.modele_case=='hunliu_Xlarge':
        from model_hunliu_Xlarge import MultiModal
    elif args.modele_case=='hunliu_large':
        from model_hunliu_large import MultiModal
    elif args.modele_case=='hunliu_small':
        from model_hunliu_small import MultiModal
    else:
        print('请填入模型的case：：：：：：：：：：：：：：：')
    print('+++++++++++++++正在进行的模型是：', args.modele_case)
    # 2. build model and optimizers
    model = MultiModal(args)
    if args.pre:
        checkpoint = torch.load(args.pre_ckpt_file)['model_state_dict']
        # model.load_state_dict(checkpoint['model_state_dict'])
        model_dict = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items()
                           if k in model_dict.keys()}

        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        
        print('+++++++++++++++正在进行的模型是：', args.modele_case, '+++++++++++++加载的预训练模型是：', args.pre_ckpt_file)
        
    optimizer, scheduler = build_optimizer(args, model)
    
    
    
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    cirtion = nn.CrossEntropyLoss()

    # for name, paramer in model.named_parameters():
    #     print(name)
    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()  
    
    num_total_steps = len(train_dataloader) * args.max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_steps*0.1, num_training_steps=num_total_steps)
    
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5,
#                                                                          last_epoch=-1)
    
    if args.attack_func == 'fgm':
        attack_func = FGM(model)
        print('Enable FGM')
    elif args.attack_func == 'pgd':
        attack_func = PGD(model)
        print('Enable PGD')
    elif args.attack_func == 'awp':
        attack_func = AWP(model)
        print('Enable AWP')
    elif args.attack_func == 'awp_fast':    
        attack_func = AWP_fast(model, optimizer, adv_lr=0.00001, adv_eps=0.00001)
        print('Enable AWP_fast')
        
    if args.ema:
        ema = EMA(model, 0.999)
        ema.register()
        print('Enable EMA--0.999')
    

    for epoch in range(args.max_epochs):
        if epoch>=6:
                break
        for batch in train_dataloader:
            model.train()
            targets = batch['label'].squeeze(dim=1).to(args.device)
            if args.attack_func == 'awp_fast':                
                attack_func.perturb()

                
            pred = model(batch)

            loss = cirtion(pred, targets)

            accuracy, pred_label_id, label = cal_accuracy(pred, targets)

            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if args.attack_func == 'fgm':
                attack_func.attack()
                # optimizer.zero_grad()
                loss_adv = cirtion(model(batch), targets)
                loss_adv.backward()
                attack_func.restore()
            elif args.attack_func == 'awp_fast':
                attack_func.restore()      
            elif args.attack_func == 'pgd' or args.attack_func == 'awp':
                attack_func.backup_grad()
                awp_k = 1
                for t in range(awp_k):
                    attack_func.attack(is_first_attack=(t == 0))
                    if t != awp_k - 1:
                        optimizer.zero_grad()
                    else:
                        attack_func.restore_grad()
                    loss_adv = cirtion(model(batch), targets)
                    loss_adv.backward()  
                attack_func.restore()  
                
            optimizer.step()
            if args.ema:
                ema.update()
                
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if epoch==3 or epoch==4 or epoch==5:
                args.print_steps = 100
            else:
                args.print_steps = 1500

            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
        # scheduler.step()
        # 4. validation
                if args.ema:
                    ema.apply_shadow()
                results = validate(model, val_dataloader, args)
                if args.ema:
                    ema.restore()
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: {results}")
                
                # 5. save checkpoint
                mean_f1 = results['mean_f1']
                if mean_f1 >= best_score:
                    if args.ema:
                        ema.apply_shadow()
                    best_score = mean_f1
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    
                    
                    if args.modele_case=='double':
                        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                               f'{args.savedmodel_path}/model_double_best_{args.seed}.bin')
                    elif args.modele_case=='hunliu_Xlarge':
                        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                               f'{args.savedmodel_path}/model_hunliu_Xlarge_best_{args.seed}.bin')
                    elif args.modele_case=='hunliu_large':
                        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                               f'{args.savedmodel_path}/model_hunliu_large_best_{args.seed}.bin')
                    elif args.modele_case=='hunliu_small':
                        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                               f'{args.savedmodel_path}/model_hunliu_small_best_{args.seed}.bin')
                    else:
                        print('请填入模型的case：：：：：：：：：：：：：：：')
                    
                    if args.ema:
                        ema.restore()


def train_mm(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    if args.modele_case == 'double':
        from model_double import MultiModal
    elif args.modele_case == 'hunliu_large':
        from model_hunliu_large import MultiModal
    elif args.modele_case == 'hunliu_small':
        from model_hunliu_small import MultiModal
    else:
        print('请填入模型的case：：：：：：：：：：：：：：：')
    print('+++++++++++++++正在进行的模型是：', args.modele_case)
    model = MultiModal(args)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    cirtion = nn.CrossEntropyLoss()
    # 3. training
    step = 0
    best_loss = 1e12
    start_time = time.time()

    num_total_steps = len(train_dataloader) * args.max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_steps * 0.1,
                                                num_training_steps=num_total_steps)

    #     if args.attack_awp:
    #         # attack_func = AWP(net)
    #         print('Enable AWP_fast')
    #         attack_func = AWP_fast(model, optimizer, adv_lr=0.00001, adv_eps=0.00001)

    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            # if args.attack_awp:
            #     attack_func.perturb()

            loss, masked_lm_loss, itm_loss = model(batch)

            loss = loss.mean()
            loss.backward()

            # if args.attack_awp:
            #     attack_func.restore()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(
                    f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f} --mlmloss {masked_lm_loss:.3f} --itmloss {itm_loss:.3f}")
        # scheduler.step()
        # 5. save checkpoint
        # if loss < best_loss:
        #     best_loss= loss
        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        if epoch == 0 or epoch >= 15:
            if args.modele_case == 'double':
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'loss': loss},
                           f'{args.savedmodel_path}/model_double_pretain_epoch_{epoch}.bin')
            elif args.modele_case == 'hunliu_Xlarge':
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'loss': loss},
                           f'{args.savedmodel_path}/model_hunliu_Xlarge_pretain_epoch_{epoch}.bin')
            elif args.modele_case == 'hunliu_large':
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'loss': loss},
                           f'{args.savedmodel_path}/model_hunliu_large_pretain_epoch_{epoch}.bin')
            elif args.modele_case == 'hunliu_small':
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'loss': loss},
                           f'{args.savedmodel_path}/model_hunliu_small_pretain_epoch_{epoch}.bin')
            else:
                print('请填入模型的case：：：：：：：：：：：：：：：')
        
def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)


    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    
    task = set(args.task)

    if 'mlm' in task or 'itm' in task:
        train_mm(args)
    else:
        train_and_validate(args)
    

if __name__ == '__main__':
    main()
