import json
import pandas as pd
import torch
import numpy as np
import os
import random
# Utils
from utils.utils import DataLoader, compute_pna_degrees, CustomWeightedRandomSampler
#compute_pna_degrees: 计算图神经网络的节点度数（用于PNA聚合器）。
# virtual_screening: 虚拟筛选函数，用于模型预测。
# CustomWeightedRandomSampler: 自定义加权随机采样器，处理数据不平衡。
from utils.dataset import *  # data
from utils.trainer import Trainer
from utils.metrics import *

# Preprocessing
from utils import protein_init, ligand_init

# Model
from models.net import net

#net: PSICHIC的主网络类，定义图卷积层、注意力机制等。
import argparse
import ast


def tuple_type(s):
    try:
        # Safely evaluate the string as a tuple
        value = ast.literal_eval(s)
        if not isinstance(value, tuple):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {s}")
    return value


def list_type(s):
    try:
        # Safely evaluate the string as a tuple
        value = ast.literal_eval(s)
        if not isinstance(value, list):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list value: {s}")
    return value


parser = argparse.ArgumentParser()

### Seed and device
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--config_path', type=str, default='config.json')
### Data and Pre-processing
parser.add_argument('--datafolder', type=str, default='./dataset/Davis', help='protein data path')
parser.add_argument('--result_path', type=str, default='./result/Davis', help='path to save results')
parser.add_argument('--save_interpret', type=bool, default=True, help='path to save results')



parser.add_argument('--epochs', type=int, default=30, help='')
parser.add_argument('--evaluate_epoch', type=int, default=1)

parser.add_argument('--total_iters', type=int, default=None)
parser.add_argument('--evaluate_step', type=int, default=500)

# optimizer params - only change this for PDBBind v2016
parser.add_argument('--lrate', type=float, default=1e-4,
                    help='learning rate for PSICHIC')  # change to 1e-5 for LargeScaleInteractionDataset
parser.add_argument('--eps', type=float, default=1e-8, help='higher = closer to SGD')  # change to 1e-5 for PDBv2016

parser.add_argument('--betas', type=tuple_type, default="(0.9,0.999)")  # change to (0.9,0.99) for PDBv2016


# batch size
parser.add_argument('--batch_size', type=int, default=16)
# sampling method - only used for pretraining large-scale interaction dataset ; allow self specified weights to the samples
parser.add_argument('--sampling_col', type=str, default='')

parser.add_argument('--trained_model_path', type=str, default='',
                    help='This does not need to be perfectly aligned, as you can add prediction head for some other tasks as well!')
parser.add_argument('--finetune_modules', type=list_type, default=None)

# # notebook mode?
# parser.add_argument('--nb_mode', type=bool, default=False)

args = parser.parse_args()

if args.trained_model_path:
    with open(os.path.join(args.trained_model_path, 'config.json'), 'r') as f:
        config = json.load(f)
else:
    with open(args.config_path, 'r') as f:
        config = json.load(f)


config['optimizer']['lrate'] = args.lrate
config['optimizer']['eps'] = args.eps
config['optimizer']['betas'] = args.betas


# device
device = torch.device(args.device)
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

model_path = os.path.join(args.result_path, 'save_model_seed{}'.format(args.seed))
if not os.path.exists(model_path):
    os.makedirs(model_path)

# interpret_path = os.path.join(args.result_path, 'interpretation_result_seed{}'.format(args.seed))
# if not os.path.exists(interpret_path):
#     os.makedirs(interpret_path)

if args.epochs is not None and args.total_iters is not None:
    print('If epochs and total iters are both not None, then we only use iters.')
    args.epochs = None


print(args)
with open(os.path.join(model_path, 'model_params.txt'), 'w') as f:
    f.write(str(args))


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)


# train_df = pd.read_csv(os.path.join(args.datafolder, 'train.csv'))
# test_df = pd.read_csv(os.path.join(args.datafolder, 'test.csv'))
# # valid_df = pd.read_csv(os.path.join(args.datafolder, 'val.csv'))

all_df = pd.read_csv(os.path.join(args.datafolder, 'Metz.csv'))
all_df = all_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)


protein_seqs = list(set(all_df['Protein'].tolist()  ))
ligand_smiles = list(set(all_df['Ligand'].tolist()  ))
# protein_seqs = list(set(train_df['Protein'].tolist() + test_df['Protein'].tolist() + valid_df['Protein'].tolist()))
# ligand_smiles = list(set(train_df['Ligand'].tolist() + test_df['Ligand'].tolist() + valid_df['Ligand'].tolist()))



protein_path = os.path.join(args.datafolder, 'protein.pt')
if os.path.exists(protein_path):
    protein_dict = torch.load(protein_path, weights_only=True)
else:
    protein_dict = protein_init(protein_seqs)
    torch.save(protein_dict, protein_path)

ligand_path = os.path.join(args.datafolder, 'ligand.pt')
if os.path.exists(ligand_path):
    ligand_dict = torch.load(ligand_path, weights_only=True)
else:
    ligand_dict = ligand_init(ligand_smiles)
    torch.save(ligand_dict, ligand_path)

torch.cuda.empty_cache()

# 5折交叉验证
k_fold = 5
fold_size = len(all_df) // k_fold
fold_list = []
for i in range(k_fold):
    start = i * fold_size
    end = (i + 1) * fold_size if i < k_fold - 1 else len(all_df)
    fold_list.append(all_df.iloc[start:end])

for fold_idx in range(k_fold):
    print(f"Fold {fold_idx + 1}/{k_fold}")
    with open(args.result_path + '/save_model_seed{}'.format(args.seed) + '/full_result-{}.txt'.format(args.seed),
              'a+') as f:
        f.write('-' * 50 + '\n')
        f.write(f"Fold {fold_idx + 1}/{k_fold} begin\n")
        f.write('-' * 50 + '\n')

    test_df = fold_list[fold_idx]
    train_df = pd.concat([df for i, df in enumerate(fold_list) if i != fold_idx])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)\

    ## training loader
    train_shuffle = True
    train_sampler = None

    if args.sampling_col:
        train_shuffle = False
        train_weights = torch.from_numpy(train_df[args.sampling_col].values)
        train_sampler = CustomWeightedRandomSampler(train_weights, len(train_weights), replacement=True)


    train_dataset = ProteinMoleculeDataset(train_df, ligand_dict, protein_dict, device=args.device)
    test_dataset = ProteinMoleculeDataset(test_df, ligand_dict, protein_dict, device=args.device)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle,
                              sampler=train_sampler, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])


    # valid_dataset = ProteinMoleculeDataset(valid_df, ligand_dict, protein_dict, device=args.device)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
    #                               follow_batch=['mol_x', 'clique_x', 'prot_node_aa']
    #                               )



    if not args.trained_model_path:

        degree_path = os.path.join(args.result_path, 'save_model_seed{}'.format(args.seed), 'degree.pt')

        if not os.path.exists(degree_path):
            mol_deg, clique_deg, prot_deg = compute_pna_degrees(train_loader)
            degree_dict = {'ligand_deg': mol_deg, 'clique_deg': clique_deg, 'protein_deg': prot_deg}

            torch.save(degree_dict, os.path.join(args.result_path, 'save_model_seed{}'.format(args.seed), 'degree.pt'))
        else:
            degree_dict = torch.load(degree_path, weights_only=True)
            mol_deg, clique_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['clique_deg'], degree_dict['protein_deg']


    else:
        degree_dict = torch.load(os.path.join(args.trained_model_path, 'degree.pt'))
        param_dict = os.path.join(args.trained_model_path, 'model.pt')
        mol_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['protein_deg']




    model = net(mol_deg, prot_deg,
                # MOLECULE
                mol_in_channels=config['params']['mol_in_channels'], prot_in_channels=config['params']['prot_in_channels'],
                prot_evo_channels=config['params']['prot_evo_channels'],
                hidden_channels=config['params']['hidden_channels'], pre_layers=config['params']['pre_layers'],
                post_layers=config['params']['post_layers'], aggregators=config['params']['aggregators'],
                scalers=config['params']['scalers'], total_layer=config['params']['total_layer'],
                K=config['params']['K'], heads=config['params']['heads'],
                dropout=config['params']['dropout'],
                dropout_attn_score=config['params']['dropout_attn_score'],
                # output
                regression_head=True,
                device=device).to(device)
    model.reset_parameters()

    if args.trained_model_path:
        model.load_state_dict(torch.load(param_dict, map_location=args.device), strict=False)
        print('Pretrained model loaded!')


    with open(os.path.join(args.result_path, 'save_model_seed{}'.format(args.seed), 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


    engine = Trainer(model=model, lrate=config['optimizer']['lrate'], min_lrate=config['optimizer']['min_lrate'],
                     wdecay=config['optimizer']['weight_decay'], betas=config['optimizer']['betas'],
                     eps=config['optimizer']['eps'], amsgrad=config['optimizer']['amsgrad'],
                     clip=config['optimizer']['clip'], steps_per_epoch=len(train_loader),
                     num_epochs=args.epochs,
                     warmup_iters=config['optimizer']['warmup_iters'],
                     lr_decay_iters=config['optimizer']['lr_decay_iters'],
                     schedule_lr=config['optimizer']['schedule_lr'], regression_weight=1,
                     evaluate_metric='rmse', result_path=args.result_path, runid=args.seed,
                     finetune_modules=args.finetune_modules,
                     device=device)

    print('-' * 50)
    print('start training model')

    valid_loader=None
    engine.train_epoch(train_loader, val_loader=valid_loader, test_loader=test_loader,
                           evaluate_epoch=args.evaluate_epoch)

    print('finished training model')
    print('-' * 50)
    with open(args.result_path + '/save_model_seed{}'.format(args.seed) + '/full_result-{}.txt'.format(args.seed),
              'a+') as f:
        f.write('-' * 50 + '\n')
        f.write(f"Fold {fold_idx + 1}/{k_fold} over\n")
        f.write('-' * 50 + '\n')
