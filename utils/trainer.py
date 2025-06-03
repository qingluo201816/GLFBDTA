import torch
import torch.nn as nn
from utils.metrics import evaluate_reg
import json 
from utils import unbatch
from reprint import output
import math
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, model, lrate, min_lrate, wdecay, betas, eps, amsgrad, clip, steps_per_epoch, num_epochs,
                warmup_iters=2000, lr_decay_iters=None, schedule_lr=False, regression_weight=1,
                evaluate_metric='rmse',
                result_path='', runid=0, device='cuda:0',
                finetune_modules=None):
                
        self.model = model
        self.model.to(device)
        self.optimizer = self.model.configure_optimizers(weight_decay=wdecay, learning_rate=lrate, 
                                                         betas=betas, eps=eps, amsgrad=amsgrad)

        self.clip = clip
        self.regression_loss = nn.MSELoss()


        self.num_epochs = num_epochs
        
        self.result_path = result_path
        self.runid = runid
        self.device = device
        self.regression_weight = regression_weight
        self.evaluate_metric = evaluate_metric


        self.schedule_lr = schedule_lr
        self.total_iters = num_epochs * steps_per_epoch


        self.lrate = lrate
        self.min_lrate = min_lrate
        self.warmup_iters = warmup_iters
        if lr_decay_iters is None:
            self.lr_decay_iters = self.total_iters
        else:
            self.lr_decay_iters = lr_decay_iters

        self.train_losses = []
        self.val_mses = []


    def train_epoch(self, train_loader, val_loader = None, test_loader = None, evaluate_epoch = 1):

        best_result = float('inf')
        best_test_epoch = -1
        patience = 100
        best_test_result_str = ""

        pbar = tqdm(total=self.total_iters, desc='training')
        iter_num = 0
        val_str = ''
        test_str = ''
        with output(initial_len=11, interval=0) as output_lines:
            for epoch in range(1, self.num_epochs+1):
                running_reg_loss = 0

                running_spectral_loss = 0 
                running_ortho_loss = 0 
                running_cluster_loss = 0 
                self.model.train()
                
                for data in train_loader:
                    if epoch <= 100:
                        curr_lr_rate = 0.001
                    else:
                        curr_lr_rate = self.lrate


                    self.optimizer.zero_grad()

                    data = data.to(self.device)

                    reg_pred,  sp_loss, o_loss, cl_loss, _ = self.model(
                        # Molecule
                        mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
                        atom_edge_index=data.mol_edge_index, clique_x=data.clique_x, 
                        clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
                        # Protein
                        residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                        residue_edge_index=data.prot_edge_index,
                        residue_edge_weight=data.prot_edge_weight,
                        # Mol-Protein Interaction batch
                        mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch, clique_batch=data.clique_x_batch
                    )
                    ## Loss compute

                    reg_loss = 0

                    loss_val = torch.tensor(0.).to(self.device)
                    loss_val += sp_loss
                    loss_val += o_loss
                    loss_val += cl_loss
                    sp_loss = sp_loss.item()
                    o_loss = o_loss.item()
                    cl_loss = cl_loss.item()

                    if reg_pred is not None:
                        reg_pred = reg_pred.squeeze()
                        reg_y = data.reg_y.squeeze()
                        reg_loss = self.regression_loss(reg_pred, reg_y) * self.regression_weight
                        loss_val += reg_loss
                        reg_loss = reg_loss.item()

                    loss_val.backward()

                    if self.clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.optimizer.step()
                    self.model.temperature_clamp()
                    running_reg_loss += reg_loss


                    running_spectral_loss += sp_loss
                    running_ortho_loss += o_loss
                    running_cluster_loss += cl_loss
                    pbar.update(1)
                    iter_num += 1

                train_reg_loss = running_reg_loss / len(train_loader)
                self.train_losses.append(train_reg_loss)
                train_spectral_loss = running_spectral_loss / len(train_loader)
                train_ortho_loss = running_ortho_loss / len(train_loader)
                train_cluster_loss = running_cluster_loss / len(train_loader)

                train_str1 = f"Train MSE Loss: {train_reg_loss:.4f}"

                if epoch % evaluate_epoch == 0  and val_loader is None:
                    test_result = self.eval(test_loader)
                    val_mse = test_result.get('mse', float('inf'))
                    self.val_mses.append(val_mse)
                    

                    if test_result[self.evaluate_metric] < best_result:
                        better_than_previous = True
                    else:
                        better_than_previous = False


                    if better_than_previous:
                        best_result = test_result[self.evaluate_metric]
                        best_test_epoch = epoch
                        best_test_result_str = f'Test Results at Epoch {epoch}: ' + json.dumps(test_result, indent=4,
                                                                                               sort_keys=True)
                        with open(self.result_path + '/save_model_seed{}'.format(
                                self.runid) + '/full_result-{}.txt'.format(self.runid),
                                  'a+') as f:
                            f.write(f"MSE improved at epoch {best_test_epoch}; best_mse: {test_result['mse']}\n")
                        # torch.save(self.model.state_dict(), os.path.join(self.result_path,'save_model_seed{}'.format(self.runid),'model.pt'))
                        patience = 100
                    else:
                        patience -= 1

                    if patience <= 0:
                        print(f"Early stopping at epoch {epoch} due to no improvement in the last 100 epochs.")
                        break


                test_result = {k:round(v,6) for k, v in test_result.items()}
                test_str = f'Test Results: ' + json.dumps(test_result, indent=4, sort_keys=True)
                output_lines[3] = f'Epoch {epoch:03d} with LR {curr_lr_rate:.6f}: Model Results'
                output_lines[4] = '-'*40
                output_lines[10] = test_str

                with open(self.result_path +'/save_model_seed{}'.format(self.runid)+'/full_result-{}.txt'.format(self.runid),'a+') as f:
                    f.write('-'*30 + f'\nEpoch: {epoch:03d} - Model Results\n' + '-'*30 + '\n')
                    f.write(train_str1 +'\n')
                    f.write(test_str +'\n')

        with open(self.result_path + '/save_model_seed{}'.format(self.runid) + '/full_result-{}.txt'.format(self.runid),
                  'a+') as f:
            f.write(f"Best Test Results at Epoch {best_test_epoch}:\n")
            f.write(best_test_result_str + "\n")


        self.plot_loss_curve()

    def plot_loss_curve(self):
        start_epoch = 10

        epochs = range(start_epoch + 1, len(self.train_losses) + 1)
        val_epochs = range(start_epoch + 1, len(self.val_mses) + 1)


        train_losses_trimmed = self.train_losses[start_epoch:]
        val_mses_trimmed = self.val_mses[start_epoch:]

        plt.figure(figsize=(10, 5))


        plt.plot(epochs, train_losses_trimmed, label='Train Loss', marker='o')


        plt.plot(val_epochs, val_mses_trimmed, label='Validation MSE', marker='x')

        plt.title('Training Loss and Validation MSE (from Epoch 10)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss/MSE')
        plt.legend()
        plt.grid(True)


        plt.savefig(os.path.join(self.result_path + '/save_model_seed{}'.format(self.runid)+ f'/train_val_curve_from_epoch10_seed{self.runid}.png'))
        plt.show()

    def eval(self, data_loader):
        reg_preds = []
        reg_truths = []

        running_reg_loss = 0
        running_spectral_loss = 0 
        running_ortho_loss = 0 
        running_cluster_loss = 0  

        self.model.eval()
        eval_result = {}
        with torch.no_grad():
            for data in tqdm(data_loader, leave=False, desc='evaluating'):
                data = data.to(self.device)
                reg_pred, sp_loss, o_loss, cl_loss, _ = self.model(
                        # Molecule
                        mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
                        atom_edge_index=data.mol_edge_index, clique_x=data.clique_x, 
                        clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
                        # Protein
                        residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                        residue_edge_index=data.prot_edge_index,
                        residue_edge_weight=data.prot_edge_weight,
                        # Mol-Protein Interaction batch
                        mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch, clique_batch=data.clique_x_batch
                    )
                ## Loss compute
                reg_loss = 0

                loss_val = 0
                loss_val += sp_loss
                loss_val += o_loss
                loss_val += cl_loss
                sp_loss = sp_loss.item()
                o_loss = o_loss.item()
                cl_loss = cl_loss.item()

                if reg_pred is not None:
                    reg_pred = reg_pred.squeeze().reshape(-1)
                    reg_y = data.reg_y.squeeze().reshape(-1)
                    reg_loss = self.regression_loss(reg_pred, reg_y) * self.regression_weight
                    loss_val += reg_loss
                    reg_loss = reg_loss.item()
                    reg_preds.append(reg_pred)
                    reg_truths.append(reg_y)

                running_reg_loss += reg_loss
            eval_reg_loss = running_reg_loss / len(data_loader)
            eval_result['regression_loss'] = eval_reg_loss


        if len(reg_truths) > 0:
            reg_preds = torch.cat(reg_preds).detach().cpu().numpy()
            reg_truths = torch.cat(reg_truths).detach().cpu().numpy()

            eval_reg_result = evaluate_reg(reg_truths, reg_preds)
            eval_result.update(eval_reg_result)


        return eval_result









