import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, time
import pywt
from statsmodels.stats.diagnostic import acorr_ljungbox  
from fastdtw import fastdtw
from collections import defaultdict 

from model.generator import Generator
from model.discriminator import Discriminator



def sinkhorn(r, c, M, device, reg=1, error_thres=1e-4, num_iters=50):

    n, d1, d2 = M.shape

    K = (-M / reg).exp()        # (n, d1, d2)
    u = torch.ones_like(r) / d1 # (n, d1)
    u = u.to(device)
    v = torch.ones_like(c) / d2 # (n, d2)
    v = v.to(device)

    for _ in range(num_iters):
        r0 = u
        # u = r / K \cdot v
        u = r / (torch.einsum('ijk,ik->ij', [K, v]) )
        # v = c / K^T \cdot u
        v = c / (torch.einsum('ikj,ik->ij', [K, u]) )

        err = (u - r0).abs().mean()
        if err.item() < error_thres:
            break
    T = torch.einsum('ij,ik->ijk', [u, v]) * K

    distance = torch.sum(T * M)

    return distance

def sinkhorn_loss_fn(data1, data2, device, config):
    M = torch.cdist(data1, data2, p=2)  # (batch_size, var_num, var_num)

    # 定義邊際分布 r 和 c，這裡使用均勻分布
    r = torch.ones(config['batch_size'], config['var_num']).to(device) / config['var_num']
    c = torch.ones(config['batch_size'], config['var_num']).to(device) / config['var_num']

    # 使用 Sinkhorn 函數計算兩個數據之間的差異性
    distance = sinkhorn(r, c, M, device)

    return distance

def training(config, data, device, writer):
    fre_normal = data['normal']
    fre_faulty = data['faulty']

    # ---------------------------------- model, loss fn definition ---------------------------------------------
    ''' add data into TensorDataset for efficiency'''
    torch_dataset = Data.TensorDataset(torch.Tensor(fre_faulty), torch.Tensor(fre_normal))
    data_loader = Data.DataLoader(dataset = torch_dataset, batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers'], drop_last = False)

    disc_Fault = Discriminator(config['batch_size'], config['var_num'], config['seq_len']).to(device)
    disc_Normal = Discriminator(config['batch_size'], config['var_num'], config['seq_len']).to(device)
    gen_FaultToNormal = Generator(config['batch_size'], config['var_num'], config['seq_len'], config['dim_z']).to(device) # fault to normal
    gen_NormalToFault = Generator(config['batch_size'], config['var_num'], config['seq_len'], config['dim_z']).to(device) # normal to fault

    ''' optimize two Discriminator simultaneously, but i remember it doesn't work '''
    # opt_disc = optim.Adam( list(disc_Fault.parameters()) + list(disc_Normal.parameters()), lr = learning_rate, betas = (0.5, 0.999),)

    ''' optimizer for Discriminators '''
    opt_disc_Fault = optim.Adam(disc_Fault.parameters(), lr = config['learning_rate'], betas = (0.5, 0.9),)
    opt_disc_Normal = optim.Adam(disc_Normal.parameters(), lr = config['learning_rate'], betas = (0.5, 0.9),)

    ''' optimizer for Generators '''
    opt_gen = optim.Adam( list(gen_FaultToNormal.parameters()) + list(gen_NormalToFault.parameters()), lr = config['learning_rate'], betas = (0.5, 0.9),)

    L1 = nn.L1Loss()

    # ---------------------------------- training ---------------------------------------------

    time_start=time.time()
    print('training start!')

    for epoch in range(1, config['epoch'] + 1):
        G_losses = []
        D_losses = []    
        
        time_start_epoch = time.time()
        
        for i, data in enumerate(data_loader):
            x_fault, x_normal = data        
            x_fault = x_fault.to(device)
            x_normal = x_normal.to(device)

            ''' Train Discriminators_Normal'''
            fake_normal = gen_FaultToNormal(x_fault)
            D_Normal_real = disc_Normal(x_normal)
            D_Normal_fake = disc_Normal(fake_normal.detach())
            D_Normal_loss = torch.mean(-D_Normal_real + D_Normal_fake)

            opt_disc_Normal.zero_grad()
            D_Normal_loss.backward()
            opt_disc_Normal.step() 
            
            # clip critic weights between -0.01, 0.01
            for p in disc_Normal.parameters():
                p.data.clamp_(-0.01, 0.01)        
                    
            ''' Train Discriminators_Fault '''
            fake_fault = gen_NormalToFault(x_normal)
            D_Fault_real = disc_Fault(x_fault)
            D_Fault_fake = disc_Fault(fake_fault.detach())
            D_Fault_loss = torch.mean(-D_Fault_real + D_Fault_fake) # first version of WGAN loss

            opt_disc_Fault.zero_grad()
            D_Fault_loss.backward()
            opt_disc_Fault.step()      
            
            # clip critic weights between -0.01, 0.01
            for p in disc_Fault.parameters():
                p.data.clamp_(-0.01, 0.01) 

            D_losses.append(D_Normal_loss.data.cpu().numpy() + D_Fault_loss.data.cpu().numpy())
                
            '''Train Generators Fault and Normal'''
            # adversarial loss for both generators
            Disc_Normal_fake = disc_Normal(fake_normal)
            Disc_Fault_fake = disc_Fault(fake_fault)
            loss_G_Normal = -torch.mean(Disc_Normal_fake) # first version of WGAN loss
            loss_G_Fault = -torch.mean(Disc_Fault_fake)   # first version of WGAN loss

            # cycle loss
            cycle_fault = gen_NormalToFault(fake_normal)
            cycle_normal = gen_FaultToNormal(fake_fault)

            # '''use L1 as generator loss'''
            cycle_normal_loss = L1(x_normal, cycle_normal)  
            cycle_fault_loss = L1(x_fault, cycle_fault)
            # '''use sinkhorn as generator loss'''
            # cycle_normal_loss = sinkhorn_loss_fn(x_normal, cycle_normal, device, config)  
            # cycle_fault_loss = sinkhorn_loss_fn(x_fault, cycle_fault, device, config)

            #  identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_fault = gen_NormalToFault(x_fault)
            identity_normal = gen_FaultToNormal(x_normal)

            # '''use L1 as generator loss'''
            identity_fault_loss = L1(x_fault, identity_fault)
            identity_normal_loss = L1(x_normal, identity_normal)
            # '''use sinkhorn as generator loss'''
            # identity_fault_loss = sinkhorn_loss_fn(x_fault, identity_fault, device, config)
            # identity_normal_loss = sinkhorn_loss_fn(x_normal, identity_normal, device, config)



            # print(f"loss_G_Normal: {loss_G_Normal.shape}")
            # print(f"loss_G_Fault: {loss_G_Fault.shape}")
            # print(f"cycle_normal_loss: {cycle_normal_loss.shape}")
            # print(f"cycle_fault_loss: {cycle_fault_loss.shape}")
            # print(f"identity_fault_loss: {identity_fault_loss.shape}")
            # print(f"identity_normal_loss: {identity_normal_loss.shape}")

            # add all togethor
            G_loss = (
                loss_G_Normal + loss_G_Fault
                + (cycle_normal_loss + cycle_fault_loss) * config['lambda_cycle']
                + (identity_fault_loss + identity_normal_loss) * config['lambda_identity'] 
            )

            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()   
            
            G_losses.append(G_loss.data.cpu().numpy())
            
        ############        show loss      #######################
        if (epoch) % 1 == 0:
            print('[%d/%d] loss_d: %.3f, loss_g: %.3f'%((epoch + 1), config['epoch'], np.mean(D_losses), np.mean(G_losses)))
            writer.add_scalar("G_losses", np.mean(G_losses), epoch)
            writer.add_scalar("D_losses", np.mean(D_losses), epoch)

        time_end_epoch = time.time()
        print('runtime per epoch', time_end_epoch - time_start_epoch,'s')         
        
        ##########        save      #######################
        if (epoch) % config['save_step'] == 0: #5
            checkpoint_path = os.path.join(config['checkpoint'])
            print("save model in " + checkpoint_path)
            torch.save(disc_Fault.state_dict(), os.path.join(checkpoint_path, f'disc_Fault_{epoch}.bin'))
            torch.save(disc_Normal.state_dict(), os.path.join(checkpoint_path, f'disc_Normal_{epoch}.bin'))
            torch.save(gen_FaultToNormal.state_dict(), os.path.join(checkpoint_path, f'gen_FaultToNormal_{epoch}.bin'))
            torch.save(gen_NormalToFault.state_dict(), os.path.join(checkpoint_path, f'gen_NormalToFault_{epoch}.bin'))
            print('save success')    

    print("training Finished!")
    time_end = time.time()
    print('Total runtime',time_end-time_start,'s')    

    writer.close()
    torch.cuda.empty_cache()


