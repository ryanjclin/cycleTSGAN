import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import os, time
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from model.generator import Generator
from model.discriminator import Discriminator


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_constant_then_linear_decay_schedule(optimizer: Optimizer, start_decay_epoch: int, total_epochs: int, last_epoch: int = -1):
    """
    Create a schedule with a learning rate that:
    - Remains constant until start_decay_epoch
    - Then linearly decays to 0 by the end of total_epochs
    
    :param optimizer: The optimizer for which to schedule the learning rate.
    :param start_decay_epoch: The epoch to start the decay.
    :param total_epochs: The total number of epochs.
    :param last_epoch: The index of the last epoch. Default: -1.
    """
    def lr_lambda(current_epoch: int):
        if current_epoch < start_decay_epoch:
            return 1.0
        return max(
            2e-6, 
            (total_epochs - current_epoch) / (total_epochs - start_decay_epoch)
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# def sinkhorn(r, c, M, device, reg=1, error_thres=1e-4, num_iters=50):

#     n, d1, d2 = M.shape

#     K = (-M / reg).exp()        # (n, d1, d2)
#     u = torch.ones_like(r) / d1 # (n, d1)
#     u = u.to(device)
#     v = torch.ones_like(c) / d2 # (n, d2)
#     v = v.to(device)

#     for _ in range(num_iters):
#         r0 = u
#         # u = r / K \cdot v
#         u = r / (torch.einsum('ijk,ik->ij', [K, v]) )
#         # v = c / K^T \cdot u
#         v = c / (torch.einsum('ikj,ik->ij', [K, u]) )

#         err = (u - r0).abs().mean()
#         if err.item() < error_thres:
#             break
#     T = torch.einsum('ij,ik->ijk', [u, v]) * K

#     distance = torch.sum(T * M)

#     return distance

# def sinkhorn_loss_fn(real_data, fake_data, device, config):
#     """
#     real_data: [batch_size, var_num, seq_len]
#     fake_data: [batch_size, var_num, seq_len]
#     """

#     loss = torch.zeros((1)).to(device)
#     for real_dat, fake_dat in zip(real_data, fake_data):
#         for real, fake in zip(real_dat, fake_dat):
#             n = len(real)
#             dist = (torch.arange(n).view(1, n) - torch.arange(n).view(n, 1)).abs().float().to(device)
#             dist /= dist.max()
#             distance = sinkhorn(r=real.unsqueeze(dim=0), c=fake.unsqueeze(dim=0), device=device, M=dist.unsqueeze(dim=0))
#             loss += distance

#     return loss

# def custom_L1(real_data, fake_data, device):
#     """
#     real_data: [batch_size, var_num, seq_len]
#     fake_data: [batch_size, var_num, seq_len]
#     """

#     L2 = nn.MSELoss()
#     L1 = nn.L1Loss()

#     loss = torch.zeros((1)).to(device)
#     for real_dat, fake_dat in zip(real_data, fake_data):
#         for real, fake in zip(real_dat, fake_dat):
#             distance = L1(real, fake)
#             loss += distance

#     return loss

def training1(config, data, device, writer, preprocess_result):
    fre_normal = data['normal']
    fre_faulty = data['faulty']

    # ---------------------------------- model, loss fn definition ---------------------------------------------
    ''' add data into TensorDataset for efficiency'''
    torch_dataset = Data.TensorDataset(torch.Tensor(fre_faulty), torch.Tensor(fre_normal))
    data_loader = Data.DataLoader(dataset = torch_dataset, batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers'], drop_last = False)

    disc_Fault = Discriminator(config, preprocess_result['source_encoding']).to(device)
    disc_Normal = Discriminator(config, preprocess_result['source_encoding']).to(device)
    gen_FaultToNormal = Generator(config, preprocess_result['source_encoding']).to(device) # fault to normal
    gen_NormalToFault = Generator(config, preprocess_result['source_encoding']).to(device) # normal to fault

    disc_Fault.train()
    disc_Normal.train()
    gen_FaultToNormal.train()
    gen_NormalToFault.train()

    ''' optimizer for Discriminators '''
    # opt_disc = optim.Adam( list(disc_Fault.parameters()) + list(disc_Normal.parameters()), lr = config['learning_rate'], betas = (0.5, 0.999),)
    opt_disc = AdamW( list(disc_Fault.parameters()) + list(disc_Normal.parameters()), lr = config['learning_rate'], correct_bias=True)

    ''' optimizer for Generators '''
    # opt_gen = optim.Adam( list(gen_FaultToNormal.parameters()) + list(gen_NormalToFault.parameters()), lr = config['learning_rate'], betas = (0.5, 0.9),)
    opt_gen = AdamW(list(gen_FaultToNormal.parameters()) + list(gen_NormalToFault.parameters()), lr=config['learning_rate'], correct_bias=True)

    ''' learning rate scheduler'''
    # lr_scheduler_disc_Fault = get_linear_schedule_with_warmup(opt_disc_Fault, num_warmup_steps=config['warmup_steps'], num_training_steps=config['epoch'])
    # lr_scheduler_disc_Normal = get_linear_schedule_with_warmup(opt_disc_Normal, num_warmup_steps=config['warmup_steps'], num_training_steps=config['epoch'])
    # lr_scheduler_gen = get_linear_schedule_with_warmup(opt_gen, num_warmup_steps=config['warmup_steps'], num_training_steps=config['epoch'])

    lr_scheduler_disc = get_constant_then_linear_decay_schedule(opt_disc, start_decay_epoch = config['start_decay_epoch'], total_epochs = config['epoch'])
    lr_scheduler_gen = get_constant_then_linear_decay_schedule(opt_gen, start_decay_epoch = config['start_decay_epoch'], total_epochs = config['epoch'])

    L2 = nn.MSELoss()
    L1 = nn.L1Loss()

    # ---------------------------------- training ---------------------------------------------

    time_start=time.time()
    print('training start!')

    for epoch in tqdm(range(1, config['epoch'] + 1)):
        G_losses = []
        D_losses = []    
        cycle_loss = []
        
        time_start_epoch = time.time()
        
        for i, data in enumerate(data_loader):
            x_fault, x_normal = data        
            x_fault = x_fault.to(device)
            x_normal = x_normal.to(device)

            ''' Train Discriminators'''
            for _ in range(config['n_critic']):
                ''' Train Discriminators_Normal'''
                fake_normal = gen_FaultToNormal(x_fault)
                D_Normal_real = disc_Normal(x_normal)
                D_Normal_fake = disc_Normal(fake_normal.detach())
                D_Normal_loss = - (torch.mean(D_Normal_real) - torch.mean(D_Normal_fake))# first version of WGAN loss

                ''' Train Discriminators_Fault '''
                fake_fault = gen_NormalToFault(x_normal)
                D_Fault_real = disc_Fault(x_fault)
                D_Fault_fake = disc_Fault(fake_fault.detach())
                D_Fault_loss = - (torch.mean(D_Fault_real) - torch.mean(D_Fault_fake))# first version of WGAN loss

                D_loss = (D_Normal_loss + D_Fault_loss)/2

                opt_disc.zero_grad()
                D_loss.backward()
                opt_disc.step() 

                # clip critic weights between -0.01, 0.01
                for p in disc_Normal.parameters():
                    p.data.clamp_(-0.01, 0.01)        

                # clip critic weights between -0.01, 0.01
                for p in disc_Fault.parameters():
                    p.data.clamp_(-0.01, 0.01) 

                D_losses.append(D_loss.data.cpu().numpy())
                
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
            if config['lambda_identity'] != 0:
                identity_fault = gen_NormalToFault(x_fault)
                identity_normal = gen_FaultToNormal(x_normal)

                # '''use L1 as generator loss'''
                identity_fault_loss = L1(x_fault, identity_fault)
                identity_normal_loss = L1(x_normal, identity_normal)
                # '''use sinkhorn as generator loss'''
                # identity_fault_loss = sinkhorn_loss_fn(x_fault, identity_fault, device, config)
                # identity_normal_loss = sinkhorn_loss_fn(x_normal, identity_normal, device, config)


            # add all togethor
            if config['lambda_identity'] != 0:
                G_loss = (
                    loss_G_Normal + loss_G_Fault
                    + (cycle_normal_loss + cycle_fault_loss) * config['lambda_cycle']
                    + (identity_fault_loss + identity_normal_loss) * config['lambda_identity'] 
                )
            else:
                G_loss = (
                    loss_G_Normal + loss_G_Fault
                    + (cycle_normal_loss + cycle_fault_loss) * config['lambda_cycle']
                )

            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()   

            G_losses.append(G_loss.data.cpu().numpy())
            cycle_loss.append((cycle_normal_loss + cycle_fault_loss).data.cpu().numpy())

        lr_scheduler_disc.step()
        lr_scheduler_gen.step()


        ############        show loss      #######################
        if (epoch) % 1 == 0:
            print(f"epoch [{(epoch + 1)}/{config['epoch']}]: d_loss: {np.mean(D_losses):.6f} g_loss: {np.mean(G_losses):.6f}, cycle_loss: {np.mean(cycle_loss):.6f}")
            
            lr_disc = lr_scheduler_disc.get_last_lr()[0]
            lr_gen = lr_scheduler_gen.get_last_lr()[0]
            print(f"epoch [{epoch + 1}/{config['epoch']}]: lr_disc: {lr_disc}, lr_gen: {lr_gen}")
            
            writer.add_scalar("G_losses", np.mean(G_losses), epoch)
            writer.add_scalar("D_losses", np.mean(D_losses), epoch)
            writer.add_scalar("cycle_loss", np.mean(cycle_loss), epoch)

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


