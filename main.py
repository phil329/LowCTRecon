import argparse
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from dataset import CTDataset
from logger import WANDBLoggerX
from networks import RED_CNN
from unet import UNet
from metrics.measure import compute_measure


'''
CUDA_VISIBLE_DEVICES=3 python main.py \
    --data_root=./CT_Reconstruction_256x256_1m \
    --print_freq=10 --save_freq=100 --img_size=256 \
    --epochs=100 --lr=1e-4 --num_workers=32 --device=cuda \
    --model_name=unet4 --batch=16 --run_name=noflip_L2_0427

'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_root', type=str, default='./CT_Reconstruction_256x256_1m')
    parser.add_argument('--print_freq',type=int,default=10)
    parser.add_argument('--save_freq',type=int,default=1000)
    parser.add_argument('--model_name',type=str,default='unet')
    parser.add_argument('--run_name',type=str,default='default')
    
    parser.add_argument('--img_size',type=int,default=256)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patch_num',type=int,default=10)
    parser.add_argument('--patch_size',type=int,default=64)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--decay_iters', type=int, default=3000)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--device', type=str,default='cuda')
    parser.add_argument('--num_workers', type=int, default=32)

    args = parser.parse_args()
    
    print(args)
    
    logger = WANDBLoggerX(save_root=os.path.join('./output',args.model_name,args.run_name),
                          print_freq=args.print_freq,
                          config=args,
                          project='LowCTRecon',
                          entity='blli',
                          name='{}_{}'.format(args.model_name,args.run_name)
                          )
    
    augment_dict = {'flip':0.5}
    train_transform = T.Compose([
        T.ToTensor(),
        #T.Resize((args.img_size,args.img_size)),
        T.Normalize(mean=0.5,std=0.5)
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        #T.Resize((args.img_size,args.img_size)),
        T.Normalize(mean=0.5,std=0.5)
    ])
    
    
    train_dataset = CTDataset(args.data_root,
                              mode='train',
                              transform=train_transform,
                              augment_dict=augment_dict,
                              patches=(True,args.patch_num,args.patch_size)
                              )
    test_dataset = CTDataset(args.data_root,mode='val',transform=test_transform)
    
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=args.batch,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    
    test_batchsize = 8
    test_loader = DataLoader(test_dataset,
                             shuffle=True,
                             batch_size=test_batchsize,
                             num_workers=args.num_workers,
                             pin_memory=False,
                             drop_last=True)
    
    show_quarter,show_full = next(iter(test_loader))
    show_quarter,show_full = show_quarter.to(args.device),show_full.to(args.device)
    
    print('** dataloader initialized successfully !')
    
    #model = RED_CNN()
    model = UNet(repeat_num=4,conv_dim=64)
    model.to(args.device)
    
    L1_loss = nn.L1Loss()
    L2_loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), args.lr)
    logger.modules = [model,optimizer]
    
    for epoch in range(args.epochs):
        for i, (quarter_img,full_img) in tqdm(enumerate(train_loader),total=len(train_loader)):
            n_iter = epoch*len(train_loader)+i
            
            if quarter_img.dim() == 5:
                b,p,c,h,w = quarter_img.shape
                quarter_img = quarter_img.view(b*p,c,h,w)
                full_img = full_img.view(b*p,c,h,w)
                
            quarter_img,full_img = quarter_img.to(args.device),full_img.to(args.device)
            
            pred_img = model(quarter_img)
            
            # loss = L1_loss(pred_img,full_img) + L2_loss(pred_img,full_img)
            loss = L2_loss(pred_img,full_img)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                # compute PSNR, SSIM, RMSE
                if n_iter % (args.save_freq) == 0:
                    
                    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
                    
                    cnt = 0
                    for test_i,(test_quarter_img,test_full_img) in tqdm(enumerate(test_loader),total=len(test_loader)):
                        cnt += 1
                        test_quarter_img,test_full_img = test_quarter_img.to(args.device),test_full_img.to(args.device)
                        
                        test_pred_img = model(test_quarter_img)
                        
                        result = compute_measure(test_pred_img,test_full_img)
                        
                        pred_psnr_avg += result['psnr']
                        pred_ssim_avg += result['ssim']
                        pred_rmse_avg += result['rmse']
                    
                    pred_psnr_avg /= cnt
                    pred_ssim_avg /= cnt
                    pred_rmse_avg /= cnt
                    
                    # show case
                    img1 = torchvision.utils.make_grid(show_quarter,normalize=True,range=(-1,1))
                    img2 = torchvision.utils.make_grid(show_full,normalize=True,range=(-1,1))
                    img3_tensor = model(show_quarter)
                    img3 = torchvision.utils.make_grid(img3_tensor,normalize=True,range=(-1,1))
                    grid_img = to_pil_image(torch.cat([img2,img1,img3],dim=1))
                    
                    logger.save_image(grid_img,n_iter,'full-quarter-pred')
                    
                    logger.msg([pred_psnr_avg,pred_ssim_avg,pred_rmse_avg],n_iter)
                    if n_iter % (args.save_freq*10) == 0:
                        logger.checkpoints(n_iter)
                
            logger.msg([loss,],n_iter)
    
    
    
    
    
    