import argparse
import torch
import dataset
import model
import cv2 as cv
import HeadPoseEstimation
import vgg_face
import numpy as np
import torch.nn as nn
import patchGAN
import random
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from EMODataset import EMODataset
import torch.nn.functional as F

data_set_length = 42
img_size = 512

hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
L1_loss = nn.L1Loss(reduction='mean')
feature_matching_loss = nn.MSELoss()
cosine_dist = nn.CosineSimilarity()

patch = (1, img_size // 2 ** 4, img_size // 2 ** 4)

def cosine_distance(args, z1, z2):
  
  res = args.s_cos * (torch.sum(cosine_dist(z1[0], z2[0])) - args.m_cos)
  res += args.s_cos * (torch.sum(cosine_dist(z1[1], z2[1])) - args.m_cos)   
  res += args.s_cos * (torch.sum(cosine_dist(z1[2], z2[2])) - args.m_cos)
  
  return res

def cosine_loss(args, descriptor_driver, 
                descriptor_source_rand, descriptor_driver_rand):
  
  z_dri = descriptor_driver
  z_dri_rand = descriptor_driver_rand
  
  # Create descriptors to form the pairs
  # z_s*->d
  z_src_rand_dri = [descriptor_source_rand[0], 
                    z_dri[1], 
                    z_dri[2]]

  # Form the pairs
  pos_pairs = [(z_dri, z_dri), (z_src_rand_dri, z_dri)]
  neg_pairs = [(z_dri, z_dri_rand), (z_src_rand_dri, z_dri_rand)]

  # Calculate cos loss
  sum_neg_paris = torch.exp(cosine_distance(args, neg_pairs[0][0], neg_pairs[0][1])) + torch.exp(cosine_distance(args, neg_pairs[1][0], neg_pairs[1][1]))
  
  L_cos = torch.zeros(dtype=torch.float)
  for i in range(len(pos_pairs)):
    L_cos += torch.log(torch.exp(cosine_distance(args, pos_pairs[0][0], pos_pairs[0][1])) / (torch.exp(cosine_distance(args, pos_pairs[0][0], pos_pairs[0][1])) + sum_neg_paris))
  
  return L_cos
      
def warp_3d(x, theta):
  # Generate 3D grid
  grid = F.affine_grid(theta, x.size())
  
  # Sample the input tensor using the grid
  warped_x = F.grid_sample(x, grid)
  
  return warped_x


def train(args, models, device, driver_loader, source_loader, optimizers, schedulers, source_img_random, driver_img_random,source_loader_origin, driver_loader_origin):

  # Instantiate each model.
  Eapp1 = models['Eapp1'] 
  Eapp2 = models['Eapp2']
  Emtn_facial = models['Emtn_facial']
  Warp_G = models['Warp_G']
  G3d = models['G3d']
  G2d = models['G2d']
  vgg_IN = models['Vgg_IN']
  vgg_face = models['Vgg_face']
  discriminator = models['patchGAN']
         
  train_loss = 0.0

  # Training procedure starts here.
  for idx in range(args.iteration):


    # pass the data through Eapp1 & 2.
    v_s = Eapp1(source_img) 
    e_s = Eapp2(source_img)

    # Emtn.

    # Second part of Emtn : Generate facial expression latent vector z
    # based on a ResNet-18 network
    z_s = Emtn_facial(source_img)
    z_d = Emtn_facial(driver_img)
    
    # Warp_Generator
    
    # First part of Warp Generator: Generate warping matrix  
    # based on its transformation matrix.
    # Note: the head pose prediction is also completed in this function.
    W_rt_s = HeadPoseEstimation.head_pose_estimation(source_imgs_origin)
    W_rt_d = HeadPoseEstimation.head_pose_estimation(driver_imgs_origin)
    
    # Second part of Warp Generator: Generate emotion warper.
    W_em_s = Warp_G(z_s + e_s)
    W_em_d = Warp_G(z_d + e_s)
  

    # Replace cv.warpPerspective with warp_3d function
    warp_3d_vs = warp_3d(v_s, W_rt_s)
    warp_3d_vs = warp_3d(warp_3d_vs, W_em_s)

    vs_d = warp_3d(warp_3d_vs, W_rt_d)
    vs_d = warp_3d(vs_d, W_em_d)
    # Pass data into G3d
    # output = G3d(warp_3d_vs)
    
  # In the train function:
    # Pass data into G3d
    output_3d = G3d(warp_3d_vs)

    # Orthographically project the 3D features into 2D
    output_3d = output_3d.squeeze(2)

    # Pass the projected features into G2d
    output = G2d(output_3d)
    # IN loss
    L_IN = L1_loss(vgg_IN(output), vgg_IN(driver_img))
    
    # face loss
    L_face = L1_loss(vgg_face(output), vgg_face(driver_img))
    
    # adv loss
    # Adversarial ground truths
    valid = Variable(torch.Tensor(np.ones((driver_img.size(0), *patch))), requires_grad=False)
    fake = Variable(torch.Tensor(-1*np.ones((driver_img.size(0), *patch))), requires_grad=False)
        
    # real loss
    pred_real = discriminator(driver_img, source_img)
    loss_real = hinge_loss(pred_real, valid)

    # fake loss        
    pred_fake = discriminator(output.detach(), source_img)
    loss_fake = hinge_loss(pred_fake, fake)
        
    L_adv = 0.5 * (loss_real + loss_fake)
    
    # feature mapping loss
    L_feature_matching = feature_matching_loss(output, driver_img)
    
    # Cycle consistency loss 
    # Feed base model with randomly sampled image.
    e_s_rand = Eapp2(source_img_random)
    trans_mat_source_rand = HeadPoseEstimation.head_pose_estimation(source_img_random)
    z_s_rand = Emtn_facial(source_img_random)

    trans_mat_driver_rand = HeadPoseEstimation.head_pose_estimation(driver_img_random)
    z_d_rand = Emtn_facial(driver_img_random)
    
    descriptor_driver = [e_s, W_rt_d, z_d]
    descriptor_source_rand = [e_s_rand, trans_mat_source_rand, z_s_rand]
    descriptor_driver_rand = [e_s_rand, trans_mat_driver_rand, z_d_rand]
    
    L_cos = cosine_loss(args,descriptor_driver, 
                            descriptor_source_rand, descriptor_driver_rand)
    
    L_per = args.weight_IN * L_IN + args.weight_face * L_face
        
    L_gan = args.weight_adv * L_adv + args.weight_FM * L_feature_matching
        
    L_final = L_per + L_gan + args.weight_cos * L_cos
        
    # Optimizer and Learning rate scheduler.
    # optimizer
    for i in range(len(optimizers)):
      optimizers[i].zero_grad()
        
    L_final.backward()
        
    for i in range(len(optimizers)):  
      optimizers[i].step()
      schedulers[i].step()
        
    train_loss += L_final
    train_loss /= idx
    
    # Print log 
    print('Iteration: {} / {} : train loss is: {}'.format(idx, args.iteration, train_loss))

def distill(args, teacher, student, device, driver_loader):

    # Set teacher model to eval mode
    teacher.eval()

    # Training loop for student
    for idx in range(args.iteration):

        # Sample driver frame and index
        driver_img = next(iter(driver_loader)).to(device)
        idx = random.randint(0, args.num_avatars-1)

        # Generate pseudo-ground truth with teacher
        with torch.no_grad():
            output_HR = teacher(driver_img, idx) 

        # Get student prediction
        output_DT = student(driver_img, idx)

        # Calculate losses
        L_per = perceptual_loss(output_DT, output_HR) 
        L_adv = adversarial_loss(output_DT, output_HR)

        L_final = L_per + L_adv

        # Optimize student
        student_optimizer.zero_grad()
        L_final.backward()
        student_optimizer.step()

        # Print log
        if idx % args.print_freq == 0:
            print('Iteration: {} / {} : distillation loss is: {}'.format(idx, args.iteration, L_final.item()))
            
            
def main():
    parser = argparse.ArgumentParser(description="Megaportrait Pytorch implementation.")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training. default=16')
    parser.add_argument('--iteration', type=int, default=20000, metavar='N',
                        help='input batch size for training. default=20000')  
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train. default = 50')
    parser.add_argument('--weight-IN', type=int, default=20, metavar='N',
                        help='weight parameter for IN loss. default = 20')
    parser.add_argument('--weight-face', type=int, default=5, metavar='N',
                        help='weight parameter for face loss. default = 5')
    parser.add_argument('--weight-adv', type=int, default=1, metavar='N',
                        help='weight parameter for adv loss. default = 1')
    parser.add_argument('--weight-FM', type=int, default=40, metavar='N',
                        help='weight parameter for feature matching loss. default = 40')
    parser.add_argument('--weight-cos', type=int, default=2, metavar='N',
                        help='weight parameter for cos loss. default = 2')
    parser.add_argument('--s-cos', type=int, default=5, metavar='N',
                        help='s parameter in cos loss. default = 5')
    parser.add_argument('--m-cos', type=float, default=0.2, metavar='N',
                        help='m parameter in cos loss. default = 0.2')
    parser.add_argument('--lr', type = float, default=2e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--num-avatars', type=int, default=100, metavar='N',
                        help='number of avatars for distillation')
    parser.add_argument('--print-freq', type=int, default=100, metavar='N',
                        help='print frequency for distillation')               
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
  


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
  
    # Transformation on images.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter()
    ])

    # Initialize EMODataset
    dataset = EMODataset(
        use_gpu=use_cuda,
        width=img_size,
        height=img_size,
        n_sample_frames=args.n_sample_frames,
        sample_rate=args.sample_rate,
        img_scale=(1.0, 1.0),
        data_dir=args.data_dir,
        video_dir=args.video_dir,
        json_file=args.json_file,
        stage='stage1-0-framesencoder',
        transform=transform
    )

    # Configuration and Hyperparameters
    num_epochs = 10  # Example number of epochs
    learning_rate = 1e-3  # Example learning rate

    # Initialize DataLoader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Model, Criterion, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    Eapp1 = model.Eapp1().to(device)
    # Eapp2 = model.Eapp2().to(device)
    Eapp2 = model.Eapp2(repeat=[3, 4, 6, 3])
    # Emtn_facial = model.Emtn_facial().to(device)

    Emtn_facial = model.Emtn_facial(in_channels=3).to(device)
    Emtn_head = model.Emtn_head(in_channels=3, resblock=model.ResBlock).to(device)
    Warp_G = model.WarpGenerator(input_channels=1).to(device)
    G3d = model.G3d(input_channels=96).to(device)
    G2d = model.G2d(input_channels=32).to(device)
    Vgg_IN = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
    Vgg_face = vgg_face.VggFace().to(device)
    discriminator = patchGAN.Discriminator().to(device)

    models = {
        'Eapp1': Eapp1,
        'Eapp2': Eapp2,
        'Emtn_facial': Emtn_facial,
        'Emtn_head': Emtn_head,
        'Warp_G': Warp_G,
        'G3d': G3d,
        'G2d': G2d,
        'Vgg_IN': Vgg_IN,
        'Vgg_face': Vgg_face,
        'patchGAN': discriminator
    }

    optimizers = [
        optim.Adam(models['Eapp1'].parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Eapp2'].parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Emtn_facial'].parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Emtn_head'].parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Warp_G'].parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['G3d'].parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['G2d'].parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['patchGAN'].parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2)
    ]

    schedulers = []

    for i in range(len(optimizers)):
        scheduler = CosineAnnealingLR(optimizers[0], T_max=args.iteration, eta_min=1e-6)
        schedulers.append(scheduler)

    train(args, models, device,  optimizers, schedulers,data_loader )

    

# Start Training.
main()
