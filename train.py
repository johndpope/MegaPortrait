import argparse
import torch
import model
import cv2 as cv
import HeadPoseEstimation
import vgg_face
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from EmoDataset import EMODataset
import torch.nn.functional as F
import decord
from omegaconf import OmegaConf


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

def cosine_loss(args, descriptor_driver, descriptor_source_rand, descriptor_driver_rand):
    z_dri = descriptor_driver
    z_dri_rand = descriptor_driver_rand
    
    z_src_rand_dri = [descriptor_source_rand[0], z_dri[1], z_dri[2]]

    pos_pairs = [(z_dri, z_dri), (z_src_rand_dri, z_dri)]
    neg_pairs = [(z_dri, z_dri_rand), (z_src_rand_dri, z_dri_rand)]

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

def train_base(cfg, Gbase, Dbase, dataloader, epochs):
    Gbase.train()
    Dbase.train()
    
    for epoch in range(epochs):
        for batch in dataloader:
            video_frames = batch['images']
            video_id = batch['video_id']
            
            # Training loop for base model
            # ...

def train_hr(cfg, GHR, Dhr, dataloader_hr):
    GHR.train()
    Dhr.train()
    
    for epoch in range(cfg.training.hr_epochs):
        for batch in dataloader_hr:
            video_frames = batch['images']
            video_id = batch['video_id']
            
            # Load pre-trained base model Gbase
            # Freeze Gbase
            # Training loop for high-resolution model
            # ...

def train_student(cfg, Student, GHR, dataloader_avatars):
    Student.train()
    
    for epoch in range(cfg.training.student_epochs):
        for batch in dataloader_avatars:
            video_frames = batch['images']
            video_id = batch['video_id']
            
            # Load pre-trained high-resolution model GHR
            # Freeze GHR
            # Training loop for student model
            # ...

def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter()
    ])

    dataset = EMODataset(
        use_gpu=use_cuda,
        width=img_size,
        height=img_size,
        n_sample_frames=cfg.training.n_sample_frames,
        sample_rate=cfg.training.sample_rate,
        img_scale=(1.0, 1.0),
        video_dir=cfg.training.video_dir,
        json_file=cfg.training.json_file,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    Gbase = model.Gbase()
    Dbase = model.Discriminator()
    train_base(cfg, Gbase, Dbase, dataloader, epochs=100)
    
    GHR = model.GHR()
    GHR.Gbase.load_state_dict(Gbase.state_dict())
    Dhr = model.Discriminator()
    train_hr(cfg, GHR, Dhr, dataloader, epochs=50)
    
    Student = model.Student(num_avatars=100)
    train_student(cfg, Student, GHR, dataloader, epochs=100)
    
    torch.save(Gbase.state_dict(), 'Gbase.pth')
    torch.save(GHR.state_dict(), 'GHR.pth')
    torch.save(Student.state_dict(), 'Student.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1.yaml")
    main(config)