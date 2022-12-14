import numpy as np
import random
import torchvision.transforms as Ttorch
import torch
from glob import glob
import cv2
from torch import Tensor
from scipy.special import softmax
from scipy.ndimage import rotate
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import PIL
import torch.nn.functional as F
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_param(degree, size):
    """
    Generate random angle for rotation and define the extension box for define their
    center
    """
    angle = float(torch.empty(1).uniform_(float(degree[0]), float(degree[1])).item())
    extent = int(np.ceil(np.abs(size*np.cos(np.deg2rad(angle)))+np.abs(size*np.sin(np.deg2rad(angle))))/2)
    return angle, extent

def subimage(image, center, theta, width, height):
   """
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   """
   shape = (image.shape[1], image.shape[0]) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
   image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

   x = int(center[0] - width/2)
   y = int(center[1] - height/2)

   image = image[y:y+height, x:x+width]
   return image

def warp(x, flo):

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    flo = flo.permute(0 ,2 ,3 ,1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size())#.cuda()
    mask = F.grid_sample(mask, vgrid)

    mask[mask <0.9999] = 0
    mask[mask >0] = 1

    return output*mask

class RandomRotation_crop(torch.nn.Module):
  def __init__(self, degrees, size):
       super().__init__()
       self.degree = [float(d) for d in degrees]
       self.size = int(size)

  def forward(self, img, pmap):
      """Rotate the image by a random angle.
         If the image is torch Tensor, it is expected
         to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

        Args:
            degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
            size (single value): size of the squared croped box

      Transformation that selects a randomly rotated region in the image within a specific 
      range of degrees and a fixed squared size.
      """      
      angle, extent = get_param(self.degree, self.size)
      
      if isinstance(img, Tensor):
        d_1=img.size(dim=1)
        d_2=img.size(dim=2)
      else:
        raise TypeError("Img should be a Tensor")

      ext_1 = [float(extent), float(d_1-extent)]
      ext_2 = [float(extent), float(d_2-extent)]

      end = time.time()
      print('2 -> ', end-start)
      start = end
      
      cut_pmap = softmax(pmap[int(ext_1[0]): int(ext_1[1]), int(ext_2[0]): int(ext_2[1])])
      end = time.time()
      print('3 -> ', end-start)
      start = end

      ind = np.array(list(np.ndindex(cut_pmap.shape)))
      end = time.time()
      print('4 -> ', end-start)
      start = end

      pos = np.random.choice(np.arange(len(cut_pmap.flatten())), 1, p=cut_pmap.flatten())
      end = time.time()
      print('5 -> ', end-start)
      start = end
      
      c = (int(ind[pos[0],1])+int(ext_1[0]), int(ind[pos[0],0])+int(ext_2[0]))

      img_raw=img.cpu().detach().numpy()

      cr_image_0 = subimage(img_raw[0], c, angle, self.size, self.size)
      cr_image_1 = subimage(img_raw[1], c, angle, self.size, self.size)

      end = time.time()
      print('6 -> ', end-start)
      start = end
    
      return torch.Tensor(np.array([cr_image_0,cr_image_1]), device='cpu')

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return Ttorch.functional.rotate(x, angle)

class SRS_crop(torch.nn.Module):
  def __init__(self, size):
       super().__init__()
       self.size = int(size)

  def forward(self, img, pmap, ind):
      counter = np.arange(len(pmap))
      pos = np.random.choice(counter, 1, p=pmap)      
      c = (int(ind[pos[0],0])+int(self.size/2), int(ind[pos[0],1])+int(self.size/2))
      img_raw=img.cpu().detach().numpy()

      x = int(c[0] - self.size/2)
      y = int(c[1] - self.size/2)

      res_array = []
      for img in img_raw:
        res_array.append(img[y:y+self.size, x:x+self.size]) # complete sequence + mask

      return torch.Tensor(np.array(res_array), device='cpu'), c

class Secuential_trasn(torch.nn.Module):
    """Generates a secuential transformation"""
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, img, pmap, ind):
      t_list=[img]      
      for t in range(len(self.transforms)):
        if t == 1:
          rotation, c = self.transforms[t](t_list[-1], pmap, ind)
          t_list.append(rotation)
        else:
          t_list.append(self.transforms[t](t_list[-1]))

      return t_list[-1], c

class segDataset(torch.utils.data.Dataset):
  def __init__(self, root, l=1000, s=128, seq_len=5):
    super(segDataset, self).__init__()
    start = time.time()
    self.root = root
    self.size = s
    self.l = l
    self.seq_len = seq_len
    self.classes = {'Intergranular lane' : 0,
                    'Uniform-shape granules': 1,
                    'Granules with dots' : 2,
                    'Granules with a lane' : 3,
                    'Complex-shape granules' : 4}

    self.bin_classes = ['Intergranular lane', 'Uniform-shape granules', 'Granules with dots', 'Granules with a lane',
                        'Complex-shape granules']

    self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                            SRS_crop(self.size),
                                            RotationTransform(angles=[0, 90, 180, 270]),
                                            #Ttorch.RandomPerspective(0.3,p=0.5, interpolation=Ttorch.InterpolationMode.NEAREST),
                                            Ttorch.RandomHorizontalFlip(p=0.5),
                                            Ttorch.RandomVerticalFlip(p=0.5)
                                            ])
    
    self.file_list = sorted(glob(self.root+'*.npz'))

    print("Reading images...")
    self.ts_smap = []
    self.mask_smap = []
    self.weight_maps = []
    self.index_list = []
    
    for f in self.file_list:
      file = np.load(f)
      psmap = file['ts_smap'].astype(np.float32)
      pmsmap = file['cmask_map'].astype(np.float32)
      psmap = psmap/psmap.max()

      pad_value = int(((np.sqrt(2*(psmap.shape[-1]**2))-psmap.shape[-1]))/2)

      #Padding for rotation
      pad_psmap = np.array([np.pad(psmap[i,:,:], ((pad_value,pad_value),(pad_value,pad_value)), mode='reflect') for i in range(self.seq_len)])
      pad_pmsmap = np.pad(pmsmap, ((pad_value,pad_value),(pad_value,pad_value)), mode='reflect')

      self.rot_angle = np.arange(0,90,5)
      for a in self.rot_angle:
        # Continumm image
        p_map1=[]
        for s in range(self.seq_len):
          pad_psmap_s = pad_psmap[s]
          vis1 = PIL.Image.fromarray(pad_psmap_s)
          p_map1_p = rotate(vis1,a)
          x01 = int(abs(p_map1_p.shape[0]/2) - (psmap.shape[-1]/2))
          x02 = int(abs(p_map1_p.shape[0]/2) + (psmap.shape[-1]/2))
          p_map1.append(p_map1_p[x01:x02,x01:x02])
        p_map1 = np.array(p_map1)

        #Mask image
        vis2 = PIL.Image.fromarray(pad_pmsmap)
        p_map2 = np.asarray(vis2.rotate(a))
        x11 = int(abs(p_map2.shape[0]/2) - (pmsmap.shape[0]/2))
        x12 = int(abs(p_map2.shape[0]/2) + (pmsmap.shape[0]/2))
        p_map_mask = p_map2[x11:x12,x11:x12]

        self.ts_smap.append(p_map1)
        self.mask_smap.append(p_map_mask)

        weight_maps = np.zeros_like(p_map_mask[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)]).astype(np.float32)
        weight_maps[p_map_mask[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 0.0] = 1
        weight_maps[p_map_mask[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 4.0] = 1
        weight_maps[p_map_mask[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 1.0] = 10
        weight_maps[p_map_mask[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 2.0] = 10
        weight_maps[p_map_mask[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 3.0] = 10

        wm_blurred = gaussian_filter(weight_maps, sigma=14)

        self.weight_maps.append(softmax(wm_blurred.flatten()))
        self.index_list.append(np.array(list(np.ndindex(weight_maps.shape))))

    print("Done!")
        
  def __getitem__(self, idx):
    
    ind = np.random.randint(low=0, high=len(self.ts_smap))
    ts_smap = self.ts_smap[ind]
    mask_smap = self.mask_smap[ind]

    #Full probability maps calculation
    weight_map = self.weight_maps[ind]
    index_l = self.index_list[ind]

    to_trans_map =[]
    for i in ts_smap:
      to_trans_map.append(i)
    to_trans_map.append(mask_smap)

    img_t, c = self.transform_serie(np.array(to_trans_map).transpose(), weight_map, index_l)

    self.images = img_t[0:-1].unsqueeze(0)
    self.mask = img_t[-1].type(torch.int64)
    #return self.image, self.mask, ind, c  #for test central points
    return self.images, self.mask
  
  def __len__(self):
        return self.l

class segDataset_val(torch.utils.data.Dataset):

  def __init__(self, root, l=1000, s=128, seq_len=5):
    super(segDataset_val, self).__init__()
    self.root = root
    self.size = s
    self.l = l
    self.classes = {'Intergranular lane' : 0,
                    'Normal-shape granules': 1,
                    'Granules with dots' : 2,
                    'Granules with lanes' : 3,
                    'Complex-shape granules' : 4}

    self.bin_classes = ['Intergranular lane', 'Normal-shape granules', 'Granules with dots', 'Granules with lanes',
                        'Complex-shape granules']

    self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                            SRS_crop(self.size),
                                            Ttorch.RandomHorizontalFlip(p=0.5),
                                            Ttorch.RandomVerticalFlip(p=0.5)
                                            ])
    
    self.file_list = sorted(glob(self.root+'*.npz'))

    print("Reading images...")
    self.ts_smap = []
    self.mask_smap = []
    self.weight_maps = []
    self.index_list = []

    for f in self.file_list:
      file = np.load(f)
      psmap = file['ts_smap'].astype(np.float32)
      pmsmap = file['cmask_map'].astype(np.float32)
      psmap = psmap/psmap.max()

      self.ts_smap.append(psmap)
      self.mask_smap.append(pmsmap)

      weight_maps = np.zeros_like(pmsmap[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)]).astype(np.float32)
      weight_maps[pmsmap[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 0] = 1
      weight_maps[pmsmap[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 4] = 1
      weight_maps[pmsmap[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 1] = 10
      weight_maps[pmsmap[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 2] = 10
      weight_maps[pmsmap[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 3] = 10

      wm_blurred = gaussian_filter(weight_maps, sigma=14)

      self.weight_maps.append(softmax(wm_blurred.flatten()))
      self.index_list.append(np.array(list(np.ndindex(weight_maps.shape))))

    print("Done!")
        
  def __getitem__(self, idx):
    
    if len(self.ts_smap) > 3:
      ind = np.random.randint(low=0, high=len(self.ts_smap))
    else:
      ind = 0
    ts_smap = self.ts_smap[ind]
    mask_smap = self.mask_smap[ind]

    #Full probability maps calculation
    weight_map = self.weight_maps[ind]
    index_l = self.index_list[ind]

    to_trans_map =[]
    for i in ts_smap:
      to_trans_map.append(i)
    to_trans_map.append(mask_smap)

    img_t, c = self.transform_serie(np.array(to_trans_map).transpose(), weight_map, index_l)

    self.images = img_t[0:-1].unsqueeze(0)
    self.mask = img_t[-1].type(torch.int64)
    #return self.image, self.mask, ind, c  #for test central points
    return self.images, self.mask
  
  def __len__(self):
        return self.l