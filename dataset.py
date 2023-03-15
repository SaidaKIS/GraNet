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
import torch.nn.functional as F
import sys
import PIL
from einops import repeat, rearrange

#For run 20 GBt memory free

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

def rotate_CV(image,angel,interpolation=cv2.INTER_LINEAR):
    '''
        input :
        image           :  image                    : ndarray
        angel           :  rotation angel           : int
        interpolation   :  interpolation mode       : cv2 Interpolation object
        
                                                        Interpolation modes :
                                                        interpolation cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR
                                                        https://theailearner.com/2018/11/15/image-interpolation-using-opencv-python/
                                                        
        returns : 
        rotated image   : ndarray
        '''
    #in OpenCV we need to form the tranformation matrix and apply affine calculations
    #
    h,w = image.shape[:2]
    cX,cY = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),angel,1)
    rotated = cv2.warpAffine(image,M,(w,h),flags=interpolation)
    return rotated

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
  def __init__(self, root, l=1000, s=96, channels=4, seq_len=4):
    super(segDataset, self).__init__()
    self.root = root
    self.size = s
    self.l = l
    self.channels = channels

    self.seq_len = seq_len
    self.classes = {'Intergranular lane' : 0,
                    'Granule': 1,
                    'Exploding granule' : 2}

    self.bin_classes = ['Intergranular lane', 'Granule', 'Exploding granule']

    self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                            SRS_crop(self.size),
                                            RotationTransform(angles=[0, 90, 180, 270]),
                                            Ttorch.RandomHorizontalFlip(p=0.5),
                                            Ttorch.RandomVerticalFlip(p=0.5)
                                            ])
    
    print("Reading files...")
    
    self.file_list = sorted(glob(self.root+'*.npz'))
    
    self.ts_smap = []
    self.mask_smap = []
    self.weight_maps = []
    self.index_list = []
    
    for f in self.file_list:
      if "training" in f:
        file = np.load(f)
        raw_cube = file['raw'].astype(np.float32)
        mask_cube = file['segmented'].astype(np.float32)
        weight_cube = file['weights'].astype(np.float32)

        self.size_cube = mask_cube.shape
        pad_value = int(((np.sqrt(2*(self.size_cube[-1]**2))-self.size_cube[-1]))/2)

        self.rot_angle = np.arange(0,90,45)
        for a in self.rot_angle:
          #print(a)
          start = time.time()
          for t in range(self.size_cube[0]):
            #Padding for rotation
            #Original maps - channels (Cont, vlos, LP, CP)
            pad_map = np.array([np.pad(raw_cube[t,c,:,:], ((pad_value,pad_value),(pad_value,pad_value)), mode='reflect') for c in range(self.channels)])
            #Segmentated maps
            pad_mmap = np.pad(mask_cube[t,:,:], ((pad_value,pad_value),(pad_value,pad_value)), mode='reflect')
            #Weigted maps
            pad_wmap = np.pad(weight_cube[t,:,:], ((pad_value,pad_value),(pad_value,pad_value)), mode='reflect')
            
            #Original image rotation
            c_maps=[]
            for c in range(self.channels):
              img = pad_map[c]
              rot_img = rotate_CV(img,a)
              x01 = int(abs(rot_img.shape[0]/2) - (self.size_cube[-1]/2))
              x02 = int(abs(rot_img.shape[0]/2) + (self.size_cube[-1]/2))
              c_maps.append(rot_img[x01:x02,x01:x02])
            
            #Segmented map rotation
            img_seg = pad_mmap
            rot_img_seg = rotate_CV(img_seg,a)
            x01 = int(abs(rot_img_seg.shape[0]/2) - (self.size_cube[-1]/2))
            x02 = int(abs(rot_img_seg.shape[0]/2) + (self.size_cube[-1]/2))
            s_map = rot_img_seg[x01:x02,x01:x02]

            #Weighted map rotation
            img_wei = pad_wmap
            rot_img_wei = rotate_CV(img_wei,a)
            x01 = int(abs(rot_img_wei.shape[0]/2) - (self.size_cube[-1]/2))
            x02 = int(abs(rot_img_wei.shape[0]/2) + (self.size_cube[-1]/2))
            w_map = rot_img_wei[x01:x02,x01:x02]

            #Sigue estando aqui
            w_map_cut = w_map[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)]
            wm_blurred = gaussian_filter(w_map_cut, sigma=14)
                  
            self.ts_smap.append(c_maps)
            self.mask_smap.append(s_map)

            self.weight_maps.append(softmax(wm_blurred.flatten()))
            self.index_list.append(np.array(list(np.ndindex(w_map_cut.shape))))

          #print(time.time() - start)
    del file
    del raw_cube
    del mask_cube
    del weight_cube
    print("Done!")
        
  def __getitem__(self, idx):
    #Temporal jump adjustment
    #r = len(self.rot_angle)
    #x1 = np.concatenate(np.array([np.arange(4,36,1,dtype=np.int16)+i*100 for i in range(r)]))
    #x2 = np.concatenate(np.array([np.arange(46,96,1,dtype=np.int16)+i*100 for i in range(r)]))
    #x = np.concatenate((x1,x2))
    #Use the frames with exploding granules detection
    x = np.array([4,8,11,19,26,28,34,35,33,36,50,58,56,63,68,83,78,87,90])
    ind = np.random.randint(low=0, high=len(x))
    val_ind = x[ind]

    ind_list = np.arange(val_ind-self.seq_len,val_ind+self.seq_len+1,1, dtype=np.int8)
    
    seq_smap=[]
    for st in ind_list:
      seq_smap.append(self.ts_smap[st])

    seq_smap = np.array(seq_smap)
    mask_smap = self.mask_smap[val_ind]
    c_mask_smap = repeat(mask_smap, 'h w -> 1 '+str(self.channels)+' h w')     
    to_trans_map = np.concatenate((seq_smap, c_mask_smap))
    to_trans_map_arrange = rearrange(to_trans_map, 's c h w -> (s c) h w')
    #(1 channels h w) -> (10,channels,796,796)
    # 10 + channels -> ((10 channels) h w)
    weight_map = self.weight_maps[val_ind]
    index_l = self.index_list[val_ind]

    img_t, c = self.transform_serie(np.array(to_trans_map_arrange).transpose(), weight_map, index_l)

    img_t_rearange = rearrange(img_t, '(s c) h w -> s c h w', c=self.channels)

    self.images = img_t_rearange[0:-1]
    self.mask = img_t_rearange[-1].type(torch.int64)[0]
    #return self.image, self.mask, ind, c  #for test central points
    return self.images, self.mask
  
  def __len__(self):
        return self.l

class segDataset_val(torch.utils.data.Dataset):
  def __init__(self, root, l=1000, s=96, channels=4, seq_len=4):
    super(segDataset_val, self).__init__()
    start = time.time()
    self.root = root
    self.size = s
    self.l = l
    self.channels = channels

    self.seq_len = seq_len
    self.classes = {'Intergranular lane' : 0,
                    'Granule': 1,
                    'Exploding granule' : 2}

    self.bin_classes = ['Intergranular lane', 'Granule', 'Exploding granule']

    self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                            SRS_crop(self.size),
                                            Ttorch.RandomHorizontalFlip(p=0.5),
                                            Ttorch.RandomVerticalFlip(p=0.5)
                                            ])
    
    print("Reading files...")
    
    self.file_list = sorted(glob(self.root+'*.npz'))
    
    self.ts_smap = []
    self.mask_smap = []
    self.weight_maps = []
    self.index_list = []
    
    for f in self.file_list:
        #print(f)
        if "validation" in f:
            file = np.load(f)
            raw_cube = file['raw'].astype(np.float32)
            mask_cube = file['segmented'].astype(np.float32)
            weight_cube = file['weights'].astype(np.float32)

            self.size_cube = mask_cube.shape

            for t in range(self.size_cube[0]):
                self.ts_smap.append([raw_cube[t,c,:,:] for c in range(self.channels)])
                self.mask_smap.append(mask_cube[t,:,:])
                w_map = weight_cube[t,:,:]
                wm_blurred = gaussian_filter(w_map, sigma=14)
                self.weight_maps.append(softmax(wm_blurred.flatten()))
                self.index_list.append(np.array(list(np.ndindex(w_map.shape))))

    del file
    del raw_cube
    del mask_cube
    del weight_cube
    print("Done!")
        
  def __getitem__(self, idx):
    #Use the frames with exploding granules detection
    x = np.array([5,10,10,24,26,27,33,34])
    ind = np.random.randint(low=0, high=len(x))
    val_ind = x[ind]

    ind_list = np.arange(val_ind-self.seq_len,val_ind+self.seq_len+1,1, dtype=np.int8)
    
    seq_smap=[]
    for st in ind_list:
      seq_smap.append(self.ts_smap[st])

    seq_smap = np.array(seq_smap)
    mask_smap = self.mask_smap[val_ind]
    c_mask_smap = repeat(mask_smap, 'h w -> 1 '+str(self.channels)+' h w')     
    to_trans_map = np.concatenate((seq_smap, c_mask_smap))
    to_trans_map_arrange = rearrange(to_trans_map, 's c h w -> (s c) h w')
    #(1 channels h w) -> (10,channels,796,796)
    # 10 + channels -> ((10 channels) h w)
    weight_map = self.weight_maps[val_ind]
    index_l = self.index_list[val_ind]

    img_t, c = self.transform_serie(np.array(to_trans_map_arrange).transpose(), weight_map, index_l)

    img_t_rearange = rearrange(img_t, '(s c) h w -> s c h w', c=self.channels)

    self.images = img_t_rearange[0:-1]
    self.mask = img_t_rearange[-1].type(torch.int64)

    #return self.image, self.mask, val_ind, c  #for test central points
    return self.images, self.mask
  
  def __len__(self):
        return self.l
