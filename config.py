import torch

class Config:
    def __init__(self):
        self.gpu_name = 'cuda:7'
        self.device = torch.device(self.gpu_name if torch.cuda.is_available() else "cpu")
        
        # BraTS2018
        self.root_dir = './data/HGG/'        
        self.modalities = ['t1', 't1ce', 't2', 'flair']
        self.modality_separator = "_"
        self.image_suffix = ".nii"
        self.et_label = 4
        
        # # BraTS2023
        # self.root_dir = './data/BraTS2023/'
        # self.modalities = ['t1n', 't1c', 't2w', 't2f']
        # self.modality_separator = "-" 
        # self.image_suffix = ".nii.gz"    
        # self.et_label = 3
        
        self.encode_method = 'poisson'  # poisson, latency, weighted_phase
        
        self.patch_size = [128, 128, 128]
        self.window_size = [it // 32 for it in self.patch_size]
        
        self.num_classes = 3
        self.T = 6
        self.num_epochs = 300
        self.batch_size = 1
        self.k_folds = 5
        self.loss_weights = [2.0, 1.0, 4.0]
        self.crop_mode = "tumor_aware_random"  # tumor_aware_random, warmup_weighted_random, random
        self.num_workers = 8

        self.compute_hd = False

        self.base_lr = 1e-3
        self.num_warmup_epochs = 15
        self.min_lr = 1e-6

        self.step_mode = 'm'


# 全局单例，方便import时直接用
config = Config()
