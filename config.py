import torch

class Config:
    def __init__(self):
        self.gpu_name = 'cuda:8'
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
        self.embed_dim = 96
        self.num_heads = [4, 6, 8, 12]  # 96-[4, 6, 8, 12]  64-[4, 4, 8, 8]
        self.num_classes = 3
        self.T = 4
        self.num_epochs = 300
        self.batch_size = 1
        self.k_folds = 5
        self.loss_function = 'dice' # dice, focal
        self.loss_weights = [2.0, 1.0, 4.0]
        self.train_crop_mode = "tumor_aware_random"  # tumor_aware_random, warmup_weighted_random, random, tumor_center
        self.val_crop_mode = 'tumor_aware_random' # tumor_aware_random, sliding_window, random, tumor_center
        self.overlap = 0.5
        self.num_workers = 8

        self.compute_hd = False

        self.scheduler = 'polynomial' # cosine, polynomial
        self.num_warmup_epochs = -1  # -1表示不使用warmup
        self.early_stop_patience = 50
        
        self.base_lr = 1e-3
        self.min_lr = 1e-6

        self.step_mode = 'm'


# 全局单例，方便import时直接用
config = Config()
