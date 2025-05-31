import torch

class Config:
    def __init__(self):
        self.gpu_name = 'cuda:6'
        self.device = torch.device(self.gpu_name if torch.cuda.is_available() else "cpu")
        self.root_dir = './data/HGG'
        
        self.modalities = ['t1', 't1ce', 't2', 'flair']
        self.encode_method = 'poisson'
        
        self.patch_size = [128, 128, 128]
        self.window_size = [it // 32 for it in self.patch_size]

        self.T = 8
        self.num_epochs = 150
        self.batch_size = 1
        self.k_folds = 5
        self.loss_weights = [2.0, 1.0, 4.0]

        self.num_workers = 8

        self.compute_hd = False

        self.base_lr = 1e-3
        self.num_warmup_epochs = 8
        self.min_lr = 1e-7

        self.step_mode = 'm'


# 全局单例，方便import时直接用
config = Config()
