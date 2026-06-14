import torch
import numpy as np
import os.path as osp
import os
from PIL import Image
import cv2
from romatch.utils import get_depth_tuple_transform_ops, get_tuple_transform_ops
from ..utils import dense_benchmark

class ScanNetPairsDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, ht=384, wt=512, use_horizontal_flip_aug = False):
        super().__init__()
        
        self.name = "ScanNet-pairs"
        self.root = data_root
        self.split = "test"
        self.wt, self.ht = wt, ht
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.instances = self.get_instances(self.root)
        print(f"{self.name} | {len(self.instances)} pairs")

        self.im_transform_ops = get_tuple_transform_ops(resize=(ht, wt), normalize=True)
        self.depth_transform_ops = get_depth_tuple_transform_ops(resize=(ht, wt), normalize=False)

    def get_instances(self, root_path):
        K_dict = dict(np.load(f"{root_path}/intrinsics.npz"))
        data = np.load(f"{root_path}/test.npz")["name"]
        instances = []

        for i in range(len(data)):
            room_id, seq_id, ins_0, ins_1 = data[i]
            scene_id = f"scene{room_id:04d}_{seq_id:02d}"
            K_i = torch.tensor(K_dict[scene_id]).float()

            instances.append((scene_id, ins_0, ins_1, K_i))

        return instances

    def __len__(self):
        return len(self.instances)
    
    def read_scannet_intrinsic(self,path):
        """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
        """
        intrinsic = np.loadtxt(path, delimiter=' ')
        return torch.tensor(intrinsic[:-1, :-1], dtype = torch.float)
    
    def read_scannet_pose(self,path):
        """ Read ScanNet's Camera2World pose and transform it to World2Camera.
        
        Returns:
            pose_w2c (np.ndarray): (4, 4)
        """
        cam2world = np.loadtxt(path, delimiter=' ')
        world2cam = np.linalg.inv(cam2world)
        return world2cam
    
    def load_im(self, im_B, crop=None):
        im = Image.open(im_B)
        return im
    
    def load_depth(self, depth_ref, crop=None):
        depth = cv2.imread(str(depth_ref), cv2.IMREAD_UNCHANGED)
        depth = depth / 1000
        depth = torch.from_numpy(depth).float()  # (h, w)
        return depth

    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.wt / wi, self.ht /  hi
        sK = torch.tensor([[sx, 0, 0],
                        [0, sy, 0],
                        [0, 0, 1]])
        return sK@K
    
    def __getitem__(self, index):
        scene_name, stem_name_1, stem_name_2, K = self.instances[index]

        K1 = K2 =  self.read_scannet_intrinsic(osp.join(self.root,
                       scene_name,
                       'intrinsic', 'intrinsic_color.txt'))

        T1 =  self.read_scannet_pose(osp.join(self.root,
                       scene_name,
                       'pose', f'{stem_name_1}.txt'))
        T2 =  self.read_scannet_pose(osp.join(self.root,
                       scene_name,
                       'pose', f'{stem_name_2}.txt'))
        T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[:4, :4]  # (4, 4)


        # Load positive pair data
        im_A_ref = os.path.join(self.root, scene_name, 'color', f'{stem_name_1}.jpg')
        im_B_ref = os.path.join(self.root, scene_name, 'color', f'{stem_name_2}.jpg')
        depth_A_ref = os.path.join(self.root, scene_name, 'depth', f'{stem_name_1}.png')
        depth_B_ref = os.path.join(self.root, scene_name, 'depth', f'{stem_name_2}.png')

        im_A = self.load_im(im_A_ref)
        im_B = self.load_im(im_B_ref)
        depth_A = self.load_depth(depth_A_ref)
        depth_B = self.load_depth(depth_B_ref)

        # Recompute camera intrinsic matrix due to the resize
        K1 = self.scale_intrinsic(K1, im_A.width, im_A.height)
        K2 = self.scale_intrinsic(K2, im_B.width, im_B.height)
        # Process images
        im_A, im_B = self.im_transform_ops((im_A, im_B))
        depth_A, depth_B = self.depth_transform_ops((depth_A[None,None], depth_B[None,None]))
        if self.use_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_A, im_B, depth_A, depth_B, K1, K2 = self.horizontal_flip(im_A, im_B, depth_A, depth_B, K1, K2)

        data_dict = {'im_A': im_A,
                    'im_B': im_B,
                    'im_A_depth': depth_A[0,0],
                    'im_B_depth': depth_B[0,0],
                    'K1': K1,
                    'K2': K2,
                    'T_1to2':T_1to2,
                    }
        return data_dict
    

class ScannetDenseBenchmark:
    def __init__(self, data_root="data/scannet_test_1500", h = 384, w = 512) -> None:
        self.dataset = ScanNetPairsDataset(
            data_root, ht=h, wt=w
        )  # fixed resolution of 384,512
        self.num_samples = len(self.dataset)

    def benchmark(self, model, batch_size=8):
        return dense_benchmark(model, self.dataset, batch_size=batch_size, num_samples=self.num_samples)