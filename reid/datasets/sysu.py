from __future__ import print_function, absolute_import
import os.path as osp
import numpy
from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class SYSU(Dataset):
    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(SYSU, self).__init__(root, split_id=split_id)

        self.root += "/SYSU-MM01"
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. ")

        self.load(num_val)

    def download(self):

        import shutil
        from glob import glob

        # Format
        images_dir = osp.join(self.root+'/images')
        mkdir_if_missing(images_dir)

        # gain the spilt from .mat
        import scipy.io as scio
        data = scio.loadmat(self.root+'/exp/train_id.mat')
        train_id = data['id'][0]
        data = scio.loadmat(self.root+'/exp/val_id.mat')
        val_id = data['id'][0]
        data = scio.loadmat(self.root+'/exp/test_id.mat')
        test_id = data['id'][0]

        # 533 identities with 6 camera views each
        identities = [[[] for _ in range(6)] for _ in range(533)]
        for pid in range(1, 534):
            for cam in range(1,7):
                images_path = self.root+"/cam"+str(cam)+"/"+str(pid).zfill(4)
                fpaths = sorted(glob(images_path+"/*.jpg"))
                for fpath in fpaths:
                    # print(fpath)
                    fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid-1, cam-1, len(identities[pid-1][cam-1])))
                    identities[pid-1][cam-1].append(fname)
                    shutil.copy(fpath, osp.join(images_dir, fname))

        trainval_pids = set()
        gallery_pids = set()
        query_pids = set()
        train_val_ = numpy.concatenate((train_id,val_id))
        for i in (train_val_):
            trainval_pids.add(int(i) - 1)
        for i in test_id:
            gallery_pids.add(int(i) - 1)
            query_pids.add(int(i) - 1)

        # Save meta information into a json file
        meta = {'name': 'sysu', 'shot': 'multiple', 'num_cameras': 6,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))
