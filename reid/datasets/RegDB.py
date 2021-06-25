from __future__ import print_function, absolute_import
import os.path as osp
from ..utils.data import Dataset
from ..utils.serialization import write_json

class RegDB(Dataset):
    def __init__(self, root, split_id=0, ii=0, num_val=100, download=True):
        super(RegDB, self).__init__(root, split_id=split_id)
        self.ii = ii
        if download:
            self.download()

        self.load(num_val)

    def download(self):
        index_train_RGB = open('./data/RegDB/idx/train_visible_{}.txt'.format(self.ii),'r')
        index_train_IR = open('./data/RegDB/idx/train_thermal_{}.txt'.format(self.ii),'r')
        index_test_RGB = open('./data/RegDB/idx/test_visible_{}.txt'.format(self.ii),'r')
        index_test_IR = open('./data/RegDB/idx/test_thermal_{}.txt'.format(self.ii),'r')

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        index_train_RGB = loadIdx(index_train_RGB)
        index_train_IR = loadIdx(index_train_IR)
        index_test_RGB = loadIdx(index_test_RGB)
        index_test_IR = loadIdx(index_test_IR)

        # 412 identities with 3 camera views each
        identities = [[[] for _ in range(3)] for _ in range(412)]
        def insertToMeta(index, cam, delta):
            for idx in index:
                fname = osp.basename(idx[0])

                pid = int(idx[1]) + delta

                identities[pid][cam].append(fname)

        insertToMeta(index_train_RGB, 0, 0)
        insertToMeta(index_train_IR, 2, 0)
        insertToMeta(index_test_RGB, 0, 206)
        insertToMeta(index_test_IR, 2, 206)

        trainval_pids = set()
        gallery_pids = set()
        query_pids = set()
        for i in range(206):
            trainval_pids.add(i)
            gallery_pids.add(i + 206)
            query_pids.add(i + 206)

        # Save meta information into a json file
        meta = {'name': 'RegDB', 'shot': 'multiple', 'num_cameras': 3,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))
