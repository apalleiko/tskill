from torch.utils.data import Dataset
import numpy as np


class MultitaskDataset(Dataset):
    def __init__(self, sequence_datasets):
        self.sequence_datasets = sequence_datasets
        # self.task_embs = task_embs
        self.group_size = len(sequence_datasets)
        self.lengths = [len(x) for x in self.sequence_datasets]
        self.task_group_size = len(self.sequence_datasets)

        self.action_scaling = sequence_datasets[0].action_scaling
        self.state_scaling = sequence_datasets[0].state_scaling

        # create a map that maps the current idx of dataloader to original task data idx
        # imagine we have task 1,2,3, with sizes 3,5,4, then the idx looks like
        # task-1  task-2  task-3
        #   0       1       2
        #   3       4       5
        #   6       7       8
        #           9       10
        #           11
        # by doing so, when we concat the dataset, every task will have equal number of demos
        
        self.map_dict = {}
        sizes = np.array(self.lengths)
        row = 0
        col = 0
        for i in range(sum(sizes)):
            while sizes[col] == 0:
                col = col + 1
                if col >= self.task_group_size:
                    col -= self.task_group_size
                    row += 1
            self.map_dict[i] = (row, col)
            sizes[col] -= 1
            col += 1
            if col >= self.task_group_size:
                col -= self.task_group_size
                row += 1
        self.n_total = sum(self.lengths)

    def __len__(self):
        return self.n_total

    def __get_original_task_idx(self, idx):
        return self.map_dict[idx]

    def __getitem__(self, idx):
        oi, oti = self.__get_original_task_idx(idx)
        return_dict = self.sequence_datasets[oti].__getitem__(oi)
        # return_dict["task_emb"] = self.task_embs[oti]
        return return_dict
