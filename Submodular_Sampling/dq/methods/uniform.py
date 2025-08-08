import numpy as np
from .coresetmethod import CoresetMethod


class Uniform(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=False, replace=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)
        print("Uniform")
    def select_balance(self, IPC=None):
        """The same sampling proportions were used in each class separately.
        
        Parameters:
        IPC (int, optional): The number of samples to select. If provided, this overrides the fraction.
        """
        #print("select_balance")
        np.random.seed(self.random_seed)
        self.index = np.array([], dtype=np.int64)
        all_index = np.arange(self.n_train)
        #print("IPC",IPC)
        for c in range(self.num_classes):
            c_index = (self.dst_train.dataset.targets[self.dst_train.indices] == c)
            # Determine the number of samples to select
            if IPC is not None:
                if IPC > 1:
                    n_samples = int(IPC/10 ) # Use IPC value if provided     /bin
                else:
                     n_samples = int(IPC)
            else:
                n_samples = round(self.fraction * c_index.sum().item())  # Default to fraction-based selection
            #print("n_samples",n_samples)
            self.index = np.append(self.index,
                                np.random.choice(all_index[c_index], n_samples, replace=self.replace))
            #print("index",len(self.index))
        return self.index

    def select_no_balance(self):
        np.random.seed(self.random_seed)
        self.index = np.random.choice(np.arange(self.n_train), round(self.n_train * self.fraction),
                                      replace=self.replace)

        return  self.index

    def select(self, IPC=None, **kwargs):
        print("Uniform select")
        return {"indices": self.select_balance(IPC=IPC) if self.balance else self.select_no_balance()}
