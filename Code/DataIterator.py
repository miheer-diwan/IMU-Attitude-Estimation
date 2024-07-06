from DataLoader import IMUDataset, ViconDataset
import numpy as np

class IMUDataIterator:
    def __init__(self, IMUDataset):
        self.dataset = IMUDataset
        self.itr = 0
    
    def step(self):
        if self.itr < self.dataset.data_size - 1:
            self.itr += 1
            return True
        else:
            return False

    def getCurrentData(self):
        if self.itr > 0:
            dt = self.dataset.ts[self.itr] - self.dataset.ts[self.itr - 1]
        else:
            dt = 0

        return {"t": self.dataset.ts[self.itr],
                "dt": dt,
                "omega_xyz": np.array(self.dataset.gyro.getDataAtIdx(self.itr)),
                "accel_xyz": np.array(self.dataset.accel.getDataAtIdx(self.itr))}

class ViconDataIterator:
    def __init__(self, ViconDataset):
        self.dataset = ViconDataset
        self.itr = 0
    
    def step(self):
        if self.itr < self.dataset.data_size - 1:
            self.itr += 1
            return True
        else:
            return False
        
    def getCurrentData(self):
        if self.itr > 0:
            dt = self.dataset.ts[self.itr] - self.dataset.ts[self.itr - 1]
        else:
            dt = 0

        return {"t": self.dataset.ts[self.itr],
                "dt": dt,
                "rot_matrix": self.dataset.getDataAtIdx(self.itr)}
        