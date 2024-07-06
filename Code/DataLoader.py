import os
from scipy import io
import numpy as np
class IMUDataset(object):
    def __init__(self, data_loader):
        raw_data=data_loader.loadData()
        self.gyro = self.GyroData(raw_data["vals"][3:6,:])
        self.accel = self.AccelData(raw_data["vals"][0:3,:])
        self.ts = raw_data["ts"][0]
        self.bias_corrected = False
        self.data_size = self.ts.shape[0]
    
    def correctBias(self, accel_param):
        if not self.bias_corrected:
            self.gyro.correctBias()
            self.accel.correctBias(accel_param)
            self.bias_corrected = True
    
    def removeDataBeforeIdx(self, idx):
        self.gyro.removeDataBeforeIdx(idx)
        self.accel.removeDataBeforeIdx(idx)
        self.ts = self.ts[idx:]
        self.data_size = self.ts.shape[0]
        

    class GyroData:
        def __init__(self, raw_gyro_data):
            self.w_x = raw_gyro_data[1,:] # raw data order[wz,wx,wy]
            self.w_y = raw_gyro_data[2,:]
            self.w_z = raw_gyro_data[0,:]
        
        def removeDataBeforeIdx(self, idx):
            self.w_x = self.w_x[idx:]
            self.w_y = self.w_y[idx:]
            self.w_z = self.w_z[idx:]

        def correctBias(self):
            bias_gwx = np.average(self.w_x[0:100])
            bias_gwy = np.average(self.w_y[0:100])
            bias_gwz = np.average(self.w_z[0:100])
        
            self.w_x = (3300/1023)*(np.pi/180)*0.3*(self.w_x - bias_gwx)
            self.w_y = (3300/1023)*(np.pi/180)*0.3*(self.w_y - bias_gwy)
            self.w_z = (3300/1023)*(np.pi/180)*0.3*(self.w_z - bias_gwz)

        def getDataAtIdx(self, idx):
            return [self.w_x[idx], self.w_y[idx], self.w_z[idx]]

    
    class AccelData:
        def __init__(self, raw_accel_data):
            self.accel_x = raw_accel_data[0,:]
            self.accel_y = raw_accel_data[1,:]
            self.accel_z = raw_accel_data[2,:]

        def removeDataBeforeIdx(self, idx):
            self.accel_x = self.accel_x[idx:]
            self.accel_y = self.accel_y[idx:]
            self.accel_z = self.accel_z[idx:]

        def correctBias(self, accel_param):
            self.accel_x = self.accel_x*accel_param.scale_x + accel_param.bias_ax
            self.accel_y = self.accel_y*accel_param.scale_y + accel_param.bias_ay
            self.accel_z = self.accel_z*accel_param.scale_z + accel_param.bias_az

        def getDataAtIdx(self, idx):
            return [self.accel_x[idx], self.accel_y[idx], self.accel_z[idx]]

class AccelParam(object):
    def __init__(self, data_loader):
        raw_data = data_loader.loadData()
        self.bias_ax = raw_data["IMUParams"][1,0]
        self.bias_ay = raw_data["IMUParams"][1,1]
        self.bias_az = raw_data["IMUParams"][1,2]
        self.scale_x = raw_data["IMUParams"][0,0]
        self.scale_y = raw_data["IMUParams"][0,1]
        self.scale_z = raw_data["IMUParams"][0,2]

class ViconDataset(object):
    def __init__(self, data_loader):
        raw_data = data_loader.loadData()
        self.rot_matrices = raw_data["rots"][:,:,:]
        self.ts = raw_data["ts"][0]
        self.data_size = self.ts.shape[0]
    
    def removeDataBeforeIdx(self, idx):
        self.rot_matrices = self.rot_matrices[:,:,idx:]
        self.ts = self.ts[idx:]
        self.data_size = self.ts.shape[0]

    def getDataAtIdx(self, idx):
        return self.rot_matrices[:,:,idx]

class MatDataLoader(object):
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    def loadData(self):
        return io.loadmat(self.file_path)

def findCommonStartTimeIdx(imu_ts, vicon_ts, time_alignment_threshold = 0.001):
    imu_idx = 0
    vicon_idx = 0
    while imu_idx < imu_ts.size and vicon_idx < vicon_ts.size:
        if abs(imu_ts[imu_idx] - vicon_ts[vicon_idx]) < time_alignment_threshold:
            break

        if imu_ts[imu_idx] < vicon_ts[vicon_idx]:
            imu_idx += 1
        else:
            vicon_idx += 1

    return imu_idx, vicon_idx
