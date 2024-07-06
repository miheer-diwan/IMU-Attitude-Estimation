from DataIterator import *
from DataLoader import *

from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from rotplot import rotplot
import time
from pyquaternion import Quaternion
from Filter import MadgwickFilter,UKFilter
# import keyboard

def wrap_angle(prev,cur):
    for i in range(cur.shape[0]):
        if abs(prev[i]-cur[i])>6.0:
            if cur[i]>6.0:
                cur[i]= cur[i]-2*np.pi
            else:
                cur[i] = cur[i] + 2*np.pi
    return cur

def quat_conjugate(q):
    q_conj = np.array([q[0],-q[1],-q[2],-q[3]])

    return q_conj

def quat_inverse(q):
    return quat_conjugate(q)/((np.linalg.norm(q))**2)

def quat_product(q1,q2):
    q1_w = q1[0]
    q1_x = q1[1]
    q1_y = q1[2]
    q1_z = q1[3]
    
    q2_w = q2[0]
    q2_x = q2[1]
    q2_y = q2[2]
    q2_z = q2[3]

    return np.array([q1_w*q2_w - q1_x*q2_x - q1_y*q2_y - q1_z*q2_z,
                     q1_w*q2_x + q1_x*q2_w + q1_y*q2_z - q1_z*q2_y,
                     q1_w*q2_y - q1_x*q2_z + q1_y*q2_w + q1_z*q2_x,
                     q1_w*q2_z + q1_x*q2_y - q1_y*q2_x + q1_z*q2_w])


def quaternion_to_euler(q):
    rm = quaternion_to_RM(q)

    # Convert rotation matrix to Rotation object
    rotation = R.from_matrix(rm)

    # Convert Rotation object to Euler angles
    euler_angles = rotation.as_euler('ZYX', degrees=False)

    roll = euler_angles[2]
    pitch = euler_angles[1]
    yaw = euler_angles[0]

    return roll, pitch, yaw

def quaternion_to_RM(q):
    quaternion = Quaternion(q[0], q[1], q[2], q[3])  # Create a Quaternion object (w,x,y,z)
    rm = quaternion.rotation_matrix # Convert to rotation matrix
    return rm

def RM_to_quaternion(rm):
    q = Quaternion(matrix=rm) # w,x,y,z
    return np.array([q[0], q[1], q[2], q[3]]).T # w,x,y,z

class Phase1:
    def __init__(self,base_path, dataset_num, train = True):
        self.dataset_num = dataset_num
        if(train):
            data_folder = "Train"
        else:
            data_folder = "Test"

        imu_path = "{}/{}/IMU/imuRaw{}.mat".format(base_path, data_folder, dataset_num)
        imu_data_loader = MatDataLoader(imu_path)
        imu_data = IMUDataset(imu_data_loader)

        if train:
            vicon_path =  "{}/{}/Vicon/viconRot{}.mat".format(base_path,data_folder, dataset_num)
            vicon_data_loader = MatDataLoader(vicon_path)
            vicon_data = ViconDataset(vicon_data_loader)

            imu_idx, vicon_idx = findCommonStartTimeIdx(imu_data.ts, vicon_data.ts)
            imu_data.removeDataBeforeIdx(imu_idx)
            vicon_data.removeDataBeforeIdx(vicon_idx)
            self.vicon_data = vicon_data
            self.yaw_values_vicon = []
            self.pitch_values_vicon = []
            self.roll_values_vicon = []
            self.load_vicon_data()
        else:
            self.vicon_data = None

        accel_param_path = "{}/IMUParams.mat".format(base_path)
        accel_param_loader = MatDataLoader(accel_param_path)
        accel_param = AccelParam(accel_param_loader)

        imu_data.correctBias(accel_param)
        self.imu_data = imu_data

        self.roll_accl = []
        self.pitch_accl = []
        self.yaw_accl = []

        self.roll_gyro = []
        self.pitch_gyro = []
        self.yaw_gyro = []
        self.time_values_gyro = [] 


        self.cf_roll=[]
        self.cf_pitch=[]
        self.cf_yaw=[]

        self.gyro_rotation_matrices=[]
        self.accl_rotation_matrices=[]
        self.cf_rotation_matrices=[]

        self.q_madg=[]
        self.roll_madg=[]
        self.pitch_madg=[]
        self.yaw_madg=[]
        
        self.q_ukf=[]
        self.roll_UKF = []
        self.pitch_UKF = []
        self.yaw_UKF = []

        # self.q_diff=[]

        self.start_index_imu = 0
        self.start_index_vicon = 0
        

    def load_vicon_data(self):
        prev_euler_angles = None
        vicon_itr = ViconDataIterator(self.vicon_data)
        while True:
            vicon_measurement = vicon_itr.getCurrentData()

            rotation_matrix = vicon_measurement["rot_matrix"]

            rotation = R.from_matrix(rotation_matrix)

            euler_angles = rotation.as_euler('ZYX', degrees=False)

            # if prev_euler_angles is not None:
            #     euler_angles = wrap_angle(prev_euler_angles,euler_angles)
            # prev_euler_angles=euler_angles

            self.roll_values_vicon.append(euler_angles[2])
            self.pitch_values_vicon.append(euler_angles[1])
            self.yaw_values_vicon.append(euler_angles[0])

            if not vicon_itr.step():
                break


    def gyro(self):
        if self.vicon_data:
            initial_roll = self.roll_values_vicon[0]  
            initial_pitch = self.pitch_values_vicon[0] 
            initial_yaw = self.yaw_values_vicon[0]  
        else:
            initial_roll = 0
            initial_pitch = 0
            initial_yaw = 0

        self.roll_gyro = [initial_roll]
        self.pitch_gyro = [initial_pitch]
        self.yaw_gyro = [initial_yaw]
        imu_data_itr = IMUDataIterator(self.imu_data)

        while imu_data_itr.step():
            imu_measurement = imu_data_itr.getCurrentData()
            
            delta_time = imu_measurement["dt"]

            current_rotation = R.from_euler('zyx',[self.yaw_gyro[-1],self.pitch_gyro[-1],self.roll_gyro[-1]],degrees=False).as_matrix()
            current_gyro_data = np.array([imu_measurement["omega_xyz"][0],
                                          imu_measurement["omega_xyz"][1],
                                          imu_measurement["omega_xyz"][2]]).reshape([3,1])
            delta_rotation = np.matmul(current_rotation,current_gyro_data).T*delta_time
            delta_roll = delta_rotation[0,0]
            delta_pitch = delta_rotation[0,1]
            delta_yaw = delta_rotation[0,2]

            new_roll = self.roll_gyro[-1] + delta_roll
            new_pitch = self.pitch_gyro[-1] + delta_pitch
            new_yaw = self.yaw_gyro[-1] + delta_yaw

            self.roll_gyro.append(new_roll)
            self.pitch_gyro.append(new_pitch)
            self.yaw_gyro.append(new_yaw)

    def accl(self):
        imu_data_itr = IMUDataIterator(self.imu_data)
        while True:
            imu_measurement = imu_data_itr.getCurrentData()
            new_roll = np.arctan2(imu_measurement["accel_xyz"][1],
                                  np.sqrt(imu_measurement["accel_xyz"][0]**2 + imu_measurement["accel_xyz"][2]**2))
            new_pitch = np.arctan2(-1*imu_measurement["accel_xyz"][0], 
                                   np.sqrt(imu_measurement["accel_xyz"][1]**2 + imu_measurement["accel_xyz"][2]**2))
            new_yaw = np.arctan2(np.sqrt(imu_measurement["accel_xyz"][0]**2 + imu_measurement["accel_xyz"][1]**2), imu_measurement["accel_xyz"][2])

            self.roll_accl.append(new_roll)
            self.pitch_accl.append(new_pitch)
            self.yaw_accl.append(new_yaw)

            if not imu_data_itr.step():
                break


    def com_filter(self):
        if self.vicon_data:
            initial_roll = self.roll_values_vicon[0]  
            initial_pitch = self.pitch_values_vicon[0] 
            initial_yaw = self.yaw_values_vicon[0]  
        else:
            initial_roll = 0
            initial_pitch = 0
            initial_yaw = 0

        self.filtered_roll_accl = [initial_roll]
        self.filtered_pitch_accl = [initial_pitch]
        self.filtered_yaw_accl = [initial_yaw]

        self.filtered_roll_gyro = [initial_roll]
        self.filtered_pitch_gyro = [initial_pitch]
        self.filtered_yaw_gyro = [initial_yaw]

        alpha_lp = 0.2
        alpha_hp = 0.9999
        alpha_cf = 0.5

        for i in range(1, self.imu_data.data_size):
            self.filtered_roll_accl.append((1-alpha_lp)*self.filtered_roll_accl[-1] + alpha_lp*self.roll_accl[i])
            self.filtered_pitch_accl.append((1-alpha_lp)*self.filtered_pitch_accl[-1] + alpha_lp*self.pitch_accl[i])
            self.filtered_yaw_accl.append((1-alpha_lp)*self.filtered_yaw_accl[-1] + alpha_lp*self.yaw_accl[i])

            self.filtered_roll_gyro.append((alpha_hp)*(self.filtered_roll_gyro[-1] + self.roll_gyro[i] - self.roll_gyro[i-1]))
            self.filtered_pitch_gyro.append((alpha_hp)*(self.filtered_pitch_gyro[-1] + self.pitch_gyro[i] - self.pitch_gyro[i-1]))
            self.filtered_yaw_gyro.append((alpha_hp)*(self.filtered_yaw_gyro[-1] + self.yaw_gyro[i] - self.yaw_gyro[i-1]))

        for i in range(self.imu_data.data_size):
            self.cf_roll.append((1-alpha_cf)*self.filtered_roll_gyro[i] + alpha_cf*self.filtered_roll_accl[i])
            self.cf_pitch.append((1-alpha_cf)*self.filtered_pitch_gyro[i] + alpha_cf*self.filtered_pitch_accl[i])
            self.cf_yaw.append((1-alpha_cf)*self.filtered_yaw_gyro[i] + alpha_cf*self.filtered_yaw_accl[i])

    def madgwick(self):

        beta = 0.1  # Beta parameter for Madgwick filter
        count=1

        if self.vicon_data:
            RM_VinW = self.vicon_data.rot_matrices[:,:,0]
        else:
            RM_VinW = np.identity(3)

        initial_q = RM_to_quaternion(RM_VinW) # this is in world frame
        
        initial_state = {"q_wxyz": initial_q}
        self.q_madg = [initial_q]

        roll, pitch, yaw = quaternion_to_euler(initial_q)
        self.roll_madg = [roll]
        self.pitch_madg = [pitch]
        self.yaw_madg = [yaw]

        imu_data_itr = IMUDataIterator(self.imu_data)
        madgwick_filter = MadgwickFilter(initial_state)

        # self.q_diff.append(0)
        while imu_data_itr.step():
            # print(count)
            # vicon_data_itr.step()
            imu_measurement = imu_data_itr.getCurrentData()

            madgwick_filter.updateIMUMeasurement(imu_measurement)
            madgwick_filter.step()

            state = madgwick_filter.getCurrentState()

            self.q_madg.append(state["q_wxyz"])
            
            # vicon_rot = vicon_data_itr.getCurrentData()["rot_matrix"]
            # vicon_q = RM_to_quaternion(vicon_rot)
            # q_diff = np.linalg.norm(1 - quat_product(state["q_wxyz"], quat_conjugate(vicon_q)))

            # self.q_diff.append(q_diff)
            roll, pitch, yaw = quaternion_to_euler(state["q_wxyz"])
            self.roll_madg.append(roll)
            self.pitch_madg.append(pitch)
            self.yaw_madg.append(yaw)
            # count+=1

    def UKF(self):
        count=1
        if self.vicon_data:
            RM_VinW = self.vicon_data.rot_matrices[:,:,0]
        else:
            RM_VinW = np.identity(3)
        
        imu_data_itr = IMUDataIterator(self.imu_data)

        initial_q = RM_to_quaternion(RM_VinW) # this is in world frame
        
        initial_state_quat = {"q_wxyz": initial_q}

        initial_state_omega = imu_data_itr.getCurrentData() 
        # print(initial_omega)
        initial_state = {"wxyzw1w2w3": np.append(initial_state_quat["q_wxyz"],initial_state_omega["omega_xyz"])}
        # print(initial_state)
        self.q_ukf = [initial_q]

        roll, pitch, yaw = quaternion_to_euler(initial_q)
        self.roll_UKF = [roll]
        self.pitch_UKF = [pitch]
        self.yaw_UKF = [yaw]

        self.P = np.zeros((6,6))
        np.fill_diagonal(self.P, [0.01,0.01,0.01,0.01,0.01,0.01])
        # Process Model noise
        self.Q = np.zeros((6,6))
        np.fill_diagonal(self.Q, [105,105,105,0.1,0.1,0.1])
        # np.fill_diagonal(self.Q, [100,100,100,0.1,0.1,0.1])

        # Measurement Model noise
        self.R = np.zeros((6,6))
        np.fill_diagonal(self.R, [11.2,11.2,11.2,0.01,0.01,0.01])
        # np.fill_diagonal(self.R, [0.5,0.5,0.5,0.01,0.01,0.01])


        UK_filter = UKFilter(initial_state, self.Q, self.R, self.P)

        while imu_data_itr.step():
            # print(count)
            # vicon_data_itr.step()
            imu_measurement = imu_data_itr.getCurrentData()

            UK_filter.updateIMUMeasurement(imu_measurement)
            UK_filter.step()

            state = UK_filter.getCurrentState()
            # print('state=', state["wxyzw1w2w3"][0:4])

            self.q_ukf.append(state["wxyzw1w2w3"][0:4])
            
            roll, pitch, yaw = quaternion_to_euler(state["wxyzw1w2w3"][0:4])
            self.roll_UKF.append(roll)
            self.pitch_UKF.append(pitch)
            self.yaw_UKF.append(yaw)
            count+=1
        
        # print(len(self.q_ukf))
        # print(self.q_ukf)

    def plot(self, plot_gyro, plot_accel, plot_com, plot_madg, plot_UKF, plot_vicon):
        plt.figure(figsize=(20, 15))

        plt.subplot(3,1,1)

        
        if self.vicon_data and plot_vicon:
            plt.plot(self.vicon_data.ts, self.roll_values_vicon, label='X Orientation - Vicon',color='k')
        if plot_accel:
            plt.plot(self.imu_data.ts, self.roll_accl, label='X Orientation - Accel',color='b')
        if plot_gyro:
            plt.plot(self.imu_data.ts, self.roll_gyro, label='X Orientation - gyro',color='r')
        if plot_com:
            plt.plot(self.imu_data.ts, self.cf_roll, label='X Orientation - comp filter',color='m')
        if plot_madg:
            plt.plot(self.imu_data.ts, self.roll_madg, label='X Orientation - madg filter',color='g')
        if plot_UKF:
            plt.plot(self.imu_data.ts, self.roll_UKF, label='X Orientation -  UKF',color='y')


        plt.xlabel('Time')
        plt.ylabel('Roll Orientation')
        plt.title('Roll Orientation Over Time')
        plt.legend()

        plt.subplot(3,1,2)
        if self.vicon_data and plot_vicon:
            plt.plot(self.vicon_data.ts, self.pitch_values_vicon, label='Y Orientation - Vicon',color='k')
        if plot_accel:
            plt.plot(self.imu_data.ts, self.pitch_accl, label='Y Orientation - Accel',color='b')
        if plot_gyro:
            plt.plot(self.imu_data.ts, self.pitch_gyro, label='Y Orientation - gyro',color='r')
        if plot_com:
            plt.plot(self.imu_data.ts, self.cf_pitch, label='Y Orientation - comp filter',color='m')
        if plot_madg:
            plt.plot(self.imu_data.ts, self.pitch_madg, label='Y Orientation - madg filter',color='g')
        if plot_UKF:
            plt.plot(self.imu_data.ts, self.pitch_UKF, label='Y Orientation -  UKF',color='y')

        plt.xlabel('Time')
        plt.ylabel('Pitch Orientation')
        plt.title('Pitch Orientation Over Time')
        plt.legend()

        plt.subplot(3,1,3)
        if self.vicon_data and plot_vicon:
            plt.plot(self.vicon_data.ts, self.yaw_values_vicon, label='Z Orientation - Vicon',color='k')
        if plot_accel:
            plt.plot(self.imu_data.ts, self.yaw_accl, label='Z Orientation - Accel',color='b')
        if plot_gyro:
            plt.plot(self.imu_data.ts, self.yaw_gyro, label='Z Orientation - gyro',color='r')
        if plot_com:
            plt.plot(self.imu_data.ts, self.cf_yaw, label='Z Orientation - comp filter',color='m')
        if plot_madg:
            plt.plot(self.imu_data.ts, self.yaw_madg, label='Z Orientation - madg filter',color='g')
        if plot_UKF:
            plt.plot(self.imu_data.ts, self.yaw_UKF, label='Z Orientation -  UKF',color='y')

        plt.xlabel('Time')
        plt.ylabel('Yaw Orientation')
        plt.legend()
        plt.title('Yaw Orientation Over Time')
        
        if self.vicon_data:
            if not os.path.exists('./Outputs/Train/'):
                os.makedirs('./Outputs/Train/')
            plt.savefig('./Outputs/Train/CompVsMadgVsVicon{}.eps'.format(self.dataset_num), format = 'eps')
        else:
            if not os.path.exists('./Outputs/Test/'):
                os.makedirs('./Outputs/Test/')
            plt.savefig('./Outputs/Test/CompVsMadg{}.eps'.format(self.dataset_num), format = 'eps')
        plt.show()

    def rotplotter(self):
        fig = plt.figure(figsize=(25, 5))
        if self.vicon_data:
            fig.canvas.manager.set_window_title('Train Dataset{}'.format(self.dataset_num))
        else:
            fig.canvas.manager.set_window_title('Test Dataset{}'.format(self.dataset_num))
        myAxis1 = fig.add_subplot(161, projection='3d')
        myAxis2 = fig.add_subplot(162, projection="3d")
        myAxis3 = fig.add_subplot(163, projection="3d")
        myAxis4 = fig.add_subplot(164, projection="3d")
        myAxis5 = fig.add_subplot(165, projection="3d")
        if self.vicon_data:
            myAxis6 = fig.add_subplot(166, projection="3d")
            length = min(self.imu_data.data_size, self.vicon_data.data_size)
        else:
            length= self.imu_data.data_size
        for i in range(0,length,20):
            imu_idx = i + self.start_index_imu
            vicon_idx = i + self.start_index_vicon

            rotation_gyro = R.from_euler('zyx', [self.yaw_gyro[imu_idx], self.pitch_gyro[imu_idx], self.roll_gyro[imu_idx]], degrees=False)
            rotation_matrix_gyro = rotation_gyro.as_matrix()

            rotation_accl = R.from_euler('zyx', [self.yaw_accl[imu_idx], self.pitch_accl[imu_idx], self.roll_accl[imu_idx]], degrees=False)
            rotation_matrix_accl = rotation_accl.as_matrix()

            rotation_cf = R.from_euler('zyx', [self.cf_yaw[imu_idx], self.cf_pitch[imu_idx], self.cf_roll[imu_idx]], degrees=False)
            rotation_matrix_cf = rotation_cf.as_matrix()

            rotation_matrix_madg = quaternion_to_RM(self.q_madg[imu_idx])

            rotation_matrix_ukf = quaternion_to_RM(self.q_ukf[imu_idx])
            # print('\n', rotation_matrix_ukf)

            myAxis1.clear()
            rotplot(rotation_matrix_gyro, myAxis1)
            plt.sca(myAxis1)
            plt.title("Gyroscope")

            myAxis2.clear()
            rotplot(rotation_matrix_accl, myAxis2)
            plt.sca(myAxis2)
            plt.title("Accelerometer")

            myAxis3.clear()
            rotplot(rotation_matrix_cf, myAxis3)
            plt.sca(myAxis3)
            plt.title("Complementary")

            myAxis4.clear()
            rotplot(rotation_matrix_madg, myAxis4)
            plt.sca(myAxis4)
            plt.title("Madgwick")

            myAxis5.clear()
            rotplot(rotation_matrix_ukf, myAxis5)
            plt.sca(myAxis5)
            plt.title("UKF")

            if self.vicon_data:
                myAxis6.clear()
                rotation_matrix_vicon = self.vicon_data.getDataAtIdx(vicon_idx)
                rotplot(rotation_matrix_vicon, myAxis6)
                plt.sca(myAxis6)
                plt.title("Vicon")

            plt.pause(0.005)

        # plt.show()
        plt.close()
        

def main():
    dataset_num = 10
    base_folder = "./Data"
    train = False
    
    phase1 = Phase1(base_folder, dataset_num, train)

    plot_gyro = False
    plot_accel = False
    plot_com= True
    plot_madg = True
    plot_UKF = True
    plot_vicon = True

    phase1.gyro()
    phase1.accl()
    phase1.com_filter()
    phase1.madgwick()
    phase1.UKF()
    phase1.plot(plot_gyro, plot_accel, plot_com, plot_madg, plot_UKF, plot_vicon)
    # phase1.rotplotter()


    ### For train set
    # for dataset_num in range(1,7):
    #     base_folder = "./Data"
    #     train = True
        
    #     phase1 = Phase1(base_folder, dataset_num, train)

    #     plot_gyro = False
    #     plot_accel = False
    #     plot_com= True
    #     plot_madg = True
    #     plot_UKF = True
    #     plot_vicon = True

    #     phase1.gyro()
    #     phase1.accl()
    #     phase1.com_filter()
    #     phase1.madgwick()
    #     phase1.UKF()
    #     phase1.plot(plot_gyro, plot_accel, plot_com, plot_madg, plot_UKF, plot_vicon)
        # phase1.rotplotter()

        # print("Press 'n' key for next data set")
        # while True:
        #     if keyboard.read_key() == "n":
        #         break
    

    ### For test set
    # for dataset_num in range(7,11):
    #     base_folder = "./Data"
    #     train = False
    #     phase1 = Phase1(base_folder, dataset_num, train)

    #     plot_gyro = False
    #     plot_accel = False
    #     plot_com= True
    #     plot_madg = True
    #     plot_UKF = True
    #     plot_vicon = True
        
    #     phase1.gyro()
    #     phase1.accl()
    #     phase1.com_filter()
    #     phase1.madgwick()
    #     phase1.UKF()
    #     # phase1.sync_time()
    #     phase1.plot(plot_gyro, plot_accel, plot_com, plot_madg, plot_vicon)
    #     # phase1.rotplotter()
    #     # print("Press 'n' key for next data set")
    #     # while True:
    #     #     if keyboard.read_key() == "n":
    #     #         break

if __name__ == "__main__":
    main()