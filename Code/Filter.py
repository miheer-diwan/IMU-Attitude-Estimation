import numpy as np
from scipy.spatial.transform import Rotation as R
# from wrapper import *
import math

def quaternion_to_euler(q):

    rotation = R.from_quat(q)

    # Convert Rotation object to Euler angles
    euler_angles = rotation.as_euler('xyz', degrees=False)


    roll = euler_angles[2]
    pitch = euler_angles[1]
    yaw = euler_angles[0]
    
    return roll, pitch, yaw

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

def quat_conjugate(q):
    q_conj = np.array([q[0],-q[1],-q[2],-q[3]])
    return q_conj

def quat_inverse(q):
    return quat_conjugate(q)/((np.linalg.norm(q))**2)

def calc_q_delta(w,dt):
    norm = np.linalg.norm(w)
    if(norm == 0.0):
        q = np.array([1,0,0,0],dtype=np.float64)
        return q
    alpha = norm*dt
    # print(alpha)
    axis = w/norm
    # print(axis)
    q = np.array([math.cos(alpha/2),axis[0]*math.sin(alpha/2),axis[1]*math.sin(alpha/2),axis[2]*math.sin(alpha/2)])
    q = q/np.linalg.norm(q)
    # print(q)
    return q

def RV2quat(w):
    norm = np.linalg.norm(w)
    if(norm == 0.0):
        q = np.array([1,0,0,0],dtype=np.float64)
        return q
    alpha = norm
    axis = w/alpha
    # print(axis)
    q = np.array([math.cos(alpha/2),axis[0]*math.sin(alpha/2),axis[1]*math.sin(alpha/2),axis[2]*math.sin(alpha/2)])
    q = q/norm
    # print(q)
    return q

def quat2RV(q):
    sinalpha = np.linalg.norm(q[1:4])
    cosalpha = q[0]

    alpha = math.atan2(sinalpha,cosalpha)
    if (sinalpha==0):
        rv = np.array([0,0,0],dtype=np.float64)
        return rv

    e = q[1:4]/float(sinalpha)
    rv = e*2.*alpha
    return rv





class Filter():
    def __init__(self, initial_states):
        self.count=0
        self.current_state = initial_states
        self.t = 0.0
        self.current_measurement = None

    def updateIMUMeasurement(self, imu_measurement):
        self.imu_measurement = imu_measurement

    def step(self):
        raise NotImplementedError
    
    def getCurrentState(self):
        return self.current_state
    
class MadgwickFilter(Filter):
    def step(self):
        beta = 0.1
        gyro_measurement = np.append(np.array([0]),self.imu_measurement["omega_xyz"])
        accel_measurement = np.append(np.array([0]), self.imu_measurement["accel_xyz"])
        accel_measurement /= np.linalg.norm(accel_measurement)

        q_wxyz = self.current_state["q_wxyz"]
        q_w = q_wxyz[0]
        q_x = q_wxyz[1]
        q_y = q_wxyz[2]
        q_z = q_wxyz[3]


        f = np.array([2.0 * (q_x * q_z - q_w * q_y) - accel_measurement[1],
                          2.0 * (q_w * q_x + q_y * q_z) - accel_measurement[2],
                          2*(0.5 - q_x**2 - q_y**2) - accel_measurement[3]]).T# 3x1
            
        J = np.array([[-2*q_y, 2*q_z, -2*q_w, 2*q_x],
                        [ 2*q_x, 2*q_w, 2*q_z, 2*q_y],
                        [ 0, -4*q_x, -4*q_y, 0]]) # 3x4
        
        delta_f = np.dot(J.T, f) # 4x1

        delta_q_accel = - beta * (delta_f/np.linalg.norm(f)) # 4x1

        q_wxyz_norm = q_wxyz / np.linalg.norm(q_wxyz)
        # print(q_wxyz_norm)
        delta_q_gyro = 0.5 * quat_product(q_wxyz_norm,gyro_measurement.T)

        q_dot = delta_q_gyro + delta_q_accel
        # print(q_dot)
        # print(q_wxyz_norm + q_dot*self.imu_measurement["dt"])

        self.current_state["q_wxyz"] = q_wxyz_norm + q_dot*self.imu_measurement["dt"]

        return
    
class UKFilter(Filter):
    def __init__(self, initial_states, Q, R, P):
        super().__init__(initial_states)
        self.Q = Q #Q
        self.R = R #R
        self.P = P #P

    def compute_sigma_points(self):
        # Implement sigma point generation here
        self.S = np.linalg.cholesky(self.P + self.Q)
        self.W = np.concatenate([np.sqrt(self.S.shape[1])* self.S,
                                 -np.sqrt(self.S.shape[1]) * self.S],axis=1)
        
        X = np.zeros((7, self.W.shape[1])) #Quaternion part of Xi
        for i in range(self.W.shape[1]):
            X[:4,i] = quat_product(self.current_state["wxyzw1w2w3"][:4],RV2quat(self.W[:3,i]))
            X[4:,i] = self.current_state["wxyzw1w2w3"][4:] + self.W[3:,i]
        # print(X2.shape)
        self.X = X
        # print(self.X.shape)
    
    def IGD(self,Y):
        qbar = Y[:4,0]
        num_pts = Y.shape[1]
        err_vectors =np.zeros((num_pts,3))  #12x3
        iter_ = 1
        mean_err = np.array([np.Infinity,np.Infinity,np.Infinity])
        thresh = 1e-5
        max_iter = 2000
        
        while(np.linalg.norm(mean_err)>thresh and iter_<=max_iter):
            for i in range(num_pts):
                qi = Y[:4,i]
                err_quat = quat_product(qi, quat_inverse(qbar))
                err_vectors[i,:] = quat2RV(err_quat)
            # compute mean of all the err_vectors
            mean_err = np.mean(err_vectors,axis = 0)
            # convert mean error rotation vector to quaternion
            mean_err_quat = RV2quat(mean_err)
            qbar = quat_product(mean_err_quat,qbar)
            qbar = np.divide(qbar,np.linalg.norm(qbar))
            iter_ += 1

        omega_bar = (1/float(num_pts))*np.mean(Y[4:,:], axis=1)
        return qbar,omega_bar   #xkbar
    
    def get_W_dash(self, qbar, omega_bar):
        num_pts = self.Y.shape[1]
        W_dash_RV = np.zeros((3,num_pts))
        W_dash_omega = np.zeros((3,num_pts))
        for i in range(num_pts):
            quat = quat_product(self.Y[:4,i],quat_inverse(qbar))
            W_dash_RV[:,i] = quat2RV(quat)
            W_dash_omega[:,i] = self.Y[4:,i]- omega_bar
        
        W_dash = np.concatenate([W_dash_RV,W_dash_omega],axis=0)

        return W_dash

    def get_apriori_cov(self,W_dash):
        num_pts = W_dash.shape[1]
        Pk_bar = np.zeros((6,6))
        for i in range(num_pts):
            W_dash_i = W_dash[:,i].reshape((6,1))
            Pk_bar += np.matmul(W_dash_i,W_dash_i.T)
        Pk_bar /= num_pts
        return Pk_bar

    def compute_Z(self):
        num_pts = self.Y.shape[1]
        # Gravity vector in quaternion
        g = np.array([0,0,0,1],dtype=np.float64)
        Z = np.zeros((6,num_pts))
        for i in range(num_pts):
            quat = quat_product(quat_inverse(self.Y[:4,i]),g)
            quat = quat_product(quat,self.Y[:4,i])
            quat /= np.linalg.norm(quat)
            Z[:,i] = np.concatenate([quat2RV(quat),self.Y[4:,i]],axis=0)
        return Z

    def compute_Z_mean(self,Z):
        return np.mean(Z,axis=1)
    
    def compute_Z_mean_centered(self, Z):
        Z_mean = self.compute_Z_mean(Z)

        Z_centered = Z - Z_mean[:,np.newaxis]
        return Z_centered

    def compute_Pzz(self,Z):
        # center the Z around mean
        Z_centered = self.compute_Z_mean_centered(Z)

        Pzz = self.get_apriori_cov(Z_centered)
        return Pzz
    
    def Compute_vk(self, acc, gyro, Z):
        #compute the innovation term vk
        Zk = np.concatenate((acc,gyro),axis=0)
        vk = Zk - self.compute_Z_mean(Z)
        return vk

    def Compute_cross_corr_mat(self,W_dash, Z_mean_centered):
        num_pts = W_dash.shape[1]
        Pxz = np.zeros((6,6))
        for i in range(num_pts):
            W_dash_i = W_dash[:,i].reshape((6,1))
            Z_mean_centered_i = Z_mean_centered[:,i].reshape((6,1))
            Pxz += np.matmul(W_dash_i,Z_mean_centered_i.T)
        Pxz /= num_pts
        return Pxz

    def predict(self):
        Y = np.zeros((7, self.W.shape[1]))
        for i in range(self.W.shape[1]):
            Y[:4,i] = quat_product(self.X[:4,i],calc_q_delta(self.current_state["wxyzw1w2w3"][4:],self.imu_measurement["dt"]))
            Y[4:,i] = self.current_state["wxyzw1w2w3"][4:] + self.W[3:,i]
        self.Y = Y
        # print(self.Y.shape)

        qbar,omega_bar = self.IGD(self.Y)
        xkbar = np.concatenate((qbar,omega_bar),axis=0)
        W_dash = self.get_W_dash(qbar,omega_bar)
        Pk_bar = self.get_apriori_cov(W_dash)
        Z = self.compute_Z()
        z_mean_centered = self.compute_Z_mean_centered(Z)
        Pzz = self.compute_Pzz(Z)
        Vk = self.Compute_vk(self.imu_measurement["accel_xyz"], self.imu_measurement["omega_xyz"], Z)
        Pvv = Pzz + self.R
        Pxz = self.Compute_cross_corr_mat(W_dash, z_mean_centered)
        return Pxz,Pvv,Vk,xkbar,Pk_bar

    def State_update(self,Pxz,Pvv,Vk,xkbar,Pk_bar):
        #Compute Kalman Gain and update state
        Kk = np.matmul(Pxz,np.linalg.inv(Pvv))
        Kkvk = np.matmul(Kk,Vk)
        quat = quat_product(xkbar[:4],RV2quat(Kkvk[:3]))
        # quat = xkbar[:4] + RV2quat(Kkvk[:3])
        # quat = np.divide(quat,np.linalg.norm(quat))
        quat /= np.linalg.norm(quat)
        xkbarhat = np.concatenate((quat,xkbar[-3:]+Kkvk[-3:]),axis=0)
        Pk = Pk_bar - np.matmul(Kk,np.matmul(Pvv,Kk.T))
        return Pk, xkbarhat
        # Implement measurement update step of UKF here
        
    def step(self):
        self.compute_sigma_points()
        Pxz,Pvv,Vk,xkbar,Pk_bar = self.predict()
        Pk,xkbarhat = self.State_update(Pxz,Pvv,Vk,xkbar,Pk_bar)
        self.current_state["wxyzw1w2w3"] = np.zeros(7)
        self.current_state["wxyzw1w2w3"][:4] = xkbarhat[:4]
        # print(self.current_state["wxyzw1w2w3"][:4])
        self.current_state["wxyzw1w2w3"][4:7] = self.imu_measurement["omega_xyz"]
        self.P = Pk




