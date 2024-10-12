from aubo_simulation.robot import Auboi16
from aubo_simulation.env import AuboEnv
from aubo_simulation.utilities import Camera
from aubo_simulation.utilities import CameraIntrinsic
from aubo_simulation.sample import shell_section
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_end_matrix(end_pos, end_tar, cam_up_vector):
    y_vector=np.array(end_tar-end_pos)/np.linalg.norm(end_tar-end_pos)
    x_vector=np.cross(cam_up_vector,y_vector)/np.linalg.norm(np.cross(cam_up_vector,y_vector))
    z_vector=np.cross(x_vector,y_vector)/np.linalg.norm(np.cross(x_vector,y_vector))

    Tend2world=np.array([x_vector,y_vector,z_vector,np.array(end_pos)]).transpose()
    Tend2world=np.concatenate([Tend2world,[[0,0,0,1]]],axis=0)

    return Tend2world

robot=Auboi16((0,0,0),(0,0,0))
env=AuboEnv(robot,vis=True)

end_pos,end_ori=env.reset()
intrinsic=CameraIntrinsic(1280,720,918.27160645,918.02313232,643.14483643,357.28491211)

up_vector=np.array([0,0,1])
distance2end=0.7
camera_pos=end_pos-np.array([0,distance2end,0])
camera=Camera(camera_pos,end_pos,up_vector,0.1,1000,intrinsic)
env.reset_camera(camera)


end_poses=shell_section(camera_pos,distance2end,distance2end,1,-15,15,10,60,120,10)

while True: 
    for end_pos in end_poses:
        Tend2world=compute_end_matrix(end_pos,camera_pos,up_vector)
        euler=R.from_matrix(Tend2world[:3,:3]).as_euler('xyz')
        trans=Tend2world[:3,3]
        action=np.concatenate((trans,euler,[0]))
        env.step(action,'end')
        image_dict=env.get_observation()
        cv2.imshow("rgb",image_dict['rgb'])
        cv2.waitKey(1)