# Local files
import pyClientRLagentPytorch
import utils

# Numpy, OpenCV
import numpy as np
import cv2

# Connect to Unity environment
ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 13000       # Port number that the TCP/IP interface listens to
size       = 128
screen_height = screen_width = size
environment = pyClientRLagentPytorch.Environment(ip = ip, port = port, size = size)
print(environment.client)

# reset the environment
end, reward, state_raw = environment.reset()

# Create window to display the frames
class Window:
    def __init__(self,windowname='Hallway', size=(1200,1200), frame=None):
        self._size          = size
        self._windowname    = windowname

    def show(self,frame):
        frame = cv2.resize(frame, self._size)
        cv2.imshow(self._windowname, frame)

disp = Window(windowname='Hallway', size=(600,600))


# Initialize some phosphene simulations
class Simulator():
    def __init__(self, low=(26,26),high=(60,60),**kwargs):
        self.low_res  = utils.PhospheneSimulator(phosphene_resolution=low,**kwargs)
        self.high_res = utils.PhospheneSimulator(phosphene_resolution=high,**kwargs)
        self.sim_mode = 0

    def __call__(self,frame):
        if self.sim_mode==0:
            return frame[:,:,::-1].astype('uint8')
        elif self.sim_mode == 1:
            frame = cv2.resize(frame, (480,480))
            contours = cv2.Canny(frame,35,70)
            phosphenes = self.low_res(contours)
            return (255*phosphenes/phosphenes.max()).astype('uint8')
        elif self.sim_mode == 2:
            frame = cv2.resize(frame, (480,480))
            contours = cv2.Canny(frame,35,70)
            phosphenes = self.high_res(contours)
            return (255*phosphenes/phosphenes.max()).astype('uint8')

simulator = Simulator(low=(26,26),high=(60,60),sigma=1.2)

while environment.client:

    # Display current state
    frame = environment.state2usableArray(state_raw)
    frame = simulator(frame)
    disp.show(frame)

    # Get key
    key = cv2.waitKey(0)
    if key == ord('w'):
        end, reward, state_raw = environment.step(0)
        print(reward)

    if key == ord('a'):
        end, reward, state_raw = environment.step(1)
        print(reward)

    if key == ord('d'):
        end, reward, state_raw = environment.step(2)
        print(reward)

    if key == ord('r'):
        end, reward, state_raw = environment.reset()
        print(reward)

    if key == 49:
        simulator.sim_mode = 0

    if key == 50:
        simulator.sim_mode = 1

    if key == 51:
        simulator.sim_mode = 2

    if key == ord('q'):
        break
