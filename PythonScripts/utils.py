import numpy as np
import skimage.filters
import cv2
from threading import Thread


class Window:
    def __init__(self,windowname='Hallway', size=(1200,1200), frame=None):
        self._size          = size
        self._windowname    = windowname
        self.frame          = frame if frame is not None else np.zeros(size+(3,))
        self.stopped        = False
        self.keypress       = None

    def start(self):
        Thread(target=self.show, args=()).start()
        return self


    def show(self):
        while not self.stopped:
            img = cv2.resize(self.frame, self._size)
            cv2.imshow(self._windowname, img)
            key = cv2.waitKey(1)
            if key != -1:
                self.keypress = key

    def getKey(self):
        key = self.keypress
        self.keypress = None
        return key

    def stop(self):
        self.stopped = True

class PhospheneSimulator(object):
    def __init__(self,phosphene_resolution=(50,50), size=(480,480),  jitter=0.35, intensity_var=0.9, aperture=.66, sigma=0.8, custom_grid=None):
        """Phosphene simulator class to create gaussian-based phosphene simulations from activation mask
        on __init__, provide custom phosphene grid or use the grid parameters to create one
        - aperture: receptive field of each phosphene (uses dilation of the activation mask to achieve this)
        - sigma: the size parameter for the gaussian phosphene simulation """
        if custom_grid is None:
            self.phosphene_resolution = phosphene_resolution
            self.size = size
            self.phosphene_spacing = np.divide(size,phosphene_resolution)
            self.jitter = jitter
            self.intensity_var = intensity_var
            self.grid = self.create_regular_grid(self.phosphene_resolution,self.size,self.jitter,self.intensity_var)
            self.aperture = np.round(aperture*self.phosphene_spacing[0]).astype(int) #relative aperture > dilation kernel size
        else:
            self.grid = custom_grid
            self.aperture = aperture
        self.sigma = sigma
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.aperture,self.aperture))
        self.k_size = 11 #np.round(4*sigma+1).astype(int) # rule of thumb: choose k_size>3*sigma

    def __call__(self,activation_mask):
        """ returns the phosphene simulation (image), given an activation mask"""
        assert self.grid.shape == activation_mask.shape
        self.mask = cv2.dilate(activation_mask, self.dilation_kernel, iterations=1)
        phosphenes = self.grid * self.mask
        phosphenes = cv2.GaussianBlur(phosphenes,(self.k_size,self.k_size),self.sigma)
        return phosphenes

    def create_regular_grid(self, phosphene_resolution, size, jitter, intensity_var):
        """Returns regular eqiodistant phosphene grid of shape <size> with resolution <phosphene_resolution>
         for variable phosphene intensity with jitterred positions"""
        grid = np.zeros(size)
        phosphene_spacing = np.divide(size,phosphene_resolution)
        for x in np.linspace(0,size[0],num=phosphene_resolution[0],endpoint=False)+0.5*phosphene_spacing[0] :
            for y in np.linspace(0,size[1],num=phosphene_resolution[1],endpoint=False)+0.5*phosphene_spacing[0]:
                deviation = np.multiply(jitter*(2*np.random.rand(2)-1),phosphene_spacing)
                intensity = intensity_var*(np.random.rand()-0.5)+1
                rx = np.clip(np.round(x+deviation[0]),0,size[0]-1).astype(int)
                ry = np.clip(np.round(y+deviation[1]),0,size[1]-1).astype(int)
                grid[rx,ry]= intensity
        return grid


def maximum_gradient(edges, phase):
    """Finds the maximum value for line-drawing <edges> in gradient direction <phase>"""
    gmax = np.zeros(edges.shape)
    for i in range(gmax.shape[0]):
        for j in range(gmax.shape[1]):
            if phase[i][j] < 0:
                phase[i][j] += 360

            if ((j + 1) < gmax.shape[1]) and ((j - 1) >= 0) and ((i + 1) < gmax.shape[0]) and ((i - 1) >= 0):
                # 0 degrees
                if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
                    if edges[i][j] >= edges[i][j + 1] and edges[i][j] >= edges[i][j - 1]:
                        gmax[i][j] = edges[i][j]
                # 45 degrees
                if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
                    if edges[i][j] >= edges[i - 1][j + 1] and edges[i][j] >= edges[i + 1][j - 1]:
                        gmax[i][j] = edges[i][j]
                # 90 degrees
                if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
                    if edges[i][j] >= edges[i - 1][j] and edges[i][j] >= edges[i + 1][j]:
                        gmax[i][j] = edges[i][j]
                # 135 degrees
                if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
                    if edges[i][j] >= edges[i - 1][j - 1] and edges[i][j] >= edges[i + 1][j + 1]:
                        gmax[i][j] = edges[i][j]
    return gmax


def non_maximum_supression(img, edges, low=None, high=None):
    """ Performs non-maximum suppresion (line-thinning) on line-drawing <edges> using gradients on original image <img>,
    followed by hysteresis thresholding with thresholds <low> and <high>. If no thresholds are provided, adaptive
    thresholding is performed instead """
    Gy = skimage.filters.sobel_h(img)
    Gx = skimage.filters.sobel_v(img)
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    thin = maximum_gradient(edges, theta)
    if low is None or high is None:
        mmax = np.max(thin)
        thin = skimage.filters.apply_hysteresis_threshold(thin, low=.33*mmax, high=.85*mmax).astype(float)
    else:
        thin = skimage.filters.apply_hysteresis_threshold(thin, low=low, high=high).astype(float)
    return thin
