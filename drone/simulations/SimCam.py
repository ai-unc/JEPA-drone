"""Camera outputs: uint8 grayscale, metric depth, and 32x32x2 depth/class tensor.
Class IDs: 0 ignored, 1 wall/ground, 2 obstacle, 3 goal. Empty depth is NaN.
"""
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from projectairsim.utils import unpack_image
from drone.Subsystems.data.data import State

class SimCamera:
    FREQUENCY = 10
    IMAGE_RESOLUTION = (640, 480)
    DEPTH_RESOLUTION = (32, 32)
    GRID_RESOLUTION = (20, 15)
    MAX_DEPTH = 20.
    MAX_RELIABLE_DEPTH = 8.
    MIN_DEPTH = .5
    CLASS_NAMES = {0:'ignored', 1:'wall', 2:'obstacle', 3:'goal'}
    PALETTE_RGB = np.array([[55,181,57],[153,108,6],[112,105,191],[89,121,72]],dtype=np.uint8)

    def __init__(self):
        self.grayscale_image = self.depth_image = self.obstacle = None
        self.display_raw = self.display_depth = self.display_tensorized = False
        self.timestamp = None
        self._pending = {}
        self._ready = None
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sim-camera')
        self._future = None
        self._goal_cell = None

    def attach(self, simulation):
        topics = simulation.drone.sensors['Vision']
        self._required = ({0} if self.display_raw else set()) | ({1} if self.display_depth or self.display_tensorized else set()) | ({3} if self.display_tensorized else set())
        for kind, topic in ((0,'scene_camera'),(1,'depth_planar_camera'),(3,'segmentation_camera')):
            if kind in self._required:
                simulation.client.subscribe(topics[topic], lambda _, msg, k=kind: self._receive(k,msg))

    def _receive(self, kind, msg):
        stamp = msg['time_stamp']
        with self._lock:
            frame = self._pending.setdefault(stamp,{})
            frame[kind] = msg
            if self._required.issubset(frame) and (self._ready is None or stamp>self._ready[0]):
                self._ready = (stamp,frame.copy())
            for old in sorted(self._pending)[:-2]:
                del self._pending[old]

    @classmethod
    def cell_to_depth_and_object(cls, cell, simulation=None):
        depth, labels = cell
        y, x = depth.shape[0]//2, depth.shape[1]//2
        value, label = depth[y,x], labels[y,x]
        if not (np.isfinite(value) and cls.MIN_DEPTH<=value<=cls.MAX_RELIABLE_DEPTH and label!=0):
            return np.array([np.nan,0],dtype=np.float32)
        return np.array([value,label],dtype=np.float32)

    @classmethod
    def tensorize(cls, image, simulation=None):
        depth, labels = image
        if labels.shape!=depth.shape:
            raise ValueError('Depth and labels must have matching dimensions')
        ys = ((np.arange(32)+.5)*depth.shape[0]/32).astype(int)
        xs = ((np.arange(32)+.5)*depth.shape[1]/32).astype(int)
        sampled_depth = depth[np.ix_(ys,xs)]
        sampled_labels = labels[np.ix_(ys,xs)]
        valid = np.isfinite(sampled_depth)&(sampled_depth>=cls.MIN_DEPTH)&(sampled_depth<=cls.MAX_RELIABLE_DEPTH)&(sampled_labels!=0)
        result = np.empty((32,32,2),dtype=np.float32)
        result[:,:,0] = np.where(valid,sampled_depth,np.nan)
        result[:,:,1] = np.where(valid,sampled_labels,0)
        return result

    @staticmethod
    def sim_to_image(simulation):
        return simulation.camera.grayscale_image

    def _process_frame(self, frame):
        grayscale = depth_image = obstacle = goal_cell = None
        if self.display_raw:
            grayscale = cv2.cvtColor(unpack_image(frame[0])[:,:,:3],cv2.COLOR_BGR2GRAY)
        if self.display_depth or self.display_tensorized:
            if frame[1]['encoding']!='16FC1':
                raise ValueError('Expected metric 16FC1 depth; ambiguous depth units rejected')
            depth_image = unpack_image(frame[1]).astype(np.float32)
        if self.display_tensorized:
            ys = ((np.arange(32)+.5)*depth_image.shape[0]/32).astype(int)
            xs = ((np.arange(32)+.5)*depth_image.shape[1]/32).astype(int)
            segmentation = unpack_image(frame[3])[:,:,:3][:,:,::-1]
            goal_pixels = np.argwhere(np.all(segmentation==self.PALETTE_RGB[3],axis=-1))
            if goal_pixels.size:
                center = goal_pixels.mean(axis=0)
                goal_cell = (min(31,int(center[0]*32/depth_image.shape[0])),
                             min(31,int(center[1]*32/depth_image.shape[1])))
            rgb = segmentation[np.ix_(ys,xs)]
            labels = np.zeros((32,32),dtype=np.uint8)
            for label in (1,2,3):
                labels[np.all(rgb==self.PALETTE_RGB[label],axis=-1)] = label
            sampled_depth = depth_image[np.ix_(ys,xs)]
            valid = np.isfinite(sampled_depth)&(sampled_depth>=self.MIN_DEPTH)&(sampled_depth<=self.MAX_RELIABLE_DEPTH)&(labels!=0)
            obstacle = np.stack((np.where(valid,sampled_depth,np.nan),np.where(valid,labels,0)),axis=-1).astype(np.float32)
        return grayscale, depth_image, obstacle, goal_cell

    def update(self, simulation=None):
        if self._future is not None and self._future.done():
            self.grayscale_image, self.depth_image, self.obstacle, self._goal_cell = self._future.result()
            self._future = None
            self._display(simulation)
        if self._future is not None:
            return False
        with self._lock:
            ready, self._ready = self._ready, None
        if ready is None or ready[0]==self.timestamp:
            return False
        self.timestamp, frame = ready
        self._future = self._executor.submit(self._process_frame, frame)
        return True

    def _display(self, simulation=None):
        if self.display_raw:
            cv2.imshow('Grayscale',self.grayscale_image)
        if self.display_depth:
            d = np.nan_to_num(self.depth_image,nan=self.MAX_DEPTH,posinf=self.MAX_DEPTH)
            cv2.imshow('Depth',cv2.applyColorMap((np.clip(d/self.MAX_DEPTH,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO))
        if self.display_tensorized:
            colors = np.array([[0,0,0],[255,180,80],[60,60,255],[60,255,60]],dtype=np.float32)
            image = colors[self.obstacle[:,:,1].astype(int)]
            depth = self.obstacle[:,:,0]
            closeness = np.clip((self.MAX_RELIABLE_DEPTH-depth)/
                                (self.MAX_RELIABLE_DEPTH-self.MIN_DEPTH),0,1)
            brightness = np.where(np.isfinite(depth),0.15+0.85*closeness,0)
            image = (image*brightness[:,:,None]).astype(np.uint8)
            if self._goal_cell is not None:
                image[self._goal_cell] = [60,255,60]
            panel = cv2.resize(image,(640,640),interpolation=cv2.INTER_NEAREST)
            if self._goal_cell is None and simulation is not None and simulation.obstacle_course is not None:
                position = State.get_position(simulation.state)
                rotation = State.get_rotation(simulation.state)
                camera_position = position + rotation @ np.array([0.5,0,-0.1])
                direction = rotation.T @ (simulation.obstacle_course.goal - camera_position)
                # Camera looks along body +X; +Y is right. Map heading to a
                # compass arrow: ahead=up, right=right, behind=down.
                arrow = np.array([direction[1], -direction[0]],dtype=float)
                length = np.linalg.norm(arrow)
                if length > 1e-6:
                    arrow /= length
                    tip = (np.array([320,320])+arrow*285).astype(int)
                    tail = (tip-arrow*35).astype(int)
                    cv2.arrowedLine(panel,tuple(tail),tuple(tip),(60,255,60),4,tipLength=.45)
            cv2.imshow('Depth + class: wall blue, obstacle red, goal green',panel)
        if self.display_raw or self.display_depth or self.display_tensorized:
            cv2.waitKey(1)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/f'{self.timestamp}.npz',grayscale=self.grayscale_image,depth=self.depth_image,obstacle=self.obstacle,timestamp=self.timestamp)

    def close(self):
        self._executor.shutdown(wait=True)
        if self.display_raw or self.display_depth or self.display_tensorized:
            cv2.destroyAllWindows()
