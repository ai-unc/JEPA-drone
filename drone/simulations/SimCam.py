"""Camera outputs: uint8 grayscale, metric depth, and 32x32x2 depth/class tensor.
Class IDs match skyjepa.models.encoders.CLASSES. Empty depth is NaN.
"""
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from projectairsim.utils import unpack_image
from drone.Subsystems.data.data import State

class SimCamera:
    FREQUENCY = 100
    IMAGE_RESOLUTION = (640, 480)
    DEPTH_RESOLUTION = (32, 32)
    GRID_RESOLUTION = (20, 15)
    MAX_DEPTH = 20.
    MAX_RELIABLE_DEPTH = 10.
    MIN_DEPTH = .5
    CLASS_NAMES = {0:'unknown', 1:'wall', 2:'floor', 3:'ceiling',
                   4:'door/goal', 5:'window', 6:'furniture', 7:'obstacle'}
    DEPTH_INVALID = 0
    DEPTH_LEVELS = 255
    STEP_NS = 10_000_000
    PALETTE_RGB = np.array([[55,181,57],[153,108,6],[112,105,191],[89,121,72]],dtype=np.uint8)
    SCENE_TO_CLASS = {1: 1, 2: 7, 3: 4}

    def __init__(self):
        self.grayscale_image = self.depth_image = self.obstacle = None
        self.display_raw = self.display_depth = self.display_tensorized = False
        self.tensor_enabled = False
        self.log_vision = False
        self.logged_steps = []
        self.logged_depth = []
        self.logged_classes = []
        self.tensorized_history = []
        self.timestamp = None
        self._pending = {}
        self._ready = None
        self._ready_queue = deque()
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sim-camera')
        self._future = None
        self._goal_cell = None

    def attach(self, simulation):
        topics = simulation.drone.sensors['Vision']
        self._required = ({0} if self.display_raw else set()) | ({1} if self.display_depth or self.tensor_enabled or self.display_tensorized or self.log_vision else set()) | ({3} if self.tensor_enabled or self.display_tensorized or self.log_vision else set())
        for kind, topic in ((0,'scene_camera'),(1,'depth_planar_camera'),(3,'segmentation_camera')):
            if kind in self._required:
                simulation.client.subscribe(topics[topic], lambda _, msg, k=kind: self._receive(k,msg))

    def _receive(self, kind, msg):
        stamp = msg['time_stamp']
        with self._lock:
            frame = self._pending.setdefault(stamp,{})
            frame[kind] = msg
            if self._required.issubset(frame) and (self._ready is None or stamp>self._ready[0]):
                if self.log_vision:
                    if not frame.get('_queued'):
                        self._ready_queue.append((stamp, frame.copy()))
                        frame['_queued'] = True
                else:
                    self._ready = (stamp,frame.copy())
            retained = 100 if self.log_vision else 2
            for old in sorted(self._pending)[:-retained]:
                del self._pending[old]

    @classmethod
    def cell_to_depth_and_object(cls, cell):
        depth, labels = cell
        y, x = depth.shape[0]//2, depth.shape[1]//2
        value, label = depth[y,x], labels[y,x]
        if not (np.isfinite(value) and cls.MIN_DEPTH<=value<=cls.MAX_RELIABLE_DEPTH and label!=0):
            return np.array([np.nan,0],dtype=np.float32)
        return np.array([value,label],dtype=np.float32)

    @classmethod
    def tensorize(cls, image):
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
        if self.display_depth or self.tensor_enabled or self.display_tensorized or self.log_vision:
            if frame[1]['encoding']!='16FC1':
                raise ValueError('Expected metric 16FC1 depth; ambiguous depth units rejected')
            depth_image = unpack_image(frame[1]).astype(np.float32)
        if self.tensor_enabled or self.display_tensorized or self.log_vision:
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
            for scene_id, class_id in self.SCENE_TO_CLASS.items():
                labels[np.all(rgb==self.PALETTE_RGB[scene_id],axis=-1)] = class_id
            sampled_depth = depth_image[np.ix_(ys,xs)]
            valid = np.isfinite(sampled_depth)&(sampled_depth>=self.MIN_DEPTH)&(sampled_depth<=self.MAX_RELIABLE_DEPTH)&(labels!=0)
            obstacle = np.stack((np.where(valid,sampled_depth,np.nan),np.where(valid,labels,0)),axis=-1).astype(np.float32)
        return grayscale, depth_image, obstacle, goal_cell

    def update(self, simulation=None):
        if self._future is not None and self._future.done():
            self.grayscale_image, self.depth_image, self.obstacle, self._goal_cell = self._future.result()
            self._future = None
            if self.log_vision:
                self._record_tensor(self._processing_stamp)
            self._display(simulation)
        if self._future is not None:
            return False
        with self._lock:
            if self.log_vision:
                ready = self._ready_queue.popleft() if self._ready_queue else None
            else:
                ready, self._ready = self._ready, None
        if ready is None or ready[0]==self.timestamp:
            return False
        self.timestamp, frame = ready
        self._processing_stamp = self.timestamp
        self._future = self._executor.submit(self._process_frame, frame)
        return True

    def _record_tensor(self, stamp):
        depth = self.obstacle[:, :, 0]
        labels = self.obstacle[:, :, 1].astype(np.uint8)
        valid = np.isfinite(depth) & (labels != 0)
        encoded = np.zeros(depth.shape, dtype=np.uint8)
        encoded[valid] = 1 + np.rint((np.clip(depth[valid], self.MIN_DEPTH, self.MAX_RELIABLE_DEPTH)
            - self.MIN_DEPTH) / (self.MAX_RELIABLE_DEPTH - self.MIN_DEPTH) * 254).astype(np.uint8)
        self.logged_steps.append(int(round(stamp / self.STEP_NS)))
        self.logged_depth.append(encoded)
        self.logged_classes.append(labels)

    def save_vision(self, directory):
        """Write frames indexed by 100 Hz step.

        Depth 0 means invalid; valid metres d map to
        1 + round((d - MIN_DEPTH) / (MAX_RELIABLE_DEPTH - MIN_DEPTH) * 254).
        Classification keeps the class IDs in CLASS_NAMES.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        steps = np.asarray(self.logged_steps, dtype=np.int64)
        shape = (0, *self.DEPTH_RESOLUTION)
        depth = np.stack(self.logged_depth) if self.logged_depth else np.empty(shape, dtype=np.uint8)
        classes = np.stack(self.logged_classes) if self.logged_classes else np.empty(shape, dtype=np.uint8)
        np.savez_compressed(directory / 'depth.npz', frames=depth, steps=steps,
                            min_depth=self.MIN_DEPTH, max_depth=self.MAX_RELIABLE_DEPTH)
        np.savez_compressed(directory / 'classification.npz', frames=classes, steps=steps)

    @staticmethod
    def load_tensor(directory):
        """Recreate the float32 depth/class tensor from a flight folder."""
        directory = Path(directory)
        with np.load(directory / 'depth.npz') as depth_file, np.load(directory / 'classification.npz') as class_file:
            if not np.array_equal(depth_file['steps'], class_file['steps']):
                raise ValueError('Depth and classification step indices do not match')
            encoded = depth_file['frames']
            labels = class_file['frames']
            if encoded.shape != labels.shape:
                raise ValueError('Depth and classification shapes do not match')
            minimum, maximum = float(depth_file['min_depth']), float(depth_file['max_depth'])
            depth = minimum + (encoded.astype(np.float32) - 1) * (maximum - minimum) / 254
            depth[encoded == 0] = np.nan
            return np.stack((depth, labels.astype(np.float32)), axis=-1)

    @classmethod
    def tensor_to_bgr(cls, tensor):
        """Render only the measured depth/class cells, without display overlays."""
        if tensor.shape[-1] != 2:
            raise ValueError('Tensor must have depth and class channels')
        colors = np.zeros((max(cls.CLASS_NAMES) + 1, 3), dtype=np.float32)
        colors[1] = [255,180,80]
        colors[7] = [60,60,255]
        colors[4] = [60,255,60]
        labels = tensor[...,1].astype(np.intp)
        if np.any(labels < 0) or np.any(labels >= len(colors)):
            raise ValueError('Tensor contains an unknown class ID')
        depth = tensor[...,0]
        closeness = np.clip((cls.MAX_RELIABLE_DEPTH - depth) /
                            (cls.MAX_RELIABLE_DEPTH - cls.MIN_DEPTH), 0, 1)
        brightness = np.where(np.isfinite(depth), 0.15 + 0.85 * closeness, 0)
        return (colors[labels] * brightness[...,None]).astype(np.uint8)

    def _display(self, simulation=None):
        if self.display_raw:
            cv2.imshow('Grayscale',self.grayscale_image)
        if self.display_depth:
            d = np.nan_to_num(self.depth_image,nan=self.MAX_DEPTH,posinf=self.MAX_DEPTH)
            cv2.imshow('Depth',cv2.applyColorMap((np.clip(d/self.MAX_DEPTH,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO))
        if self.display_tensorized:
            image = self.tensor_to_bgr(self.obstacle)
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

    def close(self):
        if self.log_vision:
            while self._future is not None or self._ready_queue:
                if self._future is not None:
                    self._future.result()
                self.update()
        self._executor.shutdown(wait=True)
        if self.display_raw or self.display_depth or self.display_tensorized:
            cv2.destroyAllWindows()
