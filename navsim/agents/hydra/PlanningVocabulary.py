from navsim.common.dataclasses import Scene
import os
from pathlib import Path
import pickle
from sklearn.cluster import KMeans
import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class PlanningVocabulary: 

    def __init__(self, vocab_size = 100000, k = 1000, 
                 trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
                 ):

        self._k = k
        self._vocab_size = vocab_size
        self.vocab = [] # (vocab_size)
        self.poses = [] # (vocab_size)
        self.anchors = [] # 1000 (K)
        self.anchor_poses = [] # 1000 (K)
        self._trajectory_sampling = trajectory_sampling

        ###
    def getSceneLoader(self, split = "trainval", filter = "navtrain"):
        
        SPLIT = split
        FILTER = filter

        hydra.initialize(config_path="../../planning/script/config/common/train_test_split/scene_filter")
        cfg = hydra.compose(config_name=FILTER)
        scene_filter: SceneFilter = instantiate(cfg)
        openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

        self.scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{SPLIT}", # data_path
        openscene_data_root / "synthetic_scenes/synthetic_sensor", # sensor_blobs_path
        openscene_data_root / f"sensor_blobs/{SPLIT}", # navsim_blobs_path
        openscene_data_root / "synthetic_scenes/scene_pickles", # synthetic_scenes_path
        scene_filter,
        sensor_config=SensorConfig.build_hydra_sensors(),
        )
        print("Scene loader has been built.")

    def createTrajectories(self):
        self.sampleTrajectories()
        print("Planning Vocab created")
        if len(self.vocab) == 0:
            print("Vocab is empty.")
        self.selectKMeans()
        self.writeTrajectories()

    def writeTrajectories(self):
        print("Write out trajectories")
        with open("trajectories.pkl", "wb") as f:
            pickle.dump(self.anchors, f)
    
    def readTrajectories(self):
       with open("/root/workdir/navsim/navsim/navsim/agents/hydra/trajectories.pkl", "rb") as f:
            loaded_anchors = pickle.load(f) 
            self.anchors = loaded_anchors
            self.anchor_poses = np.array([traj.poses for traj in self.anchors])
            print("Read in vocab")
            if len(self.anchors) == 0:
                print("Vocab is empty!")

    def sampleTrajectories(self):

        self.getSceneLoader()
        print("Start Sampling")
        print(len(self.scene_loader.tokens))
        tokens = np.random.choice(self.scene_loader.tokens, size=self._vocab_size, replace=False)

        def load_trajectory(token):
            print("X")
            scene = self.scene_loader.get_scene_from_token(token)
            return scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)

        with ThreadPoolExecutor() as executor:
            self.vocab = list(executor.map(load_trajectory, tokens))

    def selectKMeans(self):
        
        self.poses = np.array([traj.poses for traj in self.vocab])

        #print(self.poses.shape)
        reshaped_trajectories = self.poses.reshape(self.poses.shape[0], -1)  # (1000, 30)

        kmeans = KMeans(n_clusters=self._k, random_state=42, n_init=10)
        _labels = kmeans.fit_predict(reshaped_trajectories)

        centroids = kmeans.cluster_centers_
        distances = cdist(centroids, reshaped_trajectories)
        representative_indices = np.argmin(distances, axis=1)

        self.anchor_poses = self.poses[representative_indices]
        self.anchors = np.array(self.vocab)[representative_indices]
        print("Selected representative trajectory indices:", representative_indices)
        print("Representative trajectories shape:", self.anchors.shape)  # (K, 8, 3)
        print("Representative trajectories:", self.anchors)

    def visualizeTrajectories(self):
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))

        for i in range(self._k):
            trajectory = self.anchor_poses[i]  # (8, 3)
            #print(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Representative Trajectories Projected onto the XY Plane')
        # Save the plot to a file
        plt.savefig('representative_trajectories.png', dpi=300)  # Save as PNG with high resolution

        # Show the plot
        plt.show()

                
#pv = PlanningVocabulary()
#pv.createTrajectories()
#pv.readTrajectories()
#pv.visualizeTrajectories()
