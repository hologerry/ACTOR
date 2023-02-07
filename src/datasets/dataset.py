import random

import numpy as np
import torch

import src.utils.rotation_conversions as geometry

from src.datasets.tools import parse_info_name
from src.utils.misc import to_torch
from src.utils.tensors import collate


POSE_REPS = ["xyz", "rotvec", "rotmat", "rotquat", "rot6d"]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_frames=1,
        sampling="conseq",
        sampling_step=1,
        split="train",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        max_len=-1,
        min_len=-1,
        num_seq_max=-1,
        **kwargs,
    ):
        self.num_frames = num_frames
        self.sampling = sampling
        self.sampling_step = sampling_step
        self.split = split
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.max_len = max_len
        self.min_len = min_len
        self.num_seq_max = num_seq_max

        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"{self.split} is not a valid split")

        super().__init__()

        # to remove shuffling
        self._original_train = None
        self._original_test = None

        # TBD
        # self._actions
        # self._train/self._test
        # self._num_frames_in_video[data_index]
        # self._action_to_label[action]
        # self._label_to_action[label]
        # self._load_pose(data_index, frame_ix)
        # self._actions[ind] # => carefully changed here
        # self._action_classes[action]

    def action_to_label(self, action):
        return self._action_to_label[action]

    def label_to_action(self, label):
        import numbers

        if isinstance(label, numbers.Integral):
            return self._label_to_action[label]
        else:  # if it is one hot vector
            label = np.argmax(label)
            return self._label_to_action[label]

    def get_pose_data(self, data_index, frame_ix):
        pose = self._load(data_index, frame_ix)
        label = self.get_label(data_index)
        return pose, label

    def get_label(self, ind):
        action = self.get_action(ind)
        return self.action_to_label(action)

    def parse_action(self, path, return_int=True):
        info = parse_info_name(path)["A"]
        if return_int:
            return int(info)
        else:
            return info

    def get_action(self, ind):
        return self._actions[ind]

    def action_to_action_name(self, action):
        return self._action_classes[action]

    def label_to_action_name(self, label):
        action = self.label_to_action(label)
        return self.action_to_action_name(action)

    def __getitem__(self, index):
        if self.split == "train":
            data_index = self._train[index]
        else:
            data_index = self._test[index]

        inp, target = self._get_item_data_index(data_index)
        return inp, target

    def _load(self, ind, frame_ix):
        pose_rep = self.pose_rep
        if pose_rep == "xyz" or self.translation:
            if getattr(self, "_load_joints3D", None) is not None:
                # Locate the root joint of initial pose at origin
                joints3D = self._load_joints3D(ind, frame_ix)
                joints3D = joints3D - joints3D[0, 0, :]
                ret = to_torch(joints3D)
                if self.translation:
                    ret_tr = ret[:, 0, :]
            else:
                if pose_rep == "xyz":
                    raise ValueError("This representation is not possible.")
                if getattr(self, "_load_translation") is None:
                    raise ValueError("Can't extract translations.")
                ret_tr = self._load_translation(ind, frame_ix)
                ret_tr = to_torch(ret_tr - ret_tr[0])

        if pose_rep != "xyz":
            if getattr(self, "_load_rotvec", None) is None:
                raise ValueError("This representation is not possible.")
            else:
                pose = self._load_rotvec(ind, frame_ix)
                if not self.glob:
                    pose = pose[:, 1:, :]
                pose = to_torch(pose)
                if pose_rep == "rotvec":
                    ret = pose
                elif pose_rep == "rotmat":
                    ret = geometry.axis_angle_to_matrix(pose).view(*pose.shape[:2], 9)
                elif pose_rep == "rotquat":
                    ret = geometry.axis_angle_to_quaternion(pose)
                elif pose_rep == "rot6d":
                    ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
        if pose_rep != "xyz" and self.translation:
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
            padded_tr[:, :3] = ret_tr
            ret = torch.cat((ret, padded_tr[:, None]), 1)
        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float()

    def _get_item_data_index(self, data_index):
        n_frames = self._num_frames_in_video[data_index]

        if self.num_frames == -1 and (self.max_len == -1 or n_frames <= self.max_len):
            frame_ix = np.arange(n_frames)
        else:
            if self.num_frames == -2:
                if self.min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if self.max_len != -1:
                    max_frame = min(n_frames, self.max_len)
                else:
                    max_frame = n_frames

                num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else self.max_len
            # sampling goal: input: ----------- 11 n_frames
            #                       o--o--o--o- 4  n_inputs
            #
            # step number is computed like that: [(11-1)/(4-1)] = 3
            #                   [---][---][---][-
            # So step = 3, and we take 0 to step*n_inputs+1 with steps
            #                   [o--][o--][o--][o-]
            # then we can randomly shift the vector
            #                   -[o--][o--][o--]o
            # If there are too much frames required
            if num_frames > n_frames:
                fair = False  # True
                if fair:
                    # distills redundancy everywhere
                    choices = np.random.choice(range(n_frames), num_frames, replace=True)
                    frame_ix = sorted(choices)
                else:
                    # adding the last frame until done
                    n_to_add = max(0, num_frames - n_frames)
                    last_frame = n_frames - 1
                    padding = last_frame * np.ones(n_to_add, dtype=int)
                    frame_ix = np.concatenate((np.arange(0, n_frames), padding))

            elif self.sampling in ["conseq", "random_conseq"]:
                step_max = (n_frames - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= n_frames:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                last_one = step * (num_frames - 1)
                shift_max = n_frames - last_one - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, last_one + 1, step)

            elif self.sampling == "random":
                choices = np.random.choice(range(n_frames), num_frames, replace=False)
                frame_ix = sorted(choices)

            else:
                raise ValueError("Sampling not recognized.")

        inp, target = self.get_pose_data(data_index, frame_ix)
        return inp, target

    def get_label_sample(self, label, n=1, return_labels=False, return_index=False):
        if self.split == "train":
            index = self._train
        else:
            index = self._test

        action = self.label_to_action(label)
        choices = np.argwhere(np.array(self._actions)[index] == action).squeeze(1)

        if n == 1:
            data_index = index[np.random.choice(choices)]
            x, y = self._get_item_data_index(data_index)
            assert label == y
            y = label
        else:
            data_index = np.random.choice(choices, n)
            x = np.stack([self._get_item_data_index(index[di])[0] for di in data_index])
            y = label * np.ones(n, dtype=int)
        if return_labels:
            if return_index:
                return x, y, data_index
            return x, y
        else:
            if return_index:
                return x, data_index
            return x

    def get_label_sample_batch(self, labels):
        samples = [self.get_label_sample(label, n=1, return_labels=True, return_index=False) for label in labels]
        batch = collate(samples)
        x = batch["x"]
        mask = batch["mask"]
        lengths = mask.sum(1)
        return x, mask, lengths

    def get_mean_length_label(self, label):
        if self.num_frames != -1:
            return self.num_frames

        if self.split == "train":
            index = self._train
        else:
            index = self._test

        action = self.label_to_action(label)
        choices = np.argwhere(self._actions[index] == action).squeeze(1)
        lengths = self._num_frames_in_video[np.array(index)[choices]]

        if self.max_len == -1:
            return np.mean(lengths)
        else:
            # make the lengths less than max_len
            lengths[lengths > self.max_len] = self.max_len
        return np.mean(lengths)

    def get_stats(self):
        if self.split == "train":
            index = self._train
        else:
            index = self._test

        num_frames = self._num_frames_in_video[index]
        all_means = np.array([self.get_mean_length_label(x) for x in range(self.num_classes)])

        stats = {
            "name": self.data_name,
            "number of classes": self.num_classes,
            "number of sequences": len(self),
            "duration: min": int(num_frames.min()),
            "duration: max": int(num_frames.max()),
            "duration: mean": int(num_frames.mean()),
            "duration mean/action: min": int(all_means.min()),
            "duration mean/action: max": int(all_means.max()),
            "duration mean/action: mean": int(all_means.mean()),
        }
        return stats

    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf

            num_seq_max = inf

        if self.split == "train":
            return min(len(self._train), num_seq_max)
        else:
            return min(len(self._test), num_seq_max)

    def __repr__(self):
        return f"{self.data_name} dataset: ({len(self)}, _, ..)"

    def update_parameters(self, parameters):
        # this method is for first loading, self[0] -> dataset[0]: (input, target)
        # input is the pose data, target is the label
        # input: (num_joints, num_features, num_frames), (25, 6, 60) for humanact12
        self.njoints, self.nfeats, _ = self[0][0].shape
        parameters["num_classes"] = self.num_classes
        parameters["nfeats"] = self.nfeats
        parameters["njoints"] = self.njoints

    def shuffle(self):
        if self.split == "train":
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == "train":
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test
