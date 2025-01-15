import os.path as osp
import glob
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from PIL import Image

class CityScapesRGBSegmDataset(data.Dataset):
    """
    A custom dataset class for CityScapes segmentation dataset.
    Args:

        data_path (str): The path to the data folder.
        sequence_length (int): The length of each sequence.
        shape (tuple): The desired shape of the frames.
        downsample_factor (int): The factor by which the frames should be downsampled.
        subset (str, optional): The subset of the dataset to use (train, val, test). Defaults to "train".

    Attributes:
        data_path (str): The path to the data folder.
        sequence_length (int): The length of each sequence.
        shape (tuple): The desired shape of the frames.
        downsample_factor (int): The factor by which the frames should be downsampled.
        subset (str): The subset of the dataset being used.
        num_frames (int): The total number of frames in the dataset.
        sequences (list): The list of unique sequence names in the dataset.
        
    Methods:
        __len__(): Returns the number of sequences in the dataset.
        __getitem__(idx): Retrieves the frames and their corresponding file paths for a given index.
    """

    def __init__(self, data_path, args, sequence_length, shape, downsample_factor, subset="train", eval_mode=False, eval_midterm=False):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.shape = shape
        self.downsample_factor = downsample_factor
        self.subset = subset 
        self.eval_mode = eval_mode
        self.eval_midterm = eval_midterm
        self.modality = args.modality
        self.num_frames = len(glob.glob(osp.join(self.data_path, subset, '**',"*.png")))
        self.sequences = set() # Each sequence name consists of city name and sequence id i.e. aachen_00001, hanover_00013
        self.augmentations = {
                "random_crop" : args.random_crop,
                "random_horizontal_flip" : args.random_horizontal_flip,
                "random_time_flip" : args.random_time_flip,
                "timestep_augm" : args.timestep_augm,
                "no_timestep_augm" : args.no_timestep_augm}
        for city_folder in glob.glob(osp.join(self.data_path, subset, '*')):
            city_name = osp.basename(city_folder)
            frames_in_city = glob.glob(osp.join(city_folder, '*'))
            city_seqs = set([f"{city_name}_{osp.basename(frame).split('_')[1]}" for frame in frames_in_city])
            if len(city_seqs)<10: # Note that in some cities very few, though very long sequences were recorded
                for seq in city_seqs:
                    sub_seqs = sorted(glob.glob(osp.join(self.data_path, subset, city_name, seq+'*.png')))
                    sub_seq_startframe_ids = [osp.basename(sub_seqs[i])[:-16] for i in range(len(sub_seqs)) if i%30==0]
                    self.sequences.update(sub_seq_startframe_ids)
            else:
                self.sequences.update(city_seqs)
                continue
        self.sequences = sorted(list(self.sequences))
        # self.sequences = [self.sequences[i] for i in [0, 3, 272, 326]] #[0, 3, 272, 326] # For visualization purposes

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        Returns:
            int: The number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves the frames and their corresponding file paths for a given index.

        Args:
            idx (int): The index of the sequence.

        Returns:
            tuple: A tuple containing the frames and their corresponding file paths.
        """
        sequence_name = self.sequences[idx]
        splits = sequence_name.split("_")
        if len(splits)==2: # Sample from Short sequences
            frames_filepaths = sorted(glob.glob(osp.join(self.data_path, self.subset, splits[0], sequence_name+'*.png'))) # Sequence of 30 frames    
        elif len(splits)==3: # Sample from Long sequences
            frames_filepaths = [osp.join(self.data_path, self.subset,splits[0], splits[0]+"_"+splits[1]+"_"+'{:06d}'.format(int(splits[2])+i)+'_leftImg8bit.png') for i in range(30)]
        # Load, process and Stack to Tensor
        if self.eval_mode:
            if self.modality in ["segmaps","depth"]:
                frames, gt_frame  = process_evalmode(frames_filepaths,self.modality, self.shape, self.subset, self.downsample_factor, self.sequence_length, self.eval_midterm)
                return frames, gt_frame # B,T,C,H,W
            elif self.modality == "segmaps_depth":
                frames, seg_gt, depth_gt = process_evalmode(frames_filepaths,self.modality, self.shape, self.subset, self.downsample_factor, self.sequence_length, self.eval_midterm)
                return frames, seg_gt, depth_gt
                # frames, seg_gt, depth_gt, seg_gt_path = process_evalmode(frames_filepaths,self.modality, self.shape, self.subset, self.downsample_factor, self.sequence_length, self.eval_midterm)
                # return frames, seg_gt, depth_gt, seg_gt_path
            else:
                frames = process_evalmode(frames_filepaths,self.modality, self.shape, self.subset, self.downsample_factor, self.sequence_length, self.eval_midterm)
                return frames
        else:
            frames = process_trainmode(frames_filepaths,self.modality, self.shape, self.subset, self.downsample_factor, self.augmentations, self.sequence_length)
            return frames

class CityScapesDepthDataset(data.Dataset):
    """
    A custom dataset class for CityScapes depth dataset.
    Args:

        data_path (str): The path to the data folder.
        sequence_length (int): The length of each sequence.
        shape (tuple): The desired shape of the frames.
        downsample_factor (int): The factor by which the frames should be downsampled.
        subset (str, optional): The subset of the dataset to use (train, val, test). Defaults to "train".

    Attributes:
        data_path (str): The path to the data folder.
        sequence_length (int): The length of each sequence.
        shape (tuple): The desired shape of the frames.
        downsample_factor (int): The factor by which the frames should be downsampled.
        subset (str): The subset of the dataset being used.
        num_frames (int): The total number of frames in the dataset.
        sequences (list): The list of unique sequence names in the dataset.
        
    Methods:
        __len__(): Returns the number of sequences in the dataset.
        __getitem__(idx): Retrieves the frames and their corresponding file paths for a given index.
    """

    def __init__(self, data_path, args, sequence_length, shape, downsample_factor, subset="train", eval_mode=False, eval_midterm=False):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.shape = shape
        self.downsample_factor = downsample_factor
        self.subset = subset 
        self.eval_mode = eval_mode
        self.eval_midterm = eval_midterm
        self.modality = args.modality
        self.num_frames = len(glob.glob(osp.join(self.data_path, subset, '**',"*.png")))
        self.sequences = set() # Each sequence name consists of city name and sequence id i.e. aachen_00001, hanover_00013
        self.augmentations = {  "random_crop" : args.random_crop,
                                "random_horizontal_flip" : args.random_horizontal_flip,
                                "random_time_flip" : args.random_time_flip,
                                "timestep_augm" : args.timestep_augm,
                                "no_timestep_augm" : args.no_timestep_augm}
        for city_folder in glob.glob(osp.join(self.data_path, subset, '*')):
            city_name = osp.basename(city_folder)
            frames_in_city = glob.glob(osp.join(city_folder, '*'))
            city_seqs = set([f"{city_name}_{osp.basename(frame).split('_')[1]}" for frame in frames_in_city])
            if len(city_seqs)<10: # Note that in some cities very few, though very long sequences were recorded
                for seq in city_seqs:
                    sub_seqs = sorted(glob.glob(osp.join(self.data_path, subset, city_name, seq+'*.png')))
                    sub_seq_startframe_ids = [osp.basename(sub_seqs[i])[:-22] for i in range(len(sub_seqs)) if i%30==0]
                    self.sequences.update(sub_seq_startframe_ids)
            else:
                self.sequences.update(city_seqs)
                continue
        self.sequences = sorted(list(self.sequences))


    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        Returns:
            int: The number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves the frames and their corresponding file paths for a given index.

        Args:
            idx (int): The index of the sequence.

        Returns:
            tuple: A tuple containing the frames and their corresponding file paths.
        """
        sequence_name = self.sequences[idx]
        splits = sequence_name.split("_")
        
        if len(splits)==2: # Sample from Short sequences
            frames_filepaths = sorted(glob.glob(osp.join(self.data_path, self.subset, splits[0], sequence_name+'*.png'))) # Sequence of 30 frames    
        elif len(splits)==3: # Sample from Long sequences
            frames_filepaths = [osp.join(self.data_path, self.subset,splits[0], splits[0]+"_"+splits[1]+"_"+'{:06d}'.format(int(splits[2])+i)+'_leftImg8bit_depth.png') for i in range(30)]
        # Load, process and Stack to Tensor
        if self.eval_mode:
            frames, gt_frame  = process_evalmode(frames_filepaths,self.modality, self.shape, self.subset, self.downsample_factor, self.sequence_length, self.eval_midterm)
            return frames, gt_frame
        else:
            frames = process_trainmode(frames_filepaths, self.modality, self.shape, self.subset, self.downsample_factor, self.augmentations, self.sequence_length)
            return frames
       
def process_trainmode(frames_path, modality, shape, subset, downsample_factor, augmentations, sequence_length=4, num_frames_skip=0):
    if subset=="val":
        num_frames_skip = 2 
        step = num_frames_skip + 1  
        start_idx = 20 - step*sequence_length + num_frames_skip
    else:
        if augmentations["no_timestep_augm"] is True:
            num_frames_skip = 2
        elif augmentations["timestep_augm"] is not None:
            num_frames_skip = np.random.choice(list(range(1,len(augmentations["timestep_augm"])+1),p=augmentations["timestep_augm"]))
        else:
            num_frames_skip = np.random.randint(1,4) # [1,3] with equal probabilities 4 is excluded
        step = num_frames_skip + 1 # [2,4] with equal probabilities or [2,N] with non-equal probabilities 
        start_idx = np.random.randint(0, len(frames_path) - step*sequence_length + num_frames_skip + 1) # Comment FOR NOISE AUGMENTATION
    sequence_frames_path = frames_path[start_idx : start_idx + step*sequence_length : step] # Comment FOR NOISE AUGMENTATION
    if augmentations["random_time_flip"] == True and subset=="train":
        sequence_frames_path = sequence_frames_path[::-1] if np.random.rand()>0.5 else sequence_frames_path
    if modality=="segmaps" or modality == 'depth':
         sequence_frames = [Image.open(frame) for frame in sequence_frames_path]
    elif modality=="segmaps_depth":
        sequence_frames = [Image.open(frame) for frame in sequence_frames_path]
        sequence_frames += [Image.open(frame.replace("leftImg8bit_sequence_segmaps_ids","leftImg8bit_sequence_depthv2").replace("leftImg8bit.png","leftImg8bit_depth.png")) for frame in sequence_frames_path]
    else:
        assert False, "Invalid modality"
    # Random Crop
    H, W = shape
    f = downsample_factor
    if augmentations["random_crop"] == True and subset=="train":
        s_f = np.random.rand()/2 + 0.5 # [0.5, 1]
        size = (int(H*s_f),int(W*s_f))
        i, j, h, w = T.RandomCrop(size).get_params(sequence_frames[0], output_size=size)
        sequence_frames = [TF.crop(frame,i,j,h,w) for frame in sequence_frames]
    if augmentations["random_horizontal_flip"] == True and subset=="train":
        sequence_frames = [TF.hflip(frame) for frame in sequence_frames] if np.random.rand()>0.5 else sequence_frames
    if modality=="segmaps":
        transform_segmaps = T.Compose([T.Resize((H//f,W//f),interpolation=T.InterpolationMode.NEAREST),T.PILToTensor()])
        sequence_tensors = [transform_segmaps(frame) for frame in sequence_frames]
    elif modality=="depth":
        transforms_depth = T.Compose([T.Resize((H//f,W//f)),T.PILToTensor()])
        sequence_tensors = [transforms_depth(frame) for frame in sequence_frames]
    elif modality=="segmaps_depth":
        transform_segmaps = T.Compose([T.Resize((H//f,W//f),interpolation=T.InterpolationMode.NEAREST),T.PILToTensor()])
        transform_depth = T.Compose([T.Resize((H//f,W//f)),T.PILToTensor()])
        sequence_tensors = [transform_segmaps(frame) if i<len(sequence_frames)/2 else transform_depth(frame) for i, frame in enumerate(sequence_frames)]
    if modality == "segmaps_depth":
        stacked_first_part = torch.stack(sequence_tensors[:sequence_length], dim=0)
        stacked_second_part = torch.stack(sequence_tensors[sequence_length:], dim=0)
        sequence_tensor = torch.cat((stacked_first_part, stacked_second_part), dim=1)
    else:
        sequence_tensor = torch.stack(sequence_tensors, dim=0)
    return sequence_tensor

def process_evalmode(frames_path,modality, shape, subset, downsample_factor, sequence_length=4, eval_midterm=False):
    num_frames_skip = 2 
    step = num_frames_skip + 1  
    if eval_midterm and sequence_length<7:
        start_idx = 20 - step*sequence_length + num_frames_skip - 6
    else:
        start_idx = 20 - step*sequence_length + num_frames_skip
    sequence_frames_path = frames_path[start_idx : start_idx + step*sequence_length : step]
    H, W = shape
    f = downsample_factor
    if modality=="segmaps":
        sequence_frames = [Image.open(frame) for frame in sequence_frames_path]
        transform_segmaps = T.Compose([T.Resize((H//f,W//f),interpolation=T.InterpolationMode.NEAREST),T.PILToTensor()])
        sequence_tensors = [transform_segmaps(frame) for frame in sequence_frames]
        sequence_tensor = torch.stack(sequence_tensors, dim=0)
    elif modality=="depth":
        sequence_frames = [Image.open(frame) for frame in sequence_frames_path]
        sequence_tensors = [T.Compose([T.Resize((H//f,W//f)),T.PILToTensor()])(frame) for frame in sequence_frames]
        sequence_tensor = torch.stack(sequence_tensors, dim=0)
    elif modality=="segmaps_depth":
        sequence_frames = [Image.open(frame) for frame in sequence_frames_path]
        sequence_frames += [Image.open(frame.replace("leftImg8bit_sequence_segmaps_ids","leftImg8bit_sequence_depthv2").replace("leftImg8bit.png","leftImg8bit_depth.png")) for frame in sequence_frames_path]
        transform_segmaps = T.Compose([T.Resize((H//f,W//f),interpolation=T.InterpolationMode.NEAREST),T.PILToTensor()])
        transform_depth = T.Compose([T.Resize((H//f,W//f)),T.PILToTensor()])
        sequence_tensors = [transform_segmaps(frame) if i<len(sequence_frames)/2 else transform_depth(frame) for i, frame in enumerate(sequence_frames)]
        stacked_first_part = torch.stack(sequence_tensors[:sequence_length], dim=0)
        stacked_second_part = torch.stack(sequence_tensors[sequence_length:], dim=0)
        sequence_tensor = torch.cat((stacked_first_part, stacked_second_part), dim=1)
    if modality=="segmaps" :
        gt_path = frames_path[19].replace("leftImg8bit_sequence_segmaps_ids","gtFine").replace("leftImg8bit","gtFine_labelTrainIds")
        gt = np.array(Image.open(gt_path),dtype=np.uint8)
        return sequence_tensor, gt
    elif modality =="depth":
        gt_path = frames_path[19]
        gt = np.array(Image.open(gt_path))
        return sequence_tensor, gt
    elif modality=="segmaps_depth":
        seg_gt_path = frames_path[19].replace("leftImg8bit_sequence_segmaps_ids","gtFine").replace("leftImg8bit","gtFine_labelTrainIds")
        seg_gt = np.array(Image.open(seg_gt_path),dtype=np.uint8)
        depth_gt_path = frames_path[19].replace("leftImg8bit_sequence_segmaps_ids","leftImg8bit_sequence_depthv2").replace("leftImg8bit.png","leftImg8bit_depth.png")
        depth_gt = np.array(Image.open(depth_gt_path),dtype=np.uint8)
        return sequence_tensor, seg_gt, depth_gt
    else:
        return sequence_tensor

class CS_VideoData(pl.LightningDataModule):
    """
    LightningDataModule for loading CityScapes video data.

    Args:
        arguments: An object containing the required arguments for data loading.
        subset (str): The subset of the data to load. Default is "train".

    Attributes:
        data_path (str): The path to the data folder.
        subset (str): The subset of the data being loaded.
        sequence_length (int): The length of the video sequence.
        batch_size (int): The batch size for data loading.
        shape (tuple): The shape of the video frames.
        downsample_factor (int): The factor by which to downsample the frames.
    """

    def __init__(self, arguments, subset="train", batch_size=8):
        super().__init__()
        self.data_path = arguments.data_path
        self.subset = subset  # ["train","val","test"]
        self.sequence_length = arguments.sequence_length
        self.batch_size = batch_size
        self.shape = arguments.shape
        self.arguments = arguments
        self.eval_midterm = arguments.eval_midterm
        self.downsample_factor = arguments.downsample_factor
        self.num_workers = arguments.num_workers
        self.num_workers_val = arguments.num_workers if arguments.num_workers_val is None else arguments.num_workers_val
        self.eval_mode = arguments.eval_mode
        self.use_val_to_train = arguments.use_val_to_train
        self.use_train_to_val = arguments.use_train_to_val
        self.modality = arguments.modality


    def _dataset(self, subset, eval_mode):
        """
        Private method to create and return the CityScapesDataset object.

        Args:
            subset (str): The subset of the data to load.

        Returns:
            CityScapesDataset: The dataset object.
        """
        if self.modality in ["segmaps","dinov2","segmaps_depth","dinov2_segmaps_depth"]:
            dataset = CityScapesRGBSegmDataset(self.data_path, self.arguments, self.sequence_length, self.shape, self.downsample_factor, subset, eval_mode, self.eval_midterm)
        elif self.modality=="depth":
            dataset = CityScapesDepthDataset(self.data_path, self.arguments, self.sequence_length, self.shape, self.downsample_factor, subset, eval_mode, self.eval_midterm)
        return dataset

    def _dataloader(self, subset, shuffle=True, drop_last=False, eval_mode=False):
        """
        Private method to create and return the DataLoader object.

        Args:
            subset (str): The subset of the data to load.
            shuffle (bool): Whether to shuffle the data. Default is True.

        Returns:
            DataLoader: The dataloader object.
        """
        dataset = self._dataset(subset, eval_mode)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last)
        return dataloader

    def train_dataloader(self):
        """
        Method to return the dataloader for the training subset.

        Returns:
            DataLoader: The dataloader object for training data.
        """
        train_subset = "val" if self.use_val_to_train else "train"
        return self._dataloader(subset=train_subset, drop_last=True, eval_mode=False)

    def val_dataloader(self):
        """
        Method to return the dataloader for the validation subset.

        Returns:
            DataLoader: The dataloader object for validation data.
        """
        val_subset = "train" if self.use_train_to_val else "val"
        dataset = self._dataset(val_subset, self.eval_mode)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers_val, pin_memory=True, shuffle=False, drop_last=False)
        return dataloader

    def test_dataloader(self):
        """
        Method to return the dataloader for the test subset.

        Returns:
            DataLoader: The dataloader object for test data.
        """
        return self._dataloader(subset="test")