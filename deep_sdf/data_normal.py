#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import trimesh

import deep_sdf.workspace as ws


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames_ply = list(glob.iglob(shape_dir + "/**/*.ply")) + list(
        glob.iglob(shape_dir + "/*.ply")
    )
    mesh_filenames_obj = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    mesh_filenames = mesh_filenames_obj + mesh_filenames_ply
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamplesWithNormals(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        print_filename=False,
        num_files=1000000,
    ):
        print(f'[In data_normal] Printing dataset attributes...')
        self.subsample = subsample
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        print(f'[In data_normal] data_source = {data_source}') # dataset_processed
        #print(f'[In data_normal] npyfiles = {self.npyfiles}') # instance file paths, see below
        # npyfiles, a list of file paths, each looks like
        # 'shapenetv1/shapenet_planes/2c932237239e4d22181acc12f598af7.npz'

        print(f'[In data_normal] split = [SEE COMMENT]') # a dict, see below.
        # split info:
        # top_key = dataset_name
        # sec_key = list of class names
        # 3rd key = list of instance names
        print(f'[In data_normal] subsample = {subsample}') # sample points per object per batch
        
        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        filename_surface_points = os.path.join(
            self.data_source, ws.normal_samples_subdir, self.npyfiles[idx][:-4] + '.obj'
        )
        filename_surface_normals = os.path.join(
            self.data_source, ws.normal_samples_subdir, self.npyfiles[idx][:-4] + '_normal.obj'
        )
        
        surf_points_mesh = trimesh.load(filename_surface_points)
        surf_points = torch.from_numpy(np.array(surf_points_mesh.vertices).copy()).float()
        del surf_points_mesh

        surf_normals_mesh = trimesh.load(filename_surface_normals)
        surf_normals = torch.from_numpy(np.array(surf_normals_mesh.vertices).copy()).float()
        del surf_normals_mesh

        # subsample
        sdf_samples = unpack_sdf_samples(filename, self.subsample)

        random_index = (torch.rand(self.subsample) * surf_points.shape[0]).long()

        surf_point_samples = torch.index_select(surf_points, 0, random_index)
        surf_normal_samples = torch.index_select(surf_normals, 0, random_index)

        #print(f'[In data_normal] sdf_samples.shape = {sdf_samples.shape}, surf_points.shape = {surf_point_samples.shape}, surf_normals.shape = {surf_normal_samples.shape}')
        
        return sdf_samples, surf_point_samples, surf_normal_samples, idx
