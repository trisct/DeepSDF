#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import torch.nn.functional as F
import signal
import sys
import os
import logging
import math
import json
import time

import deep_sdf
import deep_sdf.workspace as ws


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, continue_from, batch_split):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = deep_sdf.data_normal.SDFSamplesWithNormals(
        data_source, train_split, num_samp_per_scene
    )
    # print('[HERE: In train_deep_sdf.main_function] sdf_dataset len =', len(sdf_dataset))

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )
    # print('[HERE: In train_deep_sdf.main_function] sdf_loader len =', len(sdf_loader))

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    # builing losses
    L1_criterion = torch.nn.L1Loss(reduction="sum")
    mse_criterion = torch.nn.MSELoss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    ### Initializing variables
    pred_surf_grad = None

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()
        logging.info("epoch {}...".format(epoch))
        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        for sdf_data, surf_points, surf_normals, indices in sdf_loader:
            #print('[HERE: In train_deep_sdf.LOOPsdf_loader] indices =', indices)

            # Process the input data

            sdf_data = sdf_data.reshape(-1, 4) # [N_pts, 4]
            # print(f'[In train_grad_superv] sdf_data.device = {sdf_data.device}')
            surf_points = surf_points.reshape(-1, 3) # [N_pts, 1]
            surf_normals = surf_normals.reshape(-1, 3) # [N_pts, 1]
            # print(f'[In train_grad_superv] sdf_data.shape = {sdf_data.shape}')
            # print(f'[In train_grad_superv] surf_points.shape = {surf_points.shape}')
            # print(f'[In train_grad_superv] surf_normals.shape = {surf_normals.shape}')

            num_sdf_samples = sdf_data.shape[0]
            num_surf_samples = surf_points.shape[0]

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3:4]

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            ### chunking data (what is torch.chunk: splitting into chunks)
            xyz = torch.chunk(xyz, batch_split)
            sdf_gt = torch.chunk(sdf_gt, batch_split)
            surf_points = torch.chunk(surf_points, batch_split)
            surf_normals = torch.chunk(surf_normals, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            batch_loss_total = 0.0
            batch_loss_sdf = 0.0
            batch_loss_normal = 0.0
            batch_loss_surf = 0.0
            
            optimizer_all.zero_grad()

            for i in range(batch_split):
                #print('[HERE: In train_deep_sdf.LOOPbatch_split] i/batch_split = %d/%d'%(i, batch_split))

                batch_vecs = lat_vecs(indices[i])
                input = torch.cat([batch_vecs, xyz[i]], dim=1).cuda()

                # NN optimization
                # print(f'[In train_grad_superv] Network forward is at line 500')
                # print(f'[In train_grad_superv] input.device = {input.device}')
                pred_sdf = decoder(input)
                                
                # print(f'[In train_grad_superv] input.shape = {input.shape}')
                # print(f'[In train_grad_superv] pred_sdf.shape = {pred_sdf.shape}')
                # print(f'[In train_grad_superv] sdf_gt[i].shape = {sdf_gt[i].shape}')
                # print(f'[In train_grad_superv] num_sdf_samples = {num_sdf_samples}')

                
                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                chunk_loss_sdf = L1_criterion(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples
                # print(f'[In train_grad_superv] pred_sdf.device = {pred_sdf.device}')
                # print(f'[In train_grad_superv] sdf_gt[i].device = {sdf_gt[i].device}')

                if specs["UseNormalLoss"]:
                    # Another forward from surface samples
                    input_surf = torch.cat([batch_vecs, surf_points[i]], dim=1).cuda()
                    pred_surf = decoder(input_surf)
                    #print(f'[In train_grad_superv] decoder.device = {decoder.device}')
                    #print(f'[In train_grad_superv] input_surf.device = {input_surf.device}')

                    if pred_surf_grad is None or pred_surf_grad.shape != pred_surf.shape:
                        pred_surf_grad = torch.ones_like(pred_surf, device=pred_surf.device)
                    
                    grad_surf = torch.autograd.grad(outputs=pred_surf, inputs=input_surf, grad_outputs=pred_surf_grad, create_graph=True, retain_graph=True)
                    grad_surf = grad_surf[0][:, -3:]
                    #print(f'[In train_grad_superv] grad_surf.shape = {grad_surf.shape}')
                    #print(f'[In train_grad_superv] grad_surf.device = {grad_surf.device}')
                    #print(f'[In train_grad_superv] surf_normals.device = {surf_normals[i].device}')

                    grad_surf = F.normalize(grad_surf, dim=1)
                    grad_surf_loss = mse_criterion(grad_surf, surf_normals[i].cuda()) / num_surf_samples * 0.03
                    
                    if specs["UseSurfLoss"]:
                        zero_surf_loss = (pred_surf ** 2).mean() * 5e2

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples

                    loss = chunk_loss_sdf + reg_loss.cuda()
                else:
                    loss = chunk_loss_sdf

                if specs["UseNormalLoss"]:
                    loss = loss + grad_surf_loss
                    if specs["UseSurfLoss"]:
                        loss = loss + zero_surf_loss

                loss.backward()

                batch_loss_total += loss.item()
                batch_loss_sdf += chunk_loss_sdf.item()
                if specs["UseNormalLoss"]:
                    batch_loss_normal += grad_surf_loss.item()
                    if specs["UseSurfLoss"]:
                        batch_loss_surf += zero_surf_loss.item()
                

            logging.debug(f"loss = {batch_loss_total:.8f}, loss_sdf = {batch_loss_sdf:.8f}, loss_normal = {batch_loss_normal:.8f}, loss_surf = {batch_loss_surf:.8f}")
            logging.info(f"loss = {batch_loss_total:.8f}, loss_sdf = {batch_loss_sdf:.8f}, loss_normal = {batch_loss_normal:.8f}, loss_surf = {batch_loss_surf:.8f}")

            loss_log.append(batch_loss_total)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        end = time.time()
        

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    arg_parser.add_argument("--gpu", type=str, default="0")

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
