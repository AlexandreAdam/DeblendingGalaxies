from score_models import NCSNppLog
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from definitions import preprocessing_nonlinear_sde
from datetime import datetime
from tqdm import tqdm
from torch.nn.functional import avg_pool2d
import time
import json
import numpy as np
import torch
import os
from glob import glob
import re
import h5py
from torch_ema import ExponentialMovingAverage
from functorch import vjp

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG10 = np.log(10.)


def sliced_score_matching_loss(model, samples, t, lambda_t, n_cotangent_vectors=1,  noise_type="rademacher"):
    """
    Score matching loss with the Hutchinson trace estimator trick. See Theorem 1 of
    Hyvärinen (2005). Estimation of Non-Normalized Statistical Models by Score Matching,
    (https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf).

    We implement an unbiased estimator of this loss with reduced variance reported in
    Y. Song et al. (2019). A Scalable Approach to Density and Score Estimation
    (https://arxiv.org/abs/1905.07088).

    Inspired from the official implementation of Sliced Score Matching at https://github.com/ermongroup/sliced_score_matching
    We also implement the weighting scheme for NCSN (Song & Ermon 2019 https://arxiv.org/abs/1907.05600)
    """
    if noise_type not in ["gaussian", "rademacher"]:
        raise ValueError("noise_type has to be either 'gaussian' or 'rademacher'")
    B, *D = samples.shape
    # duplicate noisy samples across the number of particle for the Hutchinson trace estimator
    samples = torch.tile(samples, [n_cotangent_vectors, *[1]*len(D)])
    t = torch.tile(t, [n_cotangent_vectors])

    # sample cotangent vectors
    vectors = torch.randn_like(samples)
    if noise_type == 'rademacher':
        vectors = vectors.sign()
    score, vjp_func = vjp(lambda x: model(x, t), samples)
    trace_estimate = vectors * vjp_func(vectors)[0]
    loss = (lambda_t(samples, t) * (0.5 * torch.sum(score**2, dim=-1) + torch.sum(trace_estimate, dim=-1))).mean()
    return loss


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_h5, key, channels, device=DEVICE):
        self.filepath = path_to_h5
        self.device = device
        self.key = key
        self.channels = channels
        with h5py.File(self.filepath, "r") as hf:
            self.size = hf[self.key].shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        with h5py.File(self.filepath, "r") as hf:
            im = torch.tensor(hf[self.key][index, :, :, self.channels]).to(self.device)
            # put channels first for Conv2D score model
            return torch.permute(im, (2, 0, 1))


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    with open(args.model_parameters, "r") as f:
        hyperparameters = json.load(f)
    if args.model_architecture.lower() == "ncsnpplog":
        # Cannot use DataParallel for this script, because of functorch VJP
        model = NCSNppLog(**hyperparameters).to(DEVICE)
    else:
        raise ValueError

    hyperparameters["minimum_flu"] = args.minimum_flux
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
   
    def loss_fn(x):
        B, *D = x.shape
        broadcast = [-1, *[1] * len(D)]  # used to broadcast scalars to image shape
        z = torch.randn_like(x)
        t = torch.rand(B).to(DEVICE)
        x_log = torch.log(x + model.beta0 + model.beta1 * t.view(*broadcast) + z * model.sde.sigma(t).view(*broadcast))
        x_log_detrended = x_log - torch.log(model.beta0 + model.beta1 * t.view(*broadcast))
        # We pass x_log_detrended (model input) in the loss, but weight must be computed at detrended value (or x_log)
        lambda_t = lambda x, t: model.sde.sigma(t).view(*broadcast) ** 2 * torch.exp(-2 * (x + torch.log(model.beta0 + model.beta1 * t.view(*broadcast))))
        return sliced_score_matching_loss(
            model=model.score,
            samples=x_log_detrended,
            t=t,
            lambda_t=lambda_t,
            noise_type=args.hutchinson_noise_type,
            n_cotangent_vectors=args.n_cotangent_vectors
        )

    dataset = Dataset(args.dataset_path, args.dataset_key, args.dataset_channels, device=DEVICE)
    dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ==== Take care of where to write logs and stuff =================================================================
    if args.model_id.lower() != "none":
        logname = args.model_id
    elif args.logname is not None:
        logname = args.logname
    else:
        logname = args.logname_prefixe + "_" + datetime.now().strftime("%y%m%d%H%M%S")
    if args.logdir.lower() != "none":
        logdir = os.path.join(args.logdir, logname)
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = SummaryWriter()
    # ===== Make sure directory and checkpoint manager are created to save model ===================================
    if args.model_dir.lower() != "none":
        checkpoints_dir = os.path.join(args.model_dir, logname)
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            with open(os.path.join(checkpoints_dir, "script_params.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
            with open(os.path.join(checkpoints_dir, "model_hparams.json"), "w") as f:
                json.dump(hyperparameters, f, indent=4)
        save_checkpoint = True
        # ======= Load model if model_id is provided ===============================================================
        paths = glob(os.path.join(checkpoints_dir, "checkpoint*.pt"))
        opt_paths = glob(os.path.join(checkpoints_dir, "optimizer*.pt"))
        checkpoints = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
        scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]
        if args.model_id.lower() != "none" and checkpoints != []:
            if args.model_checkpoint is not None:
                model.load_state_dict(torch.load(paths[checkpoints == args.model_checkpoint], map_location=DEVICE))
                optimizer.load_state_dict(torch.load(opt_paths[checkpoints == args.model_checkpoint], map_location=DEVICE))
                print(f"Loaded checkpoint {args.model_checkpoint} of {args.model_id}")
                lastest_checkpoint = args.model_checkpoint
            else:
                model.load_state_dict(torch.load(paths[np.argmax(checkpoints)], map_location=DEVICE))
                optimizer.load_state_dict(torch.load(opt_paths[np.argmax(checkpoints)], map_location=DEVICE))
                print(f"Loaded checkpoint {max(checkpoints)} of {args.model_id}")
                lastest_checkpoint = max(checkpoints)
        else:
            lastest_checkpoint = 0
    else:
        save_checkpoint = False

    # ====== Training loop ============================================================================================
    history = {  # recorded at the end of an epoch only
        "cost": [],
        "learning_rate": [],
        "time_per_step": [],
        "step": [],
        "wall_time": []
    }
    best_loss = np.inf
    patience = args.patience
    step = 0
    global_start = time.time()
    estimated_time_for_epoch = 0
    out_of_time = False
    for epoch in tqdm(range(args.epochs)):
        if (time.time() - global_start) > args.max_time * 3600 - estimated_time_for_epoch:
            break
        epoch_start = time.time()
        time_per_step_epoch_mean = 0
        cost = 0
        for batch, x in enumerate(dataset):
            start = time.time()
            if args.downsample > 0:
                x = avg_pool2d(x, kernel_size=2**args.downsample, stride=2**args.downsample)
            x = preprocessing_nonlinear_sde(x, minimum_flux=args.minimum_flux)
            # optimize network
            optimizer.zero_grad()
            loss = loss_fn(x)
            loss.backward()
            # warmup learning rate
            if step <= args.warmup:
                for g in optimizer.param_groups:
                    g['lr'] = args.learning_rate * np.minimum(step / args.warmup, 1.0)
            # gradient clipping
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            optimizer.step()
            ema.update()
            # ========== Summary and logs ==============================================================================
            _time = time.time() - start
            time_per_step_epoch_mean += _time
            cost += float(loss)
            step += 1
            if args.epoch_iterations is not None:
                if batch >= args.epoch_iterations:
                    break

        time_per_step_epoch_mean /= len(dataset)
        cost /= len(dataset)
        writer.add_scalar("MSE", cost, step)
        print(f"epoch {epoch} | cost {cost:.3e} "
              f"| time per step {time_per_step_epoch_mean:.2e} s")
        history["cost"].append(cost)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        history["time_per_step"].append(time_per_step_epoch_mean)
        history["step"].append(step)
        history["wall_time"].append(time.time() - global_start)

        writer.flush()
        if np.isnan(cost):
            print("Training broke the Universe")
            break
        if cost < (1 - args.tolerance) * best_loss:
            best_loss = cost
            patience = args.patience
        else:
            patience -= 1
        if (time.time() - global_start) > args.max_time * 3600:
            out_of_time = True
        if save_checkpoint:
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1 or out_of_time:
                lastest_checkpoint += 1
                with open(os.path.join(checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    f.write(f"{lastest_checkpoint} {cost}\n")
                with ema.average_parameters():  # save EMA parameters
                    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"checkpoint_{cost:.4e}_{lastest_checkpoint:03d}.pt"))
                torch.save(optimizer.state_dict(), os.path.join(checkpoints_dir, f"optimizer_{cost:.4e}_{lastest_checkpoint:03d}.pt"))
                checkpoints.append(lastest_checkpoint)
                scores.append(cost)
                print("Saved checkpoint for step {}".format(step))
            if len(checkpoints) >= args.models_to_keep + 1:
                # remove the worst score checkpoint (excluding the one we just saved)
                index_to_delete = np.argmax(scores[:-1])
                os.remove(os.path.join(checkpoints_dir, f"checkpoint_{scores[index_to_delete]:.4e}_{checkpoints[index_to_delete]:03d}.pt"))
                os.remove(os.path.join(checkpoints_dir, f"optimizer_{scores[index_to_delete]:.4e}_{checkpoints[index_to_delete]:03d}.pt"))
                del scores[index_to_delete]
                del checkpoints[index_to_delete]
        if patience == 0:
            print("Reached patience")
            break
        if out_of_time:
            break
        if epoch > 0:  # First epoch is always very slow and not a good estimate of an epoch time.
            estimated_time_for_epoch = time.time() - epoch_start
    print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
    return history, best_loss


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_architecture",     required=True,                  help="Either 'ddpm' or 'ncsnpp'")
    parser.add_argument("--dataset_path",       required=True)
    parser.add_argument("--dataset_key",        required=True)
    parser.add_argument("--dataset_channels",   nargs="+", required=True, type=int, help="Channels of the dataset to use. ")
    parser.add_argument("--model_id",           default="none",                     help="The script will search in provided model_dir argument for model_id and load checkpoint if it exists.")
    parser.add_argument("--model_checkpoint",   default=None,       type=int,       help="Index of the checkpoint to load.")
    parser.add_argument("--minimum_flux",		default=1e-3,		type=float)

    # Model parameters
    parser.add_argument("--model_parameters",               required=True,                  help="Path to model parameter json file.")

    # Optimization params
    parser.add_argument("--epochs",                         default=10,     type=int,       help="Number of epochs for training.")
    parser.add_argument("--epoch_iterations",               default=None,   type=int,       help="Number of iterations to do in an epoch")
    parser.add_argument("--learning_rate",                  default=1e-4,   type=float,     help="Initial learning rate.")
    parser.add_argument("--patience",                       default=np.inf, type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",                      default=0,      type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--max_time",                       default=np.inf, type=float,     help="Time allowed for the training, in hours.")
    parser.add_argument("--ema_decay",                      default=0.999,  type=float)
    parser.add_argument("--warmup",                         default=0,      type=int,       help="Warmup the learning up to the target learning rate over this amount of iterations")
    parser.add_argument("--clip",                           default=0.,     type=float,     help="Gradient clipping")
    parser.add_argument("--hutchinson_noise_type",          default="rademacher",           help="Noise used to compute the trace in SSM with Hutchinson's estimator")
    parser.add_argument("--n_cotangent_vectors",            default=1,      type=int,       help="Number of samples to use for the trace estimator")

    # Training set params
    parser.add_argument("--batch_size",             default=1,      type=int,       help="Number of images in a batch.")
    parser.add_argument("--downsample",             default=0,   type=int,       help="Average pooling, if zero, no downsampling, if 1, then downsample by a factor of 2, etc. ")
    # logs
    parser.add_argument("--logdir",             default="None",                     help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname",            default=None,                       help="Overwrite name of the log with this argument")
    parser.add_argument("--logname_prefixe",    default="score_model",                  help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",          default="None",                     help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",        default=10, type=int,               help="Save a checkpoint of the models each {%} epoch.")
    parser.add_argument("--models_to_keep",     default=2,  type=int,               help="Only keep 3 best model, on top of the last checkpoint")

    # Reproducibility params
    parser.add_argument("--seed",                   default=None,   type=int,       help="Random seed for numpy and tensorflow.")

    args = parser.parse_args()
    main(args)
