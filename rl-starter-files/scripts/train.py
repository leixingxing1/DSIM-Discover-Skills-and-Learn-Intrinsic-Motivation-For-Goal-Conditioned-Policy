import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys
import os
import shutil
import json
import tqdm

import utils
from model import ACModel
from model import VQVAEDiscriminator
from base.modules.generic import CategoricalWithoutReplacement
from utils.replay_memory import ReplayMemory

import numpy as np

def open_experiment(dataset):
    parser = argparse.ArgumentParser("Train VQ-VAE.")
    parser.add_argument('--config-path', type=str, default='config/vqvae_representation.json',help='Path to experiment config file (expecting a json)')
    parser.add_argument('--log-dir', type=str, default='logs',help='Parent directory that holds experiment log directories')
    parser.add_argument('--dur', type=int, default=50000, help='Number of training iterations')
    args = parser.parse_args()

    config_path = args.config_path
    assert os.path.isfile(config_path)
    config = json.load(open(config_path))

    exp_name = config_path.split('/')[-1][:-5]
    exp_dir = os.path.join(args.log_dir, exp_name)

    print('Experiment directory is: {}'.format(exp_dir), flush=True)

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
        shutil.copyfile(config_path, os.path.join(exp_dir, 'config.json'))

    state_size = dataset.ndim

    # Create VQ-VAE model and compute moments for the normalizer module
    model = VQVAEDiscriminator(state_size=state_size, **config['vae_args'])
    model.update_normalizer(dataset=dataset)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    return model, optimizer, dataset,state_size,config, args, exp_dir

def sample_skill(skill_n):
        skill_dist = CategoricalWithoutReplacement(skill_n)
        return skill_dist.sample(sample_shape=(1,)).view([])

# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

parser.add_argument("--use_entropy_reward", action="store_true", default=False)
parser.add_argument("--beta", type=float, default=0.00001)
parser.add_argument("--use_batch", action="store_true", default=False)

args = parser.parse_args()

args.mem = args.recurrence > 1

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    envs.append(utils.make_env(args.env, args.seed + 10000 * i))
txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model

acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

# Memory for store transitions
replay_memory = ReplayMemory(args.replay_size)

# Load algo

if args.algo == "a2c":
    algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss, 
                            use_entropy_reward=args.use_entropy_reward,replay_memory=replay_memory)
elif args.algo == "ppo":
    algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,replay_memory=replay_memory)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

algo.replay_buffer = np.zeros((10000, 64))
algo.idx = 0
algo.full = False

algo.beta = args.beta
algo.use_batch = args.use_batch

#Explore with a policy and self-update
while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    if args.use_entropy_reward:
        for i in range(len(exps.obs)):
            # store low-dimensional features from random encoder
            algo.replay_buffer[algo.idx] = algo.random_encoder(exps.obs[i].image.unsqueeze(0).transpose(1, 3).transpose(2, 3))[0,:,0,0].detach().cpu().numpy()
            algo.idx = (algo.idx + 1) % algo.replay_buffer.shape[0]
            algo.full = algo.full or algo.idx == 0
    
    #store transitions to repaly memory
    for i in range(len(exps.obs)):
        for j in range(args.num_frames_per_proc):
            mask = 0 if j == logs1["num_frames_per_episode"] else float(exps.mask[i][j])
            replay_memory.push(exps.obs[i][j],exps.action[i][j],exps.reward[i][j],mask)

    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

    # Save status

    if (args.save_interval > 0 and update % args.save_interval == 0):
        status = {"num_frames": num_frames, "update": update,
                  "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")

    #train vqvae for encode collect_transitions
    #Interpret the arguments. Create the model, optimizer and dataset. Fetch the config file.
    vae_model, optim, dataset, state_size,config, args, save_dir = open_experiment(replay_memory.out())

    # Training loop
    indices = list(range(dataset.size(0)))
    vae_loss_list = []
    vae_model.train()

    #Train VQ-VAE Model
    for iter_idx in tqdm(range(args.dur), desc="Training"):
        # Make batch
        batch_indices = np.random.choice(indices, size=config['batch_size'])
        batch = dict(next_state=dataset[batch_indices])

        # Forward + backward pass
        optim.zero_grad()
        vae_loss = vae_model(batch)
        vae_loss.backward()
        optim.step()

        # Log progress
        vae_loss_list.append(vae_loss.item())

    # Save model, config and losses
    vae_model.eval()
    vae_model_path = os.path.join(save_dir, "model.pth.tar")
    config_path = os.path.join(save_dir, "config.json")
    loss_path = os.path.join(save_dir, "loss.json")
    torch.save(vae_model.state_dict(), vae_model_path)
    with open(config_path, 'wt') as f:
        json.dump(config, f)
    with open(loss_path, 'wt') as f:
        json.dump(json.dumps(vae_loss_list), f)


