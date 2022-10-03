import os
import random
import time
import argparse
from distutils.util import strtobool

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import gym
import gym.wrappers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed used for random number generators")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", 
                        const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled False, GPU will not be used if available")
    
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="gym ID of environment to train in")
    parser.add_argument("--total-timesteps", type=int, default=200_000, 
                        help="num of agent-environment interaction to train for")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="learning rate for the actor and critic")
    parser.add_argument("--num-envs", type=int, default=4, help="number of envs to run in parallel")
    parser.add_argument("--num-steps", type=int, default=8, 
                        help="number of agent-environment steps before each policy update")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles learning rate annealing for policy and value network")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage estimation")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="lambda for GAE")
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches for PPO")
    parser.add_argument("--update-epochs", type=int, default=4, help="the number of epochs to update policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantage normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, 
                        help="clipping coefficient for the clipped surrogate objective")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient for the entropy term in ppo loss")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value loss term in ppo loss")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="max norm for gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the kl divergence threshold for early stopping")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

# Key Implementation Detail: Orthogonal initialization for last layer
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def mlp(sizes, activation, output_activation=nn.Identity, 
        last_layer_init_std=None, 
        last_layer_init_bias=0.0):
    """Utility function to create a multilayer perceptron"""
    layers = []
    
    # for n sizes, there will be n-1 layers. A layer connect two sizes
    # Note: A size here refers to the input, hidden or output activations in the MLP network 
    for layer in range(len(sizes)-1):
        # create linear layer
        linear_layer = nn.Linear(sizes[layer], sizes[layer+1])
        # apply orthogonal initialization for the weights of the last linear layer if requested
        if last_layer_init_std:
            linear_layer = linear_layer if layer < (len(sizes)-2) else \
                            layer_init(linear_layer, last_layer_init_std, last_layer_init_bias)
            
        # use the output_activation when creating the last layer which is at index n-2
        act = activation if layer < (len(sizes)-2) else output_activation
        
        # create a layer connecting this size and the next size, and add its activation
        layers += [linear_layer, act()]
    return nn.Sequential(*layers)

class Agent(nn.Module):
    def __init__(self, envs, hidden_sizes, activation):
        """Create an actor-critic model"""
        super().__init__()
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
            "Only environments with discrete actions asre supported"

        obs_dim = envs.single_observation_space.shape[0]
        act_dim = envs.single_action_space.n
        self.actor_net = mlp([obs_dim]+list(hidden_sizes)+[act_dim], activation, last_layer_init_std=0.01)
        self.critic_net = mlp([obs_dim]+list(hidden_sizes)+[1], activation, last_layer_init_std=1.0)

    def get_value(self, obs):
        return self.critic_net(obs)

    def get_action_and_value(self, obs, act=None):
        logits = self.actor_net(obs)
        probs = Categorical(logits=logits)
        if act is None:
            act = probs.sample()
        log_prob = probs.log_prob(act)
        return act, log_prob, probs.entropy(), self.get_value(obs)

def make_env(env_id, seed, idx, capture_video=False, run_name=None):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"video/{run_name}")
        env.seed(seed)
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"

    # Initialize Tensorboard and log hyperparams
    writer = SummaryWriter(f"tensorboard/{run_name}")
    writer.add_text("hyperparameters", 
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # Set Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Choose Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create Env and Agent
    # Key Implementation Detail: Vectorized Environment
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i) for i in range(args.num_envs)])
    agent = Agent(envs, [64, 64], nn.Tanh).to(device)
    # Key Implementation Detail: Adam's eps
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Buffers
    obs = torch.zeros((args.num_steps, args.num_envs)+envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs)+envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    
    # Set tracking variables 
    global_step = 0
    start_time = time.time()
    
    # Get initial observation and done state
    next_obs = torch.as_tensor(envs.reset(), device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    
    # Run agent in the env and update after every `num_steps` steps 
    num_updates = args.total_timesteps // args.batch_size
    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0)/num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # collect experiences for num_steps steps
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, rew, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.as_tensor(rew, device=device).flatten()
            next_obs, next_done = torch.as_tensor(next_obs, device=device), torch.Tensor(done, device=device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step: {global_step}, episodic_return: {item['episode']['r']}")
                    writer.add_scalar("metrics/episodic_return", item['episode']['r'], global_step)
                    writer.add_scalar("metrics/episodic_length", item['episode']['l'], global_step)
                    break
        
        # Bootstrap the values if not done and compute advantages
        with torch.no_grad():
            # Only bootstrap if not done, else next_value==0
            # TODO: Walk through the return and both advantages computation
            next_value = agent.get_value(next_obs).reshape(1, -1)*(1 - next_done)
            # Generalized advantage estimation
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
        
        # flatten the batch data
        b_obs = obs.reshape((-1,)+envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,)+envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            
            # take a minibatch per time
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                # ratio = (new_prob/old_prob), log(ratio) = log(new_prob/old_prob) = log(new_prob) - log(old_prob)
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Useful info
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # 1e-8 added at denominator for numerical stability. E.g. std can be possibly zero
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # Clipped Surrogate Objective Function
                # instead of the min as in the paper, we compute the max of the negative
                pg_loss1 = -mb_advantages*ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                new_value = new_value.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_value - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_inds])**2).mean()

                entropy_loss = entropy.mean()
                
                optimizer.zero_grad()
                loss = pg_loss - (args.ent_coef*entropy_loss) + (v_loss*args.vf_coef)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
            
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log to Tensorboard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()