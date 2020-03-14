from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm

import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update,hard_update


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            alpha=0.2,
            automatic_entropy_tuning=True,
            policy_lr=3e-4,
            qf_lr=3e-4,
            lr=3e-4,
            context_lr=3e-4,
            kl_lambda=1.,
            policy_mean_reg_weight=3e-4,
            policy_std_reg_weight=3e-4,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=0.005,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).to("cuda").item()  # torch.prod(input) : 返回所有元素的乘积
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        # self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.critic, self.target_critic = nets[1:]#nets[agent{policy},critic{q1,q2}]
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(),lr=qf_lr)
        self.policy_optimizer = Adam(self.agent.policy.parameters(), lr=policy_lr)
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.critic, self.target_critic]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))                                                                     #14

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]                                          #14
            self._take_step(indices, context)


            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    # def _update_target_network(self):
    #     ptu.soft_update_from_to(self.critic, self.target_critic, self.soft_target_tau)

    def _take_step(self, indices, context):
        # print("alpha:",self.alpha)
        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        _, policy_mean, policy_log_std, _ = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.agent.policy.sample(torch.cat([next_obs, task_z], 1))#策略网络前向传播，获得下一个状态下的动作以及下一个状态下采取的动作
            target_qf1 , target_qf2 = self.target_critic(next_obs, next_state_action, task_z.detach())#计算目标Q值Q(st+1,at+1)获得目标值
            min_target_qf = torch.min(target_qf1,target_qf2) - self.alpha * next_state_log_pi#计算小的Q值，并加上熵
            next_q_value = rewards_flat + (1. - terms_flat) * self.discount * (min_target_qf)#Bellman update
        #利用Q和targetQ计算出Q的Loss
        qf1, qf2 = self.critic(obs, actions, task_z)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        #计算policy的loss
        pi, log_pi, _ = self.agent.policy.sample(torch.cat([obs, task_z],1))#a,logπ(s)
        qf1_pi, qf2_pi = self.critic(obs, pi, task_z)#Q(s,a)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)#min Q(s,a)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()#E[αlogπ(at|st)-Q(st,at)]
        #不明觉厉
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)
        self.context_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.critic_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()  # E[-αlogπ(at|st)-αH]
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optimizer.step()#log alpha更新了
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            self.alpha = 1

        # target update
        soft_update(self.target_critic, self.critic, self.soft_target_tau)

        # policy update
        # n.b. policy update includes dQ/da



        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            if self.automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = np.mean(ptu.get_numpy(self.alpha))
                self.eval_statistics['Alpha Loss'] = np.mean(ptu.get_numpy(alpha_loss))
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(qf1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(qf2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            critic=self.critic.state_dict(),
            policy=self.agent.policy.state_dict(),
            target_critic=self.target_critic.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
        )
        return snapshot
