import os
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import set_random_seed
from poolenv import PoolEnv
from agents import BasicAgentPro, BasicAgent
from agents.new_agent import my_analyze_shot_for_reward
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def save_checkpoint(ckpt_path, actor, critic, opt, scaler, steps, config, last_metrics):
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'optimizer': opt.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'steps': steps,
        'config': config,
        'last_metrics': last_metrics,
        'rng': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }, ckpt_path)

def load_checkpoint(ckpt_path, actor, critic, opt, scaler, device):
    data = torch.load(ckpt_path, map_location=device)
    if 'actor' in data and data['actor'] is not None:
        actor.load_state_dict(data['actor'])
    if 'critic' in data and data['critic'] is not None:
        critic.load_state_dict(data['critic'])
    if 'optimizer' in data and data['optimizer'] is not None:
        opt.load_state_dict(data['optimizer'])
    if scaler is not None and data.get('scaler') is not None:
        scaler.load_state_dict(data['scaler'])
    steps = int(data.get('steps', 0))
    last_metrics = data.get('last_metrics', {})
    rng = data.get('rng', None)
    try:
        if rng is not None:
            rs = rng.get('python', None)
            if rs is not None:
                random.setstate(rs)
            nrs = rng.get('numpy', None)
            if nrs is not None:
                np.random.set_state(nrs)
            trs = rng.get('torch_cpu', None)
            if trs is not None:
                torch.set_rng_state(trs)
            crs = rng.get('torch_cuda', None)
            if crs is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(crs)
    except Exception:
        pass
    return steps, last_metrics

def find_latest_checkpoint(dir_path):
    if not os.path.isdir(dir_path):
        return None
    files = [f for f in os.listdir(dir_path) if f.startswith("new_agent_step_") and f.endswith(".pt")]
    if not files:
        return None
    def _step(f):
        try:
            return int(f.split("_")[-1].split(".")[0])
        except Exception:
            return -1
    files.sort(key=_step, reverse=True)
    latest = files[0]
    return os.path.join(dir_path, latest)

class Actor(nn.Module):
    def __init__(self, in_dim=66, hidden=256, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, out_dim)
        self.log_std = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        std = self.log_std.exp()
        return mu, std

class Critic(nn.Module):
    def __init__(self, in_dim=66, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def encode_state(balls, my_targets, table):
    ids = ["cue"] + [str(i) for i in range(1, 16)]
    w = float(table.w)
    l = float(table.l)
    sx = []
    for bid in ids:
        pos = balls[bid].state.rvw[0]
        x = float(pos[0]) / l
        y = float(pos[1]) / w
        pocketed = 1.0 if balls[bid].state.s == 4 else 0.0
        sx.extend([x, y, pocketed])
    target_flags = []
    target_set = set(my_targets)
    for bid in ids:
        target_flags.append(1.0 if bid in target_set else 0.0)
    sx.extend(target_flags)
    sx.extend([l, w])
    return np.array(sx, dtype=np.float32)

def scale_action(z):
    v = 0.5 + (z[0].item() + 1.0) * 0.5 * (8.0 - 0.5)
    phi = (z[1].item() + 1.0) * 0.5 * 360.0
    theta = (z[2].item() + 1.0) * 0.5 * 30.0
    a = -0.5 + (z[3].item() + 1.0) * 0.5
    b = -0.5 + (z[4].item() + 1.0) * 0.5
    return {
        'V0': float(np.clip(v, 0.5, 8.0)),
        'phi': float(phi % 360.0),
        'theta': float(np.clip(theta, 0.0, 90.0)),
        'a': float(np.clip(a, -0.5, 0.5)),
        'b': float(np.clip(b, -0.5, 0.5)),
    }

def select_action(actor, x):
    mu, std = actor(x)
    dist = Normal(mu, std)
    a = dist.rsample()
    z = torch.tanh(a)
    logp = dist.log_prob(a).sum(-1)
    return z, logp, a.detach()

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (0.0 if dones[t] else next_value) - values[t]
        gae = delta + gamma * lam * (0.0 if dones[t] else gae)
        adv.append(gae)
        next_value = values[t]
    adv.reverse()
    returns = [a + v for a, v in zip(adv, values)]
    return np.array(adv, dtype=np.float32), np.array(returns, dtype=np.float32)

def compute_drill_reward(step_info, last_state_before, player_targets):
    own = step_info.get('ME_INTO_POCKET', []) or []
    enemy = step_info.get('ENEMY_INTO_POCKET', []) or []
    scratch = bool(step_info.get('WHITE_BALL_INTO_POCKET', False))
    eight = bool(step_info.get('BLACK_BALL_INTO_POCKET', False))
    remaining_own_before = [bid for bid in player_targets if last_state_before[bid].state.s != 4]
    illegal8 = eight and (len(remaining_own_before) > 0)
    foul_first = bool(step_info.get('FOUL_FIRST_HIT', False))
    foul_norail = bool(step_info.get('NO_POCKET_NO_RAIL', False))
    foul_nohit = bool(step_info.get('NO_HIT', False))
    foul = foul_first or foul_norail or scratch or illegal8 or foul_nohit
    P_own = 50.0
    R_8_legal = 100.0
    N_enemy = 30.0
    C_survive = 10.0
    F_base = 120.0
    F_first = 40.0
    F_norail = 40.0
    F_scratch = 80.0
    F_illegal8 = 300.0
    if foul:
        add = 0.0
        if foul_first:
            add += F_first
        if foul_norail:
            add += F_norail
        if scratch:
            add += F_scratch
        if illegal8:
            add += F_illegal8
        r = 0.0
        r -= (F_base + add)
        r -= N_enemy * float(len(enemy))
        r -= C_survive
        return r
    else:
        r = P_own * float(len(own))
        r -= N_enemy * float(len(enemy))
        if eight and (len(remaining_own_before) == 0):
            r += R_8_legal
        r -= C_survive
        return r

def train(total_steps=10000000, horizon=256, batch_size=128, init_lr=3e-4, min_lr=1e-5, clip_eps=0.15, gamma=0.995, lam=0.95, vf_coef=0.5, init_ent=0.01, final_ent=0.0, epochs=5, log_interval=4096, use_amp=True, use_compile=True, fast_mode=False, ckpt_interval=50000, resume_path=None, auto_resume=True, solo=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(enable=True, seed=42)
    env = PoolEnv()
    actor = Actor().to(device)
    critic = Critic().to(device)
    if use_compile and hasattr(torch, "compile"):
        actor = torch.compile(actor)
        critic = torch.compile(critic)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=init_lr)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")
    ckpt_dir = os.path.join(os.path.dirname(__file__), "ckpts_refined_rl_1")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "new_agent.pt")
    final_full_path = os.path.join(ckpt_dir, "new_agent_final.pt")
    pbar = tqdm(total=total_steps, desc="train", ncols=100)
    obs_buf = []
    act_buf = []
    logp_buf = []
    a_pre_buf = []
    val_buf = []
    rew_buf = []
    done_buf = []
    steps = 0
    last_metrics = {"rew": 0.0, "pl": 0.0, "vf": 0.0, "ent": 0.0, "ratio": 1.0}
    config = {
        'total_steps': total_steps, 'horizon': horizon, 'batch_size': batch_size,
        'init_lr': init_lr, 'min_lr': min_lr, 'clip_eps': clip_eps, 'gamma': gamma, 'lam': lam,
        'vf_coef': vf_coef, 'init_ent': init_ent, 'final_ent': final_ent, 'epochs': epochs,
        'log_interval': log_interval, 'use_amp': use_amp, 'use_compile': use_compile,
        'fast_mode': fast_mode, 'ckpt_interval': ckpt_interval
    }
    resume_file = resume_path if resume_path is not None else (find_latest_checkpoint(ckpt_dir) if auto_resume else None)
    if resume_file is not None:
        loaded_steps, loaded_metrics = load_checkpoint(resume_file, actor, critic, opt, scaler, device)
        steps = loaded_steps
        last_metrics = loaded_metrics or last_metrics
        if steps > 0:
            pbar.update(min(steps, total_steps))
    while steps < total_steps:
        if solo:
            opp_tag = "SOLO"
        elif fast_mode:
            if steps < total_steps * 0.5:
                opponent = BasicAgent()
                opp_tag = "BA"
            elif steps < total_steps * 0.8:
                opponent = BasicAgent() if random.random() < 0.8 else BasicAgentPro()
                opp_tag = "BA" if isinstance(opponent, BasicAgent) else "BAP"
            else:
                opponent = BasicAgentPro()
                opp_tag = "BAP"
        else:
            if steps < 100000:
                opponent = BasicAgent()
                opp_tag = "BA"
            elif steps < 300000:
                opponent = BasicAgent() if random.random() < 0.7 else BasicAgentPro()
                opp_tag = "BA" if isinstance(opponent, BasicAgent) else "BAP"
            else:
                opponent = BasicAgentPro()
                opp_tag = "BAP"
        env.reset(target_ball=random.choice(['solid', 'stripe']))
        while True:
            if solo:
                env.curr_player = 0
                player = 'A'
            else:
                player = env.get_curr_player()
            balls, my_targets, table = env.get_observation(player)
            if player == 'A':
                x_np = encode_state(balls, my_targets, table)
                x = torch.from_numpy(x_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    v = critic(x).cpu().item()
                    z, logp, a_pre = select_action(actor, x)
                action = scale_action(z.squeeze(0))
                last_state_before = copy.deepcopy(env.last_state)
                step_info = env.take_shot(action)
                reward = compute_drill_reward(step_info, last_state_before, my_targets)
                done, _ = env.get_done()
                obs_buf.append(x_np)
                act_buf.append(np.array([action['V0'], action['phi'], action['theta'], action['a'], action['b']], dtype=np.float32))
                logp_buf.append(logp.cpu().item())
                a_pre_buf.append(a_pre.cpu().numpy().squeeze(0))
                val_buf.append(v)
                rew_buf.append(reward)
                done_buf.append(1.0 if done else 0.0)
                steps += 1
                pbar.update(1)
                lr_now = init_lr + (min_lr - init_lr) * (steps / float(total_steps))
                opt.param_groups[0]['lr'] = lr_now
                if steps % ckpt_interval == 0:
                    save_checkpoint(os.path.join(ckpt_dir, f"new_agent_step_{steps:06d}.pt"),
                                    actor, critic, opt, scaler, steps, config, last_metrics)
                if steps % horizon == 0:
                    o = torch.from_numpy(np.stack(obs_buf)).to(device)
                    a = torch.from_numpy(np.stack(act_buf)).to(device)
                    old_logp = torch.from_numpy(np.array(logp_buf, dtype=np.float32)).to(device)
                    a_pre_t = torch.from_numpy(np.stack(a_pre_buf)).to(device)
                    v_np = np.array(val_buf, dtype=np.float32)
                    r_np = np.array(rew_buf, dtype=np.float32)
                    d_np = np.array(done_buf, dtype=np.float32)
                    adv_np, ret_np = compute_gae(r_np, v_np, d_np, gamma=gamma, lam=lam)
                    adv = torch.from_numpy((adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)).to(device)
                    ret = torch.from_numpy(ret_np).to(device)
                    policy_loss_val = 0.0
                    vf_loss_val = 0.0
                    entropy_val = 0.0
                    ratio_mean = 1.0
                    ent_w = init_ent + (final_ent - init_ent) * (steps / float(total_steps))
                    for _ in range(epochs):
                        with autocast(enabled=use_amp and device.type == "cuda"):
                            mu, std = actor(o)
                            dist = Normal(mu, std)
                            new_logp = dist.log_prob(a_pre_t).sum(-1)
                            ratio = torch.exp(new_logp - old_logp)
                            surr1 = ratio * adv
                            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                            policy_loss = -torch.min(surr1, surr2).mean()
                            v_pred = critic(o)
                            vf_loss = F.mse_loss(v_pred, ret)
                            entropy = dist.entropy().sum(-1).mean()
                            loss = policy_loss + vf_coef * vf_loss - ent_w * entropy
                        opt.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                        policy_loss_val = policy_loss.item()
                        vf_loss_val = vf_loss.item()
                        entropy_val = entropy.item()
                        ratio_mean = ratio.mean().item()
                    last_metrics = {
                        "rew": float(r_np.mean()),
                        "pl": float(policy_loss_val),
                        "vf": float(vf_loss_val),
                        "ent": float(entropy_val),
                        "ratio": float(ratio_mean),
                    }
                    pbar.set_postfix({
                        "rew": f"{last_metrics['rew']:.2f}",
                        "pl": f"{last_metrics['pl']:.3f}",
                        "vf": f"{last_metrics['vf']:.3f}",
                        "ent": f"{last_metrics['ent']:.3f}",
                        "lr": f"{lr_now:.1e}",
                        "opp": opp_tag,
                    })
                    obs_buf.clear()
                    act_buf.clear()
                    logp_buf.clear()
                    a_pre_buf.clear()
                    val_buf.clear()
                    rew_buf.clear()
                    done_buf.clear()
                if steps % log_interval == 0:
                    tqdm.write(f"steps={steps} rew={last_metrics['rew']:.2f} pl={last_metrics['pl']:.3f} vf={last_metrics['vf']:.3f} ent={last_metrics['ent']:.3f} ratio={last_metrics['ratio']:.3f} lr={lr_now:.1e} opp={opp_tag}")
                if done:
                    break
            else:
                if solo:
                    env.curr_player = 0
                    continue
                action = opponent.decision(balls, my_targets, table)
                env.take_shot(action)
                done, _ = env.get_done()
                if done:
                    break
    save_checkpoint(final_full_path, actor, critic, opt, scaler, steps, config, last_metrics)
    torch.save(actor.state_dict(), ckpt_path)
    pbar.close()

if __name__ == "__main__":
    train(solo=True)
