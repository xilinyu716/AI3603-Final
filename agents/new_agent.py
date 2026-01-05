import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime

from .agent import Agent

def my_analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0:
            if first_contact_ball_id in opponent_plus_eight:
                foul_first_hit = True
        else:
            if first_contact_ball_id != '8':
                foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
    
       
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 1000
    elif cue_pocketed:
        score -= 40
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -1000
            
    if foul_first_hit:
        score -= 40
    if foul_no_rail:
        score -= 40
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 50
    
    survive_cost = 10.0
    base_legal = 5.0
    motion_penalty_coeff = 1.5
    if (not cue_pocketed) and (not eight_pocketed) and (not foul_first_hit) and (not foul_no_rail) and (len(own_pocketed) == 0) and (len(enemy_pocketed) == 0):
        score += base_legal
    if 'cue' in last_state and 'cue' in shot.balls:
        p0 = last_state['cue'].state.rvw[0]
        p1 = shot.balls['cue'].state.rvw[0]
        dx = float(p1[0]) - float(p0[0])
        dy = float(p1[1]) - float(p0[1])
        dist = math.sqrt(dx * dx + dy * dy)
        score -= motion_penalty_coeff * dist
    score -= survive_cost
        
    return score


import numpy as np
import pooltool as pt
import copy
import multiprocessing as mp
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import os
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None


# ==========================================
#  全局辅助函数 (用于多进程 MC 模拟)
# ==========================================
def _mc_worker_process(args):
    """
    MC 模拟的工作单元
    Args: (table_state, balls_state, action_dict, noise_std, target_id)
    """
    table_state, balls_state, action, noise_std, target_id = args
    
    # 1. 构建模拟环境
    # 为了效率，这里假定传入的 state 已经是 pickle 友好的数据，或者在此处重建
    sim_sys = pt.System(
        table=copy.deepcopy(table_state),
        balls=copy.deepcopy(balls_state),
        cue=pt.Cue(cue_ball_id="cue")
    )
    
    # 2. 注入噪声 (模拟真实误差)
    v_noisy = np.clip(action['V0'] + np.random.normal(0, noise_std['V0']), 0.1, 8.0)
    phi_noisy = (action['phi'] + np.random.normal(0, noise_std['phi'])) % 360
    theta_noisy = np.clip(action['theta'] + np.random.normal(0, noise_std['theta']), 0, 90)
    a_noisy = np.clip(action['a'] + np.random.normal(0, noise_std['a']), -0.5, 0.5)
    b_noisy = np.clip(action['b'] + np.random.normal(0, noise_std['b']), -0.5, 0.5)
    
    # 3. 执行物理模拟
    sim_sys.cue.set_state(V0=v_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
    pt.simulate(sim_sys, inplace=True)
    
    # 4. 提取关键风险指标
    # (这里我们只关心“是否进球”和“是否发生灾难”，具体的 Reward 计算留给贝叶斯)
    
    # 检查目标球是否进袋
    success = False
    if target_id in sim_sys.balls and sim_sys.balls[target_id].state.s == 4:
        success = True
        
    # 检查白球洗袋 (灾难)
    cue_scratch = (sim_sys.balls['cue'].state.s == 4)
    
    # 检查黑8误进 (如果在打普通球时进黑8 -> 判负 -> 极大灾难)
    fatal_8 = False
    if target_id != '8' and '8' in sim_sys.balls and sim_sys.balls['8'].state.s == 4:
        fatal_8 = True
        
    # 检查打黑8时洗袋 (判负 -> 极大灾难)
    fatal_8_scratch = False
    if target_id == '8' and cue_scratch:
        fatal_8_scratch = True
        
    return {
        'success': success,
        'scratch': cue_scratch,
        'fatal': fatal_8 or fatal_8_scratch
    }



def strip_compile_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_sd[k[len("_orig_mod."):]] = v
        else:
            new_sd[k] = v
    return new_sd


# ==========================================
#  NewComAgent 类定义
# ==========================================
class NewAgent(Agent):
    """
    NewComAgent (Combined Agent):
    1. 几何筛选 (Candidate Generation) -> 选出 Top 3 路径
    2. 贝叶斯优化 (Optimization) -> 对路径进行精细微调
    3. 蒙特卡洛验证 (Verification) -> 压力测试，剔除不稳健的解
    """
    
    def __init__(self, root=None):
        super().__init__()
        self.BALL_RADIUS = 0.028575
        
        # 贝叶斯参数
        self.INITIAL_SEARCH = 2
        self.OPT_SEARCH = 5
        self.ALPHA = 1e-2
        
        # 蒙特卡洛参数
        self.MC_CANDIDATE_NUM = 3   # 对几何最好的前3个方案进行 优化+验证
        self.MC_SIM_COUNT = 20      # 每个方案验证 20 次
        self.MC_FATAL_TOLERANCE = 0.05 # 致命错误容忍度 (5%)
        self.MC_SUCCESS_THRESHOLD = 0.05 # 最小成功率要求
        self.CPU_CORES = max(1, mp.cpu_count() - 2)
        
        # 内部噪声模型 (应与环境保持一致)
        self.noise_std = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }
        self.rl_enabled = False
        self.policy = None
        self.state_dim = 66
        self._init_rl(root)
        # print(f"NewComAgent (Geo -> BayesOpt -> MC Verify) Initialized.")

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else np.zeros_like(v)

    def _get_angle(self, v):
        rad = np.arctan2(v[1], v[0])
        deg = np.degrees(rad)
        return deg % 360
    
    def _init_rl(self, root=None):
        if torch is None or nn is None:
            return
        class PolicyNet(nn.Module):
            def __init__(self, in_dim, hidden=256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                )
                self.mu = nn.Linear(hidden, 5)
                self.log_std = nn.Parameter(torch.zeros(5))
            def forward(self, x):
                h = self.net(x)
                return self.mu(h), self.log_std.exp()
            
        ckpt_dir = root
        model = PolicyNet(self.state_dim)
        loaded = False
        
        path_final = os.path.join(ckpt_dir, "new_agent_step_1700000.pt")

        if os.path.exists(path_final):
            data = torch.load(path_final, map_location="cpu",weights_only=False)
            
            
            
            if isinstance(data, dict) and "actor" in data and data["actor"]:
                # print(f"[debug] data: {data}")
                # print(f"[debug] keys: {data.keys()}")
                
                actor_state = strip_compile_prefix(data["actor"])
                model.load_state_dict(actor_state)
                loaded = True
        if loaded:
            self.rl_enabled = True
            self.policy = model.eval()
        else:
            self.rl_enabled = False
            self.policy = None
            
        # print(f"[debug]: In init_rl: self.rl_enabled = {self.rl_enabled}")
    
    def _encode_state(self, balls, my_targets, table):
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
    
    def _scale_action(self, raw):
        v = 0.5 + (raw[0] + 1.0) * 0.5 * (8.0 - 0.5)
        phi = (raw[1] + 1.0) * 0.5 * 360.0
        theta = (raw[2] + 1.0) * 0.5 * 30.0
        a = -0.5 + (raw[3] + 1.0) * 0.5 * (1.0)
        b = -0.5 + (raw[4] + 1.0) * 0.5 * (1.0)
        return {
            'V0': float(np.clip(v, 0.5, 8.0)),
            'phi': float(phi % 360.0),
            'theta': float(np.clip(theta, 0.0, 90.0)),
            'a': float(np.clip(a, -0.5, 0.5)),
            'b': float(np.clip(b, -0.5, 0.5))
        }
    
    def _policy_infer(self, balls, my_targets, table):
        if not self.rl_enabled or self.policy is None:
            return None
        x = self._encode_state(balls, my_targets, table)
        with torch.no_grad():
            t = torch.from_numpy(x).unsqueeze(0)
            mu, std = self.policy(t)
            raw = torch.tanh(mu).squeeze(0).numpy()
        return self._scale_action(raw)
    
    def _select_target_by_phi(self, balls, valid_targets, cue_pos, phi):
        dir_vec = np.array([np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))], dtype=np.float32)
        best = None
        best_score = 1e9
        for tid in valid_targets:
            tpos = balls[tid].state.rvw[0][:2]
            v = tpos - cue_pos[:2]
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                continue
            vdir = v / nv
            ang = np.degrees(np.arccos(np.clip(np.dot(dir_vec, vdir), -1.0, 1.0)))
            score = ang + nv
            if score < best_score:
                best_score = score
                best = tid
        return best

    def _check_collision(self, start_pos, end_pos, obstacle_balls, clearance_factor=2.0, strict_8_avoid=False):
        """碰撞检测 (保持原逻辑)"""
        path_vec = end_pos - start_pos
        path_len = np.linalg.norm(path_vec)
        if path_len < 1e-6: return False 
        path_dir = path_vec / path_len
        base_safe_dist = self.BALL_RADIUS * clearance_factor
        
        for bid, ball in obstacle_balls.items():
            if ball.state.s == 4: continue
            current_safe_dist = self.BALL_RADIUS * 2.8 if (strict_8_avoid and bid == '8') else base_safe_dist
            obs_pos = ball.state.rvw[0]
            vec_to_obs = obs_pos - start_pos
            proj_len = np.dot(vec_to_obs, path_dir)
            
            if -self.BALL_RADIUS < proj_len < path_len + self.BALL_RADIUS:
                dist_sq = np.linalg.norm(vec_to_obs)**2 - proj_len**2
                if dist_sq < 0: dist_sq = 0
                if np.sqrt(dist_sq) < current_safe_dist:
                    return True
        return False

    def _run_bayesian_optimization(self, candidate, balls, table, base_v0):
        """对单个候选方案运行贝叶斯优化"""
        
        # 定义优化目标函数
        def eval_shot(delta_v, delta_phi, a, b):
            sim_table = copy.deepcopy(table)
            sim_balls = {k: copy.deepcopy(v) for k,v in balls.items()}
            sim_cue = pt.Cue(cue_ball_id="cue")
            sim_sys = pt.System(table=sim_table, balls=sim_balls, cue=sim_cue)
            
            
            v_eval = np.clip(base_v0 + delta_v, 0.5, 5.5)
            phi_eval = (candidate['phi_geo'] + delta_phi) % 360
            
            try:
                sim_sys.cue.set_state(V0=v_eval, phi=phi_eval, theta=0, a=a, b=b)
                pt.simulate(sim_sys, inplace=True)
            except:
                return -1000.0
            
            # 使用题目给定的 my_analyze_shot_for_reward
            state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            return my_analyze_shot_for_reward(sim_sys, state_snapshot, [candidate['tid']])

        # 配置优化器
        pbounds = {
            'delta_v': (-0.5, 0.5), 
            'delta_phi': (-2.5, 2.5), # 搜索范围稍微放大一点
            'a': (-0.05, 0.05),
            'b': (-0.05, 0.05) 
        }
        
        optimizer = BayesianOptimization(
            f=eval_shot, pbounds=pbounds, verbose=0,
            random_state=np.random.randint(1000)
        )
        optimizer._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), alpha=self.ALPHA, n_restarts_optimizer=2
        )
        
        # 执行优化
        optimizer.maximize(init_points=self.INITIAL_SEARCH, n_iter=self.OPT_SEARCH)
        
        # 提取结果
        params = optimizer.max['params']
        optimized_action = {
            'V0': np.clip(base_v0 + params['delta_v'], 0.5, 5.5),
            'phi': (candidate['phi_geo'] + params['delta_phi']) % 360,
            'theta': 0, 'a': params['a'], 'b': params['b']
        }
        return optimized_action, optimizer.max['target'] # 返回动作和理论最高分

    def _run_monte_carlo_verification(self, action, table, balls, target_id):
        """对优化后的动作进行蒙特卡洛压力测试"""
        tasks = []
        for _ in range(self.MC_SIM_COUNT):
            tasks.append((table, balls, action, self.noise_std, target_id))
            
        with mp.Pool(processes=self.CPU_CORES) as pool:
            results = pool.map(_mc_worker_process, tasks)
            
        stats = {'success': 0, 'scratch': 0, 'fatal': 0}
        for res in results:
            if res['success']: stats['success'] += 1
            if res['scratch']: stats['scratch'] += 1
            if res['fatal']: stats['fatal'] += 1
            
        # 计算比率
        total = self.MC_SIM_COUNT
        return {
            'win_rate': stats['success'] / total,
            'scratch_rate': stats['scratch'] / total,
            'fatal_rate': stats['fatal'] / total
        }

    def decision(self, balls=None, my_targets=None, table=None):
        """决策主循环"""
        if balls is None or my_targets is None:
            return self._random_action()

        # 1. 目标识别
        normal_balls = [b for b in my_targets if b != '8' and balls[b].state.s != 4]
        valid_targets = normal_balls if normal_balls else (['8'] if balls['8'].state.s != 4 else [])
        
        if not valid_targets: return self._random_action()

        cue_pos = balls['cue'].state.rvw[0]
        candidates = []
        
        # print(f"rl_enabled = {self.rl_enabled}")
        
        if self.rl_enabled:
            
            # print("[Info]: RL Enabled")
            rl_action = self._policy_infer(balls, my_targets, table)
            if rl_action is not None:
                tid_rl = self._select_target_by_phi(balls, valid_targets, cue_pos, rl_action['phi'])
                if tid_rl is None and valid_targets:
                    tid_rl = valid_targets[0]
                if tid_rl is not None:

                    cand_rl = {'tid': tid_rl, 'pid': None, 'score': 1.0, 'dist': 0.0, 'phi_geo': rl_action['phi']}
                    base_v0 = rl_action['V0']
                    opt_action_rl, theory_reward_rl = self._run_bayesian_optimization(cand_rl, balls, table, base_v0)
                    mc_stats_rl = self._run_monte_carlo_verification(opt_action_rl, table, balls, tid_rl)
                    win_rate_rl = mc_stats_rl['win_rate']
                    fatal_rate_rl = mc_stats_rl['fatal_rate']
                    scratch_rate_rl = mc_stats_rl['scratch_rate']
                    verify_score_rl = win_rate_rl - (scratch_rate_rl * 1.5) - (fatal_rate_rl * 8)
                    if win_rate_rl > 0.88 and fatal_rate_rl == 0 and scratch_rate_rl < 0.05:
                        print("[Info] Decision by RL")
                        return opt_action_rl
                    if win_rate_rl >= self.MC_SUCCESS_THRESHOLD and fatal_rate_rl <= self.MC_FATAL_TOLERANCE:
                        print("[Info] RL Participation")
                        candidates.append({'tid': tid_rl, 'pid': None, 'score': verify_score_rl, 'dist': 0.0, 'phi_geo': opt_action_rl['phi']})
                        candidates[-1]['opt_action_prefill'] = opt_action_rl

        # 2. 几何扫描 (Candidate Generation)
        for tid in valid_targets:
            target_pos = balls[tid].state.rvw[0]
            strict_avoid_8 = (tid != '8')
            
            for pid, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # A. 幽灵球
                t_to_p = pocket_pos - target_pos
                t_to_p_dir = self._normalize(t_to_p)
                ghost_pos = target_pos - t_to_p_dir * (2 * self.BALL_RADIUS)
                
                # B. 几何碰撞检测 (快速剪枝)
                obstacles = {k:v for k,v in balls.items() if k != tid and k != 'cue'}
                if self._check_collision(target_pos, pocket_pos, obstacles, 1.8, strict_avoid_8): continue
                if self._check_collision(cue_pos, ghost_pos, obstacles, 1.9, strict_avoid_8): continue
                
                # C. 角度检查
                aim_vec = ghost_pos - cue_pos
                dist_aim = np.linalg.norm(aim_vec)
                aim_dir = self._normalize(aim_vec)
                cut_angle_cos = np.dot(aim_dir, t_to_p_dir)
                
                if cut_angle_cos < 0.2: continue 
                
                # D. 启发式打分 (用于选出 Top N)
                dist_total = dist_aim + np.linalg.norm(t_to_p)
                score = (1.0 / (dist_total + 0.1)) + (2.0 * cut_angle_cos)
                
                candidates.append({
                    'tid': tid, 'pid': pid, 'score': score,
                    'dist': dist_total, 'phi_geo': self._get_angle(aim_vec)
                })

        # 3. 如果没有几何解，直接防守
        if not candidates:
            # print("[NewComAgent] 无几何解，转入防守。")
            return self._safety_action(balls, normal_balls)

        # 4. 排序并取前 N 个候选 (Candidate Selection)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:self.MC_CANDIDATE_NUM]
        
        print(f"[NewComAgent] 选中 {len(top_candidates)} 个候选方案进行 [优化+验证]...")

        # 5. 生成-验证循环 (Generate & Verify Loop)
        best_verified_action = None
        best_verified_score = -1.0 # 用 win_rate - risk 衡量

        for i, cand in enumerate(top_candidates):
            tid = cand['tid']
            # 基础速度设定
            if tid == '8':
                base_v0 = np.clip(0.8 + cand['dist'] * 1.0, 0.5, 2.5) # 黑8求稳
            else:
                base_v0 = np.clip(1.0 + cand['dist'] * 1.5, 1.5, 5.5)
            
            # Step A: 贝叶斯优化 (Optimization)
            # # print(f"  > 正在优化候选 #{i} (Target: {tid})...")
            if 'opt_action_prefill' in cand:
                opt_action = cand['opt_action_prefill']
                theory_reward = 0.0
            else:
                opt_action, theory_reward = self._run_bayesian_optimization(cand, balls, table, base_v0)
            
            # Step B: 蒙特卡洛验证 (Verification)
            # # print(f"  > 正在验证候选 #{i}...")
            mc_stats = self._run_monte_carlo_verification(opt_action, table, balls, tid)
            
            # Step C: 评估
            win_rate = mc_stats['win_rate']
            fatal_rate = mc_stats['fatal_rate']
            scratch_rate = mc_stats['scratch_rate']
            
            print(f"  Result #{i} (T:{tid}): Win {win_rate:.2f} | Fatal {fatal_rate:.2f} | Scratch {scratch_rate:.2f} | TheoryReward {theory_reward:.1f}")
            
            # 判决逻辑：
            # 1. 致命错误 (误打黑8) 必须低于容忍度
            # 2. 成功率必须高于阈值 (除非是最后没办法了)
            if fatal_rate > self.MC_FATAL_TOLERANCE:
                # print(f"    -> ❌ 驳回: 致命风险过高 ({fatal_rate:.2f})")
                continue # 尝试下一个候选
            
            # 综合评分: 进球率优先，但严重惩罚洗袋
            # score = Win% - Scratch% * 1.5
            verify_score = win_rate - (scratch_rate * 1.5) - (fatal_rate * 8)
            
            # 如果这球非常稳 (win > 70% 且无风险)，直接采纳，不再算后面的
            if win_rate > 0.88 and fatal_rate == 0 and scratch_rate < 0.05:
                print(f"    -> ✅ 完美方案，直接采纳！")
                return opt_action
            
            # 否则，记录下来，找分最高的
            if verify_score > best_verified_score and win_rate >= self.MC_SUCCESS_THRESHOLD:
                best_verified_score = verify_score
                best_verified_action = opt_action
        
        # 6. 最终决策
        if best_verified_action is not None:
            print(f"[NewComAgent] 最终选择: V0={best_verified_action['V0']:.2f}, Score={best_verified_score:.2f}")
            return best_verified_action
        
        # 如果所有候选方案都被 MC 否决了 (例如都有风险，或者进球率都极低)
        # print("[NewComAgent] 所有进攻方案经 MC 验证均不可行/高风险，执行防守。")
        return self._safety_action(balls, normal_balls)

    def _safety_action(self, balls, normal_balls):
        """防守策略 (同 NewORAgent)"""
        cue_pos = balls['cue'].state.rvw[0]
        pos_8 = balls['8'].state.rvw[0]
        
        best_target = None
        max_dist_8 = -1
        
        for tid in normal_balls:
            t_pos = balls[tid].state.rvw[0]
            d = np.linalg.norm(t_pos - pos_8)
            obstacles = {k:v for k,v in balls.items() if k != tid and k != 'cue'}
            if not self._check_collision(cue_pos, t_pos, obstacles, strict_8_avoid=True):
                if d > max_dist_8:
                    max_dist_8 = d
                    best_target = t_pos
        
        if best_target is not None:
            aim = best_target - cue_pos
            dist = np.linalg.norm(aim)
            return {
                'V0': np.clip(dist * 0.8, 0.5, 1.5),
                'phi': self._get_angle(aim), 'theta': 0, 'a': 0, 'b': 0
            }
            
        # 盲打防守
        angle_to_8 = self._get_angle(pos_8 - cue_pos)
        return {
            'V0': 0.5, 'phi': (angle_to_8 + 180) % 360, 'theta': 0, 'a': 0, 'b': 0
        }

    def _random_action(self):
        return {"V0": np.random.uniform(0.5, 5.0), "phi": np.random.uniform(0, 360), "theta": 0, "a": 0, "b": 0}
