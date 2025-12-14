"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
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
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

# 假设必要的库和 BasicAgent 中的 analyze_shot_for_reward, BayesianOptimization 等都已导入

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
    
    
    '''
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
    '''
       
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score


class NewAgent(Agent):
    """
    NewAgent: 几何物理计算 + 局部贝叶斯优化 (增强版)
    
    修改特性：
    1. 严格避让黑8：打普通球时，黑8的避让半径提升至5倍。
    2. 智能防守：无进攻路线时，朝远离黑8的方向轻推防守。
    3. 稳健收官：打黑8时自动降速，防止母球洗袋。
    """
    
    def __init__(self):
        super().__init__()
        # 手动定义标准台球半径 (单位: 米)
        self.BALL_RADIUS = 0.028575
        
        # 贝叶斯优化参数
        self.INITIAL_SEARCH = 3
        self.OPT_SEARCH = 5
        self.ALPHA = 1e-2
        print("NewAgent (Geometry + Local Opt + Safety) 已初始化。")

    def _normalize(self, v):
        """向量归一化"""
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else np.zeros_like(v)

    def _get_angle(self, v):
        """计算向量在xy平面的角度 (0-360度)"""
        rad = np.arctan2(v[1], v[0])
        deg = np.degrees(rad)
        return deg % 360

    def _check_collision(self, start_pos, end_pos, obstacle_balls, clearance_factor=2.0, strict_8_avoid=False):
        """
        检测从 start_pos 到 end_pos 的线段路径上是否有障碍球
        
        参数:
            clearance_factor: 默认安全因子
            strict_8_avoid: 是否对黑8进行超严格避让 (5倍半径)
        """
        path_vec = end_pos - start_pos
        path_len = np.linalg.norm(path_vec)
        if path_len < 1e-6:
            return False 
            
        path_dir = path_vec / path_len
        
        # 基础安全距离
        base_safe_dist = self.BALL_RADIUS * clearance_factor
        
        for bid, ball in obstacle_balls.items():
            if ball.state.s == 4: continue # 已进袋的球忽略
            
            # --- 修改点1：对黑8的特殊判定 ---
            if strict_8_avoid and bid == '8':
                current_safe_dist = self.BALL_RADIUS * 5.0 # 5倍半径避让
            else:
                current_safe_dist = base_safe_dist
            
            obs_pos = ball.state.rvw[0]
            vec_to_obs = obs_pos - start_pos
            proj_len = np.dot(vec_to_obs, path_dir)
            
            # 障碍球在路径范围内
            if -self.BALL_RADIUS < proj_len < path_len + self.BALL_RADIUS:
                dist_sq = np.linalg.norm(vec_to_obs)**2 - proj_len**2
                if dist_sq < 0: dist_sq = 0
                dist = np.sqrt(dist_sq)
                
                if dist < current_safe_dist:
                    return True # 发生碰撞或距离太近
        return False

    def decision(self, balls=None, my_targets=None, table=None):
        """
        核心决策逻辑
        """
        if balls is None or my_targets is None:
            return self._random_action()

        # --- 1. 确定当前合法的目标球 ---
        normal_balls = [b for b in my_targets if b != '8' and balls[b].state.s != 4]
        
        valid_targets = []
        is_shooting_8 = False # 标记当前是否在打黑8
        
        if len(normal_balls) > 0:
            valid_targets = normal_balls
        else:
            if balls['8'].state.s != 4:
                valid_targets = ['8']
                is_shooting_8 = True
        
        if not valid_targets:
            return self._random_action()

        cue_pos = balls['cue'].state.rvw[0]
        candidates = []

        # --- 2. 遍历所有击球方案 (几何筛选) ---
        for tid in valid_targets:
            target_pos = balls[tid].state.rvw[0]
            
            # 如果当前目标不是黑8，则需要严格避让黑8
            strict_avoid_8 = (tid != '8')
            
            for pid, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # A. 计算 幽灵球位置
                t_to_p = pocket_pos - target_pos
                t_to_p_dir = self._normalize(t_to_p)
                ghost_pos = target_pos - t_to_p_dir * (2 * self.BALL_RADIUS)
                
                # B. 碰撞检测
                # 1. 检查 目标球 -> 袋口
                obstacles_t_p = {k:v for k,v in balls.items() if k != tid and k != 'cue'}
                if self._check_collision(target_pos, pocket_pos, obstacles_t_p, 
                                       clearance_factor=1.8, strict_8_avoid=strict_avoid_8):
                    continue 
                    
                # 2. 检查 白球 -> 幽灵球
                obstacles_c_g = {k:v for k,v in balls.items() if k != tid and k != 'cue'}
                if self._check_collision(cue_pos, ghost_pos, obstacles_c_g, 
                                       clearance_factor=1.9, strict_8_avoid=strict_avoid_8):
                    continue 

                # C. 角度检查
                aim_vec = ghost_pos - cue_pos
                aim_dist = np.linalg.norm(aim_vec)
                aim_dir = self._normalize(aim_vec)
                
                cut_angle_cos = np.dot(aim_dir, t_to_p_dir)
                if cut_angle_cos < 0.15: continue
                    
                # D. 评分
                total_dist = aim_dist + np.linalg.norm(t_to_p)
                score = (1.0 / (total_dist + 0.1)) + (2.0 * cut_angle_cos)
                
                candidates.append({
                    'tid': tid,
                    'pid': pid,
                    'score': score,
                    'aim_vec': aim_vec,
                    'ghost_pos': ghost_pos,
                    'dist': total_dist,
                    'phi_geo': self._get_angle(aim_vec)
                })

        # --- 修改点2：防守策略 (Safety Play) ---
        # 如果没有几何上可行的方案，执行防守，而不是随机击球
        if not candidates:
            print("[NewAgent] 无直击路径，执行防守策略(Safety)。")
            
            # 获取黑8位置
            pos_8 = balls['8'].state.rvw[0]
            vec_c_8 = pos_8 - cue_pos
            angle_to_8 = self._get_angle(vec_c_8)
            
            # 策略：朝黑8的反方向打(偏160度)，最小力度
            safety_phi = (angle_to_8 + 160) % 360
            safety_v0 = 0.5 # 最小速度
            
            return {
                'V0': safety_v0,
                'phi': safety_phi,
                'theta': 0,
                'a': 0,
                'b': 0
            }
        
        # 选出分最高的方案
        best_geo = max(candidates, key=lambda x: x['score'])
        # --- 3. 局部贝叶斯优化 ---
        
        # --- 修改点3：打黑8时的速度控制 ---
        if best_geo['tid'] == '8':
            # 打黑8：速度更慢，防止洗袋，求稳
            # 基础速度降低，且上限设为 2.5
            base_v0 = 0.8 + best_geo['dist'] * 1.0 
            base_v0 = np.clip(base_v0, 2.0, 2.1)
            print(f"[NewAgent] 决胜球(黑8)！启用低速模式: V0={base_v0:.2f}")
        else:
            # 普通球：正常力度
            base_v0 = 1.0 + best_geo['dist'] * 1.5 
            base_v0 = np.clip(base_v0, 1.5, 4.5)
        
        pbounds = {
            'delta_v': (-0.6, 0.6), 
            'delta_phi': (-1.5, 1.5), 
            'a': (-0.1, 0.1),
            'b': (-0.1, 0.1) 
        }
        
        state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        target_list_for_eval = [best_geo['tid']] 
        
        def eval_shot(delta_v, delta_phi, a, b):
            sim_table = copy.deepcopy(table)
            sim_balls = {k: copy.deepcopy(v) for k,v in balls.items()}
            sim_cue = pt.Cue(cue_ball_id="cue")
            sim_sys = pt.System(table=sim_table, balls=sim_balls, cue=sim_cue)
            
            v_eval = np.clip(base_v0 + delta_v, 0.5, 8.0)
            phi_eval = (best_geo['phi_geo'] + delta_phi) % 360
            
            try:
                sim_sys.cue.set_state(V0=v_eval, phi=phi_eval, theta=0, a=a, b=b)
                pt.simulate(sim_sys, inplace=True)
            except:
                return -1000
                
            return my_analyze_shot_for_reward(sim_sys, state_snapshot, target_list_for_eval)

        optimizer = BayesianOptimization(
            f=eval_shot,
            pbounds=pbounds,
            verbose=0,
            random_state=np.random.randint(1000)
        )
        optimizer._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=2
        )
        
        optimizer.maximize(init_points=self.INITIAL_SEARCH, n_iter=self.OPT_SEARCH)
        
        best_res = optimizer.max
        params = best_res['params']
        
        final_action = {
            'V0': np.clip(base_v0 + params['delta_v'], 0.5, 8.0),
            'phi': (best_geo['phi_geo'] + params['delta_phi']) % 360,
            'theta': 0, 
            'a': params['a'],
            'b': params['b']
        }
        
        print(f"[NewAgent] 目标:{best_geo['tid']} 距离:{best_geo['dist']:.2f} "
              f"修正后:{final_action['phi']:.1f}")
              
        return final_action
        '''
        # --- 3. 无贝叶斯优化：直接使用几何方案 ---
        final_action = {
            'V0': base_v0,
            'phi': best_geo['phi_geo'],
            'theta': 0,
            'a': 0,
            'b': 0
        }

        print(f"[NewAgent] 目标:{best_geo['tid']} 距离:{best_geo['dist']:.2f} phi={final_action['phi']:.1f}")
        return final_action
        '''
