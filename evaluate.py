import time
import argparse
from utils import set_random_seed
from poolenv import PoolEnv
from agents import BasicAgent, BasicAgentPro, NewAgent

def parse_args():
    parser = argparse.ArgumentParser(description="AI Billiards Agent Evaluation Script")
    
    parser.add_argument('--n_games', type=int, default=120, help='Total number of games to play')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--no_seed', action='store_false', dest='enable_seed', help='Disable fixed random seed')
    
    parser.add_argument('--ckpt_path', type=str, 
                        default="/home/xlyu/AI-Billiard/AI3603-Billiards-RL/AI3603-Billiards-main/ckpts/agent_last.pt", 
                        help='Checkpoint path for NewAgent')
    
    parser.add_argument('--log_file', type=str, default="logging.txt", help='Path to log file')
    
    parser.set_defaults(enable_seed=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    set_random_seed(enable=args.enable_seed, seed=args.seed)

    env = PoolEnv()
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    
    agent_a = BasicAgentPro()
    agent_b = NewAgent(args.ckpt_path)

    players = [agent_a, agent_b]  
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe'] 

    new_agent_total_time = 0.0
    new_agent_shot_count = 0

    for i in range(args.n_games): 
        print(f"------- 第 {i} 局比赛开始 -------")
        
        current_ball_type = target_ball_choice[i % 4]
        env.reset(target_ball=current_ball_type)
        
        player_a_class = players[i % 2].__class__.__name__
        print(f"本局 Player A: {player_a_class}, 目标球型: {current_ball_type}")
        
        with open(args.log_file, "a", encoding="utf-8") as f: 
            f.write(f"第 {i} 局 - Player A: {player_a_class}\n")

        while True:
            player_label = env.get_curr_player() 
            print(f"[第{env.hit_count}次击球] player: {player_label}")
            obs = env.get_observation(player_label)

            is_new_agent = (player_label == 'A' and i % 2 == 1) or (player_label == 'B' and i % 2 == 0)


            current_agent = players[i % 2] if player_label == 'A' else players[(i + 1) % 2]

            t_start = time.time()
            action = current_agent.decision(*obs)
            t_dur = time.time() - t_start

            if is_new_agent:
                new_agent_total_time += t_dur
                new_agent_shot_count += 1

            step_info = env.take_shot(action)
            
            done, info = env.get_done()
            if not done:
                for foul_key in ['FOUL_FIRST_HIT', 'NO_POCKET_NO_RAIL', 'NO_HIT']:
                    if step_info.get(foul_key):
                        print(f"本杆判罚：{foul_key}，交换球权。")
                
                if step_info.get('ME_INTO_POCKET'):
                    print(f"我方球入袋：{step_info['ME_INTO_POCKET']}")
                if step_info.get('ENEMY_INTO_POCKET'):
                    print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
            
            if done:
                winner = info['winner']
                if winner == 'SAME':
                    results['SAME'] += 1
                elif winner == 'A':
                    results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
                else:
                    results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
                break


    results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] + results['SAME'] * 0.5
    results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] + results['SAME'] * 0.5

    print("\n" + "="*40)
    print("最终结果:", results)
    win_rate = results['AGENT_B_WIN'] / float(args.n_games)
    score_rate = results['AGENT_B_SCORE'] / float(args.n_games)
    
    print(f"NewAgent 胜:{results['AGENT_B_WIN']}/{args.n_games} 平:{results['SAME']} 胜率:{win_rate:.3f} 计分胜率:{score_rate:.3f}")
    print("="*40)

if __name__ == '__main__':
    main()