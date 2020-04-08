from unityagents import UnityEnvironment
import numpy as np
import time

from agent import Agent
from epsilon_greedy import EpsilonGreedy
from plot import save_plot_results
from utils import StateAggregator

# Parameters
approach_title = "RANDOM"
episodes = 5000
frames = 4
target_avg_score = 30
target_score_episodes = 100
eps_start = 1
eps_stop = 0.01
eps_decay = 0.9975

# Create environment
env = UnityEnvironment(file_name='./Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations

num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = states.shape[1]

print(f"Estimated epsilon on end: {eps_start*(eps_decay**episodes):0.6f} Min:{eps_stop:0.3f}")
agent = Agent(state_size*frames, action_size)

def play(brain_name, agent, env, eps):
    # Reset environment and variables
    env_info = env.reset(train_mode=True)[brain_name]      
    states = env_info.vector_observations                  
    scores = np.zeros(num_agents)                          

    state_agg = StateAggregator(states, frames)
    next_state_agg = StateAggregator(states, frames)

    while True:
        aggregated_states = state_agg.to_input()
        
        # Choose actions
        actions = []
        for n_agent in range(num_agents):
            action = agent.act(aggregated_states[n_agent], eps)
            actions.append(action)
        actions = np.array(actions)
        actions = np.clip(actions, -1, 1)

        # Interact with env
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment

        # Gather data
        rewards = env_info.rewards                        
        dones = env_info.local_done                        
        scores += env_info.rewards                        
        next_states = env_info.vector_observations         # get next state (for each agent)

        next_state_agg.push(next_states)

        # Inform agent about the results
        aggregated_next_states = next_state_agg.to_input()
        for n_agent in range(num_agents):
            agent.step(aggregated_states[n_agent], actions[n_agent], rewards[n_agent], 
                        aggregated_next_states[n_agent], dones[n_agent])

        # Finish episode step
        states = next_states 
        state_agg.push(states)
        if np.any(dones):
            break
    return scores

# Try to solve environment
episode_scores = []
epsgreedy = EpsilonGreedy(eps_start, eps_stop, eps_decay)
for episode in range(episodes):
    e_start = time.time()
    eps = epsgreedy.sample()
    scores = play(brain_name, agent, env, eps)
    avg_score = np.mean(scores)
    episode_scores.append(scores)

    # Solve rule
    mean_target_score = np.mean(episode_scores[-target_score_episodes:])
    if len(episode_scores) >= target_score_episodes and mean_target_score>=target_avg_score:
        print(f"Environment solved after : {episode+1} episodes.")
        print(f"Mean score: {mean_target_score:.3f} over last {target_score_episodes} episodes.")
        break

    agent.save()
    e_stop = time.time()
    seconds = e_stop-e_start
    print(f"[Episode {episode}, Time(s): {seconds:.1f}, Eps:{eps:.5f}] Score: {avg_score:.3f}, MeanOver{target_score_episodes}: {mean_target_score:.3f}")

# Finish
env.close()
save_plot_results(approach_title, np.mean(episode_scores, axis=1), target_score_episodes, target_avg_score)



