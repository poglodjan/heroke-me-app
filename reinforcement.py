import autograd.numpy.random as npr
import numpy as np
import random
import gym
import plotly.graph_objects as go

def init_game():
    env = gym.make("FrozenLake8x8-v1", desc=None, map_name=None, is_slippery=True)
    prob = {i: {j: {a: 0 for a in range(env.action_space.n)}
                for j in range(env.observation_space.n)}
            for i in range(env.observation_space.n)}
    rewd = {i: {j: {a: 0 for a in range(env.action_space.n)}
                for j in range(env.observation_space.n)}
            for i in range(env.observation_space.n)}
    
    for i in range(env.observation_space.n):
        for a in range(env.action_space.n):
            for (p, j, r, d) in env.P[i][a]:
                prob[i][j][a] += p
                rewd[i][j][a] += r

    gamma, theta, T = 0.9, 1e-5, 50
    V = np.zeros(env.observation_space.n)

    for _ in range(1000):
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            Q_values = []
            for a in range(env.action_space.n):
                q = 0
                for (p, next_state, reward, done) in env.P[s][a]:
                    q += p * (reward + gamma * V[next_state])
                Q_values.append(q)
            V[s] = max(Q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    policy = {t: {} for t in range(T)}
    for s in range(env.observation_space.n):
        best_action, best_value = None, float('-inf')
        for a in range(env.action_space.n):
            q = sum(p * (reward + gamma * V[next_state])
                    for (p, next_state, reward, done) in env.P[s][a])
            if q > best_value:
                best_value, best_action = q, a
        for t in range(T):
            policy[t][s] = best_action

    random_policy = {t: {i: env.action_space.sample() for i in range(env.observation_space.n)} for t in range(T)}

    # evaluate for optimal
    optimal_eval = round(evaluate_policy(env, policy, T, gamma),2)
    random_eval = round(evaluate_policy(env, random_policy, T, gamma),2)

    random_value = np.random.rand(env.observation_space.n).reshape(8, 8)
    value = np.random.rand(env.observation_space.n).reshape(8, 8)

    moves_to_str = {3: u'↑', 2: u'→', 1: u'↓', 0: u'←'}

    fig_random = create_policy_figure(random_value, "Optimal Policy", random_policy, env, moves_to_str)
    fig_optimal = create_policy_figure(value, "Random Policy", policy, env, moves_to_str)

    return fig_random.to_html(full_html=False), fig_optimal.to_html(full_html=False), optimal_eval, random_eval 


def create_policy_figure(value_matrix, title, policy, env, moves_to_str):
    fig = go.Figure(data=go.Heatmap(
    z=value_matrix,
    colorscale='Viridis'
    ))
    fig.update_layout(
        title="The map of game for " + title,
        xaxis=dict(tickmode='array', tickvals=list(range(8)), ticktext=['0', '1', '2', '3', '4', '5', '6', '7']),
        yaxis=dict(tickmode='array', tickvals=list(range(8)), ticktext=['0', '1', '2', '3', '4', '5', '6', '7']),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )

    for i in range(8):
        for j in range(8):
            t = moves_to_str[policy[0][i*8 + j]]
            if env.desc[i, j] == b'H':
                fig.add_shape(
                    type="rect",
                    x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                    fillcolor="black",
                    line=dict(color="black")
                )
                fig.add_annotation(x=j, y=i, text='H', showarrow=False, font=dict(color="white", size=15))
            elif env.desc[i, j] == b'S':
                fig.add_shape(
                    type="rect",
                    x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                    fillcolor="black",
                    line=dict(color="black")
                )
                fig.add_annotation(x=j, y=i, text='S', showarrow=False, font=dict(color="white", size=15))
            elif env.desc[i, j] == b'G':
                fig.add_shape(
                    type="rect",
                    x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                    fillcolor="black",
                    line=dict(color="black")
                )
                fig.add_annotation(x=j, y=i, text='G', showarrow=False, font=dict(color="white", size=15))
            else:
                fig.add_annotation(x=j, y=i, text=t, showarrow=False, font=dict(color="white", size=15))
    return fig

def evaluate_policy(env, policy, T, gamma):
    nS = env.observation_space.n
    V = np.zeros(nS)
    
    for t in range(T):
        V_new = np.zeros(nS)
        for s in range(nS):
            a = policy[t][s]  
            V_new[s] = sum(p * (reward + gamma * V[next_state])
                           for (p, next_state, reward, done) in env.P[s][a])
        V = V_new
    
    return np.sum(V) 
