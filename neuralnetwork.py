import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from matplotlib import pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def g(x, g_trial):
    return g_trial

def neural_network(params, x, num_hidden_layers):
    num_values = np.size(x)
    x = x.reshape(-1, num_values)
    x_input = np.concatenate((np.ones((1, num_values)), x), axis=0)
    
    # Przepuszczanie przez warstwy ukryte
    for i in range(num_hidden_layers):
        w_hidden = params[i]
        z_hidden = np.matmul(w_hidden, x_input)
        x_hidden = sigmoid(z_hidden)
        x_hidden = np.concatenate((np.ones((1, num_values)), x_hidden), axis=0)
        x_input = x_hidden
        
    
    # Wyjście z sieci neuronowej
    w_output = params[-1]
    z_output = np.matmul(w_output, x_input)
    return z_output

def g_trial(x, params, g0=1, num_hidden_layers=1):
    return g0 + x * neural_network(params, x, num_hidden_layers)
def g_trial_exp(x, params, g0=1, num_hidden_layers=1):
    return g0 + (1-np.exp(-x)) * neural_network(params, x, num_hidden_layers)

def cost_function(P, x, num_hidden_layers):
    g_t = g_trial(x, P, num_hidden_layers=num_hidden_layers)
    d_g_t = elementwise_grad(g_trial, 0)(x, P, num_hidden_layers=num_hidden_layers)
    func = g(x, g_t)

    err_sqr = (d_g_t - func) ** 2
    cost_sum = np.sum(err_sqr)

    return cost_sum / np.size(err_sqr)

def initialize_weights(method, num_neurons_hidden, input_size, num_hidden_layers):
    weights = []
    if method == 'normal':
        # Inicjacja normalna
        p0 = npr.randn(num_neurons_hidden, input_size)
        weights.append(p0)
        for _ in range(1, num_hidden_layers):
            p_hidden = npr.randn(num_neurons_hidden, num_neurons_hidden + 1)
            weights.append(p_hidden)
        p_output = npr.randn(1, num_neurons_hidden + 1)
        weights.append(p_output)
        
    elif method == 'uniform':
        # Inicjacja jednostajna
        p0 = npr.rand(num_neurons_hidden, input_size)
        weights.append(p0)
        for _ in range(1, num_hidden_layers):
            p_hidden = npr.rand(num_neurons_hidden, num_neurons_hidden + 1)
            weights.append(p_hidden)
        p_output = npr.rand(1, num_neurons_hidden + 1)
        weights.append(p_output)
        
    elif method == 'xavier':
        # Inicjacja Xavier
        scale_hidden = np.sqrt(2.0 / (input_size + num_neurons_hidden))
        p0 = npr.randn(num_neurons_hidden, input_size) * scale_hidden
        weights.append(p0)
        for _ in range(1, num_hidden_layers):
            scale_hidden = np.sqrt(2.0 / (num_neurons_hidden + num_neurons_hidden + 1))
            p_hidden = npr.randn(num_neurons_hidden, num_neurons_hidden + 1) * scale_hidden
            weights.append(p_hidden)
        scale_output = np.sqrt(2.0 / (num_neurons_hidden + 1))
        p_output = npr.randn(1, num_neurons_hidden + 1) * scale_output
        weights.append(p_output)
        
    elif method == 'he':
        # Inicjacja He
        scale_hidden = np.sqrt(6.0 / (input_size + num_neurons_hidden))
        p0 = npr.randn(num_neurons_hidden, input_size) * scale_hidden
        weights.append(p0)
        for _ in range(1, num_hidden_layers):
            scale_hidden = np.sqrt(6.0 / (num_neurons_hidden + num_neurons_hidden + 1))
            p_hidden = npr.randn(num_neurons_hidden, num_neurons_hidden + 1) * scale_hidden
            weights.append(p_hidden)
        scale_output = np.sqrt(6.0 / (num_neurons_hidden + 1))
        p_output = npr.randn(1, num_neurons_hidden + 1) * scale_output
        weights.append(p_output)
    
    else:
        raise ValueError(f"Unknown initialization method: {method}")

    return weights

def solve_ode_neural_network(x, num_neurons_hidden, num_iter, lmb, g0=1, num_hidden_layers=1, batch_size=None, use_rms=False, 
                             use_momentum=False, use_adam=False, beta1=0.9, beta2=0.999, init_method="normal", trial_fun=g_trial):
    input_size = 2
    P = initialize_weights(init_method, num_neurons_hidden, input_size, num_hidden_layers)
    errors_array = []

    momentum = [np.zeros_like(layer) for layer in P]
    v = [np.zeros_like(layer) for layer in P]
    V_weights = [np.zeros_like(layer) for layer in P]
    V_biases = [np.zeros_like(layer) for layer in P]
    M_weights = [np.zeros_like(layer) for layer in P]
    M_biases = [np.zeros_like(layer) for layer in P]

    print('Initial cost: %g' % cost_function(P, x, num_hidden_layers))
    cost_function_grad = grad(cost_function, 0)

    if batch_size is None:
        batch_size = len(x)  

    for i in range(1, num_iter + 1):
        # Shuffle data
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x_shuffled = x[indices]
        
        # Batch processing
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            x_batch = x_shuffled[start:end]

            cost_grad = cost_function_grad(P, x_batch, num_hidden_layers)

            if use_adam:
                V_weights = [beta1 * vw + (1 - beta1) * gw for vw, gw in zip(V_weights, cost_grad)]
                V_biases = [beta1 * vb + (1 - beta1) * gb for vb, gb in zip(V_biases, cost_grad)]
                M_weights = [beta2 * mw + (1 - beta2) * np.square(gw) for mw, gw in zip(M_weights, cost_grad)]
                M_biases = [beta2 * mb + (1 - beta2) * np.square(gb) for mb, gb in zip(M_biases, cost_grad)]
                epsilon = 1e-15
                P = [p - lmb * vw / (np.sqrt(mw) + epsilon) for p, vw, mw in zip(P, V_weights, M_weights)]

            elif use_momentum:
                momentum = [cg + beta1 * mom for cg, mom in zip(cost_grad, momentum)]
                P = [p - lmb * mom for p, mom in zip(P, momentum)]
                
            elif use_rms:
                v = [beta1 * v_l + (1 - beta1) * cg for v_l, cg in zip(v, cost_grad)]
                P = [p - lmb * v_l for p, v_l in zip(P, v)]

            else:
                P = [p - lmb * cg for p, cg in zip(P, cost_grad)]
            
            res = g_trial(x, P, g0=g0, num_hidden_layers=num_hidden_layers)

        errors_array.append(cost_function(P, x, num_hidden_layers))

    final_cost = cost_function(P, x, num_hidden_layers)
    return P, errors_array, final_cost

def right_side1(t, N):
    return N
def euler_method(t, N):
    dt = t[1] - t[0]
    N_values = [N]
    for i in range(1, len(t)):
        N_new = N_values[-1] + dt * right_side1(t[i-1], N_values[-1])
        N_values.append(N_new)
    return N_values

if __name__ == '__main__':
    npr.seed(15)

    N = 100
    x = np.linspace(0, 1, N)
    
    lmb = 0.001 
    layers = 1
    num_hidden_neurons = 50
    num_iter = 10
    plt.figure(figsize=(10, 10))
    plt.plot(x, euler_method(x, N=1), label='euler')
    plt.title('Your ODE soulution:')
    plt.xlabel('x')
    plt.ylabel('g(x)')
    P, errors_array, final_cost = solve_ode_neural_network(x, num_hidden_neurons, num_iter, lmb, num_hidden_layers=1, use_adam=True)
    res = g_trial(x, P, num_hidden_layers=1)

    nn_mse = np.max(np.abs(res - euler_method(x, N=1)))
