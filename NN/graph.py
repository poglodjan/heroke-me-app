import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def mlp(x_train, y_train, x_test, y_test, norm,
        activation, iters, learn, inputs_n, neurons):
    
    # normalization:
    np.random.seed(100)
    if norm == 1:
        x_train_norm = (x_train - min(x_train)) / (max(x_train)+min(x_train))
        x_test_norm = (x_test - min(x_test)) / (max(x_test)+min(x_test))
        y_train_norm = (y_train - min(y_train)) / (max(y_train)+min(y_train))
        y_test_norm = (y_test - min(y_test)) / (max(y_test)+min(y_test))
    else: 
        x_train_norm = x_train
        x_test_norm = x_test
        y_train_norm = y_train
        y_test_norm = y_test
    frames=[]
    # _______________Inicjalizacja losowych wag i biasów_______________
    weights_input_hidden = np.random.randn(inputs_n, neurons)
    biases_hidden = np.random.randn(neurons)
    weights_hidden_output = np.random.randn(neurons, inputs_n)
    bias_output = np.random.randn(1, inputs_n)

    # _______________Propagacja w przód dla danych treningowych_______________
    hidden_layer_input = np.dot(x_train_norm, weights_input_hidden) + biases_hidden
    hidden_layer_output = activation(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output

    y_predicted = output_layer_input
    error = y_predicted - y_train_norm

    frames=[]
    for i in range(iters):
        learning_rate = learn

        # _______________Aktualizacja wag w warstwie ukrytej na wyjście_______________
        weights_hidden_output -= learning_rate * np.outer(error, hidden_layer_output).T
        bias_output -= learning_rate * np.sum(np.sum(error))

        # _______________Obliczenie wagi dla warstwy ukrytej_______________
        w = np.dot(error, weights_hidden_output.T) * hidden_layer_output * (1 - hidden_layer_output)
        w = w.T * x_train_norm

        # _______________Aktualizacja wag w warstwie wejściowej na ukrytą_______________
        weights_input_hidden -= w.T
        suma = np.sum(error.T * weights_hidden_output.T * hidden_layer_output * (1 - hidden_layer_output.T))
        biases_hidden -= learning_rate * suma

        # _______________Ponowna propagacja w przód_______________
        new_hidden_layer_input = np.dot(x_train_norm, weights_input_hidden) + biases_hidden
        new_hidden_layer_output = activation(new_hidden_layer_input)
        new_output_layer_input = np.dot(new_hidden_layer_output, weights_hidden_output) + bias_output
        new_y_predicted = new_output_layer_input
        error = new_y_predicted - y_train_norm
        plt.scatter(x_train_norm,new_y_predicted)
        plt.show()
        frame_data = {'x': x_train_norm, 'y': new_y_predicted}  # Załóżmy, że y_predicted to Twoje przewidywane wartości
        frame = go.Frame(data=[go.Scatter(x=frame_data['x'], y=frame_data['y'], 
                                      mode='markers',  # Ustawienie trybu na 'markers'
                                      marker=dict(color='red', size=20))]) 
        frames.append(frame)
    fig = go.Figure(frames=frames)
    fig.update_layout(title='Scatter Plot: x_train vs. y_predicted (Animation)',
                            xaxis_title='x_train',
                            yaxis_title='y_predicted')

    fig.write_html('scatter_plot_animation.html')

def sigm(x):
    return 1 / (1 + np.exp(-x))
def mse_out(y,y_pred):
    difference = y - y_pred
    return np.mean(np.square(difference))

def main():
    simple_train = pd.read_csv('reg/square-simple-training.csv')
    simple_test = pd.read_csv('reg/square-simple-test.csv')
    x_train = np.array(simple_train['x'])
    y_train= np.array(simple_train['y'])
    x_test = np.array(simple_test['x'])
    y_test = np.array(simple_test['y'])
    activation=sigm
    norm=0
    mlp(x_train,y_train,x_test,y_test,norm, activation, iters=10,
                learn=0.1,inputs_n=len(simple_train),neurons=10) # one hidden layer 10 neurons

if __name__ == "__main__":
    main()