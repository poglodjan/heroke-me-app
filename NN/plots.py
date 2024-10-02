import plotly.graph_objs as go
import numpy as np

frames = []
for i in range(100):
    x = np.linspace(0, 10, 50)
    y = np.sin(x + i)
    noise = np.random.normal(0, 0.05, len(x))
    y_with_noise = y + noise
    frame_data = {'x': x, 'y': y_with_noise}
    frames.append(frame_data)

fig = go.Figure()
fig.add_trace(go.Scatter(x=frames[0]['x'], y=frames[0]['y'], mode='lines', line=dict(color='red')))
fig.frames = [go.Frame(data=[go.Scatter(x=frame['x'], y=frame['y'], mode='lines', line=dict(color='red'))]) for frame in frames]
for frame_data in frames:
    x_points = frame_data['x']
    y_points = frame_data['y']
    noise = np.random.normal(0, 0.05, len(x_points))  
    x_noise = x_points
    y_noise = y_points + noise
    fig.add_trace(go.Scatter(x=x_noise, y=y_noise, mode='markers', marker=dict(color='white', size=1)))

fig.update_yaxes(visible=False)  
fig.update_xaxes(visible=False)  
fig.update_layout(showlegend=False) 
fig.show(config={'displayModeBar': False}) 
fig.update_layout(title='', xaxis=dict(range=[0, 10]), 
                  yaxis=dict(range=[-1, 1]), plot_bgcolor='black', paper_bgcolor='black')
fig.write_html('animated_plot.html', full_html=False, include_plotlyjs='cdn')
