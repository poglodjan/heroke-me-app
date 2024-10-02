from flask import Flask, render_template, jsonify, Response,url_for, redirect
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from flask_sse import sse
import time
import random

app = Flask(__name__)
app.register_blueprint(sse, url_prefix='/stream')
app.config['REDIS_URL'] = 'redis://localhost:6379/0'

@app.route('/comercial_projects')
def commercial_projects():
    return redirect(url_for('commercial_projects_page'))
@app.route('/commercial_projects_page')
def commercial_projects_page():
    return render_template('corpo.html')

@app.route('/contact')
def contact():
    return redirect(url_for('contact_page'))
@app.route('/contact_page')
def contact_page():
    return render_template('contact.html')

@app.route('/students')
def students():
    return redirect(url_for('students_page'))
@app.route('/students_page')
def students_page():
    return render_template('students.html')

@app.route('/own')
def own():
    return redirect(url_for('own_page'))
@app.route('/own_page')
def own_page():
    return render_template('own.html')

@app.route('/education')
def education():
    return index()

@app.route('/')
def index():
    # Przykładowe dane
    data = {'Category': ["Python", "R", "Data bases", "C/C++", "Software Engineering", "Unix", "AI","Web Apps", "Algorithms","SQL", "SAS", "Matlab"],
            'How much do I know and like': [90, 50, 60, 60, 40, 40, 80, 80, 60,70,60,40],
             'x': ['Data analysis, softwares, web apps',
                    'Data processing, statistics',
                      'Creating client applications in azure data studio',
                        'Knowledge of object-oriented and structured programming',
                          'Software engineering projects in Python and C++',
                            'Proficiency in linux commands',
                            'Developing Machine learning and Computational intelligence projects',
                            'Much joy of creating apps with Flask,JS,CSS,html',
                              'Knowledge of tree-based and advanced algorithms',
                              'Efficiency in designing sql queries',
                              'Good knowledge of SAS language and handling of statistical events',
                              'Use for machine learning and matrix actions']}
    
    data2 = {'Category': ["Calculus", "Algebra", "Complex Analysis", "Statistics", "Computer Statictics", "Functional analysis", "Probability Calculus", "Discrete Maths", "Optimalization","Stochastic Processes", "Numerical Methods", "Convex Analysis"],
            'How much do I know and like': [100, 100, 40, 80, 70, 30, 50, 60, 80,90,80,50],
             'x': ['Theory of measure, Lebesgue integrals, ODEs',
                    'Linear operators on vector spaces, matrix operations',
                      "Complex integrals, holomorphic functions, residue theorems, Laurent's Series",
                        'Spatial data analysis, estimators, tests, distributions',
                          'Multiple regression analysis, linear models, experiment matrices',
                            'Study of Banach spaces, Hilbert spaces, linear operators, compact operators',
                            'Theory of measure in probability calculus, laws of numbers',
                            'Theory of graphs, codes, combinatorics', 
                            'Theory of methods gradient descent, symplex, coupled directions, bfgs',
                            'Markov processes, Poisson processes, Wiener processes,Brownian movements',
                            "Bysection method, Newton's method, secant method, matrix distributions",
                            'Convex optimization, linear programming']}

    data = pd.DataFrame(data).sort_values(by='How much do I know and like', ascending=False)
    data2 = pd.DataFrame(data2).sort_values(by='How much do I know and like', ascending=False)
# computer science
    fig = px.bar(data, x="x", y='How much do I know and like', text="Category",
             color_discrete_sequence=['#B21223']*len(data['Category']), hover_data={"How much do I know and like":False,"Category":False})
    fig.update_layout(paper_bgcolor='black') 
    fig.update_traces(textposition='outside', textfont=dict(size=8,family='monospace'))  
    fig.update_layout(plot_bgcolor='black')  
    fig.update_yaxes(visible=False)  
    fig.update_xaxes(visible=False)  
    fig.update_traces(textposition='outside', textfont=dict(size=14,family='monospace', color="white"),  
                        hovertemplate="<b>%{x}</b><br>%{text}")
    fig.update_layout(xaxis=dict(tickangle=45), yaxis=dict(tickfont=dict(size=14,family='monospace', color='white')),  
                    yaxis_title_font=dict(size=14,family='monospace', color='white'))  
    fig.update_layout(hoverlabel=dict(bgcolor="white", font=dict(color="black",family='monospace'), font_size=18, namelength=-1)) 
    fig.update_layout(yaxis_range=[0, 100]) 
    graph_html = fig.to_html(full_html=False)

# mathematics
    fig2 = px.bar(data2, x="x", y='How much do I know and like', text="Category",
                 color_discrete_sequence=['#1758E1']*len(data['Category']), 
                 hover_data={"How much do I know and like":False,"Category":False})
    fig2.update_layout(paper_bgcolor='black')  
    fig2.update_traces(textposition='outside', textfont=dict(size=8,family='monospace'))  
    fig2.update_layout(plot_bgcolor='black')  
    fig2.update_yaxes(visible=False)  
    fig2.update_xaxes(visible=False) 
    fig2.update_traces(textposition='outside', textfont=dict(size=14,family='monospace', color="white"),  
                        hovertemplate="<b>%{x}</b><br>%{text}")
    fig2.update_layout(xaxis=dict(tickangle=45), yaxis=dict(tickfont=dict(size=14,family='monospace', color='white')), 
                    yaxis_title_font=dict(size=8,family='monospace', color='white'))  
    fig2.update_layout(hoverlabel=dict(bgcolor="white", font=dict(color="black",family='monospace'), font_size=18, namelength=-1))  
    fig2.update_layout(yaxis_range=[0, 120])  
    graph_html2= fig2.to_html(full_html=False)

    return render_template('index.html', graph_html=graph_html, graph_html2=graph_html2)

if __name__ == '__main__':
    app.run(debug=True)
