from flask import Flask, render_template, jsonify, Response, url_for, redirect, make_response,send_file,request
import os
import zipfile
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt
import time
import random
import base64
import io
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix,roc_curve, auc
from sklearn.datasets import load_breast_cancer
import seaborn as sns
from scipy.stats import chi2_contingency
from neuralnetwork import g_trial, initialize_weights, cost_function, grad
from bioengineering import needleman_wunsch, generate_random_sequence
from reinforcement import init_game

app = Flask(__name__)
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

@app.route('/education')
def education():
    return redirect(url_for('education_page'))
@app.route('/education_page')
def education_page():
    return render_template('education.html')

@app.route('/own')
def own():
    response = make_response(render_template('projects.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/fin')
def fin():
    return redirect(url_for('fin_page'))
@app.route('/fin_page')
def fin_page():
    return render_template('serv_fin.html')
@app.route('/block')
def block():
    return redirect(url_for('block_page'))
@app.route('/block_page')
def block_page():
    return render_template('serv_block.html')

#####
# BIO
#####

@app.route('/bio')
def bio():
    return redirect(url_for('bio_page'))
@app.route('/bio_page')
def bio_page():
    # Załadowanie danych o nowotworach piersi
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Podział na zestawy treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.3, random_state=42)
    
    # Klasyfikator i obliczenia dla ROC
    clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)

    # Wykres ROC z mniejszą skutecznością
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = 0.97)', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='red'), name='Random Guess'))
    fig1.update_layout(
        title='ROC Curve for Breast Cancer Prediction (RandomForest)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )
    fig1_html = pio.to_html(fig1, full_html=False)

    fig2 = px.scatter(
    df, 
    x='mean radius', 
    y='mean texture', 
    color='target', 
    labels={'target': '', 'mean radius': 'Mean Radius', 'mean texture': 'Mean Texture'},
    color_continuous_scale=px.colors.sequential.Viridis,
    title="Scatter Plot of Mean Radius vs. Mean Texture by Diagnosis",
    symbol='target'
    )
    fig2.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        coloraxis_colorbar=dict(
            title="",
            tickvals=[0, 1],
            ticktext=['Improvement', 'Detoriation']
        )
    )
    fig2_html = pio.to_html(fig2, full_html=False)


    time = np.linspace(0, 24, 50)
    concentration = np.exp(-0.1 * time) + np.random.normal(0, 0.05, size=time.shape)
    fig3 = px.scatter(x=time, y=concentration, title="Drug Concentration over Time", labels={'x': 'Time (hours)', 'y': 'Concentration'})
    fig3.update_layout(paper_bgcolor='black', plot_bgcolor='black', font=dict(color='white'))
    fig3_html = pio.to_html(fig3, full_html=False)

    # Generowanie sekwencji i dopasowanie algorytmem Needleman-Wunsch
    seq1 = generate_random_sequence(7)
    seq2 = generate_random_sequence(7)
    score_matrix, align1, align2 = needleman_wunsch(seq1, seq2)
    fig4 = go.Figure(data=go.Heatmap(z=score_matrix, colorscale='Cividis'))
    fig4.update_layout(
        title="Needleman-Wunsch Alignment Matrix",
        xaxis=dict(title="Sequence 2", tickvals=list(range(len(seq2) + 1)), ticktext=['-'] + list(seq2)),
        yaxis=dict(title="Sequence 1", tickvals=list(range(len(seq1) + 1)), ticktext=['-'] + list(seq1)),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )
    fig4_html = pio.to_html(fig4, full_html=False)

    return render_template('serv_bio.html', fig1=fig1_html, fig2=fig2_html, fig3=fig3_html, fig4=fig4_html)

#####
# ODE
#####

@app.route('/ode')
def ode():
    return render_template('serv_ode.html')
@app.route('/update_plot')
def update_plot():
    g0 = float(request.args.get('g0', 1))
    num_neurons = int(request.args.get('num_neurons', 50))
    num_iter = int(request.args.get('num_iters', 100))
    layers = int(request.args.get('layers', 1))
    is_random = bool(int(request.args.get('is_random', 0)))
    is_adam = bool(int(request.args.get('is_adam', 1)))

    def solve_ode_neural_network(x, num_iter, lmb, g0, num_hidden_layers=1, num_neurons_hidden=50, batch_size=None, use_adam=True):
        input_size = 2
        P = initialize_weights('normal', num_neurons_hidden, input_size, num_hidden_layers)
        cost_function_grad = grad(cost_function, 0)

        if batch_size is None:
            batch_size = len(x)

        for i in range(1, num_iter + 1):
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            x_shuffled = x[indices]

            for start in range(0, len(x), batch_size):
                end = min(start + batch_size, len(x))
                x_batch = x_shuffled[start:end]

                try:
                    cost_grad = cost_function_grad(P, x_batch, num_hidden_layers)
                except ValueError as e:
                    print("Error in cost_function_grad:", e)
                    print("Shapes of variables:")
                    print("P:", [p.shape for p in P])
                    print("x_batch:", x_batch.shape)
                    raise

                P = [p - lmb * g for p, g in zip(P, cost_grad)]
                res = g_trial(x, P, g0=g0, num_hidden_layers=num_hidden_layers)
                y = res[0, :]

                data = pd.DataFrame({'x': x, 'g(x)': y})
                cost_value = cost_function(P, x, num_hidden_layers)
                fig = px.line(data, x='x', y='g(x)', 
                              title=f'ODE Solution at epoch {i}, Cost: {round(cost_value,4)}',
                              labels={'x': 'x', 'g(x)': 'g(x)'})
                fig.update_traces(line=dict(color='#B21223'))
                fig.update_layout(
                    paper_bgcolor='black',
                    plot_bgcolor='black',
                    font=dict(color='white'),
                    title_font=dict(size=16),
                    xaxis_title='x',
                    yaxis_title='g(x)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )

                buf = io.BytesIO()
                fig.write_image(buf, format='png')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')

                yield f"data: {img_base64}\n\n"

    N = 100
    if not is_random: np.random.seed(320575)
    x = np.linspace(0, 2, N)
    lmb = 0.001

    return Response(solve_ode_neural_network(x, num_iter, lmb, g0, num_hidden_layers=layers, num_neurons_hidden=num_neurons, batch_size=None, use_adam=is_adam),
                    mimetype='text/event-stream')

@app.route('/rain')
def rain():
    return redirect(url_for('rain_page'))
@app.route('/rain_page')
def rain_page():
    fig_random_html, fig_optimal_html, eval1, eval2 = init_game()
    return render_template('serv_rain.html', 
                           fig_html=fig_random_html, 
                           fig2_html=fig_optimal_html, 
                           eval1=eval1, 
                           eval2=eval2)

@app.route('/explor')
def explor():
    return redirect(url_for('explor_page'))
@app.route('/explor_page')
def explor_page():
    dataset_name = request.args.get('dataset', 'iris')  # domyślnie 'iris'

    if dataset_name == 'iris':
        data = datasets.load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        target_names = data.target_names

        # Analiza PCA dla Iris
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df.iloc[:, :-1])
        df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        df_pca['target'] = df['target']
        pca_fig = px.scatter(df_pca, x='PC1', y='PC2', color='target', title="PCA Analysis of Iris Dataset")
        pca_fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        pca_html = pio.to_html(pca_fig, full_html=False)

        # Macierz korelacji dla Iris
        corr = df.iloc[:, :-1].corr()
        corr_fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='Viridis'
        ))
        corr_fig.update_layout(
            title='Correlation Matrix of Iris Dataset',
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        corr_html = pio.to_html(corr_fig, full_html=False)

        # Feature Importance using Random Forest dla Iris
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_model.fit(df.iloc[:, :-1], df['target'])
        feature_importances = rf_model.feature_importances_
        importance_fig = px.bar(
            x=df.columns[:-1], 
            y=feature_importances, 
            title='Feature Importance using Random Forest on Iris Dataset'
        )
        importance_fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        importance_html = pio.to_html(importance_fig, full_html=False)

        # Pair Plot Analysis for Iris (scatter plot of pairwise relationships)
        pair_fig = px.scatter_matrix(df, dimensions=df.columns[:-1], color=df['target'],
                                     title="Pair Plot Analysis of Iris Dataset",
                                     labels={col: col.replace(" (cm)", "") for col in df.columns[:-1]})
        pair_fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        pair_html = pio.to_html(pair_fig, full_html=False)

        return render_template('serv_explor.html', 
                               pca_fig=pca_html, 
                               corr_fig=corr_html, 
                               importance_fig=importance_html, 
                               cm_fig=pair_html, 
                               selected_dataset=dataset_name)

    elif dataset_name == 'titanic':
        df = sns.load_dataset('titanic').dropna(subset=['survived', 'age', 'fare'])
        df['target'] = df['survived']

        # Histogram wieku dla Titanic
        age_fig = px.histogram(df, x='age', color='sex', title="Age Distribution on Titanic")
        age_fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        age_html = pio.to_html(age_fig, full_html=False)

        # Wykres przetrwania w zależności od klasy dla Titanic
        class_survival_fig = px.histogram(df, x='pclass', color='survived', title="Survival by Class on Titanic", barmode='group')
        class_survival_fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        class_survival_html = pio.to_html(class_survival_fig, full_html=False)

        # Zwracanie analiz dla Titanic
        return render_template('serv_explor.html', 
                               pca_fig=age_html, 
                               corr_fig=class_survival_html, 
                               importance_fig="", 
                               cm_fig="", 
                               selected_dataset=dataset_name)

    elif dataset_name == 'wine':
        data = datasets.load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        # PCA dla Wine
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df.iloc[:, :-1])
        df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        df_pca['target'] = df['target']
        pca_fig = px.scatter(df_pca, x='PC1', y='PC2', color='target', title="PCA Analysis of Wine Dataset")
        pca_fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        pca_html = pio.to_html(pca_fig, full_html=False)

        # Feature Importance dla Wine
        rf_model = RandomForestClassifier(n_estimators=10)
        rf_model.fit(df.iloc[:, :-1], df['target'])
        feature_importances = rf_model.feature_importances_
        importance_fig = px.bar(
            x=df.columns[:-1], 
            y=feature_importances, 
            title='Feature Importance using Random Forest on Wine Dataset'
        )
        importance_fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        importance_html = pio.to_html(importance_fig, full_html=False)

        # Zwracanie analiz dla Wine
        return render_template('serv_explor.html', 
                               pca_fig=pca_html, 
                               corr_fig=importance_html, 
                               importance_fig="", 
                               cm_fig="", 
                               selected_dataset=dataset_name)

    else:
        return "Invalid dataset selected", 400



###########
### profile
########### 

@app.route('/profile')
def profile():
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
