
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os


app = dash.Dash(__name__)
app.title = "Análise de Padrões de Sono"


DATA_PATH = os.path.join('data', 'sleep_data.csv')

def load_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    
   
    df['Bedtime'] = pd.to_datetime(df['Bedtime'], errors='coerce')
    df['Wakeup time'] = pd.to_datetime(df['Wakeup time'], errors='coerce')
    
  
    if 'Sleep duration' not in df.columns or df['Sleep duration'].isnull().any():
        df['Sleep duration'] = (df['Wakeup time'] - df['Bedtime']).dt.total_seconds() / 3600.0
    
   
    df['Caffeine consumption'] = df['Caffeine consumption'].fillna(0)
    df['Alcohol consumption'] = df['Alcohol consumption'].fillna(0)
    df['Smoking status'] = df['Smoking status'].fillna('Não')
    df['Exercise frequency'] = df['Exercise frequency'].fillna(df['Exercise frequency'].median())
    df['Sleep efficiency'] = df['Sleep efficiency'].fillna(df['Sleep efficiency'].mean())
    

    df['Gender'] = df['Gender'].map({'Male': 'Masculino', 'Female': 'Feminino'})
    df['Smoking status'] = df['Smoking status'].map({'No': 'Não', 'Yes': 'Sim'})
    
    return df

df = load_data(DATA_PATH)


df['Bed_hour'] = df['Bedtime'].dt.hour
df['Wakeup_hour'] = df['Wakeup time'].dt.hour

FEATURES = ['Age', 'Gender', 'Sleep duration', 'Caffeine consumption', 
            'Alcohol consumption', 'Smoking status', 'Exercise frequency',
            'Bed_hour', 'Wakeup_hour']
TARGET = 'Sleep efficiency'


df_model = df.copy()
df_model['Gender'] = df_model['Gender'].map({'Masculino': 0, 'Feminino': 1})
df_model['Smoking status'] = df_model['Smoking status'].map({'Não': 0, 'Sim': 1})

X = df_model[FEATURES]
y = df_model[TARGET]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='r2', verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²: {r2:.2f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


MODEL_PATH = 'model.pkl'
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(best_model, f)


with open(MODEL_PATH, 'rb') as f:
    loaded_model = pickle.load(f)


app.layout = html.Div([
    html.H1("Análise e Predição de Padrões de Sono", style={'textAlign': 'center', 'color': '#003366'}),
    

    html.Div([
        html.Div([
            html.H2("Distribuição de Idades"),
            dcc.Graph(
                id='age-distribution',
                figure=px.histogram(
                    df, 
                    x='Age', 
                    nbins=20, 
                    title='Distribuição de Idades',
                    labels={'Age': 'Idade'},
                    color_discrete_sequence=['#1f77b4']
                )
            ),
        ], className='six columns'),
        
        html.Div([
            html.H2("Proporção por Gênero"),
            dcc.Graph(
                id='gender-pie',
                figure=px.pie(
                    df, 
                    names='Gender', 
                    title='Proporção de Gênero',
                    labels={'Gender': 'Gênero'},
                    color_discrete_map={'Masculino': '#ff7f0e', 'Feminino': '#2ca02c'}
                )
            ),
        ], className='six columns'),
    ], className='row'),
    

    html.Div([
        html.Div([
            html.H2("Eficiência do Sono Média por Gênero"),
            dcc.Graph(
                id='sleep-efficiency-bar',
                figure=px.bar(
                    df.groupby('Gender')['Sleep efficiency'].mean().reset_index(),
                    x='Gender',
                    y='Sleep efficiency',
                    title='Eficiência do Sono Média por Gênero',
                    labels={'Gender': 'Gênero', 'Sleep efficiency': 'Eficiência do Sono (%)'},
                    color='Gender',
                    color_discrete_map={'Masculino': '#ff7f0e', 'Feminino': '#2ca02c'}
                )
            ),
        ], className='six columns'),
        
        html.Div([
            html.H2("Consumo de Cafeína vs Eficiência do Sono"),
            dcc.Graph(
                id='caffeine-efficiency',
                figure=px.scatter(
                    df, 
                    x='Caffeine consumption', 
                    y='Sleep efficiency',
                    trendline='ols',
                    labels={'Caffeine consumption': 'Consumo de Cafeína (mg)',
                            'Sleep efficiency': 'Eficiência do Sono (%)'},
                    title='Consumo de Cafeína vs Eficiência do Sono',
                    color='Gender',
                    color_discrete_map={'Masculino': '#ff7f0e', 'Feminino': '#2ca02c'},
                    hover_data=['Age', 'Sleep duration', 'Alcohol consumption', 'Smoking status', 'Exercise frequency']
                )
            ),
        ], className='six columns'),
    ], className='row'),
    

    html.Div([
        html.Div([
            html.H2("Consumo de Álcool vs Eficiência do Sono"),
            dcc.Graph(
                id='alcohol-efficiency',
                figure=px.scatter(
                    df, 
                    x='Alcohol consumption', 
                    y='Sleep efficiency',
                    trendline='ols',
                    labels={'Alcohol consumption': 'Consumo de Álcool (gr)',
                            'Sleep efficiency': 'Eficiência do Sono (%)'},
                    title='Consumo de Álcool vs Eficiência do Sono',
                    color='Gender',
                    color_discrete_map={'Masculino': '#ff7f0e', 'Feminino': '#2ca02c'},
                    hover_data=['Age', 'Sleep duration', 'Caffeine consumption', 'Smoking status', 'Exercise frequency']
                )
            ),
        ], className='six columns'),
        
        html.Div([
            html.H2("Frequência de Exercícios vs Eficiência do Sono"),
            dcc.Graph(
                id='exercise-efficiency',
                figure=px.scatter(
                    df, 
                    x='Exercise frequency', 
                    y='Sleep efficiency',
                    trendline='ols',
                    labels={'Exercise frequency': 'Frequência de Exercícios (dias/semana)',
                            'Sleep efficiency': 'Eficiência do Sono (%)'},
                    title='Frequência de Exercícios vs Eficiência do Sono',
                    color='Gender',
                    color_discrete_map={'Masculino': '#ff7f0e', 'Feminino': '#2ca02c'},
                    hover_data=['Age', 'Sleep duration', 'Caffeine consumption', 'Alcohol consumption', 'Smoking status']
                )
            ),
        ], className='six columns'),
    ], className='row'),
    

    
    

    html.H2("Predição da Eficiência do Sono", style={'marginTop': 50}),
    html.Div([
        html.Div([
            html.Label('Idade:', style={'font-weight': 'bold'}),
            dcc.Input(id='input-age', type='number', value=30, min=0, max=100, style={'width': '100%'}),
        ], className='three columns', style={'padding': '10px'}),
        
        html.Div([
            html.Label('Gênero:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='input-gender',
                options=[
                    {'label': 'Masculino', 'value': 'Masculino'},
                    {'label': 'Feminino', 'value': 'Feminino'}
                ],
                value='Masculino',
                clearable=False
            ),
        ], className='three columns', style={'padding': '10px'}),
        
        html.Div([
            html.Label('Duração do Sono (horas):', style={'font-weight': 'bold'}),
            dcc.Input(id='input-duration', type='number', value=7, min=0, max=24, style={'width': '100%'}),
        ], className='three columns', style={'padding': '10px'}),
        
        html.Div([
            html.Label('Consumo de Cafeína (mg):', style={'font-weight': 'bold'}),
            dcc.Input(id='input-caffeine', type='number', value=0, min=0, style={'width': '100%'}),
        ], className='three columns', style={'padding': '10px'}),
    ], className='row'),
    
    html.Div([
        html.Div([
            html.Label('Consumo de Álcool (gr):', style={'font-weight': 'bold'}),
            dcc.Input(id='input-alcohol', type='number', value=0, min=0, style={'width': '100%'}),
        ], className='three columns', style={'padding': '10px'}),
        
        html.Div([
            html.Label('Status de Tabagismo:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='input-smoking',
                options=[
                    {'label': 'Não', 'value': 'Não'},
                    {'label': 'Sim', 'value': 'Sim'}
                ],
                value='Não',
                clearable=False
            ),
        ], className='three columns', style={'padding': '10px'}),
        
        html.Div([
            html.Label('Frequência de Exercícios (dias/semana):', style={'font-weight': 'bold'}),
            dcc.Input(id='input-exercise', type='number', value=3, min=0, max=7, style={'width': '100%'}),
        ], className='three columns', style={'padding': '10px'}),
        
        html.Div([
            html.Label('Hora de Dormir (0-23):', style={'font-weight': 'bold'}),
            dcc.Input(id='input-bed-hour', type='number', value=22, min=0, max=23, style={'width': '100%'}),
        ], className='three columns', style={'padding': '10px'}),
    ], className='row'),
    
    html.Button(
        'Prever Eficiência do Sono', 
        id='predict-button', 
        n_clicks=0,
        style={'marginTop': '20px', 'padding': '10px 20px', 'fontSize': '16px'}
    ),
    
    html.Div(
        id='prediction-output', 
        style={'marginTop': 20, 'fontSize': 24, 'textAlign': 'center', 'color': '#FF5733'}
    ),
    

    html.H2("Avaliação do Modelo de Machine Learning", style={'marginTop': 50}),
    html.Div([
        html.P(f"<b>R²:</b> {r2:.2f}", style={'fontSize': '18px'}),
        html.P(f"<b>MAE:</b> {mae:.2f}", style={'fontSize': '18px'}),
        html.P(f"<b>RMSE:</b> {rmse:.2f}", style={'fontSize': '18px'})
    ], style={'textAlign': 'center'}),
    

    html.Div([
        html.H2("Importância das Features no Modelo", style={'marginTop': 50}),
        dcc.Graph(
            id='feature-importance',
            figure=px.bar(
                x=best_model.feature_importances_,
                y=FEATURES,
                orientation='h',
                labels={'x': 'Importância', 'y': 'Features'},
                title='Importância das Features no Modelo de Random Forest',
                color=best_model.feature_importances_,
                color_continuous_scale='Blues'
            )
        ),
    ], style={'marginTop': 50}),
])


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [
        Input('input-age', 'value'),
        Input('input-gender', 'value'),
        Input('input-duration', 'value'),
        Input('input-caffeine', 'value'),
        Input('input-alcohol', 'value'),
        Input('input-smoking', 'value'),
        Input('input-exercise', 'value'),
        Input('input-bed-hour', 'value')
    ]
)
def predict_sleep_efficiency(n_clicks, age, gender, duration, caffeine, alcohol, smoking, exercise, bed_hour):
    if n_clicks > 0:
        try:

            gender_num = 1 if gender == 'Feminino' else 0
            smoking_num = 1 if smoking == 'Sim' else 0
            
            # Cálculo da hora de acordar
            wakeup_hour = bed_hour + duration
            if wakeup_hour >= 24:
                wakeup_hour -= 24
            

            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender_num],
                'Sleep duration': [duration],
                'Caffeine consumption': [caffeine],
                'Alcohol consumption': [alcohol],
                'Smoking status': [smoking_num],
                'Exercise frequency': [exercise],
                'Bed_hour': [bed_hour],
                'Wakeup_hour': [wakeup_hour]
            })
            

            prediction = loaded_model.predict(input_data)[0]
            prediction = np.round(prediction, 2)
            
            return f"Eficiência do Sono Predita: {prediction}%"
        except Exception as e:
            return f"Erro na predição: {e}"
    
    return ""


if __name__ == '__main__':
    app.run_server(debug=True)
