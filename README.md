# Análise e Predição de Padrões de Sono

## Descrição

Esta aplicação web permite a análise e predição de padrões de sono utilizando um conjunto de dados fictício de padrões de sono. Através de um dashboard interativo, é possível visualizar diversas análises estatísticas e realizar predições sobre a eficiência do sono com base em diferentes parâmetros.

## Funcionalidades

- **Visão Geral dos Dados**: Distribuição de idades e eficiência do sono por gênero.
- **Análises de Correlação**: Relações entre consumo de cafeína, duração do sono e eficiência do sono.
- **Mapa de Calor das Correlações**: Visualização das correlações entre todas as variáveis.
- **Predição de Eficiência do Sono**: Previsão da eficiência do sono com base em entradas fornecidas pelo usuário.
- **Avaliação do Modelo de Machine Learning**: Exibição das métricas de desempenho do modelo treinado.
- **Importância das Features**: Visualização das features mais importantes para a predição.

## Tecnologias Utilizadas

- **Backend**: Python, Dash, scikit-learn, statsmodels
- **Frontend**: Dash, Plotly
- **Manipulação de Dados**: pandas, numpy

## Instalação

1. **Clone o repositório**

    ```bash
    git clone https://github.com/seu_usuario/sleep_analysis_app.git
    cd sleep_analysis_app
    ```

2. **Crie e ative um ambiente virtual**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux/Mac
    # ou
    venv\Scripts\activate  # Para Windows
    ```

3. **Instale as dependências**

    ```bash
    pip install -r requirements.txt
    ```

4. **Execute a aplicação**

    ```bash
    python app.py
    ```

5. **Acesse no navegador**

    Abra o navegador e vá para `http://127.0.0.1:8050/`

## Uso

- **Exploração de Dados**: Navegue pelos diferentes gráficos para entender as distribuições e correlações dos dados.
- **Predição**: Preencha os campos do formulário com os parâmetros desejados e clique em "Prever Eficiência do Sono" para obter a previsão.
- **Avaliação do Modelo**: Veja as métricas de desempenho do modelo de Machine Learning na seção de avaliação.
- **Importância das Features**: Confira quais variáveis têm maior impacto na eficiência do sono.

## Contribuição

Sinta-se à vontade para abrir issues ou pull requests para contribuir com melhorias na aplicação.

## Licença

Este projeto está licenciado sob a Licença MIT.
