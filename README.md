# Xplica.ai

Ferramenta para detecção explicável de fake news sintéticas (geradas por IA) em português brasileiro, desenvolvida como Trabalho de Conclusão de Curso (TCC).

## Descrição

Modelos de aprendizado de máquina treinados exclusivamente com fake news escritas por humanos apresentam queda de desempenho ao serem testados em notícias falsas geradas por Large Language Models (LLMs). O Xplica.ai investiga as causas dessa queda por meio de técnicas de Inteligência Artificial Explicável (XAI) aplicadas a um modelo BERTimbau treinado e avaliado em diferentes cenários de treino e teste (combinando dados de origem humana e sintética).

Além da investigação, o projeto entrega uma aplicação web simples que classifica uma notícia como verdadeira ou falsa, destaca no texto as palavras que mais influenciaram essa decisão (via LIME) e apresenta um perfil linguístico do texto (via spaCy), sinalizando características sintáticas associadas à queda de desempenho identificada na pesquisa para tornar a classificação transparente para o usuário final.

## Objetivo

- Avaliar a capacidade de generalização do BERTimbau entre fake news de origem humana e sintética em quatro cenários de treino e teste.
- Identificar, por meio de LIME (nível lexical) e spaCy (nível sintático/estrutural), quais características linguísticas e padrões textuais estão associados às classificações incorretas em cada cenário.
- Disponibilizar uma aplicação web que aplique o modelo treinado à verificação prática de notícias, exibindo ao usuário tanto a explicação lexical (LIME) quanto o perfil linguístico do texto (spaCy).

## Dados

- **Fake.Br Corpus** ([Monteiro et al., 2018](https://github.com/roneysco/Fake.br-Corpus)): 7.200 notícias em português brasileiro (3.600 verdadeiras, 3.600 falsas), de origem humana.
- **Fake.Br-LLM** ([Silva et al., 2025](https://github.com/renatosvmor/fake-news-llm-ptbr)): versão sintética do Fake.Br com notícias falsas geradas por LLM a partir das notícias verdadeiras originais.

## Tecnologias

- **Python 3.13**
- **Pandas / NumPy**: manipulação e preparo dos dados
- **Transformers (Hugging Face)**: fine-tuning do BERTimbau
- **PyTorch**: treinamento do modelo
- **scikit-learn**: métricas de avaliação
- **LIME**: interpretabilidade das predições do modelo (nível lexical)
- **spaCy**: análise linguística complementar (nível sintático/estrutural)
- **Streamlit**: interface web da aplicação

> Stack sujeita a ajustes conforme o desenvolvimento avança.

## Estrutura do projeto

```
xplica-ai/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_explainability.ipynb
├── src/
│   ├── data_loader.py
│   └── model.py
├── app/
│   └── app.py
├── models/
├── results/
├── requirements.txt
└── README.md
```

## Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/httpsemilly/xplica.ai
   cd xplica-ai
   ```

2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Execute a aplicação web:
   ```bash
   streamlit run src/app.py
   ```

## Status do projeto

🚧 Em desenvolvimento.

