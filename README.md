# Xplica.ai

Ferramenta para detecção explicável de fake news sintéticas (geradas por IA) em português brasileiro, desenvolvida como Trabalho de Conclusão de Curso (TCC).

## Descrição

Modelos de aprendizado de máquina treinados exclusivamente com fake news escritas por humanos apresentam queda de desempenho ao serem testados em notícias falsas geradas por Large Language Models (LLMs). O Xplica.ai investiga as causas dessa queda por meio de técnicas de Inteligência Artificial Explicável (XAI) aplicadas a um modelo BERTimbau treinado e avaliado em quatro cenários de treino e teste, que combinam dados de origem humana e sintética.

Além da investigação, o projeto prevê uma aplicação web simples que classifica uma notícia como verdadeira ou falsa, destaca no texto as palavras que mais influenciaram essa decisão (via LIME) e apresenta um perfil linguístico do texto (via spaCy).

## Objetivo

- Avaliar a capacidade de generalização do BERTimbau entre fake news de origem humana e sintética em quatro cenários de treino e teste.
- Identificar, por meio de LIME (nível lexical) e spaCy (nível sintático/estrutural), quais características linguísticas e padrões textuais estão associados às classificações incorretas.
- Disponibilizar uma aplicação web que aplique o modelo treinado à verificação prática de notícias, exibindo a explicação lexical (LIME) e o perfil linguístico (spaCy).

## Dados

- **Fake.Br Corpus** ([Monteiro et al., 2018](https://github.com/roneysco/Fake.br-Corpus)): 7.200 notícias em português brasileiro (3.600 verdadeiras, 3.600 falsas), de origem humana.
- **Fake.Br-LLM** ([Silva et al., 2025](https://github.com/renatosvmor/fake-news-llm-ptbr)): versão sintética do Fake.Br, com notícias falsas geradas por LLM a partir das notícias verdadeiras originais.

O dataset consolidado usado nos experimentos (10.800 registros) está em `data/processed/news_dataset.csv`.

## Cenários experimentais

| Cenário | Treino | Teste |
|---|---|---|
| 1 | Humano | Humano |
| 2 | Humano (mesmo modelo do Cenário 1) | Sintético |
| 3 | Misto (50% de falsas sintéticas) | Sintético |
| 4 | Misto (mesmo modelo do Cenário 3) | Humano |

## Resultados

| Cenário | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| 1 | 0,9972 | 0,9945 | 1,0000 | 0,9972 |
| 2 | 0,7354 | 0,6540 | 1,0000 | 0,7908 |
| 3 | 0,9972 | 0,9986 | 0,9958 | 0,9972 |
| 4 | 0,9910 | 0,9862 | 0,9958 | 0,9910 |

Os notebooks foram versionados sem saídas. Os valores acima vêm de `results/scenario_results.json`, gerado pelo notebook `03_explainability.ipynb`.

## Tecnologias

- **Python 3.13**
- **Pandas / NumPy**: manipulação e preparo dos dados
- **Transformers (Hugging Face)** e **PyTorch**: fine-tuning do BERTimbau
- **scikit-learn** e **SciPy**: métricas de avaliação e teste qui-quadrado
- **LIME**: interpretabilidade das predições (nível lexical)
- **spaCy**: análise linguística (nível sintático/estrutural)
- **Streamlit** e **Modal**: interface web e serviço de inferência

## Estrutura do projeto

```
xplica.ai/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb
│   └── 03_explainability.ipynb
├── results/
├── app/
├── models/ # (não versionados)
├── requirements.txt
└── README.md
```

## Modelos treinados

Os modelos não estão no repositório por causa do tamanho. Eles ficam no Hugging Face Hub:

- Cenários 1 e 2: [xplica-bertimbau-scenario1](https://huggingface.co/httpsemilly/xplica-bertimbau-scenario1)
- Cenários 3 e 4: [xplica-bertimbau-scenario3](https://huggingface.co/httpsemilly/xplica-bertimbau-scenario3)

Para reproduzir sem acesso ao Hub, execute o `02_training.ipynb` (requer GPU). Ele salva os modelos em `models/`, e o `03_explainability.ipynb` os usa a partir dali.

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

3. Execute os notebooks em ordem (`01` a `03`). Eles localizam a raiz do projeto automaticamente e podem ser abertos de qualquer pasta dentro do repositório.

## Status do projeto

🚧 Em desenvolvimento
