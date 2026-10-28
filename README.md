# Xplica.ai

Ferramenta para detecção explicável de fake news sintéticas (geradas por IA) em português brasileiro, desenvolvida como Trabalho de Conclusão de Curso (TCC).

🔗 **Aplicação online:** https://httpsemilly--xplica-ai-interface.us-east.modal.run

## Descrição

Modelos de aprendizado de máquina treinados exclusivamente com fake news escritas por humanos apresentam queda de desempenho ao serem testados em notícias falsas geradas por Large Language Models (LLMs). O Xplica.ai investiga as causas dessa queda por meio de técnicas de Inteligência Artificial Explicável (XAI) aplicadas a um modelo BERTimbau treinado e avaliado em quatro cenários de treino e teste, que combinam dados de origem humana e sintética.

Além da investigação, o projeto disponibiliza uma aplicação web que classifica uma notícia como verdadeira ou falsa, destaca as palavras que mais influenciaram essa decisão (via LIME) e apresenta um perfil linguístico do texto (via spaCy).

## Objetivo

- Avaliar a capacidade de generalização do BERTimbau entre fake news de origem humana e sintética em quatro cenários de treino e teste.
- Identificar, por meio de LIME (nível lexical) e spaCy (nível sintático/estrutural), quais características linguísticas e padrões textuais estão associados às classificações incorretas.
- Disponibilizar uma aplicação web que aplique o modelo treinado à verificação prática de notícias, exibindo a explicação lexical (LIME) e o perfil linguístico (spaCy).

## Dados

- **Fake.Br Corpus** ([Monteiro et al., 2018](https://github.com/roneysco/Fake.br-Corpus)): 7.200 notícias em português brasileiro (3.600 verdadeiras, 3.600 falsas), de origem humana.
- **Fake.Br-LLM** ([Silva et al., 2025](https://github.com/renatosvmor/fake-news-llm-ptbr)): versão sintética do Fake.Br, com notícias falsas geradas por LLM (Sabiá-3) a partir das notícias verdadeiras originais.

O dataset consolidado usado nos experimentos (10.800 registros: 3.600 verdadeiras, 3.600 falsas humanas e 3.600 falsas sintéticas) está em `data/processed/news_dataset.csv`. Como o Fake.Br não possui divisão nativa entre treino e teste, cada notícia falsa humana herdou o *split* de sua correspondente sintética no Fake.Br-LLM.

## Cenários experimentais

| Cenário | Treino | Teste |
|---|---|---|
| 1 | Humano | Humano |
| 2 | Humano (mesmo modelo do Cenário 1) | Sintético |
| 3 | Misto (50% de falsas sintéticas) | Sintético |
| 4 | Misto (mesmo modelo do Cenário 3) | Humano |

Em todos os cenários, o conjunto de teste é balanceado, com 720 notícias verdadeiras e 720 falsas.

## Resultados

| Cenário | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| 1 | 0,9972 | 0,9945 | 1,0000 | 0,9972 |
| 2 | 0,7354 | 0,6540 | 1,0000 | 0,7908 |
| 3 | 0,9972 | 0,9986 | 0,9958 | 0,9972 |
| 4 | 0,9910 | 0,9862 | 0,9958 | 0,9910 |

Os notebooks foram versionados sem saídas. Os valores acima vêm de `results/scenario_results.json`, gerado pelo notebook `03_explainability.ipynb`.

### Principais achados da análise de interpretabilidade

A análise foi feita sobre os erros do Cenário 2, com o modelo treinado apenas com dados humanos:

- No Cenário 2, 381 notícias sintéticas foram classificadas incorretamente como verdadeiras; o treinamento misto (Cenário 3) recuperou o desempenho com custo reduzido sobre a detecção de notícias humanas (Cenário 4).
- O **tamanho do texto** é o fator mais associado ao erro: as notícias verdadeiras do corpus são muito mais longas (mediana de 932 *tokens*) do que as falsas humanas (150), e as sintéticas (348) ficam no meio.
- Mesmo comparando textos de tamanho semelhante, **marcadores de dia da semana**, **substantivos próprios**, **tamanho médio das sentenças** e **proporção de *stop words*** aumentam a chance de uma notícia sintética enganar o modelo.
- No experimento de remoção, apagar nomes de pessoas e organizações ou marcadores temporais reduziu a confiança do modelo muito mais do que apagar palavras aleatórias.

Os resultados detalhados de LIME e spaCy estão em `results/`.

## Aplicação web

A aplicação está disponível em **https://httpsemilly--xplica-ai-interface.us-east.modal.run**. O usuário cola o texto de uma notícia e recebe:

1. **Classificação**: probabilidade de a notícia ser falsa ou verdadeira, usando o modelo do Cenário 3 (treino misto).
2. **Explicação lexical (LIME)**: as 10 palavras de maior influência na decisão, calculadas com 1.000 perturbações do texto. Pesos negativos indicam evidência de notícia falsa; pesos positivos, de notícia verdadeira.
3. **Perfil linguístico (spaCy)**: as 10 características analisadas no artigo (número de *tokens*, tamanho médio das sentenças, substantivos próprios, entidades de pessoa e de organização, adjetivos, advérbios, emotividade, *stop words* e marcadores de dia da semana), comparadas às **medianas** do conjunto de teste para notícias verdadeiras, falsas humanas e falsas sintéticas.

### Arquitetura

```
Usuário ──► Interface (Gradio, CPU) ──► Análise (Modal, GPU T4) ◄── Modelo (Hugging Face Hub)
                  ▲                         │  BERTimbau + LIME + spaCy
                  └──── resultados ◄────────┘
```

- A **interface** e a **análise** rodam separadas na plataforma [Modal](https://modal.com), de computação *serverless*. A interface usa CPU; a análise usa GPU, porque o LIME exige centenas de passagens do modelo por texto.
- O modelo, o tokenizador e o pipeline do spaCy são carregados uma única vez, na inicialização do contêiner de análise, e reutilizados entre as requisições.
- Os contêineres são desligados automaticamente quando não há uso. Por isso, a primeira análise após um período parado pode levar de 30 segundos a 1 minuto.

### Arquivos da aplicação (`app/`)

| Arquivo | Função |
|---|---|
| `core.py` | Classificação, explicação (LIME) e características linguísticas (spaCy). |
| `modal_app.py` | Define a infraestrutura no Modal (GPU e interface Gradio) e monta a tela. |
| `compute_baselines.py` | Calcula as medianas de referência a partir do conjunto de teste. |
| `baselines.json` | Medianas geradas por `compute_baselines.py` e exibidas na tabela da aplicação. |

### Limitações de uso

- O resultado indica padrões de escrita associados a notícias falsas. **Não é uma checagem de fatos.**
- O modelo analisa no máximo 512 *tokens*; o restante do texto é ignorado, e a aplicação avisa quando isso acontece. O limite de entrada é de 20.000 caracteres.
- Textos muito curtos ou trechos de notícias tendem a ser classificados como falsos, já que o tamanho do texto é um dos fatores que mais pesam na decisão do modelo. Para melhores resultados, cole a notícia completa.

## Modelos treinados

Os modelos não estão no repositório por causa do tamanho. Eles ficam no Hugging Face Hub:

- Cenários 1 e 2: [xplica-bertimbau-scenario1](https://huggingface.co/httpsemilly/xplica-bertimbau-scenario1)
- Cenários 3 e 4: [xplica-bertimbau-scenario3](https://huggingface.co/httpsemilly/xplica-bertimbau-scenario3) (usado na aplicação web)

Para reproduzir sem acesso ao Hub, execute o `02_training.ipynb` (requer GPU). Ele salva os modelos em `models/`, e o `03_explainability.ipynb` os usa a partir dali.

## Tecnologias

- **Python 3.13**
- **Pandas / NumPy**: manipulação e preparo dos dados
- **Transformers (Hugging Face)** e **PyTorch**: fine-tuning do BERTimbau
- **scikit-learn**, **SciPy** e **statsmodels**: métricas de avaliação, testes estatísticos e regressão logística
- **LIME**: interpretabilidade das predições (nível lexical)
- **spaCy**: análise linguística (nível sintático/estrutural)
- **Gradio**: interface web
- **Modal**: hospedagem *serverless* da aplicação com GPU
- **Hugging Face Hub**: hospedagem dos modelos treinados

## Estrutura do projeto

```
xplica.ai/
├── data/
│   ├── raw/                  # corpora originais (Fake_Br e Fake.Br-LLM)
│   └── processed/            # news_dataset.csv consolidado
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb
│   └── 03_explainability.ipynb
├── results/                  # métricas dos cenários e saídas de LIME/spaCy
├── app/
│   ├── core.py
│   ├── modal_app.py
│   ├── compute_baselines.py
│   └── baselines.json
├── models/                   # (não versionados)
├── requirements.txt
└── README.md
```

## Como executar

Há três formas de usar o projeto, dependendo do objetivo.

### Opção 1: Local (com GPU)

Ideal para treinar do zero e reproduzir todos os resultados.

1. Clone o repositório:

```bash
git clone https://github.com/httpsemilly/xplica.ai
cd xplica.ai
```

2. Crie um ambiente virtual e instale as dependências:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

3. Execute os notebooks em ordem (`01` a `03`). Eles localizam a raiz do projeto automaticamente e podem ser abertos de qualquer pasta dentro do repositório.

### Opção 2: Google Colab

Recomendado para treinar quando não se tem GPU local. Os notebooks detectam o Colab automaticamente. Eles clonam o repositório (apenas a pasta `data/`), instalam as dependências e montam o Google Drive para persistir `models/` e `results/` entre sessões.

Ative a GPU em: Ambiente de execução → Alterar tipo → T4 GPU.

### Opção 3: Publicar a aplicação web no Modal

Não requer GPU local: o processamento acontece nos servidores do Modal.

1. Com o ambiente virtual da Opção 1 ativo, conecte sua conta do Modal (abre o navegador para login):

```bash
modal setup
```

2. (Opcional) Recalcule as medianas de referência, caso o dataset ou as características linguísticas mudem. O resultado é salvo em `app/baselines.json`:

```bash
python app/compute_baselines.py
```

3. Entre na pasta da aplicação e teste com um link temporário (encerre com `Ctrl+C`):

```bash
cd app
modal serve modal_app.py
```

4. Publique a versão definitiva, com link fixo:

```bash
modal deploy modal_app.py
```

Cada novo `modal deploy` substitui a versão anterior mantendo o mesmo link.

## Autoras

- Emilly Cavalcante Menezes dos Santos
- Maria Luiza Marins Barbosa

Centro Universitário Jorge Amado, Salvador, BA.

## Status do projeto

✅ Aplicação publicada e disponível online · 📝 Artigo em fase final de revisão
