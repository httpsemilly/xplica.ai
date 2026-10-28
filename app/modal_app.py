"""Xplica.ai no Modal.

Duas partes:
  1. Analisador (GPU T4): roda o core.py com o modelo do cenario 3
     (classificacao + LIME com 1000 amostras/10 palavras + spaCy).
  2. Interface (CPU): tela em Gradio que chama o Analisador.

Uso (dentro da pasta app/, com o ambiente virtual ativo):
  modal serve modal_app.py    -> link temporario para testar (Ctrl+C encerra)
  modal deploy modal_app.py   -> link fixo, publicado
"""
from pathlib import Path

import modal

PASTA = Path(__file__).parent
MODEL_REPO = "httpsemilly/xplica-bertimbau-scenario3"
TOKENIZER_NAME = "neuralmind/bert-base-portuguese-cased"
SPACY_WHL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "pt_core_news_sm-3.8.0/pt_core_news_sm-3.8.0-py3-none-any.whl"
)

app = modal.App("xplica-ai")


def baixar_modelos():
    """Roda uma vez, na montagem da imagem: deixa modelo e tokenizer salvos nela."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(TOKENIZER_NAME, do_lower_case=False)
    AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)


# ---------- Parte 1: analisador na GPU ----------
imagem_gpu = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.16.1",
        "lime==0.2.0.1",
        "spacy==3.8.16",
        "numpy>=2.1,<3",
        SPACY_WHL,
    )
    .run_function(baixar_modelos)
    .add_local_file(PASTA / "core.py", "/root/core.py")
)


@app.cls(image=imagem_gpu, gpu="T4", scaledown_window=120, timeout=600)
class Analisador:
    @modal.enter()
    def carregar(self):
        from core import Analyzer

        self.analyzer = Analyzer(model_repo=MODEL_REPO)

    @modal.method()
    def analisar(self, texto: str) -> dict:
        return self.analyzer.analyze(texto)


# ---------- Parte 2: interface Gradio ----------
imagem_web = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("gradio==6.30.0", "fastapi[standard]==0.143.2", "pandas>=2.2,<4")
    .add_local_file(PASTA / "baselines.json", "/root/baselines.json")
)

# (chave no core.py, rotulo, fator de exibicao, casas decimais)
# As 10 caracteristicas analisadas na Secao 6.2 do artigo.
METRICAS = [
    ("total_tokens", "Total de tokens", 1, 0),
    ("avg_sentence_length", "Palavras por frase", 1, 1),
    ("propn_ratio", "Substantivos próprios (%)", 100, 1),
    ("per_ratio", "Entidades: pessoas (%)", 100, 2),
    ("org_ratio", "Entidades: organizações (%)", 100, 2),
    ("adj_ratio", "Adjetivos (%)", 100, 1),
    ("adv_ratio", "Advérbios (%)", 100, 1),
    ("emotiveness", "Emotividade", 1, 3),
    ("stop_ratio", "Stop words (%)", 100, 1),
    ("weekday_ratio", "Dias da semana (%)", 100, 2),
]


@app.function(image=imagem_web, max_containers=1, scaledown_window=600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def interface():
    import json

    import gradio as gr
    import pandas as pd
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    base = json.loads(Path("/root/baselines.json").read_text(encoding="utf-8"))
    analisador = Analisador()

    def formatar(valor, fator, casas):
        valor = valor * fator
        return int(round(valor)) if casas == 0 else round(valor, casas)

    def analisar(texto):
        texto = (texto or "").strip()
        if len(texto) < 20:
            raise gr.Error("Cole uma notícia com pelo menos 20 caracteres.")
        try:
            r = analisador.analisar.remote(texto)
        except Exception as erro:
            raise gr.Error(f"Não foi possível analisar o texto: {erro}")

        falsa = round(r["prob_fake"] * 100)
        verdadeira = round(r["prob_true"] * 100)
        titulo = "Provavelmente falsa" if r["prob_fake"] >= 0.5 else "Provavelmente verdadeira"
        resumo = f"## {titulo}\nFalsa: {falsa}% · Verdadeira: {verdadeira}%"
        if r["truncated"]:
            resumo += (
                f"\n\n⚠️ O texto tem {r['n_tokens']} tokens. "
                "O modelo analisou apenas os primeiros 512."
            )

        probs = {"falsa": r["prob_fake"], "verdadeira": r["prob_true"]}

        lime = sorted(r["lime"], key=lambda item: item["weight"])
        lime_df = pd.DataFrame(
            {
                "palavra": [item["word"] for item in lime],
                "influência": [round(item["weight"], 4) for item in lime],
            }
        )

        ling = r["linguistic"]
        linhas = []
        for chave, rotulo, fator, casas in METRICAS:
            linhas.append(
                {
                    "Métrica": rotulo,
                    "Seu texto": formatar(ling.get(chave, 0), fator, casas),
                    "Verdadeiras": formatar(base["true"][chave], fator, casas),
                    "Falsas humanas": formatar(base["human_fake"][chave], fator, casas),
                    "Falsas sintéticas": formatar(base["synthetic_fake"][chave], fator, casas),
                }
            )
        ling_df = pd.DataFrame(linhas)

        return resumo, probs, lime_df, ling_df

    with gr.Blocks(title="Xplica.ai") as demo:
        gr.Markdown(
            "# 🔍 Xplica.ai\n"
            "Detecção explicável de possíveis fake news em português: "
            "BERTimbau + LIME + spaCy. Texto truncado em 512 tokens.\n\n"
            "*O resultado indica padrões de escrita associados a notícias falsas. "
            "Não é uma checagem de fatos.*"
        )
        entrada = gr.Textbox(
            label="Cole a notícia para analisar",
            lines=8,
            max_lines=20,
            placeholder="Cole aqui o texto da notícia (mín. 20 caracteres)…",
        )
        botao = gr.Button("Analisar", variant="primary")
        saida_resumo = gr.Markdown()
        saida_probs = gr.Label(label="Probabilidades")
        gr.Markdown(
            "### Palavras de influência (LIME)\n"
            "Score negativo = evidência de fake · positivo = evidência de verdadeira."
        )
        saida_lime = gr.BarPlot(x="palavra", y="influência")
        gr.Markdown(
            "### Perfil linguístico (spaCy)\n"
            "Seu texto comparado às **medianas** do conjunto de teste para notícias "
            "verdadeiras, falsas humanas e falsas sintéticas."
        )
        saida_ling = gr.Dataframe()
        botao.click(
            analisar,
            inputs=entrada,
            outputs=[saida_resumo, saida_probs, saida_lime, saida_ling],
        )

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
