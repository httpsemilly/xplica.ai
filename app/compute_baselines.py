"""Gera app/baselines.json com as medianas de referencia exibidas na aplicacao.

Usa a mesma base da Secao 6.2 do artigo: o conjunto de TESTE, separado em
verdadeiras, falsas humanas e falsas sinteticas, e a mediana de cada
caracteristica linguistica (mesmas funcoes do core.py).

Uso (a partir da raiz do repositorio, com o ambiente virtual ativo):
    python app/compute_baselines.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import spacy

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from core import compute_linguistic_features  # noqa: E402

CSV_PATH = ROOT / 'data' / 'processed' / 'news_dataset.csv'
OUT_PATH = APP_DIR / 'baselines.json'

FEATURES = [
  'total_tokens', 'avg_sentence_length', 'propn_ratio', 'per_ratio', 'org_ratio',
  'adj_ratio', 'adv_ratio', 'emotiveness', 'stop_ratio', 'weekday_ratio',
]


def main():
  if not CSV_PATH.exists() or CSV_PATH.stat().st_size < 10_000:
    sys.exit(
      f'Dataset nao encontrado ou incompleto em {CSV_PATH}.\n'
      'Se o arquivo estiver no Git LFS, rode: git lfs pull'
    )

  nlp = spacy.load('pt_core_news_sm')
  df = pd.read_csv(CSV_PATH)
  test = df[df['split'] == 'test']

  groups = {
    'true': test[test['label'] == 'true'],
    'human_fake': test[(test['label'] == 'fake') & (test['origin'] == 'human')],
    'synthetic_fake': test[(test['label'] == 'fake') & (test['origin'] == 'synthetic')],
  }

  baselines = {}
  for name, group in groups.items():
    print(f'{name}: {len(group)} noticias...', flush=True)
    rows = []
    for i, text in enumerate(group['text'], start=1):
      features = compute_linguistic_features(nlp, str(text))
      if features:
        rows.append(features)
      if i % 100 == 0:
        print(f'  {i}/{len(group)}', flush=True)

    medians = pd.DataFrame(rows)[FEATURES].median()
    baselines[name] = {feature: round(float(medians[feature]), 6) for feature in FEATURES}

  baselines['_info'] = {
    'statistic': 'median',
    'split': 'test',
    'n': {name: int(len(group)) for name, group in groups.items()},
  }

  OUT_PATH.write_text(json.dumps(baselines, ensure_ascii=False, indent=2), encoding='utf-8')
  print(f'Salvo em {OUT_PATH}')


if __name__ == '__main__':
  main()
