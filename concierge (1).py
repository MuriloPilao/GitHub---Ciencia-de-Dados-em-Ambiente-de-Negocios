# -*- coding: utf-8 -*-
"""
🍽️ Concierge · recomendador diário de onde comer (CRISP-DM)

Projeto final de Ciência de Dados em Ambiente de Negócios (FGV RI, 2026.2)
Grupo: Fabrício Wenter, Gabriel Gandolfi, Luís Gustavo Barbosa e Murilo Pilão

Este arquivo junta o projeto inteiro, que antes estava espalhado em notebooks 03a, 03b e 04,
src/concierge/recomendador.py, app/app.py, scripts/ e tests/. Todas as bases de dados ficam num
arquivo só, que deve estar na MESMA pasta deste script:

    concierge_dados.sqlite

---------------------------------------------------------------------------------------------------
1. INSTALAR (uma vez, Python 3.11 ou mais novo; o 3.10 não serve)

    pip install scikit-learn==1.9.1 numpy pandas scipy matplotlib joblib streamlit

   Só o scikit-learn tem versão fixa: os modelos do banco foram treinados com o 1.9.1. As demais podem ser as que
   o pip escolher (testado com pandas 2.2 e 3.x, numpy 2.4 e 2.5). Atenção: o requirements.txt antigo fixava
   numpy==2.5.3 e scipy==1.18.1, que só instalam no Python 3.12+; era por isso que a instalação falhava no 3.11.

2. RODAR EM PARTES (no terminal, dentro da pasta do script)

    python concierge.py            mostra o que tem no banco e o que dá para rodar
    python concierge.py parte1     Data Understanding, Preparation e modelos A e B   (antigo 03a)
    python concierge.py parte2     modelo com a nota individual e RMSE               (antigo 03b)
    python concierge.py parte3     Evaluation, Responsible AI e teste A/B            (antigo 04)
    python concierge.py testes     testes do recomendador
    python concierge.py tudo       partes 1, 2 e 3 e os testes, nessa ordem
    python concierge.py app        abre o protótipo no navegador (o mesmo que: streamlit run concierge.py)

   Figuras e tabelas de cada parte vão para a pasta saidas/ (criada ao lado do script). As tabelas
   também ficam no banco, e é de lá que uma parte lê o que a anterior gerou.

3. DADOS DO YELP

    python concierge.py importar PASTA

   Lê os JSON do Yelp Open Dataset (https://www.yelp.com/dataset) que estiverem em PASTA e guarda
   dentro do banco só os restaurantes de Philadelphia: business, tip, checkin e, se existir, review
   (o arquivo de 5 GB ou o review_philadelphia.json.gz). A leitura é linha a linha, com pouca memória.
   Precisa ser feito uma vez só; depois disso o banco basta.

---------------------------------------------------------------------------------------------------
O QUE TEM NO BANCO (concierge_dados.sqlite)

  restaurantes      5.855 restaurantes de Philadelphia, prontos para o app (gerado pela parte 1 com o Yelp real)
  modelos           modelos A e B treinados com o Yelp real, guardados dentro do banco
  p1_*, p3_*        tabelas de resultado das partes 1 e 3 (a parte 3 lê as da parte 1)
  eventos_app       log do app: exibição, clique e nota (é o que o teste A/B analisa)
  yelp_business, yelp_tip, yelp_checkin, yelp_review
                    JSON do Yelp, só restaurantes de Philadelphia; vazias até rodar "importar"
  meta              de onde vieram os dados e quando

A parte 3 e o app rodam direto com os dados reais que já estão no banco. As partes 1 e 2 precisam
das tabelas yelp_*; enquanto elas estiverem vazias, essas partes rodam numa amostra sintética no
mesmo esquema do Yelp, gerada na hora (os números então só validam o código). Os resultados de uma
rodada sintética ficam com o prefixo "sint_" e nunca sobrescrevem os resultados reais.

Os termos do Yelp restringem a redistribuição dos dados: o banco tem nomes e coordenadas do Yelp,
então o repositório deve ser privado (a professora entra como colaboradora).

Também dá para rodar em partes num editor com células (VS Code, Spyder, PyCharm): cada bloco começa
com "# %%"; rode o bloco 0 e depois chame, por exemplo, parte3(conectar()).
"""
# %% 0. Configuração ===============================================================================
from __future__ import annotations

import subprocess
import sys


def _instalar_dependencias():
    """Instala sozinho, no Python que está rodando este script, as bibliotecas que faltarem."""
    import importlib.util
    from importlib import metadata
    faltando = [pacote for modulo, pacote in [("joblib", "joblib"), ("numpy", "numpy"), ("pandas", "pandas"),
                                              ("scipy", "scipy"), ("matplotlib", "matplotlib")]
                if importlib.util.find_spec(modulo) is None]
    try:
        if metadata.version("scikit-learn") != "1.9.1":
            faltando.append("scikit-learn==1.9.1")
    except metadata.PackageNotFoundError:
        faltando.append("scikit-learn==1.9.1")
    if not faltando:
        return
    print(f"Instalando bibliotecas que faltam ({', '.join(faltando)}). Só acontece na primeira vez...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *faltando])
    except Exception:
        sys.exit("Não consegui instalar sozinho. Rode no terminal:\n  "
                 f'"{sys.executable}" -m pip install scikit-learn==1.9.1 numpy pandas scipy matplotlib joblib streamlit')


if "streamlit" not in sys.modules:
    _instalar_dependencias()
for _fluxo in (sys.stdout, sys.stderr):           # acentos e símbolos no terminal do Windows
    try:
        _fluxo.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import ast
import gzip
import io
import json
import math
import os
import sqlite3
import time
import uuid
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import sklearn

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)

SEED = 42
CIDADE = "Philadelphia"
try:
    PASTA_SCRIPT = Path(__file__).resolve().parent
except NameError:                                   # rodando célula a célula num console
    PASTA_SCRIPT = Path.cwd()
BANCO_PADRAO = PASTA_SCRIPT / "concierge_dados.sqlite"
SAIDAS_PADRAO = PASTA_SCRIPT / "saidas"
MOSTRAR_FIGURAS = hasattr(sys, "ps1") or "ipykernel" in sys.modules   # num console as figuras aparecem


def cab(texto):
    print("\n" + "=" * 100 + f"\n{texto}\n" + "=" * 100)


def mostrar(obj, titulo=None):
    if titulo:
        print(f"\n{titulo}")
    print(obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else obj)


def pasta_saida(base, nome):
    p = Path(base or SAIDAS_PADRAO) / nome
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fig(pasta, nome, dpi=120):
    plt.tight_layout()
    plt.savefig(Path(pasta) / nome, dpi=dpi)
    if MOSTRAR_FIGURAS:
        plt.show()
    plt.close("all")


# %% 1. Banco de dados: um arquivo SQLite com todas as bases =======================================
ESQUEMA_YELP = {
    "yelp_business": ["business_id TEXT", "name TEXT", "address TEXT", "city TEXT", "state TEXT",
                      "postal_code TEXT", "latitude REAL", "longitude REAL", "stars REAL", "review_count INTEGER",
                      "is_open INTEGER", "attributes TEXT", "categories TEXT", "hours TEXT"],
    "yelp_tip": ["user_id TEXT", "business_id TEXT", "text TEXT", "date TEXT", "compliment_count INTEGER"],
    "yelp_checkin": ["business_id TEXT", "date TEXT"],
    "yelp_review": ["review_id TEXT", "user_id TEXT", "business_id TEXT", "stars REAL", "date TEXT"],
}
CAMPOS_EVENTOS = ["ts", "sessao", "variante", "origem", "evento", "posicao", "business_id", "n_itens", "nota",
                  "modo", "bairro", "culinarias", "teto", "raio_km"]


def conectar(banco=None):
    """Abre o banco (concierge_dados.sqlite ao lado do script, ou o caminho dado) e cria as tabelas fixas."""
    banco = Path(banco or os.environ.get("CONCIERGE_BANCO") or BANCO_PADRAO)
    con = sqlite3.connect(banco)
    criar_tabelas(con)
    return con


def criar_tabelas(con):
    for nome, cols in ESQUEMA_YELP.items():
        con.execute(f"CREATE TABLE IF NOT EXISTS {nome} ({', '.join(cols)})")
    con.execute("CREATE TABLE IF NOT EXISTS meta (chave TEXT PRIMARY KEY, valor TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS modelos (nome TEXT PRIMARY KEY, sklearn TEXT, criado_em TEXT, "
                "bytes INTEGER, blob BLOB)")
    con.execute("CREATE TABLE IF NOT EXISTS _indices (tabela TEXT PRIMARY KEY, colunas TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS eventos_app (ts TEXT, sessao TEXT, variante TEXT, origem TEXT, "
                "evento TEXT, posicao INTEGER, business_id TEXT, n_itens INTEGER, nota REAL, modo TEXT, "
                "bairro TEXT, culinarias TEXT, teto INTEGER, raio_km REAL)")
    con.commit()


def existe(con, tabela):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (tabela,)).fetchone() is not None


def n_linhas(con, tabela):
    return con.execute(f'SELECT COUNT(*) FROM "{tabela}"').fetchone()[0] if existe(con, tabela) else 0


def meta(con, chave, padrao=None):
    if not existe(con, "meta"):
        return padrao
    r = con.execute("SELECT valor FROM meta WHERE chave=?", (chave,)).fetchone()
    return r[0] if r else padrao


def gravar_meta(con, **valores):
    con.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in valores.items()])
    con.commit()


def prefixo(fonte):
    """Resultados de dados reais ficam sem prefixo; os de amostra sintética, com 'sint_'."""
    return "" if fonte == "real" else "sint_"


def salvar_tabela(con, nome, df, pasta=None):
    """Grava uma tabela de resultado no banco (com o índice) e, se pasta for dada, também em CSV."""
    d = df.to_frame() if isinstance(df, pd.Series) else df.copy()
    indices = []
    if not isinstance(d.index, pd.RangeIndex) or any(d.index.names):
        d.index = d.index.set_names([n or "chave" for n in d.index.names])
        indices = list(d.index.names)
        d = d.reset_index()
    d.to_sql(nome, con, if_exists="replace", index=False)
    con.execute("INSERT OR REPLACE INTO _indices VALUES (?, ?)", (nome, json.dumps(indices)))
    con.commit()
    if pasta is not None:
        df.to_csv(Path(pasta) / f"{nome}.csv", float_format="%.3f", index=bool(indices))


def ler_tabela(con, nome):
    if not existe(con, nome):
        raise LookupError(f"A tabela '{nome}' não está no banco. Rode antes a parte que a gera (ver: python concierge.py).")
    d = pd.read_sql(f'SELECT * FROM "{nome}"', con)
    r = con.execute("SELECT colunas FROM _indices WHERE tabela=?", (nome,)).fetchone()
    indices = json.loads(r[0]) if r else []
    if indices:
        d = d.set_index(indices)
        if indices == ["chave"]:
            d.index.name = None
    return d


def salvar_modelo(con, nome, art):
    buf = io.BytesIO()
    joblib.dump(art, buf, compress=3)
    blob = buf.getvalue()
    con.execute("INSERT OR REPLACE INTO modelos VALUES (?, ?, ?, ?, ?)",
                (nome, art.get("sklearn", sklearn.__version__), datetime.now().isoformat(timespec="seconds"),
                 len(blob), sqlite3.Binary(blob)))
    con.commit()
    return len(blob)


def carregar_modelo(con, nome):
    r = con.execute("SELECT blob, sklearn FROM modelos WHERE nome=?", (nome,)).fetchone()
    if r is None:
        raise LookupError(f"O modelo '{nome}' não está no banco. Rode a parte 1 antes.")
    if r[1] and r[1] != sklearn.__version__:
        print(f"ATENÇÃO: o modelo '{nome}' foi treinado com scikit-learn {r[1]} e aqui está o {sklearn.__version__}. "
              f"Se der erro, rode: pip install scikit-learn=={r[1]}")
    return joblib.load(io.BytesIO(r[0]))


def salvar_restaurantes(con, nome, tab):
    t = tab.copy()
    t["culinarias"] = t["culinarias"].apply(lambda cs: json.dumps([str(c) for c in cs]))
    t.to_sql(nome, con, if_exists="replace", index=False)
    con.commit()


def ler_restaurantes(con, nome):
    if not existe(con, nome):
        raise LookupError(f"A tabela '{nome}' não está no banco. Rode a parte 1 antes.")
    t = pd.read_sql(f'SELECT * FROM "{nome}"', con)
    t["culinarias"] = t["culinarias"].apply(json.loads)
    return t


# ---- Yelp: importar os JSON reais (ou gravar a amostra sintética) no mesmo esquema
def _eh_restaurante_philly(b):
    return (b.get("city") or "").strip().casefold() == CIDADE.casefold() and "Restaurants" in (b.get("categories") or "")


def _em_lotes(con, tabela, linhas, tamanho=50_000):
    cols = [c.split()[0] for c in ESQUEMA_YELP[tabela]]
    sql = f"INSERT INTO {tabela} VALUES ({', '.join('?' * len(cols))})"
    lote, n = [], 0
    for r in linhas:
        lote.append(tuple(r.get(c) for c in cols))
        if len(lote) >= tamanho:
            con.executemany(sql, lote); n += len(lote); lote = []
    if lote:
        con.executemany(sql, lote); n += len(lote)
    con.commit()
    return n


def gravar_yelp(con, business, tips, checkins, reviews=(), fonte="real", aviso=print):
    """Guarda no banco só os restaurantes de Philadelphia e as dicas, check-ins e avaliações deles.
    Cada argumento é um iterável de dicionários (linhas do JSON do Yelp)."""
    for t in ESQUEMA_YELP:
        con.execute(f"DELETE FROM {t}")
    ids, tot = set(), Counter()

    def biz():
        for b in business:
            tot["negocios"] += 1
            if "Restaurants" in (b.get("categories") or ""):
                tot["restaurantes"] += 1
            if _eh_restaurante_philly(b):
                ids.add(b["business_id"])
                yield {**b, "attributes": json.dumps(b.get("attributes")), "hours": json.dumps(b.get("hours"))}

    def filtra(linhas, chave):
        for r in linhas:
            tot[chave] += 1
            if r["business_id"] in ids:
                yield r

    n_b = _em_lotes(con, "yelp_business", biz())
    aviso(f"  business: {tot['negocios']:,} negócios lidos -> {n_b:,} restaurantes de {CIDADE}")
    n_t = _em_lotes(con, "yelp_tip", filtra(tips, "tips"))
    aviso(f"  tip:      {tot['tips']:,} dicas lidas -> {n_t:,}")
    n_c = _em_lotes(con, "yelp_checkin", filtra(checkins, "checkins"))
    aviso(f"  checkin:  {tot['checkins']:,} linhas lidas -> {n_c:,}")
    n_r = _em_lotes(con, "yelp_review", ({k: r[k] for k in ("review_id", "user_id", "business_id", "stars", "date")}
                                          for r in filtra(reviews, "reviews")))
    aviso(f"  review:   {tot['reviews']:,} avaliações lidas -> {n_r:,}" if tot["reviews"] else
          "  review:   arquivo não encontrado (só a parte 2 usa; ela roda na amostra sintética)")
    gravar_meta(con, fonte_yelp=fonte, yelp_importado_em=datetime.now().isoformat(timespec="seconds"),
                n_negocios_yelp=tot["negocios"], n_restaurantes_yelp=tot["restaurantes"],
                n_tips_yelp=tot["tips"], n_checkin_yelp=tot["checkins"], n_reviews_yelp=tot["reviews"])
    return {"business": n_b, "tip": n_t, "checkin": n_c, "review": n_r}


def _linhas_json(arq):
    abrir = gzip.open if str(arq).endswith(".gz") else open
    with abrir(arq, "rt", encoding="utf-8") as f:
        for linha in f:
            if linha.strip():
                yield json.loads(linha)


def _linhas_review(arq, ids_ok):
    """Lê o review.json (≈5 GB) sem decodificar as linhas de outros negócios (truque do antigo filtrar_reviews.py)."""
    abrir = gzip.open if str(arq).endswith(".gz") else open
    chave = '"business_id"'
    with abrir(arq, "rt", encoding="utf-8") as f:
        for linha in f:
            i = linha.find(chave)
            if i < 0:
                continue
            j = linha.find('"', linha.find(":", i + len(chave)) + 1) + 1
            if linha[j: linha.find('"', j)] in ids_ok():
                yield json.loads(linha)
            else:
                yield {"business_id": None}          # conta como lida, mas não entra


def importar_yelp(con, pasta):
    """Importa os JSON do Yelp Open Dataset de `pasta` para o banco (só restaurantes de Philadelphia)."""
    pasta = Path(pasta)
    arq = {t: pasta / f"yelp_academic_dataset_{t}.json" for t in ("business", "tip", "checkin", "review")}
    faltando = [a.name for t, a in arq.items() if t != "review" and not a.exists()]
    if faltando:
        raise FileNotFoundError(f"Faltam em {pasta}: {', '.join(faltando)}")
    filtrado = pasta / "review_philadelphia.json.gz"
    arq_rev = filtrado if filtrado.exists() else (arq["review"] if arq["review"].exists() else None)
    t0 = time.time()
    print(f"Importando o Yelp de {pasta} (pode levar alguns minutos com os arquivos completos)...")
    ids_cache = {}

    def ids_ok():
        if "ids" not in ids_cache:
            ids_cache["ids"] = {r[0] for r in con.execute("SELECT business_id FROM yelp_business")}
        return ids_cache["ids"]

    reviews = _linhas_review(arq_rev, ids_ok) if arq_rev else ()
    n = gravar_yelp(con, _linhas_json(arq["business"]), _linhas_json(arq["tip"]), _linhas_json(arq["checkin"]),
                    reviews, fonte="real")
    con.execute("VACUUM")
    print(f"Pronto em {time.time() - t0:.0f}s: {n}")
    return n


# ---- Amostra sintética no esquema do Yelp (antigo scripts/gerar_yelp_teste.py)
BAIRROS = {
    "Center City": (39.9526, -75.1652), "Old City": (39.9522, -75.1440), "Fishtown": (39.9727, -75.1340),
    "South Philly": (39.9245, -75.1690), "University City": (39.9522, -75.1932),
    "Northern Liberties": (39.9637, -75.1406), "Chinatown": (39.9555, -75.1550), "Manayunk": (40.0262, -75.2245),
}


def gerar_yelp_sintetico(n_rest=900, n_users=2500, seed=7):
    """Linhas no MESMO ESQUEMA do Yelp Open Dataset, só para testar o código. Não são dados reais."""
    culinarias = ["Pizza", "American (New)", "American (Traditional)", "Italian", "Chinese", "Mexican", "Japanese",
                  "Sandwiches", "Breakfast & Brunch", "Burgers", "Thai", "Seafood", "Vietnamese", "Indian"]
    dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pos = ["great", "love", "delicious", "amazing", "best", "fresh", "friendly"]
    neg = ["bad", "slow", "cold", "overpriced", "rude", "bland"]
    rng = np.random.default_rng(seed)

    def ts(ini, fim):
        s = rng.integers(np.datetime64(ini, "s").astype(int), np.datetime64(fim, "s").astype(int))
        return str(np.datetime64(int(s), "s")).replace("T", " ")

    qualidade = rng.normal(0, 0.6, n_rest)
    pop = (rng.pareto(1.4, n_rest) + 0.05) * np.exp(0.6 * qualidade)
    cul_rest, biz = [], []
    for i in range(n_rest):
        bairro = rng.choice(list(BAIRROS))
        lat0, lon0 = BAIRROS[bairro]
        cs = list(rng.choice(culinarias, size=rng.integers(1, 3), replace=False))
        preco = int(rng.integers(1, 5))
        cul_rest.append(cs)
        attrs = {
            "RestaurantsPriceRange2": str(preco) if rng.random() > 0.15 else "None",
            "RestaurantsDelivery": rng.choice(["True", "False", "None"], p=[.6, .35, .05]),
            "RestaurantsTakeOut": rng.choice(["True", "False"]),
            "RestaurantsReservations": "True" if preco >= 3 or rng.random() > .7 else "False",
            "OutdoorSeating": rng.choice(["True", "False"]),
            "GoodForKids": rng.choice(["True", "False"]),
            "RestaurantsGoodForGroups": rng.choice(["True", "False"]),
            "HasTV": rng.choice(["True", "False"]),
            "BusinessAcceptsCreditCards": "True",
            "RestaurantsTableService": rng.choice(["True", "False"]),
            "Alcohol": rng.choice(["u'none'", "'beer_and_wine'", "u'full_bar'"]),
            "NoiseLevel": rng.choice(["u'quiet'", "u'average'", "'loud'"]),
            "WiFi": rng.choice(["u'free'", "'no'", "None"]),
            "Ambience": "{'romantic': False, 'casual': %s, 'classy': %s, 'trendy': False}"
                        % (rng.random() > .3, rng.random() > .7),
            "GoodForMeal": "{'lunch': True, 'dinner': %s, 'brunch': False}" % (rng.random() > .2),
            "BusinessParking": "{'street': True, 'lot': %s, 'garage': False, 'valet': False}" % (rng.random() > .6),
        }
        attrs = {k: str(v) for k, v in attrs.items()}
        if rng.random() < 0.1:
            attrs = None
        horas = None if rng.random() < 0.15 else {d: str(rng.choice(["11:0-22:0", "7:0-15:0", "17:0-2:0"])) for d in dias}
        stars = float(np.clip(np.round((3.7 + qualidade[i] + rng.normal(0, .3)) * 2) / 2, 1, 5))
        biz.append({"business_id": f"B{i:05d}", "name": f"Restaurante Teste {i}", "address": f"{i} Test St",
                    "city": "Philadelphia" if rng.random() > .02 else "Philadelphia ", "state": "PA",
                    "postal_code": "19107", "latitude": float(lat0 + rng.normal(0, .006)),
                    "longitude": float(lon0 + rng.normal(0, .006)), "stars": stars,
                    "review_count": int(5 + pop[i] * 40), "is_open": int(rng.random() > .2),
                    "attributes": attrs, "categories": ", ".join(cs + ["Restaurants"]), "hours": horas})
    for j in range(40):            # negócios que não são restaurantes, para o filtro ter o que filtrar
        biz.append({"business_id": f"X{j:04d}", "name": f"Loja {j}", "address": "", "city": "Philadelphia",
                    "state": "PA", "postal_code": "19107", "latitude": 39.95, "longitude": -75.16, "stars": 4.0,
                    "review_count": 10, "is_open": 1, "attributes": None, "categories": "Shopping", "hours": None})
    afin = rng.dirichlet(np.full(len(culinarias), 0.4), n_users)
    idx = {c: k for k, c in enumerate(culinarias)}
    tips, reviews, checkins = [], [], []
    for u in range(n_users):
        k = int(np.clip(rng.zipf(1.8), 1, 60))
        gosto = np.array([max(afin[u, idx[c]] for c in cs) for cs in cul_rest])
        p = pop ** 0.7 * (0.02 + gosto) ** 1.3
        for b in rng.choice(n_rest, size=min(k, n_rest), replace=False, p=p / p.sum()):
            nota = int(np.clip(np.round(3.8 + qualidade[b] + 4 * (gosto[b] - 0.07) + rng.normal(0, .8)), 1, 5))
            data = ts("2010-01-01", "2022-01-15")
            reviews.append({"review_id": f"R{len(reviews):07d}", "user_id": f"U{u:05d}", "business_id": f"B{b:05d}",
                            "stars": float(nota), "date": data})
            if rng.random() < 0.6:
                palavra = rng.choice(pos if nota >= 4 else neg)
                tips.append({"user_id": f"U{u:05d}", "business_id": f"B{b:05d}", "text": f"The food was {palavra}.",
                             "date": data, "compliment_count": int(rng.poisson(.2))})
    reviews += reviews[:30]        # duplicatas de propósito
    for i in range(n_rest):
        n = int(np.clip(pop[i] * 60, 1, 3000))
        checkins.append({"business_id": f"B{i:05d}",
                         "date": ", ".join(sorted(ts("2010-06-01", "2022-01-15") for _ in range(n)))})
    return {"business": biz, "tip": tips, "checkin": checkins, "review": reviews}


_SINTETICO = {}


def abrir_yelp(con, precisa=("business", "tip", "checkin")):
    """Devolve (conexão com as tabelas yelp_*, fonte). Usa o banco se as tabelas tiverem dados; senão, monta a
    amostra sintética num banco em memória, com o mesmo código de importação."""
    if all(n_linhas(con, f"yelp_{t}") > 0 for t in precisa):
        return con, meta(con, "fonte_yelp", "real")
    print(f"As tabelas {', '.join('yelp_' + t for t in precisa)} estão vazias no banco "
          "(rode 'python concierge.py importar PASTA' com os JSON do Yelp).")
    print("Rodando com a AMOSTRA SINTÉTICA no esquema do Yelp: os números só validam o código.")
    if "con" not in _SINTETICO:
        mem = sqlite3.connect(":memory:")
        criar_tabelas(mem)
        s = gerar_yelp_sintetico()
        gravar_yelp(mem, s["business"], s["tip"], s["checkin"], s["review"], fonte="sintetico", aviso=lambda *_: None)
        _SINTETICO["con"] = mem
    return _SINTETICO["con"], "sintetico"


def ler_business(con):
    biz = pd.read_sql("SELECT * FROM yelp_business ORDER BY rowid", con)
    for c in ("attributes", "hours"):
        biz[c] = biz[c].apply(lambda s: json.loads(s) if isinstance(s, str) else None)
    return biz


# %% 2. Recomendador: a regra do top 5 (usada pela parte 1, pela parte 3 e pelo app) ================
"""
Etapas da variante "concierge":
1. Filtros de negócio: aberto, faz delivery (pedir) ou aceita reserva (sair), preço até o teto, culinária escolhida,
   dentro do raio.
2. Guarda de qualidade (contra o falso positivo, o erro caro do BU):
   - restaurante com pelo menos MIN_AVALIACOES avaliações: nota atual >= 4 (a nota de hoje é legítima na hora de
     recomendar; ela só não pode entrar no treino de um modelo que prevê o passado);
   - restaurante novo, com poucas avaliações: probabilidade do modelo A >= limiar de A.
3. Ordenação pelo gosto: o modelo B estima a chance de a pessoa escolher o restaurante e ordena os candidatos. Só
   entra quem passa de um piso: FATOR_LIMIAR_B x o limiar de B escolhido por F0,5 na parte 1. O piso é menor que o
   limiar cheio porque o falso positivo caro (casa mal avaliada) já é barrado na etapa 2; com o limiar cheio, 26% das
   respostas ficavam sem nenhuma recomendação (parte 3, seção 3.1). Se menos de `top` passarem, a lista fica menor.
4. Vaga de descoberta (mitigação do viés para populares): se nenhum item da lista estiver na metade de menor movimento
   da cidade, a última vaga vai para o restaurante de menor movimento com maior chance de escolha que passou na guarda
   de qualidade e no piso de B. Ele aparece marcado como "descoberta".

A variante "controle" reproduz a ordenação atual dos apps (mesmos filtros, ordenada por movimento e nota, sem modelo).
É o braço de controle do teste A/B.
"""
MIN_AVALIACOES = 20          # abaixo disso a nota média é instável e quem decide é o modelo A
FATOR_LIMIAR_B = 0.5         # piso do modelo B = 50% do limiar F0,5 (escolhido na análise de sensibilidade da parte 3)
N_PERFIL = 3                 # o questionário vale como 3 dicas (protocolo do questionário simulado do BU)


def km(lat1, lon1, lat2, lon2):
    """Distância em km entre pontos dados em radianos (fórmula de haversine)."""
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def carregar(con, conjunto="real"):
    """Lê os artefatos gravados pela parte 1: modelos + tabela de restaurantes."""
    pref = prefixo(conjunto)
    return carregar_modelo(con, pref + "concierge_yelp"), ler_restaurantes(con, pref + "restaurantes")


def conjunto_disponivel(con, sintetico=False):
    """'real' se os modelos treinados com o Yelp real estão no banco; senão, os da rodada sintética."""
    tem = {r[0] for r in con.execute("SELECT nome FROM modelos")}
    if not sintetico and "concierge_yelp" in tem:
        return "real"
    if "sint_concierge_yelp" in tem:
        return "sintetico"
    raise LookupError("Nenhum modelo no banco. Rode a parte 1 antes (python concierge.py parte1).")


def perfil_questionario(culinarias, faixa_preco, modo, bairro, n=N_PERFIL):
    """Converte as respostas do questionário no mesmo estado de usuário que o modelo B viu no treino."""
    lat0, lon0 = np.radians(BAIRROS[bairro])
    return {"n": n, "cont": {c: n / len(culinarias) for c in culinarias}, "lat": lat0, "lon": lon0,
            "preco_medio": float(faixa_preco), "share_delivery": 1.0 if modo == "pedir" else 0.0}


def atributos_par(perfil: dict, cand: pd.DataFrame) -> pd.DataFrame:
    """Atributos do par (usuário, restaurante), iguais aos de `linha()` da parte 1, calculados para 'hoje'."""
    n = perfil["n"]
    afin = [[perfil["cont"].get(c, 0.0) / n for c in cs] or [0.0] for cs in cand["culinarias"]]
    dist = km(np.radians(cand["lat"].to_numpy()), np.radians(cand["lon"].to_numpy()), perfil["lat"], perfil["lon"])
    deliv = cand["delivery"].to_numpy(dtype=float)
    usd = perfil["share_delivery"]
    return pd.DataFrame({
        "user_n_prev_log": np.log1p(n),
        "afin_max": [max(a) for a in afin],
        "afin_media": [float(np.mean(a)) for a in afin],
        "dist_centroide_km": dist,
        "dist_min_km": dist,
        "user_preco_medio": perfil["preco_medio"],
        "dif_preco": np.abs(cand["preco"].to_numpy(dtype=float) - perfil["preco_medio"]),
        "user_share_delivery": usd,
        "cand_delivery": deliv,
        "match_delivery": deliv * usd,
        "cand_preco": cand["preco"].to_numpy(dtype=float),
        "cand_chk_prev_log": cand["cand_chk_prev_log"].to_numpy(),
        "cand_chk_365d_log": cand["cand_chk_365d_log"].to_numpy(),
        "cand_tips_prev_log": cand["cand_tips_prev_log"].to_numpy(),
        "cand_idade_anos": cand["cand_idade_anos"].to_numpy(),
    }, index=cand.index)


def candidatos(tab, culinarias, faixa_preco, modo, bairro, raio_km):
    """Filtros de negócio, comuns às duas variantes."""
    lat0, lon0 = np.radians(BAIRROS[bairro])
    cand = tab[(tab["is_open"] == 1) & (tab["preco"] <= faixa_preco)]
    cand = cand[cand["delivery"] == 1] if modo == "pedir" else cand[cand["reservas"] == 1]
    cand = cand[cand["culinarias"].apply(lambda cs: bool(set(cs) & set(culinarias)))].copy()
    cand["dist_km"] = km(lat0, lon0, np.radians(cand["lat"].to_numpy()), np.radians(cand["lon"].to_numpy()))
    return cand[cand["dist_km"] <= raio_km]


def passa_qualidade(cand, limiar_a, min_avaliacoes=MIN_AVALIACOES):
    estabelecido = cand["review_count"] >= min_avaliacoes
    return np.where(estabelecido, cand["stars"] >= 4, cand["prob_aprovacao"] >= limiar_a)


def _motivo(r, culinarias):
    cul = ", ".join([c for c in r["culinarias"] if c in culinarias][:2])
    qual = (f"nota {r['stars']:.1f}" if r["review_count"] >= MIN_AVALIACOES
            else f"casa nova, aprovação prevista {r['prob_aprovacao']:.0%}")
    return f"{cul} · {'$' * int(r['preco'])} · {r['dist_km']:.1f} km · {qual}"


COLUNAS_SAIDA = ["business_id", "nome", "culinaria", "preco", "dist_km", "stars", "review_count",
                 "prob_aprovacao", "prob_escolha", "descoberta", "por que"]


def corte_movimento(tab):
    """Mediana de movimento (check-ins até hoje) dos restaurantes abertos: separa 'menor' e 'maior' movimento."""
    return float(tab.loc[tab["is_open"] == 1, "cand_chk_prev_log"].median())


def recomendar(tab, art, culinarias, faixa_preco, modo="pedir", bairro="Center City", raio_km=3.0, top=5,
               variante="concierge", descoberta=True, fator_limiar_b=FATOR_LIMIAR_B):
    """Devolve até `top` restaurantes (DataFrame; vazio se nenhum passar)."""
    if not culinarias:
        raise ValueError("Escolha pelo menos uma culinária.")
    cand = candidatos(tab, culinarias, faixa_preco, modo, bairro, raio_km)
    if variante == "controle":
        ok = cand.sort_values(["cand_chk_prev_log", "stars"], ascending=False).head(top).copy()
        ok["prob_escolha"] = np.nan
        ok["descoberta"] = False
        ok["por que"] = "entre os mais movimentados perto de você"
    else:
        cand = cand[passa_qualidade(cand, art["limiar_A"])]
        if cand.empty:
            return pd.DataFrame(columns=COLUNAS_SAIDA)
        X = atributos_par(perfil_questionario(culinarias, faixa_preco, modo, bairro), cand)
        cand["prob_escolha"] = art["modelo_B"].predict_proba(X[art["atributos_B"]])[:, 1]
        cand["descoberta"] = False
        cand = cand[cand["prob_escolha"] >= fator_limiar_b * art["limiar_B"]]
        ok = cand.sort_values("prob_escolha", ascending=False).head(top)
        if descoberta and len(ok):
            corte = corte_movimento(tab)
            if not (ok["cand_chk_prev_log"] <= corte).any():
                pequenos = cand[(cand["cand_chk_prev_log"] <= corte) & ~cand.index.isin(ok.index)]
                if len(pequenos):
                    extra = pequenos.sort_values("prob_escolha", ascending=False).head(1).assign(descoberta=True)
                    ok = pd.concat([ok.head(top - 1), extra])
        ok["por que"] = [("descoberta: casa menos conhecida e bem avaliada · " if r["descoberta"] else "")
                         + _motivo(r, culinarias) for _, r in ok.iterrows()]
    # culinárias pedidas primeiro, para a coluna mostrar o que a pessoa escolheu
    ok["culinaria"] = ok["culinarias"].apply(
        lambda cs: ", ".join(([c for c in cs if c in culinarias] + [c for c in cs if c not in culinarias])[:2]))
    return ok[COLUNAS_SAIDA].reset_index(drop=True)


# %% 3. Ferramentas comuns aos modelos ==============================================================
def melhor_limiar(y, p, beta=0.5):
    """Limiar que maximiza o F0,5 (a precisão pesa 2x o recall, porque o falso positivo é o erro caro)."""
    grade = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    if len(grade) < 2:
        return 0.0                                             # modelo constante: recomenda tudo
    f = [fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0) for t in grade]
    return float(grade[int(np.argmax(f))])


class Score:
    """Baseline sem aprendizado: usa uma coluna como pontuação (ex.: popularidade)."""
    def __init__(self, coluna, sinal=1): self.coluna, self.sinal = coluna, sinal
    def fit(self, X, y): return self
    def predict_proba(self, X):
        s = self.sinal * X[self.coluna].to_numpy(dtype=float)
        s = np.nan_to_num(s, nan=np.nanmin(s))
        p = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return np.c_[1 - p, p]


# %% 4. PARTE 1 · Data Understanding, Data Preparation e Modeling com o Yelp (antigo 03a) ===========
"""
Bases usadas: business, tip e checkin (Philadelphia + categoria Restaurants, como definido no BU).
Sem review.json não existe a nota que cada usuário deu a cada restaurante. A preferência é modelada em duas partes:

  Modelo A · Aprovação           os consumidores aprovam este restaurante?   alvo: stars >= 4 (nota média)   business
  Modelo B · Preferência pessoal este consumidor vai escolher este lugar?    alvo: deixou uma dica nele       tip

B ordena pelo gosto da pessoa e A barra restaurantes mal avaliados (defesa contra o falso positivo, o erro caro do BU).
Em cada modelo: árvore de decisão, random forest, gradient boosting, regressão logística e KNN contra baselines de
popularidade, com % de acertos, falsos positivos e falsos negativos. Esta parte grava no banco os artefatos que o app
e a parte 3 usam.
"""


def parte1(con, saidas=None):
    con_y, fonte = abrir_yelp(con, ("business", "tip", "checkin"))
    pref = prefixo(fonte)
    SAIDA = pasta_saida(saidas, pref + "parte1")
    cab("PARTE 1 · Data Understanding, Data Preparation e Modeling (fonte: "
        + ("Yelp Open Dataset (real)" if fonte == "real" else "amostra sintética (esquema do Yelp)") + ")")

    def salvar(nome, df):
        salvar_tabela(con, f"{pref}p1_{nome}", df, SAIDA)

    # ------------------------------------------------------------------ 1. Data Understanding: as três bases
    # business: 1 negócio (cadastro e nota média, alvo do modelo A) | tip: 1 dica (quem foi aonde e quando, alvo do
    # modelo B) | checkin: 1 negócio com todas as datas numa string (volume, horário, tendência).
    # stars e review_count são FOTOGRAFIAS do fim da base: alvo no modelo A, nunca atributo no modelo B.
    # is_open é o estado atual: só filtro. Check-ins em UTC; Philadelphia fica em UTC-5 (horário de verão ignorado).
    t0 = time.time()
    biz_all = ler_business(con_y)
    tip_all = pd.read_sql("SELECT * FROM yelp_tip ORDER BY rowid", con_y)
    chk_all = pd.read_sql("SELECT * FROM yelp_checkin ORDER BY rowid", con_y)
    chk_all["n_checkins"] = chk_all["date"].str.count(",") + 1
    tip_all["date"] = pd.to_datetime(tip_all["date"])

    visao = pd.DataFrame({
        "linhas": [len(biz_all), len(tip_all), len(chk_all)],
        "colunas": [biz_all.shape[1], tip_all.shape[1], chk_all.shape[1] - 1],
        "negócios distintos": [biz_all["business_id"].nunique(), tip_all["business_id"].nunique(), chk_all["business_id"].nunique()],
        "usuários distintos": [np.nan, tip_all["user_id"].nunique(), np.nan],
        "eventos": [np.nan, len(tip_all), chk_all["n_checkins"].sum()],
        "período": ["snapshot", f"{tip_all['date'].min():%Y-%m} a {tip_all['date'].max():%Y-%m}", "2010 a 2022"],
    }, index=["business", "tip", "checkin"])
    print(f"Leitura do banco: {time.time() - t0:.0f}s (o banco guarda só os restaurantes de {CIDADE})")
    mostrar(visao)

    # Recorte do BU: restaurantes de Philadelphia
    biz_all["city"] = biz_all["city"].str.strip()
    eh_rest = biz_all["categories"].fillna("").str.contains("Restaurants")
    biz = biz_all[eh_rest & (biz_all["city"].str.casefold() == CIDADE.casefold())].reset_index(drop=True)
    ids = set(biz["business_id"])
    tip = tip_all[tip_all["business_id"].isin(ids)].copy()
    chk = chk_all[chk_all["business_id"].isin(ids)].copy()

    n_neg = int(meta(con_y, "n_negocios_yelp", len(biz_all)))
    n_rest_yelp = int(meta(con_y, "n_restaurantes_yelp", eh_rest.sum()))
    print(f"\nNegócios no Yelp: {n_neg:,} | restaurantes: {n_rest_yelp:,} | restaurantes em {CIDADE}: {len(biz):,}")
    print(f"Dicas nesses restaurantes: {len(tip):,} de {tip['user_id'].nunique():,} usuários")
    print(f"Restaurantes com check-in: {len(chk):,} | check-ins: {chk['n_checkins'].sum():,}")

    def perfil(df):
        linhas = []
        for c in df.columns:
            s = df[c]
            try:
                dist = s.nunique()
            except TypeError:
                dist = s.dropna().astype(str).nunique()
            linhas.append({"coluna": c, "tipo": str(s.dtype), "nulos_%": round(100 * s.isna().mean(), 1),
                           "distintos": dist, "exemplo": str(s.dropna().iloc[0])[:50] if s.notna().any() else ""})
        return pd.DataFrame(linhas).set_index("coluna")

    mostrar(perfil(biz), "Perfil da base business")

    fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
    biz["stars"].value_counts().sort_index().plot.bar(ax=ax[0], color="#4C72B0")
    ax[0].set_title(f"Nota média dos restaurantes ({(biz['stars'] >= 4).mean():.0%} com 4+)"); ax[0].set_xlabel("stars")
    tip["date"].dt.year.value_counts().sort_index().plot.bar(ax=ax[1], color="#55A868")
    ax[1].set_title("Dicas por ano (queda forte em 2020)")
    dicas_por_usuario = tip["user_id"].value_counts()
    (dicas_por_usuario.clip(upper=10).value_counts().sort_index() / len(dicas_por_usuario) * 100).plot.bar(ax=ax[2], color="#C44E52")
    ax[2].set_title("Usuários por nº de dicas (10 = 10+)"); ax[2].set_ylabel("% dos usuários")
    _fig(SAIDA, "01_bases.png")
    print(f"Usuários com 1 dica só: {(dicas_por_usuario == 1).mean():.0%} -> cold start também aqui")

    # ------------------------------------------------------------------ 2. Data Preparation
    # Atributos em texto ('True', "u'free'", 'None') -> parser: 0/1, ordinais para barulho e álcool
    # Atributos-dicionário (Ambience, GoodForMeal, BusinessParking) -> uma coluna 0/1 por chave
    # Faixa de preço ausente (≈16%) -> mediana das culinárias do restaurante + indicador
    # Demais atributos ausentes -> mediana DENTRO do pipeline + indicador de ausência
    # categories em texto livre -> multi-hot das culinárias com 60+ restaurantes
    # Dicas duplicadas -> remover | check-ins numa string só -> explodir e converter para horário local
    # review_count e is_open -> fora dos atributos
    def limpar(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, str):
            v = v.strip()
            if v in ("None", "none", "'none'", "u'none'"):
                return np.nan
            if v in ("True", "False"):
                return float(v == "True")
            if v.startswith("{"):
                try:
                    return ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    return np.nan
            if v.startswith(("u'", "'")):
                return v.lstrip("u").strip("'")
        return v

    attrs = biz["attributes"].apply(lambda d: d if isinstance(d, dict) else {})
    A = lambda k: attrs.apply(lambda d: limpar(d.get(k)))

    rest = pd.DataFrame({"business_id": biz["business_id"], "nome": biz["name"], "stars": biz["stars"],
                         "review_count": biz["review_count"],       # fora dos atributos; só para segmentar e para o app
                         "is_open": biz["is_open"], "lat": biz["latitude"], "lon": biz["longitude"]})
    rest["preco"] = pd.to_numeric(A("RestaurantsPriceRange2"), errors="coerce")
    for col, chave in [("delivery", "RestaurantsDelivery"), ("takeout", "RestaurantsTakeOut"),
                       ("reservas", "RestaurantsReservations"), ("mesas_fora", "OutdoorSeating"),
                       ("criancas", "GoodForKids"), ("grupos", "RestaurantsGoodForGroups"), ("tv", "HasTV"),
                       ("cartao", "BusinessAcceptsCreditCards"), ("atende_mesa", "RestaurantsTableService")]:
        rest[col] = pd.to_numeric(A(chave), errors="coerce")
    rest["alcool"] = A("Alcohol").map({"none": 0, "beer_and_wine": 1, "full_bar": 2})
    rest["barulho"] = A("NoiseLevel").map({"quiet": 0, "average": 1, "loud": 2, "very_loud": 3})
    rest["wifi_gratis"] = A("WiFi").map({"free": 1.0, "no": 0.0, "paid": 0.0})

    for chave, prefixo_col, subs in [
            ("Ambience", "amb", ["casual", "classy", "trendy", "hipster", "romantic", "intimate", "divey", "upscale", "touristy"]),
            ("GoodForMeal", "ref", ["breakfast", "brunch", "lunch", "dinner", "latenight", "dessert"]),
            ("BusinessParking", "estac", ["street", "lot", "garage", "valet"])]:
        dic = A(chave)
        for s in subs:
            rest[f"{prefixo_col}_{s}"] = dic.apply(
                lambda d: float(bool(d.get(s))) if isinstance(d, dict) and d.get(s) is not None else np.nan)

    def horas(h):
        if not isinstance(h, dict):
            return np.nan, np.nan, np.nan
        total, fecha_tarde = 0.0, 0
        for intervalo in h.values():
            a, b = intervalo.split("-")
            ha, hb = (int(x.split(":")[0]) + int(x.split(":")[1]) / 60 for x in (a, b))
            dur = (hb - ha) % 24 or (24 if intervalo != "0:0-0:0" else 0)
            total += dur
            fecha_tarde |= (hb >= 22 or hb < ha)
        return total, float("Sunday" in h), float(fecha_tarde)
    rest[["horas_semana", "abre_domingo", "fecha_tarde"]] = pd.DataFrame(biz["hours"].apply(horas).tolist())
    rest["n_atributos"] = attrs.apply(len)

    # culinárias: multi-hot das categorias de comida com 60+ restaurantes
    GENERICAS = {"Restaurants", "Food", "Nightlife", "Event Planning & Services", "Caterers", "Specialty Food",
                 "Food Delivery Services", "Shopping", "Grocery", "Convenience Stores", "Arts & Entertainment",
                 "Beer", "Wine & Spirits", "Food Trucks", "Juice Bars & Smoothies", "Ethnic Food", "Venues & Event Spaces",
                 "Local Flavor", "Imported Food", "Lounges", "Sports Bars", "Wine Bars", "Cocktail Bars", "Pubs", "Dive Bars"}
    lista_cat = biz["categories"].fillna("").str.split(", ")
    contagem = Counter(c for cs in lista_cat for c in cs if c not in GENERICAS)
    CULINARIAS = [c for c, n in contagem.most_common() if n >= 60]
    rest["culinarias"] = lista_cat.apply(lambda cs: [c for c in cs if c in CULINARIAS])
    for c in CULINARIAS:
        rest[f"cul_{c}"] = rest["culinarias"].apply(lambda cs: float(c in cs))
    rest["n_culinarias"] = rest["culinarias"].apply(len)

    # preço ausente: mediana das culinárias do restaurante
    rest["preco_ausente"] = rest["preco"].isna().astype(int)
    med = {c: rest.loc[rest[f"cul_{c}"] == 1, "preco"].median() for c in CULINARIAS}

    def imputar_preco(r):
        if not np.isnan(r["preco"]):
            return r["preco"]
        vals = [med[c] for c in r["culinarias"] if not np.isnan(med[c])]
        return float(np.median(vals)) if vals else np.nan
    rest["preco"] = rest.apply(imputar_preco, axis=1).fillna(rest["preco"].median())

    print(f"\n{len(CULINARIAS)} culinárias viraram colunas: {', '.join(CULINARIAS[:12])}, ...")
    print(f"Preço ausente: {rest['preco_ausente'].mean():.1%} | delivery declarado: {(rest['delivery'] == 1).mean():.1%}")

    # ---- check-ins: uma linha por check-in, em horário local
    ck = chk[["business_id", "date"]].assign(date=chk["date"].str.split(", ")).explode("date")
    ck["ts"] = pd.to_datetime(ck["date"]) - pd.Timedelta(hours=5)          # UTC -> Philadelphia (aprox.)
    FIM = ck["ts"].max()
    g = ck.groupby("business_id")["ts"]
    feat_ck = pd.DataFrame({
        "chk_total": g.size(),
        "chk_anos_ativo": (g.max() - g.min()).dt.days / 365.25 + 0.25,
        "chk_share_jantar": g.apply(lambda s: s.dt.hour.between(17, 21).mean()),
        "chk_share_fds": g.apply(lambda s: (s.dt.dayofweek >= 5).mean()),
        "chk_share_ult2anos": g.apply(lambda s: (s >= FIM - pd.Timedelta(days=730)).mean()),
    })
    feat_ck["chk_por_ano"] = feat_ck["chk_total"] / feat_ck["chk_anos_ativo"]

    # ---- dicas: volume, elogios e tom do texto (léxico simples, só para dar uma régua)
    n_antes = len(tip)
    tip = tip.drop_duplicates(["user_id", "business_id", "date"]).sort_values("date").reset_index(drop=True)
    POS = r"\b(?:great|best|love|loved|amazing|delicious|awesome|excellent|fantastic|perfect|favorite|tasty|yummy|fresh|friendly|good)\b"
    NEG = r"\b(?:worst|bad|terrible|rude|cold|slow|avoid|horrible|awful|disgusting|overpriced|dirty|bland|never|mediocre|wrong)\b"
    txt = tip["text"].str.lower()
    tip["tom_pos"] = txt.str.contains(POS).astype(int)
    tip["tom_neg"] = txt.str.contains(NEG).astype(int)
    g = tip.groupby("business_id")
    feat_tip = pd.DataFrame({"tips_total": g.size(), "tips_usuarios": g["user_id"].nunique(),
                             "tips_elogios": g["compliment_count"].sum(),
                             "tips_tom_pos": g["tom_pos"].mean(), "tips_tom_neg": g["tom_neg"].mean()})

    rest = rest.merge(feat_ck, left_on="business_id", right_index=True, how="left") \
               .merge(feat_tip, left_on="business_id", right_index=True, how="left")
    for c in ["chk_total", "chk_por_ano", "tips_total", "tips_usuarios", "tips_elogios"]:
        rest[c] = rest[c].fillna(0)
        rest[c + "_log"] = np.log1p(rest[c])
    rest["tips_por_chk"] = rest["tips_total"] / (rest["chk_total"] + 1)
    rest["tips_saldo_tom"] = rest["tips_tom_pos"] - rest["tips_tom_neg"]
    print(f"Dicas duplicadas removidas: {n_antes - len(tip)} | check-ins processados: {len(ck):,}")
    mostrar(rest[["chk_total", "chk_por_ano", "chk_share_jantar", "chk_share_fds", "tips_total", "tips_tom_pos",
                  "tips_tom_neg"]].describe().T.round(2))

    # ------------------------------------------------------------------ 2.1 A hipótese do BU com os dados
    # Esparsidade e cauda longa; gosto concentrado (favorita definida na 1ª metade das dicas e medida na 2ª, para
    # quem tem 6+ restaurantes); satisfação pelo tom positivo das dicas na favorita contra as demais.
    pares = tip.drop_duplicates(["user_id", "business_id"])          # tip já está em ordem de data: fica a 1ª visita
    por_rest = tip.groupby("business_id").size().reindex(rest["business_id"], fill_value=0).sort_values(ascending=False)
    x = np.sort(por_rest.to_numpy(dtype=float))
    gini_tips = float((2 * np.arange(1, len(x) + 1) - len(x) - 1) @ x / (len(x) * x.sum()))
    ano = tip["date"].dt.year.value_counts()

    cul_rest = dict(zip(rest["business_id"], rest["culinarias"]))
    fatia_cidade = Counter(c for b in pares["business_id"] for c in cul_rest[b])
    fatia_cidade = {c: n / len(pares) for c, n in fatia_cidade.items()}
    contagem_u = pares["user_id"].value_counts()
    obs, esp, tom_fav, tom_outras = [], [], [], []
    for uid, gu in pares[pares["user_id"].isin(contagem_u[contagem_u >= 6].index)].groupby("user_id"):
        culs = [cul_rest[b] for b in gu["business_id"]]
        meio = len(culs) // 2
        cont = Counter(c for cs in culs[:meio] for c in cs)
        if not cont:
            continue
        fav = cont.most_common(1)[0][0]
        na_fav = np.array([fav in cs for cs in culs[meio:]])
        tom = gu["tom_pos"].to_numpy()[meio:]
        obs.append(na_fav.mean()); esp.append(fatia_cidade[fav])
        tom_fav += list(tom[na_fav]); tom_outras += list(tom[~na_fav])

    du_real = pd.Series({
        "esparsidade da matriz usuário × restaurante (%)": 100 * (1 - len(pares) / (pares["user_id"].nunique() * len(rest))),
        "usuários com 1 dica só (%)": 100 * (contagem_u == 1).mean(),
        "top 10% dos restaurantes: fatia das dicas (%)": 100 * por_rest.head(len(por_rest) // 10).sum() / por_rest.sum(),
        "Gini das dicas por restaurante": gini_tips,
        "variação das dicas 2019 -> 2020 (%)": 100 * (ano.get(2020, 0) / ano.get(2019, np.nan) - 1),
        "usuários no teste de gosto (6+ restaurantes)": len(obs),
        "fatia da culinária favorita na 2ª metade (%)": 100 * np.mean(obs),
        "fatia esperada ao acaso (%)": 100 * np.mean(esp),
        "dicas com tom positivo na favorita (%)": 100 * np.mean(tom_fav),
        "dicas com tom positivo nas demais (%)": 100 * np.mean(tom_outras),
    }, name=f"{CIDADE}, " + ("dados reais" if fonte == "real" else "amostra sintética"))
    salvar("DU_hipotese_real", du_real.round(2).to_frame())
    dif = np.array(obs) - np.array(esp)
    tt = stats.ttest_1samp(dif, 0)
    print(f"\nGosto concentrado: +{100 * dif.mean():.1f} p.p. acima do acaso (t = {tt.statistic:.1f}, p = {tt.pvalue:.1e}, "
          f"{len(dif)} usuários)")
    mostrar(du_real.round(2).to_frame())

    # ------------------------------------------------------------------ 3. Ferramentas comuns aos dois modelos
    # pipeline(): mediana com indicador de ausência + escala + modelo, tudo aprendido só no treino.
    # avaliar(): matriz de confusão em %. % do total = fatia de todos os casos; taxa de FP = FP ÷ negativos reais.
    def pipeline(modelo):
        return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("sc", StandardScaler()), ("clf", modelo)])

    def avaliar(y, p, limiar):
        y = np.asarray(y); pred = (p >= limiar).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel(); n = tn + fp + fn + tp
        return {"Acertos (% do total)": 100 * (tp + tn) / n, "Falso positivo (% do total)": 100 * fp / n,
                "Falso negativo (% do total)": 100 * fn / n, "VP (% do total)": 100 * tp / n, "VN (% do total)": 100 * tn / n,
                "Taxa de FP (FP ÷ negativos)": 100 * fp / max(fp + tn, 1), "Taxa de FN (FN ÷ positivos)": 100 * fn / max(fn + tp, 1),
                "Precisão": 100 * precision_score(y, pred, zero_division=0), "Recall": 100 * recall_score(y, pred, zero_division=0),
                "AUC": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5}

    COLS = ["Acertos (% do total)", "Falso positivo (% do total)", "Falso negativo (% do total)",
            "Taxa de FP (FP ÷ negativos)", "Taxa de FN (FN ÷ positivos)", "Precisão", "Recall", "AUC", "Limiar"]

    def tabela(res, titulo=None):
        mostrar(res[COLS].astype(float).round({c: 1 for c in COLS[:7]} | {"AUC": 3, "Limiar": 3}), titulo)

    def grafico_composicao(res, titulo, arquivo, rot_pos, rot_neg):
        comp = res[["VP (% do total)", "VN (% do total)", "Falso positivo (% do total)", "Falso negativo (% do total)"]].astype(float)
        cores = ["#2E7D32", "#81C784", "#C62828", "#F9A825"]
        rot = [f"Acerto: {rot_pos}", f"Acerto: {rot_neg}", "Falso positivo", "Falso negativo"]
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(comp) + 1.6)); esq = np.zeros(len(comp))
        for c, cor, r in zip(comp.columns, cores, rot):
            ax.barh(comp.index, comp[c], left=esq, color=cor, label=r)
            for i, (v, e) in enumerate(zip(comp[c], esq)):
                if v >= 4: ax.text(e + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
            esq += comp[c].to_numpy()
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("% dos casos do teste"); ax.set_title(titulo)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)
        _fig(SAIDA, arquivo)

    def grafico_matrizes(y, probas, limiares, rotulos, arquivo):
        nomes = list(probas); ncol = 4; nlin = int(np.ceil(len(nomes) / ncol))
        fig, axes = plt.subplots(nlin, ncol, figsize=(16, 3.8 * nlin))
        for ax, nome in zip(np.ravel(axes), nomes):
            pred = (probas[nome] >= limiares[nome]).astype(int)
            ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1], normalize="all") * 100,
                                   display_labels=rotulos).plot(ax=ax, colorbar=False, values_format=".1f", cmap="Blues")
            ax.set_title(nome, fontsize=10); ax.set_xlabel("previsto (%)"); ax.set_ylabel("real")
        for ax in np.ravel(axes)[len(nomes):]: ax.axis("off")
        _fig(SAIDA, arquivo)

    def modelos_padrao(prof_arvore):
        return {
            "Regressão logística": pipeline(LogisticRegression(max_iter=3000)),
            "Árvore de decisão": pipeline(DecisionTreeClassifier(max_depth=prof_arvore, min_samples_leaf=20, random_state=SEED)),
            "Random forest": pipeline(RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=10,
                                                             n_jobs=-1, random_state=SEED)),
            "Gradient boosting": pipeline(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)),
            "KNN (k=50)": pipeline(KNeighborsClassifier(n_neighbors=50, weights="distance")),
        }

    # ------------------------------------------------------------------ 4. Modelo A · os consumidores aprovam?
    # Alvo: aprovado = 1 se stars >= 4. Grão: 1 restaurante. stars não tem data: split aleatório estratificado
    # (75/25) e limiar escolhido com validação cruzada dentro do treino (o teste é usado uma vez).
    # Três conjuntos de atributos: Cadastro (existe desde o 1º dia), + Comportamento (check-ins, dicas), + Tom.
    # Fora: review_count (contagem das avaliações que formam o alvo) e is_open (estado atual).
    print("\n--- Modelo A · aprovação do restaurante")
    CAD = (["preco", "preco_ausente", "delivery", "takeout", "reservas", "mesas_fora", "criancas", "grupos", "tv",
            "cartao", "atende_mesa", "alcool", "barulho", "wifi_gratis", "horas_semana", "abre_domingo", "fecha_tarde",
            "n_atributos", "n_culinarias", "lat", "lon"]
           + [c for c in rest.columns if c.startswith(("amb_", "ref_", "estac_", "cul_"))])
    COMP = ["chk_total_log", "chk_por_ano_log", "chk_anos_ativo", "chk_share_jantar", "chk_share_fds", "chk_share_ult2anos",
            "tips_total_log", "tips_usuarios_log", "tips_elogios_log", "tips_por_chk"]
    TOM = ["tips_tom_pos", "tips_tom_neg", "tips_saldo_tom"]
    CONJUNTOS = {"Cadastro": CAD, "Cadastro + comportamento": CAD + COMP, "Cadastro + comportamento + tom": CAD + COMP + TOM}

    rest["aprovado"] = (rest["stars"] >= 4).astype(int)
    idx_tr, idx_te = train_test_split(rest.index, test_size=0.25, stratify=rest["aprovado"], random_state=SEED)
    yA_tr, yA_te = rest.loc[idx_tr, "aprovado"], rest.loc[idx_te, "aprovado"]
    CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
    print(f"Treino {len(idx_tr):,} | teste {len(idx_te):,} | aprovados: {rest['aprovado'].mean():.1%}")

    # profundidade da árvore por validação cruzada (conjunto completo)
    XA_full = rest.loc[idx_tr, CONJUNTOS["Cadastro + comportamento + tom"]]
    curva = []
    for prof in [2, 3, 4, 5, 6, 8, 10, 14, None]:
        m = pipeline(DecisionTreeClassifier(max_depth=prof, min_samples_leaf=20, random_state=SEED))
        cv_auc = cross_val_score(m, XA_full, yA_tr, cv=CV, scoring="roc_auc").mean()
        tr_auc = roc_auc_score(yA_tr, m.fit(XA_full, yA_tr).predict_proba(XA_full)[:, 1])
        curva.append({"max_depth": "sem limite" if prof is None else prof, "AUC treino": tr_auc, "AUC validação cruzada": cv_auc})
    curva = pd.DataFrame(curva)
    PROF_A = curva.loc[curva["AUC validação cruzada"].idxmax(), "max_depth"]
    PROF_A = None if PROF_A == "sem limite" else int(PROF_A)
    ax = curva.set_index(curva["max_depth"].astype(str))[["AUC treino", "AUC validação cruzada"]].plot(marker="o", figsize=(7, 3))
    ax.set_title("Modelo A · árvore: profundidade × overfitting"); _fig(SAIDA, "A_curva_arvore.png")
    print("Profundidade escolhida:", PROF_A)

    resA, probA, limA = {}, {}, {}
    for nome_conj, cols in CONJUNTOS.items():
        Xtr, Xte = rest.loc[idx_tr, cols], rest.loc[idx_te, cols]
        modelos = {"Chute: todos aprovados": DummyClassifier(strategy="prior"),
                   "Popularidade (check-ins)": Score("chk_total_log")} if nome_conj == "Cadastro + comportamento" else {}
        modelos |= modelos_padrao(PROF_A)
        for nome, m in modelos.items():
            chave = nome if nome.startswith(("Chute", "Popularidade")) else f"{nome} · {nome_conj}"
            cv_p = (m.predict_proba(Xtr)[:, 1] if isinstance(m, Score)
                    else cross_val_predict(m, Xtr, yA_tr, cv=CV, method="predict_proba")[:, 1])
            limA[chave] = melhor_limiar(yA_tr, cv_p)
            m.fit(Xtr, yA_tr)
            probA[chave] = m.predict_proba(Xte)[:, 1]
            resA[chave] = avaliar(yA_te, probA[chave], limA[chave]) | {"Limiar": limA[chave], "conjunto": nome_conj, "modelo": nome}
        print(f"ok: {nome_conj}")

    resA = pd.DataFrame(resA).T
    for c in resA.columns.drop(["conjunto", "modelo"]): resA[c] = resA[c].astype(float)
    salvar("A_metricas_teste", resA.sort_values("AUC", ascending=False))

    # AUC por modelo × conjunto: quanto cada fonte de dados acrescenta
    pivo = resA[~resA.index.str.startswith(("Chute", "Popularidade"))].pivot(index="modelo", columns="conjunto", values="AUC")[list(CONJUNTOS)]
    mostrar(pivo.round(3), "AUC por modelo × conjunto de atributos")
    # Leitura: o cadastro sozinho (o que existe para um restaurante novo) já separa parte dos aprovados; check-ins e
    # dicas acrescentam; o tom das dicas acrescenta mais, porque é opinião do consumidor.

    CONJ_A = "Cadastro + comportamento + tom"
    selA = resA[(resA["conjunto"] == CONJ_A) | resA.index.str.startswith(("Chute", "Popularidade"))].copy()
    selA.index = selA.index.str.replace(f" · {CONJ_A}", "", regex=False)
    selA = selA.sort_values("AUC", ascending=False)
    tabela(selA, "Modelo A no teste (conjunto completo, o usado no top 5)")

    grafico_composicao(selA, "Modelo A · aprovação do restaurante (teste)", "A_composicao_erros.png",
                       "aprovou e é bem avaliado", "barrou e é mal avaliado")
    probA_sel = {n: probA[n if n.startswith(("Chute", "Popularidade")) else f"{n} · {CONJ_A}"] for n in selA.index}
    limA_sel = {n: limA[n if n.startswith(("Chute", "Popularidade")) else f"{n} · {CONJ_A}"] for n in selA.index}
    grafico_matrizes(yA_te, probA_sel, limA_sel, ["< 4", "≥ 4"], "A_matrizes_confusao.png")

    # Segmento de interesse do negócio: restaurantes que fazem delivery
    MELHOR_A = selA.drop([i for i in selA.index if i.startswith(("Chute", "Popularidade"))])["AUC"].idxmax()
    deliv = (rest.loc[idx_te, "delivery"] == 1).to_numpy()
    seg = pd.DataFrame({rot: avaliar(yA_te.to_numpy()[m], probA_sel[MELHOR_A][m], limA_sel[MELHOR_A])
                        for rot, m in [("faz delivery", deliv), ("não faz / não informa", ~deliv)]}).T
    seg.insert(0, "n", [int(deliv.sum()), int((~deliv).sum())])
    print("\nMelhor modelo A:", MELHOR_A)
    mostrar(seg[["n"] + COLS[:8]].round({c: 1 for c in COLS[:7]} | {"AUC": 3}))

    # 4.1 Onde o modelo A decide no app: restaurantes com poucas avaliações (abaixo de 20 a nota é instável)
    rc = rest.loc[idx_te, "review_count"].to_numpy()
    yA = yA_te.to_numpy()
    linhas = []
    for rot, m in [("menos de 20 avaliações (A decide no app)", rc < 20), ("20 a 99", (rc >= 20) & (rc < 100)),
                   ("100 ou mais", rc >= 100)]:
        for conj in ["Cadastro", CONJ_A]:
            chave = f"{MELHOR_A} · {conj}"
            if m.sum() == 0 or len(np.unique(yA[m])) < 2:
                continue
            r = avaliar(yA[m], probA[chave][m], limA[chave])
            linhas.append({"faixa": rot, "atributos": conj, "restaurantes": int(m.sum()), "% aprovados": 100 * yA[m].mean(),
                           **{k: r[k] for k in ["Acertos (% do total)", "Falso positivo (% do total)",
                                                "Taxa de FP (FP ÷ negativos)", "Precisão", "AUC"]}})
    segA = pd.DataFrame(linhas).set_index(["faixa", "atributos"])
    salvar("A_por_n_avaliacoes", segA)
    mostrar(segA.round({c: 1 for c in segA.columns if c not in ("AUC", "restaurantes")} | {"AUC": 3}),
            f"Modelo A ({MELHOR_A}) por número de avaliações")

    # Interpretação: árvore (3 níveis) e importâncias
    cols_A = CONJUNTOS[CONJ_A]
    arvA = pipeline(DecisionTreeClassifier(max_depth=PROF_A, min_samples_leaf=20, random_state=SEED)).fit(rest.loc[idx_tr, cols_A], yA_tr)
    nomes_A = [n.replace("missingindicator_", "ausente_") for n in arvA.named_steps["imp"].get_feature_names_out(cols_A)]
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(arvA.named_steps["clf"], max_depth=3, feature_names=nomes_A, class_names=["< 4", "≥ 4"], filled=True,
              proportion=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Modelo A · árvore de decisão (3 primeiros níveis; valores escalonados)")
    _fig(SAIDA, "A_arvore.png", dpi=110)

    rfA = modelos_padrao(PROF_A)["Random forest"].fit(rest.loc[idx_tr, cols_A], yA_tr)
    pd.Series(rfA.named_steps["clf"].feature_importances_, index=nomes_A).sort_values().tail(15) \
      .plot.barh(figsize=(8, 5), title="Modelo A · importância das variáveis (random forest)")
    _fig(SAIDA, "A_importancias.png")

    # ------------------------------------------------------------------ 5. Modelo B · este consumidor vai escolher?
    # Preferência revelada: deixar uma dica mostra que a pessoa foi lá. Protocolo anti-vazamento:
    # 1. cada dica (a partir de 2011) de quem já tinha deixado 1+ dica antes vira positivo, se o restaurante é novo
    #    para a pessoa; 2. para cada positivo, 4 restaurantes que ela nunca visitou e que já existiam, sorteados de
    #    forma uniforme; 3. todos os atributos usam só o que existia ANTES da data da dica;
    # 4. split temporal: treino até 2016, validação em 2017 (limiar e profundidade), teste de 2018 em diante.
    # Cada positivo + 4 negativos = um caso; a seção 5.1 mede HitRate@1 (ao acaso 20%) e MRR (ao acaso 0,457).
    print("\n--- Modelo B · escolha do consumidor (montando a base de pares, pode levar alguns minutos)")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    R = rest.reset_index(drop=True)
    bid2i = {b: i for i, b in enumerate(R["business_id"])}
    cul_idx = {c: k for k, c in enumerate(CULINARIAS)}
    cats_i = [np.array([cul_idx[c] for c in cs], dtype=int) for cs in R["culinarias"]]
    LAT, LON = np.radians(R["lat"].to_numpy()), np.radians(R["lon"].to_numpy())
    PRECO = R["preco"].to_numpy(); DELIV = R["delivery"].to_numpy()

    EPOCA, UM_S = pd.Timestamp("1970-01-01"), pd.Timedelta(seconds=1)
    segundos = lambda s: ((s - EPOCA) // UM_S).astype("int64")          # datas -> segundos (independe da unidade)
    ck_ts = {bid2i[b]: np.sort(segundos(s).to_numpy()) for b, s in ck.groupby("business_id")["ts"]}
    tip["t"] = segundos(tip["date"])
    tip_ts = {bid2i[b]: np.sort(s.to_numpy()) for b, s in tip.groupby("business_id")["t"]}
    vazio = np.array([], dtype="int64")
    primeira = np.array([min(ck_ts.get(i, [np.inf])[0] if len(ck_ts.get(i, vazio)) else np.inf,
                             tip_ts.get(i, [np.inf])[0] if len(tip_ts.get(i, vazio)) else np.inf) for i in range(len(R))])
    acum = np.cumsum(np.full(len(R), 1 / len(R)))          # sorteio uniforme dos negativos (protocolo do HitRate do DU)
    ANO = 365.25 * 86400

    def antes(arr, t):
        return int(np.searchsorted(arr, t, side="left")) if arr is not None else 0

    def linha(estado, c, t):
        # atributos do par (usuário no instante t, restaurante candidato c), só com o passado
        n, cont, lats, lons = estado["n"], estado["cont"], estado["lats"], estado["lons"]
        sh = cont[cats_i[c]] / n if len(cats_i[c]) else np.array([0.0])
        d = km(LAT[c], LON[c], np.array(lats), np.array(lons))
        cent = km(LAT[c], LON[c], np.mean(lats), np.mean(lons))
        upm = estado["preco_soma"] / n
        usd = estado["deliv_soma"] / estado["deliv_n"] if estado["deliv_n"] else np.nan
        ckc, tpc = ck_ts.get(c), tip_ts.get(c)
        return {"user_n_prev_log": np.log1p(n), "afin_max": sh.max(), "afin_media": sh.mean(),
                "dist_centroide_km": cent, "dist_min_km": d.min(), "user_preco_medio": upm,
                "dif_preco": abs(PRECO[c] - upm), "user_share_delivery": usd, "cand_delivery": DELIV[c],
                "match_delivery": (DELIV[c] * usd) if not np.isnan(DELIV[c]) and not np.isnan(usd) else np.nan,
                "cand_preco": PRECO[c],
                "cand_chk_prev_log": np.log1p(antes(ckc, t)),
                "cand_chk_365d_log": np.log1p(antes(ckc, t) - antes(ckc, t - 365 * 86400)),
                "cand_tips_prev_log": np.log1p(antes(tpc, t)),
                "cand_idade_anos": (t - primeira[c]) / ANO if np.isfinite(primeira[c]) else np.nan}

    K_NEG = 4
    T_INICIO = int((pd.Timestamp("2011-01-01") - EPOCA) // UM_S)      # antes de 2011 há poucos restaurantes ativos na base
    linhas_B, n_caso = [], 0
    tip["i"] = tip["business_id"].map(bid2i)
    usuarios_multi = tip["user_id"].value_counts()
    usuarios_multi = set(usuarios_multi[usuarios_multi >= 2].index)
    for uid, grupo in tip[tip["user_id"].isin(usuarios_multi)].groupby("user_id", sort=False):
        est = {"n": 0, "cont": np.zeros(len(CULINARIAS)), "lats": [], "lons": [], "preco_soma": 0.0,
               "deliv_soma": 0.0, "deliv_n": 0}
        vistos = set()
        for i, t, data in zip(grupo["i"].to_numpy(), grupo["t"].to_numpy(), grupo["date"]):
            if est["n"] >= 1 and i not in vistos and t >= T_INICIO:
                negs, tentativas = [], 0
                while len(negs) < K_NEG and tentativas < 30:            # limite evita laço infinito
                    tentativas += 1
                    for c in np.searchsorted(acum, rng.random(3 * K_NEG)):
                        c = min(int(c), len(R) - 1)
                        if c != i and c not in vistos and primeira[c] <= t and c not in negs:
                            negs.append(c)
                            if len(negs) == K_NEG: break
                if len(negs) == K_NEG:
                    for c, y in [(i, 1)] + [(c, 0) for c in negs]:
                        linhas_B.append({"caso": n_caso, "user_id": uid, "business_id": R.at[c, "business_id"],
                                         "date": data, "escolheu": y, **linha(est, c, t)})
                    n_caso += 1
            # atualiza o histórico DEPOIS de gerar o caso (o presente não entra no passado)
            est["n"] += 1; est["cont"][cats_i[i]] += 1
            est["lats"].append(LAT[i]); est["lons"].append(LON[i]); est["preco_soma"] += PRECO[i]
            if not np.isnan(DELIV[i]): est["deliv_soma"] += DELIV[i]; est["deliv_n"] += 1
            vistos.add(i)

    B = pd.DataFrame(linhas_B)
    FEATS_B = [c for c in B.columns if c not in ("caso", "user_id", "business_id", "date", "escolheu")]
    B["periodo"] = np.select([B["date"] < "2017-01-01", B["date"] < "2018-01-01"], ["treino", "validacao"], "teste")
    print(f"Base B montada em {time.time() - t0:.0f}s: {len(B):,} pares usuário × restaurante, {B['user_id'].nunique():,} usuários")
    mostrar(B.groupby("periodo").agg(pares=("escolheu", "size"), escolhas=("escolheu", "sum"),
                                     usuarios=("user_id", "nunique")).loc[["treino", "validacao", "teste"]])

    trB, vaB, teB = (B["periodo"] == p for p in ["treino", "validacao", "teste"])
    XB_tr, yB_tr = B.loc[trB, FEATS_B], B.loc[trB, "escolheu"]
    XB_va, yB_va = B.loc[vaB, FEATS_B], B.loc[vaB, "escolheu"]
    XB_te, yB_te = B.loc[teB, FEATS_B], B.loc[teB, "escolheu"]

    # profundidade da árvore pela validação
    curvaB = []
    for prof in [2, 3, 4, 6, 8, 10, 14, None]:
        m = pipeline(DecisionTreeClassifier(max_depth=prof, min_samples_leaf=20, random_state=SEED)).fit(XB_tr, yB_tr)
        curvaB.append({"max_depth": "sem limite" if prof is None else prof,
                       "AUC treino": roc_auc_score(yB_tr, m.predict_proba(XB_tr)[:, 1]),
                       "AUC validação": roc_auc_score(yB_va, m.predict_proba(XB_va)[:, 1])})
    curvaB = pd.DataFrame(curvaB)
    PROF_B = curvaB.loc[curvaB["AUC validação"].idxmax(), "max_depth"]
    PROF_B = None if PROF_B == "sem limite" else int(PROF_B)

    modelosB = {"Chute: recomenda tudo": DummyClassifier(strategy="prior"),
                "Popularidade (check-ins até a data)": Score("cand_chk_prev_log"),
                "Mais perto (distância)": Score("dist_centroide_km", sinal=-1)} | modelos_padrao(PROF_B)
    probB_va, probB_te, limB = {}, {}, {}
    for nome, m in modelosB.items():
        if nome.startswith("KNN"):                                  # KNN: amostra do treino para caber no tempo
            amostra = XB_tr.sample(min(60_000, len(XB_tr)), random_state=SEED).index
            m.fit(XB_tr.loc[amostra], yB_tr.loc[amostra])
        else:
            m.fit(XB_tr, yB_tr)
        probB_va[nome], probB_te[nome] = m.predict_proba(XB_va)[:, 1], m.predict_proba(XB_te)[:, 1]
        limB[nome] = melhor_limiar(yB_va, probB_va[nome])
        print(f"{nome:38s} AUC validação = {roc_auc_score(yB_va, probB_va[nome]) if nome[0] != 'C' else 0.5:.3f}")

    resB = pd.DataFrame({n: avaliar(yB_te, probB_te[n], limB[n]) | {"Limiar": limB[n]} for n in modelosB}).T.sort_values("AUC", ascending=False)
    salvar("B_metricas_teste", resB)
    print("Profundidade da árvore escolhida:", PROF_B)
    tabela(resB, "Modelo B no teste (2018+)")

    grafico_composicao(resB, "Modelo B · escolha do consumidor (teste, 2018+)", "B_composicao_erros.png",
                       "recomendou e ele escolheu", "não recomendou e ele não escolheu")
    grafico_matrizes(yB_te, probB_te, limB, ["não escolheu", "escolheu"], "B_matrizes_confusao.png")

    # Por experiência do usuário: o modelo funciona para quem tem pouco histórico?
    MELHOR_B = resB.drop([i for i in resB.index if i.startswith(("Chute", "Popularidade", "Mais perto"))])["AUC"].idxmax()
    n_prev = np.expm1(XB_te["user_n_prev_log"]).round().to_numpy()
    seg = []
    for rot, m in [("1 a 2 dicas antes", n_prev <= 2), ("3 a 9 dicas antes", (n_prev >= 3) & (n_prev <= 9)), ("10+ dicas antes", n_prev >= 10)]:
        for nome in [MELHOR_B, "Popularidade (check-ins até a data)"]:
            if m.sum() == 0 or len(np.unique(yB_te.to_numpy()[m])) < 2:
                continue
            r = avaliar(yB_te.to_numpy()[m], probB_te[nome][m], limB[nome])
            seg.append({"segmento": rot, "modelo": nome, "pares": int(m.sum()),
                        **{k: r[k] for k in ["Acertos (% do total)", "Falso positivo (% do total)", "Falso negativo (% do total)", "Precisão", "AUC"]}})
    print("\nMelhor modelo B:", MELHOR_B)
    mostrar(pd.DataFrame(seg).set_index(["segmento", "modelo"]).round(
        {"Acertos (% do total)": 1, "Falso positivo (% do total)": 1, "Falso negativo (% do total)": 1, "Precisão": 1, "AUC": 3}))

    # 5.1 Ranking: o restaurante escolhido fica em 1º entre os 5 candidatos? (empates desfeitos ao acaso, em média)
    def ranking(casos, y, p):
        d = pd.DataFrame({"caso": np.asarray(casos), "y": np.asarray(y), "p": np.asarray(p, dtype=float)})
        pos = d[d["y"] == 1].set_index("caso")["p"]
        neg = d[d["y"] == 0].join(pos.rename("p_pos"), on="caso")
        gg = (neg["p"] > neg["p_pos"]).groupby(neg["caso"]).sum().reindex(pos.index).fillna(0).to_numpy()
        tt_ = (neg["p"] == neg["p_pos"]).groupby(neg["caso"]).sum().reindex(pos.index).fillna(0).to_numpy()
        hit1 = np.where(gg == 0, 1 / (1 + tt_), 0.0)                       # empate: posição sorteada
        mrr = np.array([np.mean(1 / np.arange(gi + 1, gi + ti + 2)) for gi, ti in zip(gg, tt_)])
        return {"casos": len(pos), "HitRate@1 (%)": 100 * hit1.mean(), "MRR": mrr.mean()}

    casos_te = B.loc[teB, "caso"].to_numpy()
    rankB = pd.DataFrame({n: ranking(casos_te, yB_te, probB_te[n]) for n in modelosB}).T.sort_values("MRR", ascending=False)
    salvar("B_ranking_teste", rankB)
    print("\nReferência ao acaso: HitRate@1 = 20,0% | MRR = 0,457")
    mostrar(rankB.astype({"casos": int}).round({"HitRate@1 (%)": 1, "MRR": 3}))

    # 5.2 Quanto custa baixar o limiar do modelo B (no app, a casa mal avaliada já é barrada pela guarda de qualidade)
    lb = limB[MELHOR_B]
    sensB = pd.DataFrame({f"{f:.0%} do limiar ({f * lb:.2f})": avaliar(yB_te, probB_te[MELHOR_B], f * lb)
                          for f in [1.0, 0.75, 0.5, 0.25]}).T
    sensB = sensB[["Precisão", "Recall", "Falso positivo (% do total)", "Taxa de FP (FP ÷ negativos)"]]
    salvar("B_limiar_sensibilidade", sensB)
    mostrar(sensB.round(1), f"Modelo B ({MELHOR_B}) no teste com limiares menores; referência ao acaso: precisão de 20%")

    arvB = modelosB["Árvore de decisão"]
    nomes_B = [n.replace("missingindicator_", "ausente_") for n in arvB.named_steps["imp"].get_feature_names_out(FEATS_B)]
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(arvB.named_steps["clf"], max_depth=3, feature_names=nomes_B, class_names=["não escolheu", "escolheu"],
              filled=True, proportion=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Modelo B · árvore de decisão (3 primeiros níveis; valores escalonados)")
    _fig(SAIDA, "B_arvore.png", dpi=110)
    pd.Series(modelosB["Random forest"].named_steps["clf"].feature_importances_, index=nomes_B).sort_values().tail(15) \
      .plot.barh(figsize=(8, 5), title="Modelo B · importância das variáveis (random forest)")
    _fig(SAIDA, "B_importancias.png")

    # ------------------------------------------------------------------ 6. Decisão (CRISP-DM), escrita pelos números
    def linha_txt(nome, r):
        return (f"  {nome:38s} acertos {r['Acertos (% do total)']:5.1f}% | FP {r['Falso positivo (% do total)']:5.1f}% | "
                f"FN {r['Falso negativo (% do total)']:5.1f}% | precisão {r['Precisão']:5.1f}% | AUC {r['AUC']:.3f}")

    print("\nMODELO A · aprovação do restaurante")
    for n in [MELHOR_A, "Popularidade (check-ins)", "Chute: todos aprovados"]:
        print(linha_txt(n, selA.loc[n]))
    print("\nMODELO B · escolha do consumidor")
    for n in [MELHOR_B, "Popularidade (check-ins até a data)", "Mais perto (distância)", "Chute: recomenda tudo"]:
        print(linha_txt(n, resB.loc[n]))
    print("\nMODELO B · ranking (o escolhido fica em 1º entre 5?)")
    for n in [MELHOR_B, "Popularidade (check-ins até a data)", "Mais perto (distância)", "Chute: recomenda tudo"]:
        print(f"  {n:38s} HitRate@1 {rankB.loc[n, 'HitRate@1 (%)']:5.1f}% | MRR {rankB.loc[n, 'MRR']:.3f}")

    okA = selA.loc[MELHOR_A, "AUC"] > selA.loc["Popularidade (check-ins)", "AUC"]
    okB = (resB.loc[MELHOR_B, "AUC"] > max(resB.loc["Popularidade (check-ins até a data)", "AUC"], resB.loc["Mais perto (distância)", "AUC"])
           and rankB.loc[MELHOR_B, "MRR"] > max(rankB.loc["Popularidade (check-ins até a data)", "MRR"], rankB.loc["Mais perto (distância)", "MRR"]))
    print("\nDecisão:", "APROVAR os dois modelos para a Fase 5" if okA and okB else "rever a Fase 3 no modelo que não supera os baselines")

    # ------------------------------------------------------------------ 7. Uso: top 5 a partir do questionário
    # A regra é a da seção 2 deste script (recomendar), a mesma que o app usa. A probabilidade de A na tabela do app
    # é FORA DA AMOSTRA (validação cruzada no treino), para não favorecer os restaurantes que o modelo já viu.
    modelo_A = modelos_padrao(PROF_A)[MELHOR_A].fit(rest.loc[idx_tr, cols_A], yA_tr)
    LIM_A = limA_sel[MELHOR_A]
    modelo_B, LIM_B = modelosB[MELHOR_B], limB[MELHOR_B]

    prob_ap = pd.Series(np.nan, index=rest.index)
    prob_ap[idx_tr] = cross_val_predict(modelos_padrao(PROF_A)[MELHOR_A], rest.loc[idx_tr, cols_A], yA_tr,
                                        cv=CV, method="predict_proba")[:, 1]
    prob_ap[idx_te] = modelo_A.predict_proba(rest.loc[idx_te, cols_A])[:, 1]

    # Estado de cada restaurante "hoje" (fim da base): é o que o app teria na hora de recomendar
    T_HOJE = int((FIM - EPOCA) // UM_S)
    tab = R[["business_id", "nome", "stars", "review_count", "is_open", "lat", "lon", "preco", "delivery", "reservas",
             "culinarias"]].copy()
    tab["prob_aprovacao"] = prob_ap.to_numpy()
    tab["cand_chk_prev_log"] = [np.log1p(antes(ck_ts.get(c), T_HOJE)) for c in range(len(R))]
    tab["cand_chk_365d_log"] = [np.log1p(antes(ck_ts.get(c), T_HOJE) - antes(ck_ts.get(c), T_HOJE - 365 * 86400))
                                for c in range(len(R))]
    tab["cand_tips_prev_log"] = [np.log1p(antes(tip_ts.get(c), T_HOJE)) for c in range(len(R))]
    tab["cand_idade_anos"] = [(T_HOJE - primeira[c]) / ANO if np.isfinite(primeira[c]) else np.nan for c in range(len(R))]

    art = {"modelo_A": modelo_A, "nome_A": MELHOR_A, "limiar_A": LIM_A, "atributos_A": cols_A,
           "modelo_B": modelo_B, "nome_B": MELHOR_B, "limiar_B": LIM_B, "atributos_B": FEATS_B,
           "culinarias": CULINARIAS, "data_base": str(FIM.date()), "sklearn": sklearn.__version__}

    # Checagem: o recomendador calcula os mesmos atributos que linha() usou no treino
    cul_teste = CULINARIAS[:2]
    p = perfil_questionario(cul_teste, 2, "pedir", "Center City")
    decl = np.zeros(len(CULINARIAS))
    for c in cul_teste: decl[cul_idx[c]] = 3 / len(cul_teste)
    est = {"n": 3, "cont": decl, "lats": [p["lat"]] * 3, "lons": [p["lon"]] * 3, "preco_soma": 6.0,
           "deliv_soma": 3.0, "deliv_n": 3}
    amostra = rng.choice(len(R), size=min(300, len(R)), replace=False)
    ref = pd.DataFrame([linha(est, c, T_HOJE) for c in amostra])[FEATS_B].to_numpy(dtype=float)
    mod = atributos_par(p, tab.iloc[amostra])[FEATS_B].to_numpy(dtype=float)
    assert np.allclose(ref, mod, equal_nan=True), "atributos do app diferentes dos do treino"
    print("\nChecagem ok: o app calcula os mesmos atributos do treino.")
    print(f"Guarda de qualidade: nota observada para {(tab['review_count'] >= MIN_AVALIACOES).mean():.0%} dos "
          f"restaurantes; modelo A ({MELHOR_A}, limiar {LIM_A:.2f}) para o restante.")

    def exemplo(titulo, **perfil_q):
        out = recomendar(tab, art, **perfil_q)
        mostrar("Nenhum restaurante passa nos filtros e na guarda de qualidade: aumente o raio, o teto de preço ou as culinárias."
                if out.empty else out.drop(columns=["business_id"]).round(3), "\n" + titulo)

    exemplo("Perfil: PEDIR em Center City, Japanese ou Thai, até $$",
            culinarias=["Japanese", "Thai"], faixa_preco=2, modo="pedir", bairro="Center City")
    exemplo("Mesmo perfil na variante CONTROLE (ordenação atual dos apps, sem modelo)",
            culinarias=["Japanese", "Thai"], faixa_preco=2, modo="pedir", bairro="Center City", variante="controle")
    exemplo("Perfil: SAIR em South Philly, Italian ou Seafood, até $$$",
            culinarias=["Italian", "Seafood"], faixa_preco=3, modo="sair", bairro="South Philly", raio_km=4)

    mb = salvar_modelo(con, pref + "concierge_yelp", art) / 1e6
    salvar_restaurantes(con, pref + "restaurantes", tab)
    print(f"\nGravado no banco: modelos '{pref}concierge_yelp' ({mb:.1f} MB) e tabela '{pref}restaurantes' ({len(tab):,} linhas)")
    print(f"Figuras e tabelas em {SAIDA}")
    if fonte != "real":
        print("(rodada sintética: os modelos reais do banco continuam intactos; o app e a parte 3 seguem usando os reais)")
    # Limites: sem review.json o modelo A prevê a aprovação MÉDIA do restaurante (a personalização fica no B);
    # dica ≠ gostou; negativos sorteados facilitam o papel da distância; snapshot no modelo A; sem atributos pessoais.
    return art, tab


# %% 5. PARTE 2 · Modelo com a nota individual (review.json) e RMSE (antigo 03b) ====================
"""
Pergunta de negócio: este consumidor vai GOSTAR deste restaurante (nota 4 ou 5) antes de pedir ou reservar?
1. Dados: business + review do banco (ou amostra sintética com o mesmo esquema)
2. Preparação sem vazamento: atributos calculados SÓ com o passado de cada avaliação
3. Corte temporal: treino até 2018, validação em 2019, teste a partir de 2020
4. Modelos: baselines de popularidade, regressão logística, árvore, random forest, gradient boosting e KNN
5. Avaliação: % de acertos, falsos positivos e falsos negativos, com limiar escolhido pelo custo do erro (F0,5)
6. KPI técnico do BU: RMSE da nota contra a média global e a lista por popularidade
7. Uso: top 5 para um perfil declarado no questionário
"""
CULINARIAS_P2 = ["Pizza", "American", "Italian", "Chinese", "Mexican", "Japanese", "Sandwiches", "Breakfast & Brunch",
                 "Burgers", "Thai", "Indian", "Mediterranean", "Vegan", "Seafood", "Steakhouses", "Korean", "Vietnamese"]


def parte2(con, saidas=None):
    con_y, fonte_y = abrir_yelp(con, ("business", "review"))
    fonte = fonte_y
    pref = prefixo(fonte)
    SAIDA = pasta_saida(saidas, pref + "parte2")
    FONTE = "Yelp Open Dataset (real)" if fonte == "real" else "amostra sintética (esquema do Yelp)"
    cab(f"PARTE 2 · Modelo com a nota individual (fonte: {FONTE})")

    def salvar(nome, df):
        salvar_tabela(con, f"{pref}p2_{nome}", df, SAIDA)

    # ------------------------------------------------------------------ 1. Dados
    biz = ler_business(con_y)
    biz["city"] = biz["city"].str.strip()
    biz = biz[(biz["city"].str.casefold() == CIDADE.casefold())
              & biz["categories"].fillna("").str.contains("Restaurants")].reset_index(drop=True)
    ids = set(biz["business_id"])
    partes = [b[b["business_id"].isin(ids)] for b in
              pd.read_sql("SELECT review_id, user_id, business_id, stars, date FROM yelp_review ORDER BY rowid",
                          con_y, chunksize=200_000)]                # lido em blocos: pouca memória
    rev = pd.concat(partes, ignore_index=True)
    rev["stars"] = rev["stars"].astype(float)
    rev["date"] = pd.to_datetime(rev["date"])
    print(f"Restaurantes: {len(biz):,} | avaliações: {len(rev):,} | usuários: {rev['user_id'].nunique():,}")

    # ------------------------------------------------------------------ 2.1 Limpeza determinística
    # review_id duplicado -> remover | atributos em texto -> parser | preço ausente -> mediana da culinária + indicador
    # business.stars, review_count, user.average_stars são snapshots -> NÃO usar (vazamento); recalcular com o passado
    def limpar_valor(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, str):
            v = v.strip()
            if v in ("None", "none", "'none'", "u'none'"):
                return np.nan
            if v.startswith(("u'", "'")):
                try:
                    return ast.literal_eval(v.lstrip("u"))
                except (ValueError, SyntaxError):
                    return v
            if v in ("True", "False"):
                return v == "True"
        return v

    def culinaria_principal(txt):
        if not isinstance(txt, str):
            return "Outras"
        for c in (x.strip() for x in txt.split(",")):
            if c.startswith("American"):          # no Yelp real: "American (New)" e "American (Traditional)"
                return "American"
            if c in CULINARIAS_P2:
                return c
        return "Outras"

    attrs = biz["attributes"].apply(lambda d: d if isinstance(d, dict) else {})
    rest = pd.DataFrame({
        "business_id": biz["business_id"],
        "nome": biz["name"],
        "is_open": biz["is_open"],
        "culinaria": biz["categories"].apply(culinaria_principal),
        "preco": pd.to_numeric(attrs.apply(lambda d: limpar_valor(d.get("RestaurantsPriceRange2"))), errors="coerce"),
        "delivery": attrs.apply(lambda d: limpar_valor(d.get("RestaurantsDelivery"))),
        "reserva": attrs.apply(lambda d: limpar_valor(d.get("RestaurantsReservations"))),
        "takeout": attrs.apply(lambda d: limpar_valor(d.get("RestaurantsTakeOut"))),
        "stars_snapshot": biz["stars"],              # guardado só para demonstrar o vazamento
    })
    for c in ["delivery", "reserva", "takeout"]:
        rest[c] = rest[c].map({True: 1.0, False: 0.0})

    rest["preco_ausente"] = rest["preco"].isna().astype(int)
    med_cul = rest.groupby("culinaria")["preco"].median()
    rest["preco"] = rest["preco"].fillna(rest["culinaria"].map(med_cul)).fillna(rest["preco"].median())

    n_antes = len(rev)
    rev = rev.drop_duplicates("review_id").copy()
    print(f"Duplicatas removidas: {n_antes - len(rev)}")
    print(f"Faixa de preço ausente: {rest['preco_ausente'].mean():.1%} (imputada pela mediana da culinária)")

    # ------------------------------------------------------------------ 2.2 Alvo e atributos SÓ com o passado
    # Alvo: gostou = 1 se a nota é 4 ou 5. Grão: 1 avaliação. Contagens e médias usam apenas avaliações ANTERIORES
    # (soma acumulada menos a própria linha). Os atributos de perfil são a ponte com o questionário.
    df = (rev[["review_id", "user_id", "business_id", "stars", "date"]]
          .merge(rest, on="business_id", how="inner")
          .sort_values(["date", "review_id"]).reset_index(drop=True))
    df["gostou"] = (df["stars"] >= 4).astype(int)

    CORTE_VALID = pd.Timestamp("2019-01-01")
    CORTE_TESTE = pd.Timestamp("2020-01-01")
    MU = df.loc[df["date"] < CORTE_VALID, "stars"].mean()   # média global aprendida só no treino
    K_REST, K_USER = 10, 3                                    # força da suavização (prior = MU)

    g = df.groupby("business_id")
    df["rest_n_prev"] = g.cumcount()
    soma_r = g["stars"].cumsum() - df["stars"]
    df["rest_media_prev"] = (soma_r + K_REST * MU) / (df["rest_n_prev"] + K_REST)
    P_GOSTOU = df.loc[df["date"] < CORTE_VALID, "gostou"].mean()                 # prior só do treino
    df["rest_gostou_prev"] = ((g["gostou"].cumsum() - df["gostou"]) + K_REST * P_GOSTOU) / (df["rest_n_prev"] + K_REST)

    g = df.groupby("user_id")
    df["user_n_prev"] = g.cumcount()
    soma_u = g["stars"].cumsum() - df["stars"]
    df["user_media_prev"] = np.where(df["user_n_prev"] > 0,
                                     (soma_u + K_USER * MU) / (df["user_n_prev"] + K_USER), np.nan)
    df["user_preco_medio"] = np.where(df["user_n_prev"] > 0,
                                      (g["preco"].cumsum() - df["preco"]) / df["user_n_prev"].replace(0, np.nan), np.nan)
    df["dif_preco"] = (df["preco"] - df["user_preco_medio"]).abs()
    df["usuario_novo"] = (df["user_n_prev"] == 0).astype(int)

    g = df.groupby(["user_id", "culinaria"])
    n_cul = g.cumcount()
    df["afinidade_cul"] = np.where(df["user_n_prev"] > 0, n_cul / df["user_n_prev"].replace(0, np.nan), np.nan)
    df["user_media_cul_prev"] = np.where(n_cul > 0, (g["stars"].cumsum() - df["stars"]) / n_cul.replace(0, np.nan), np.nan)

    NUM = ["preco", "preco_ausente", "delivery", "reserva", "takeout",
           "rest_n_prev", "rest_media_prev", "rest_gostou_prev",
           "user_n_prev", "user_media_prev", "user_preco_medio", "dif_preco", "usuario_novo",
           "afinidade_cul", "user_media_cul_prev"]
    CAT = ["culinaria"]

    df["periodo"] = np.select([df["date"] < CORTE_VALID, df["date"] < CORTE_TESTE], ["treino", "validacao"], "teste")
    resumo = df.groupby("periodo").agg(avaliacoes=("gostou", "size"), taxa_gostou=("gostou", "mean"),
                                       usuarios_novos=("usuario_novo", "mean")).loc[["treino", "validacao", "teste"]]
    resumo["taxa_gostou"] = (100 * resumo["taxa_gostou"]).round(1)
    resumo["usuarios_novos"] = (100 * resumo["usuarios_novos"]).round(1)
    resumo.columns = ["avaliações", "% gostou (nota ≥ 4)", "% de usuário novo"]
    mostrar(resumo)
    # Cerca de 3 em 4 avaliações são positivas: "todo mundo gosta" acerta ≈75% e não entrega nada. Matriz inteira.

    # 2.3 Checagem de vazamento: o stars snapshot do Yelp
    tr = df["periodo"] == "treino"; va = df["periodo"] == "validacao"

    def auc_rapido(cols):
        m = HistGradientBoostingClassifier(max_iter=150, random_state=SEED).fit(df.loc[tr, cols], df.loc[tr, "gostou"])
        return roc_auc_score(df.loc[va, "gostou"], m.predict_proba(df.loc[va, cols])[:, 1])

    auc_ok, auc_leak = auc_rapido(NUM), auc_rapido(NUM + ["stars_snapshot"])
    print(f"\nAUC só com o passado (correto)            : {auc_ok:.3f}")
    print(f"AUC + stars snapshot do Yelp (vazamento)  : {auc_leak:.3f}  (+{auc_leak - auc_ok:.3f})")
    print("-> o ganho vem de notas que ainda não existiam na data da avaliação; o snapshot fica fora do modelo.")

    # ------------------------------------------------------------------ 3. Modeling
    X_tr, y_tr = df.loc[df["periodo"] == "treino", NUM + CAT], df.loc[df["periodo"] == "treino", "gostou"]
    X_va, y_va = df.loc[df["periodo"] == "validacao", NUM + CAT], df.loc[df["periodo"] == "validacao", "gostou"]
    X_te, y_te = df.loc[df["periodo"] == "teste", NUM + CAT], df.loc[df["periodo"] == "teste", "gostou"]
    print(f"\ntreino {len(X_tr):,} | validação {len(X_va):,} | teste {len(X_te):,}")

    def pipeline(modelo):
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("sc", StandardScaler())]), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ])
        return Pipeline([("pre", pre), ("clf", modelo)])

    # 3.2 Árvore: profundidade × overfitting (escolhida pelo AUC na validação)
    linhas = []
    for prof in [2, 3, 4, 6, 8, 10, 14, None]:
        m = pipeline(DecisionTreeClassifier(max_depth=prof, min_samples_leaf=30, random_state=SEED)).fit(X_tr, y_tr)
        linhas.append({"max_depth": "sem limite" if prof is None else prof,
                       "AUC treino": roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1]),
                       "AUC validação": roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])})
    curva = pd.DataFrame(linhas)
    PROF_ARVORE = curva.loc[curva["AUC validação"].idxmax(), "max_depth"]
    PROF_ARVORE = None if PROF_ARVORE == "sem limite" else int(PROF_ARVORE)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(curva["max_depth"].astype(str), curva["AUC treino"], "o-", label="treino")
    ax.plot(curva["max_depth"].astype(str), curva["AUC validação"], "o-", label="validação")
    ax.set_xlabel("max_depth"); ax.set_ylabel("AUC"); ax.set_title("Árvore de decisão: profundidade × overfitting")
    ax.legend(); _fig(SAIDA, "curva_arvore.png")
    print("Profundidade escolhida pela validação:", PROF_ARVORE)
    mostrar(curva.round(3))

    # 3.3 Modelos comparados
    class Popularidade:
        # baseline: probabilidade = média suavizada do restaurante até a data, levada para 0-1
        def fit(self, X, y): return self
        def predict_proba(self, X):
            p = ((X["rest_media_prev"].to_numpy() - 1) / 4).clip(0, 1)
            return np.c_[1 - p, p]

    modelos = {
        "Chute: todo mundo gosta": pipeline(DummyClassifier(strategy="prior")),
        "Popularidade (baseline)": Popularidade(),
        "Regressão logística": pipeline(LogisticRegression(max_iter=3000)),
        "Árvore de decisão": pipeline(DecisionTreeClassifier(max_depth=PROF_ARVORE, min_samples_leaf=30, random_state=SEED)),
        "Random forest": pipeline(RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                                         n_jobs=-1, random_state=SEED)),
        "Gradient boosting": pipeline(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                     max_leaf_nodes=31, random_state=SEED)),
        "KNN (k=50)": pipeline(KNeighborsClassifier(n_neighbors=50, weights="distance")),
    }
    proba_va, proba_te = {}, {}
    for nome, m in modelos.items():
        if nome.startswith("KNN") and len(X_tr) > 60_000:   # KNN guarda o treino inteiro: amostra para caber no tempo
            a = X_tr.sample(60_000, random_state=SEED).index
            m.fit(X_tr.loc[a], y_tr.loc[a])
        else:
            m.fit(X_tr, y_tr)
        proba_va[nome] = m.predict_proba(X_va)[:, 1]
        proba_te[nome] = m.predict_proba(X_te)[:, 1]
        print(f"{nome:28s} AUC validação = {roc_auc_score(y_va, proba_va[nome]):.3f}")

    # 3.4 Limiar pelo custo do erro: F0,5 na validação, aplicado ao teste sem reajuste
    LIMIAR = {nome: melhor_limiar(y_va, p) for nome, p in proba_va.items()}
    mostrar(pd.Series(LIMIAR, name="limiar escolhido na validação").round(3).to_frame())

    # ------------------------------------------------------------------ 4. Evaluation no teste
    # Positivo = "recomendar, a pessoa vai gostar". FP = pedido ruim, cliente perdido (caro); FN = oportunidade perdida.
    def avaliar(y, p, limiar):
        pred = (p >= limiar).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        n = tn + fp + fn + tp
        return {
            "Acertos (% do total)": 100 * (tp + tn) / n,
            "Falso positivo (% do total)": 100 * fp / n,
            "Falso negativo (% do total)": 100 * fn / n,
            "VP (% do total)": 100 * tp / n,
            "VN (% do total)": 100 * tn / n,
            "Taxa de FP (FP ÷ não gostou)": 100 * fp / max(fp + tn, 1),
            "Taxa de FN (FN ÷ gostou)": 100 * fn / max(fn + tp, 1),
            "Precisão": 100 * precision_score(y, pred, zero_division=0),
            "Recall": 100 * recall_score(y, pred, zero_division=0),
            "AUC": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
        }

    resultado = pd.DataFrame({nome: avaliar(y_te, proba_te[nome], LIMIAR[nome]) for nome in modelos}).T
    resultado["Limiar"] = pd.Series(LIMIAR)
    resultado = resultado.sort_values("AUC", ascending=False)
    salvar("metricas_teste", resultado)
    cols_show = ["Acertos (% do total)", "Falso positivo (% do total)", "Falso negativo (% do total)",
                 "Taxa de FP (FP ÷ não gostou)", "Taxa de FN (FN ÷ gostou)", "Precisão", "Recall", "AUC", "Limiar"]
    mostrar(resultado[cols_show].round({c: 1 for c in cols_show[:7]} | {"AUC": 3, "Limiar": 3}), "Resultado no teste")

    comp = resultado[["VP (% do total)", "VN (% do total)", "Falso positivo (% do total)", "Falso negativo (% do total)"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    esq = np.zeros(len(comp))
    cores = {"VP (% do total)": "#2E7D32", "VN (% do total)": "#81C784",
             "Falso positivo (% do total)": "#C62828", "Falso negativo (% do total)": "#F9A825"}
    rotulos = {"VP (% do total)": "Acerto: recomendou e gostou", "VN (% do total)": "Acerto: não recomendou e não gostou",
               "Falso positivo (% do total)": "Falso positivo", "Falso negativo (% do total)": "Falso negativo"}
    for c in comp.columns:
        ax.barh(comp.index, comp[c], left=esq, color=cores[c], label=rotulos[c])
        for i, (v, e) in enumerate(zip(comp[c], esq)):
            if v >= 4:
                ax.text(e + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
        esq += comp[c].to_numpy()
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("% das avaliações do teste")
    ax.set_title("Como cada modelo erra e acerta no teste")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8)
    _fig(SAIDA, "composicao_erros.png")

    nomes = list(resultado.index)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, nome in zip(axes.ravel(), nomes):
        pred = (proba_te[nome] >= LIMIAR[nome]).astype(int)
        ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=[0, 1], normalize="all") * 100,
                               display_labels=["não gostou", "gostou"]).plot(ax=ax, colorbar=False, values_format=".1f",
                                                                             cmap="Blues")
        ax.set_title(nome, fontsize=10); ax.set_xlabel("previsto (%)"); ax.set_ylabel("real")
    for ax in axes.ravel()[len(nomes):]:
        ax.axis("off")
    _fig(SAIDA, "matrizes_confusao.png")

    # 4.0 O efeito do limiar no melhor modelo: subir o limiar troca falso positivo (caro) por falso negativo (barato)
    MELHOR = resultado.drop(["Chute: todo mundo gosta", "Popularidade (baseline)"])["AUC"].idxmax()
    p_best = proba_te[MELHOR]
    tab_lim = pd.DataFrame({f"limiar {t:.2f}" + (" (F0,5)" if t == LIMIAR[MELHOR] else ""): avaliar(y_te, p_best, t)
                            for t in [0.5, LIMIAR[MELHOR], 0.85]}).T
    mostrar(tab_lim[["Acertos (% do total)", "Falso positivo (% do total)", "Falso negativo (% do total)",
                     "Precisão", "Recall"]].round(1), f"Efeito do limiar · {MELHOR}")
    grade = np.linspace(0.3, 0.95, 66)
    curvas = pd.DataFrame([avaliar(y_te, p_best, t) for t in grade], index=grade)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(grade, curvas["Falso positivo (% do total)"], color="#C62828", label="falso positivo (% do total)")
    ax.plot(grade, curvas["Falso negativo (% do total)"], color="#F9A825", label="falso negativo (% do total)")
    ax.plot(grade, curvas["Acertos (% do total)"], color="#2E7D32", label="acertos (% do total)")
    ax.axvline(LIMIAR[MELHOR], ls="--", color="gray"); ax.text(LIMIAR[MELHOR] + 0.01, 50, "limiar F0,5", color="gray")
    ax.set_xlabel("limiar de decisão"); ax.set_ylabel("%"); ax.set_title(f"{MELHOR}: trocando FP por FN")
    ax.legend(fontsize=8); _fig(SAIDA, "efeito_limiar.png")

    # 4.1 Por segmento: usuários novos × ativos (o ganho do perfil só aparece com os grupos separados)
    novo = X_te["usuario_novo"].to_numpy() == 1
    seg = []
    for nome in [MELHOR, "Popularidade (baseline)"]:
        for rot, m in [("usuários novos", novo), ("usuários ativos", ~novo)]:
            if m.sum() == 0 or len(np.unique(y_te.to_numpy()[m])) < 2:
                continue
            r = avaliar(y_te.to_numpy()[m], proba_te[nome][m], LIMIAR[nome])
            seg.append({"modelo": nome, "segmento": rot, "n": int(m.sum()),
                        **{k: r[k] for k in ["Acertos (% do total)", "Falso positivo (% do total)",
                                             "Falso negativo (% do total)", "Taxa de FP (FP ÷ não gostou)",
                                             "Precisão", "AUC"]}})
    seg = pd.DataFrame(seg).set_index(["modelo", "segmento"])
    try:
        a_novo, a_pop_novo = seg.loc[(MELHOR, "usuários novos"), "AUC"], seg.loc[("Popularidade (baseline)", "usuários novos"), "AUC"]
        a_ativo, a_pop_ativo = seg.loc[(MELHOR, "usuários ativos"), "AUC"], seg.loc[("Popularidade (baseline)", "usuários ativos"), "AUC"]
        print(f"\nGanho de AUC sobre a popularidade -> ativos: {a_ativo - a_pop_ativo:+.3f} | novos: {a_novo - a_pop_novo:+.3f}")
        if a_novo - a_pop_novo < 0.01:
            print("Sem histórico, o modelo não supera a popularidade: o ganho vem do perfil. "
                  "É o argumento para o questionário preencher esse perfil no cold start.")
    except KeyError:
        pass
    mostrar(seg.round({c: 1 for c in seg.columns if c != "AUC"} | {"AUC": 3}))

    # 4.2 Interpretação: o que a árvore aprendeu
    arv = modelos["Árvore de decisão"]
    nomes_feat = [n.replace("num__", "").replace("cat__", "") for n in arv.named_steps["pre"].get_feature_names_out()]
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(arv.named_steps["clf"], max_depth=3, feature_names=nomes_feat, class_names=["não gostou", "gostou"],
              filled=True, impurity=True, proportion=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Árvore de decisão (3 primeiros níveis; valores escalonados)")
    _fig(SAIDA, "arvore.png", dpi=110)
    imp = pd.DataFrame({
        "Árvore de decisão": arv.named_steps["clf"].feature_importances_,
        "Random forest": modelos["Random forest"].named_steps["clf"].feature_importances_,
    }, index=nomes_feat).sort_values("Random forest", ascending=False).head(12)
    imp.iloc[::-1].plot.barh(figsize=(8, 5), title="Importância das variáveis (Gini)")
    _fig(SAIDA, "importancias.png")

    # 4.3 Decisão (CRISP-DM), escrita a partir dos números
    r_m, r_p, r_c = resultado.loc[MELHOR], resultado.loc["Popularidade (baseline)"], resultado.loc["Chute: todo mundo gosta"]
    print(f"\nFonte dos dados: {FONTE}")
    print(f"Melhor modelo (AUC no teste): {MELHOR}  ->  AUC {r_m['AUC']:.3f} vs popularidade {r_p['AUC']:.3f}")
    print(f"Limiar de decisão: {r_m['Limiar']:.3f} (F0,5 na validação)\n")
    print(f"{'':32s}{'acertos':>9s}{'FP':>8s}{'FN':>8s}{'precisão':>10s}")
    for n, r in [(MELHOR, r_m), ("Popularidade (baseline)", r_p), ("Chute: todo mundo gosta", r_c)]:
        print(f"{n:32s}{r['Acertos (% do total)']:8.1f}%{r['Falso positivo (% do total)']:7.1f}%"
              f"{r['Falso negativo (% do total)']:7.1f}%{r['Precisão']:9.1f}%")
    ganho_prec = r_m["Precisão"] - r_c["Precisão"]
    corte_fp = r_c["Falso positivo (% do total)"] - r_m["Falso positivo (% do total)"]
    print(f"\nFrente ao chute, o {MELHOR} reduz os falsos positivos em {corte_fp:.1f} p.p. do total e eleva a precisão "
          f"em {ganho_prec:.1f} p.p.;\no custo é deixar de recomendar parte do que a pessoa gostaria (FN), o erro barato.")
    veredito = "APROVAR para a Fase 5" if r_m["AUC"] > r_p["AUC"] and r_m["Precisão"] > r_p["Precisão"] else \
               "VOLTAR à Fase 3 (não supera a popularidade)"
    print("Decisão:", veredito)

    # 4.4 KPI técnico do BU: RMSE da nota (meta: abaixo da lista por popularidade)
    def rmse(y, p):
        return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))

    yr_tr = df.loc[df["periodo"] == "treino", "stars"]
    yr_te = df.loc[df["periodo"] == "teste", "stars"].to_numpy()
    reg = pipeline(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED)).fit(X_tr, yr_tr)
    prev = {
        "Média global (treino)": np.full(len(yr_te), MU),
        "Popularidade (média do restaurante)": X_te["rest_media_prev"].to_numpy(),
        "Popularidade + viés do usuário": (X_te["rest_media_prev"] + (X_te["user_media_prev"] - MU).fillna(0)).clip(1, 5).to_numpy(),
        "Gradient boosting (regressão)": reg.predict(X_te).clip(1, 5),
    }
    novo_te = X_te["usuario_novo"].to_numpy() == 1
    tab_rmse = pd.DataFrame({n: {"RMSE (todos)": rmse(yr_te, p),
                                 "RMSE (usuários novos)": rmse(yr_te[novo_te], p[novo_te]) if novo_te.any() else np.nan,
                                 "RMSE (usuários ativos)": rmse(yr_te[~novo_te], p[~novo_te]) if (~novo_te).any() else np.nan}
                             for n, p in prev.items()}).T
    salvar("rmse_teste", tab_rmse)
    meta_rmse = tab_rmse.loc["Popularidade (média do restaurante)", "RMSE (todos)"]
    ganho = meta_rmse - tab_rmse.loc["Gradient boosting (regressão)", "RMSE (todos)"]
    print(f"\nMeta do BU: RMSE abaixo de {meta_rmse:.3f} (popularidade). Modelo: {'cumpre' if ganho > 0 else 'não cumpre'} "
          f"({ganho:+.3f}).")
    mostrar(tab_rmse.round(3))

    # ------------------------------------------------------------------ 5. Uso: top 5 a partir do questionário
    # As respostas preenchem os atributos de perfil como se a pessoa tivesse 3 avaliações (questionário simulado do
    # BU). Só entra no top quem passa do limiar; se menos de 5 passarem, o top fica menor.
    hist = df.groupby("business_id").agg(n=("stars", "size"), soma=("stars", "sum"), gostou=("gostou", "sum"))
    atual = rest.set_index("business_id").join(hist).fillna({"n": 0, "soma": 0, "gostou": 0})
    atual["rest_n_prev"] = atual["n"]
    atual["rest_media_prev"] = (atual["soma"] + K_REST * MU) / (atual["n"] + K_REST)
    atual["rest_gostou_prev"] = (atual["gostou"] + K_REST * P_GOSTOU) / (atual["n"] + K_REST)
    modelo_final = modelos[MELHOR]

    def recomendar_p2(culinarias, faixa_preco, modo="pedir", top=5, n_perfil=3):
        cand = atual[atual["is_open"] == 1].copy()
        cand = cand[cand["delivery"] == 1] if modo == "pedir" else cand[cand["reserva"] == 1]
        cand = cand[cand["preco"] <= faixa_preco]        # "até $$" é teto declarado: filtro, não preferência
        cand["user_n_prev"] = n_perfil
        cand["usuario_novo"] = 0
        cand["user_media_prev"] = np.nan             # ainda sem notas: imputado pelo pipeline
        cand["user_media_cul_prev"] = np.nan
        cand["user_preco_medio"] = float(faixa_preco)
        cand["dif_preco"] = (cand["preco"] - faixa_preco).abs()
        cand["afinidade_cul"] = np.where(cand["culinaria"].isin(culinarias), 1 / len(culinarias), 0.0)
        cand["prob_gostar"] = modelo_final.predict_proba(cand[NUM + CAT])[:, 1]
        ok = cand[cand["prob_gostar"] >= LIMIAR[MELHOR]].sort_values("prob_gostar", ascending=False).head(top)
        ok["por que"] = [("culinária que você escolheu" if c in culinarias else "fora das suas culinárias, mas muito bem avaliado")
                         + f", {'$' * int(p)}" for c, p in zip(ok["culinaria"], ok["preco"])]
        return ok[["nome", "culinaria", "preco", "rest_media_prev", "rest_n_prev", "prob_gostar", "por que"]].round(3)

    mostrar(recomendar_p2(["Japanese", "Thai"], faixa_preco=2, modo="pedir"), "\nPerfil: quer PEDIR, gosta de Japanese e Thai, até $$")
    mostrar(recomendar_p2(["Italian", "Steakhouses"], faixa_preco=3, modo="sair"), "\nPerfil: quer SAIR, gosta de Italian e Steakhouses, até $$$")

    arq_modelo = SAIDA / "concierge_reviews.joblib"          # pode passar de 100 MB com os dados reais: fica fora do banco
    joblib.dump({"modelo": modelo_final, "nome": MELHOR, "limiar": LIMIAR[MELHOR], "atributos": NUM + CAT,
                 "mu": MU, "fonte": FONTE}, arq_modelo, compress=3)
    print(f"\nModelo em {arq_modelo} | figuras e tabelas em {SAIDA}")
    # Limites: com a amostra sintética os percentuais só validam o código; nenhum usuário do Yelp respondeu ao
    # questionário; o teste começa em 2020 (pandemia); sem gênero, idade ou raça; cada recomendação vem com o motivo.
    return resultado


# %% 6. PARTE 3 · Evaluation, Responsible AI e teste A/B (antigo 04) ================================
"""
Fase 5 do CRISP-DM: validar o Business Case com o produto rodando, medir os riscos de Responsible AI e desenhar
(e deixar pronta a análise do) teste A/B do piloto. Entradas: modelos, restaurantes e tabelas p1_* do banco.
  1 Os modelos superam os baselines no teste?
  2 Como o top 5 se comporta em centenas de respostas ao questionário?
  3 Business Case: quanto o Concierge reduz recomendações ruins e quanto custa em listas menores? (3.1: piso de B)
  4 Responsible AI: o top 5 favorece os grandes? Algum bairro fica sem resposta?
  5 Teste A/B: quantos usuários, por quanto tempo, e como decidir
  6 Decisão: seguir para o deploy, voltar à Fase 3 ou à Fase 1
"""


def parte3(con, saidas=None, sintetico=False):
    conjunto = conjunto_disponivel(con, sintetico)
    pref = prefixo(conjunto)
    art, tab = carregar(con, conjunto)
    SAIDA = pasta_saida(saidas, pref + "parte3")
    cab(f"PARTE 3 · Evaluation, Responsible AI e teste A/B (modelos: {conjunto})")

    def salvar(nome, df):
        salvar_tabela(con, f"{pref}p3_{nome}", df, SAIDA)

    print(f"Base do Yelp até {art['data_base']} | modelo A: {art['nome_A']} (limiar {art['limiar_A']:.2f}) | "
          f"modelo B: {art['nome_B']} (limiar {art['limiar_B']:.2f}) | {len(tab):,} restaurantes")

    # ------------------------------------------------------------------ 1. Os modelos no teste (resumo da parte 1)
    mA = ler_tabela(con, pref + "p1_A_metricas_teste")
    mB = ler_tabela(con, pref + "p1_B_metricas_teste")
    rB = ler_tabela(con, pref + "p1_B_ranking_teste")
    segA = ler_tabela(con, pref + "p1_A_por_n_avaliacoes")

    COLS = ["Acertos (% do total)", "Falso positivo (% do total)", "Falso negativo (% do total)", "Precisão", "AUC"]
    linhas_A = {f"{art['nome_A']} (completo)": mA.loc[f"{art['nome_A']} · Cadastro + comportamento + tom"],
                "Popularidade (check-ins)": mA.loc["Popularidade (check-ins)"],
                "Chute: todos aprovados": mA.loc["Chute: todos aprovados"]}
    resumoA = pd.DataFrame(linhas_A).T[COLS].astype(float)
    nomes_B = [art["nome_B"], "Popularidade (check-ins até a data)", "Mais perto (distância)", "Chute: recomenda tudo"]
    resumoB = mB.loc[nomes_B, COLS].astype(float).join(rB[["HitRate@1 (%)", "MRR"]])
    mostrar(resumoA.round(3), "MODELO A · o restaurante é aprovado (nota ≥ 4)?")
    mostrar(resumoB.round(3), "MODELO B · a pessoa escolhe este restaurante? "
                              "(1 escolhido + 4 sorteados; ao acaso HitRate@1 = 20%, MRR = 0,457)")

    # ------------------------------------------------------------------ 2. O top 5 em centenas de respostas
    # modo × bairro × culinária (12 mais comuns) × teto ($$ a $$$$), raio de 3 km. Variantes: controle (lista atual),
    # concierge sem descoberta, concierge (versão do app e braço de tratamento do A/B).
    top_cul = pd.Series([c for cs in tab["culinarias"] for c in cs]).value_counts().head(12).index.tolist()
    perfis = [dict(modo=m, bairro=b, culinarias=[c], faixa_preco=p, raio_km=3.0)
              for m in ["pedir", "sair"] for b in BAIRROS for c in top_cul for p in [2, 3, 4]]
    VARIANTES = {"controle": {"variante": "controle"},
                 "concierge sem descoberta": {"variante": "concierge", "descoberta": False},
                 "concierge": {"variante": "concierge"}}
    ORDEM = list(VARIANTES)
    sessoes, recs, pool = [], [], []
    for k, pf in enumerate(perfis):
        cand = candidatos(tab, pf["culinarias"], pf["faixa_preco"], pf["modo"], pf["bairro"], pf["raio_km"])
        pool += [{"perfil": k, "business_id": b} for b in cand["business_id"]]
        for var, kw in VARIANTES.items():
            out = recomendar(tab, art, **kw, **pf)
            sessoes.append({"perfil": k, "variante": var, "modo": pf["modo"], "bairro": pf["bairro"],
                            "culinaria": pf["culinarias"][0], "teto": pf["faixa_preco"], "candidatos": len(cand),
                            "itens": len(out)})
            recs += [{"perfil": k, "variante": var, "posicao": i + 1, "business_id": b} for i, b in enumerate(out["business_id"])]

    sessoes, pool = pd.DataFrame(sessoes), pd.DataFrame(pool)
    info = tab[["business_id", "stars", "review_count", "cand_chk_prev_log", "prob_aprovacao"]]
    recs = pd.DataFrame(recs, columns=["perfil", "variante", "posicao", "business_id"]).merge(info, on="business_id")
    print(f"\n{len(perfis)} respostas simuladas | {len(recs):,} recomendações | "
          f"{pool['business_id'].nunique():,} restaurantes elegíveis em algum perfil")
    mostrar(sessoes.groupby("variante").agg(itens_medios=("itens", "mean"),
                                            lista_vazia_pct=("itens", lambda s: 100 * (s == 0).mean()),
                                            lista_cheia_pct=("itens", lambda s: 100 * (s == 5).mean())).round(1))

    # ------------------------------------------------------------------ 3. Business Case
    # Estabelecido (20+ avaliações): a nota atual é a verdade observável; "ruim" = nota < 4 (zero no concierge por
    # construção). Casa nova (< 20): o risco é o falso positivo do modelo A nessa faixa, medido no teste da parte 1.
    estab = recs["review_count"] >= MIN_AVALIACOES
    bc = pd.DataFrame({
        "recomendações": recs.groupby("variante").size(),
        "% em casa estabelecida com nota < 4": 100 * recs[estab].assign(ruim=recs["stars"] < 4).groupby("variante")["ruim"].mean(),
        "% em casa nova (< 20 avaliações)": 100 * (~estab).groupby(recs["variante"]).mean(),
        "itens por lista": sessoes.groupby("variante")["itens"].mean(),
        "% de listas vazias": 100 * sessoes.assign(v=sessoes["itens"] == 0).groupby("variante")["v"].mean(),
    }).loc[ORDEM]
    bc["ruins a cada 1.000 recomendações"] = 10 * bc["% em casa estabelecida com nota < 4"] * (1 - bc["% em casa nova (< 20 avaliações)"] / 100)
    salvar("business_case", bc.round(1))

    faixa_nova = [i for i in segA.index if i[0].startswith("menos de 20") and i[1] != "Cadastro"]
    if faixa_nova:
        r = segA.loc[faixa_nova[0]]
        print(f"\nCasas novas: o modelo A tem precisão de {r['Precisão']:.1f}% e taxa de falso positivo de "
              f"{r['Taxa de FP (FP ÷ negativos)']:.1f}% nessa faixa ({int(r['restaurantes'])} restaurantes no teste).")
    evitadas = bc.loc["controle", "ruins a cada 1.000 recomendações"] - bc.loc["concierge", "ruins a cada 1.000 recomendações"]
    print(f"A cada 1.000 recomendações, o Concierge evita ≈{evitadas:.0f} indicações de casas mal avaliadas; "
          f"o custo é {bc.loc['concierge', '% de listas vazias'] - bc.loc['controle', '% de listas vazias']:+.1f} p.p. de listas vazias.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    bc["% em casa estabelecida com nota < 4"].plot.bar(ax=ax[0], color=["#9E9E9E", "#81C784", "#2E7D32"], rot=0)
    ax[0].set_title("Recomendações em casas com nota < 4 (%)")
    bc["% de listas vazias"].plot.bar(ax=ax[1], color=["#9E9E9E", "#81C784", "#2E7D32"], rot=0)
    ax[1].set_title("Respostas sem nenhuma recomendação (%)")
    _fig(SAIDA, "business_case.png")
    mostrar(bc.round(1))

    # 3.1 Ajuste feito nesta fase: o piso do modelo B (com o limiar cheio ≈1/4 das respostas ficava vazia)
    sensB = ler_tabela(con, pref + "p1_B_limiar_sensibilidade")
    corte = corte_movimento(tab)
    mov = tab.set_index("business_id")["cand_chk_prev_log"]
    linhas = []
    for f in [1.0, 0.75, 0.5, 0.25, 0.0]:
        itens, baixos, total, ids = [], 0, 0, set()
        for pf in perfis:
            out = recomendar(tab, art, fator_limiar_b=f, **pf)
            itens.append(len(out)); ids |= set(out["business_id"])
            baixos += int((mov.loc[out["business_id"]] <= corte).sum()); total += len(out)
        itens = np.array(itens)
        prec = sensB["Precisão"].iloc[[1.0, 0.75, 0.5, 0.25].index(f)] if f > 0 else np.nan
        linhas.append({"piso de B": f"{f:.0%} do limiar" + ("  <- app" if f == FATOR_LIMIAR_B else ""),
                       "% de listas vazias": 100 * (itens == 0).mean(), "itens por lista": itens.mean(),
                       "restaurantes distintos indicados": len(ids),
                       "% das indicações na metade de menor movimento": 100 * baixos / max(total, 1),
                       "precisão de B no teste (%)": prec})
    sens = pd.DataFrame(linhas).set_index("piso de B")
    salvar("sensibilidade_piso_B", sens.round(1))
    mostrar(sens.round(1), "Sensibilidade ao piso do modelo B")

    # ------------------------------------------------------------------ 4. Responsible AI
    # 4.1 Viés para populares e cobertura: cobertura do catálogo, exposição por quartil de movimento e Gini.
    # Mitigação: a vaga de descoberta ("concierge sem descoberta" × "concierge").
    def gini(x):
        x = np.sort(np.asarray(x, dtype=float))
        n = len(x)
        return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())) if x.sum() > 0 else 0.0

    abertos = tab[tab["is_open"] == 1]
    cortes = abertos["cand_chk_prev_log"].quantile([0.25, 0.5, 0.75]).to_numpy()
    ROT_Q = ["Q1 menor movimento", "Q2", "Q3", "Q4 maior movimento"]
    quartil = pd.Series(pd.Categorical(np.array(ROT_Q)[np.searchsorted(cortes, tab["cand_chk_prev_log"], side="right")],
                                       categories=ROT_Q), index=tab["business_id"])
    elegiveis = pool["business_id"].unique()
    expo = pd.DataFrame({"candidatos (base)": pool["business_id"].map(quartil).value_counts(normalize=True)})
    cob, gin = {}, {}
    for var in ORDEM:
        r = recs[recs["variante"] == var]
        expo[var] = r["business_id"].map(quartil).value_counts(normalize=True)
        cob[var] = 100 * r["business_id"].nunique() / len(elegiveis)
        gin[var] = gini(r["business_id"].value_counts().reindex(elegiveis, fill_value=0))
    expo = (100 * expo.reindex(ROT_Q).fillna(0))
    expo.index = list(expo.index)
    salvar("exposicao_por_porte", expo.round(1))
    ax = expo.plot.bar(figsize=(9, 3.6), rot=0, color=["#E0E0E0", "#9E9E9E", "#81C784", "#2E7D32"])
    ax.set_ylabel("% das indicações"); ax.set_title("Exposição por porte (quartis de movimento)")
    _fig(SAIDA, "exposicao_por_porte.png")
    rai = pd.DataFrame({"cobertura do catálogo (%)": cob, "Gini da exposição": gin,
                        "% das indicações na metade de menor movimento": expo.loc[["Q1 menor movimento", "Q2"], ORDEM].sum()})
    salvar("cobertura", rai.round(3))
    mostrar(rai.round(3), "Cobertura e concentração")
    mostrar(expo.round(1), "Exposição por porte (% das indicações)")

    # 4.2 Equidade geográfica: algum bairro fica sem resposta?
    geo = (sessoes.assign(util=sessoes["itens"] >= 3)
           .pivot_table(index="bairro", columns="variante", values="util", aggfunc="mean")[ORDEM] * 100)
    geo["candidatos médios"] = sessoes.groupby("bairro")["candidatos"].mean()
    geo = geo.sort_values("concierge")
    salvar("equidade_bairros", geo.round(1))
    mostrar(geo.round(1), "% das respostas com pelo menos 3 recomendações, por bairro")
    # 4.3 Checklist: sem gênero, idade, raça ou renda; localização só por bairro e raio; cada recomendação com o
    # motivo; restrições alimentares não entram no modelo nem viram dado salvo; braço controle e cobertura como
    # alarme do ciclo de retroalimentação; no Brasil os modelos precisam ser retreinados.

    # ------------------------------------------------------------------ 5. Teste A/B do piloto (MVP 2)
    # Hipótese: o top 5 gera mais escolhas que a lista atual. Sorteio por usuário (50/50, fixo). Métrica principal:
    # conversão da sessão. Secundárias: nota do feedback, posição escolhida. Guardrails: listas vazias, cobertura.
    # Decisão: bilateral, α = 5%, poder de 80%, sem espiar. Checagem de sample ratio mismatch (SRM).
    P0 = 0.10            # premissa: conversão da lista atual (trocar quando o MVP 1 tiver números)
    SESSOES_DIA = 400    # premissa: sessões por dia no piloto (somando os dois braços)

    def n_por_braco(p0, mde_rel, alpha=0.05, poder=0.80):
        p1 = p0 * (1 + mde_rel)
        za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(poder)
        pm = (p0 + p1) / 2
        n = (za * np.sqrt(2 * pm * (1 - pm)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / (p1 - p0) ** 2
        return int(np.ceil(n))

    plano = pd.DataFrame([{"conversão do controle": f"{p0:.0%}", "efeito mínimo (relativo)": f"+{m:.0%}",
                           "sessões por braço": n_por_braco(p0, m), "dias de teste": int(np.ceil(2 * n_por_braco(p0, m) / SESSOES_DIA))}
                          for p0 in [0.05, 0.10, 0.20] for m in [0.10, 0.20, 0.30]])
    salvar("ab_tamanho_amostra", plano)
    N_PLANO = n_por_braco(P0, 0.20)
    print(f"\nPlano escolhido: conversão de {P0:.0%} no controle, detectar +20% -> {N_PLANO:,} sessões por braço, "
          f"≈{np.ceil(2 * N_PLANO / SESSOES_DIA):.0f} dias com {SESSOES_DIA} sessões/dia.")
    mostrar(plano)

    # 5.2 Análise: lê o log do app (tabela eventos_app). Sem log suficiente, roda num log SIMULADO só para mostrar o
    # procedimento. Sessões com variante escolhida à mão (origem = manual) ficam fora.
    def analisar_ab(ev):
        if "origem" in ev.columns:
            ev = ev[ev["origem"] != "manual"]
        exib = ev[ev["evento"] == "exibicao"].groupby(["sessao", "variante"], as_index=False)["n_itens"].max()
        conv = ev[ev["evento"] == "clique"].groupby("sessao").size().rename("cliques")
        s = exib.join(conv, on="sessao").fillna({"cliques": 0})
        s["converteu"] = (s["cliques"] > 0).astype(int)
        nota = ev[ev["evento"] == "feedback"].groupby("sessao")["nota"].mean()
        s = s.join(nota, on="sessao")
        gg = s.groupby("variante").agg(sessoes=("converteu", "size"), conversoes=("converteu", "sum"),
                                       conversao=("converteu", "mean"), nota_media=("nota", "mean"),
                                       listas_vazias=("n_itens", lambda x: (x == 0).mean())).reindex(["controle", "concierge"])
        n0, n1 = gg["sessoes"]
        c0, c1 = gg["conversoes"]
        p0, p1 = c0 / n0, c1 / n1
        srm_p = stats.chisquare([n0, n1]).pvalue
        pp = (c0 + c1) / (n0 + n1)
        z = (p1 - p0) / np.sqrt(pp * (1 - pp) * (1 / n0 + 1 / n1))
        p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
        ep = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
        ic = (p1 - p0 - 1.96 * ep, p1 - p0 + 1.96 * ep)
        n_a, n_b = s.loc[s["variante"] == "concierge", "nota"].dropna(), s.loc[s["variante"] == "controle", "nota"].dropna()
        p_nota = stats.ttest_ind(n_a, n_b, equal_var=False).pvalue if min(len(n_a), len(n_b)) > 1 else np.nan
        return gg, {"diferença (p.p.)": 100 * (p1 - p0), "IC 95% (p.p.)": (100 * ic[0], 100 * ic[1]),
                    "p-valor conversão": p_valor, "p-valor nota": p_nota, "p-valor SRM": srm_p}

    ev = pd.read_sql("SELECT * FROM eventos_app", con)
    REAL = (len(ev) > 0 and ev.loc[ev["evento"] == "exibicao", "sessao"].nunique() >= 200
            and ev["variante"].nunique() == 2)
    if not REAL:
        rng = np.random.default_rng(SEED)
        vazias = sessoes.groupby("variante")["itens"].apply(lambda s: (s == 0).mean())
        linhas = []
        for k in range(2 * N_PLANO):
            var = "concierge" if rng.random() < 0.5 else "controle"
            vazia = rng.random() < vazias[var]
            linhas.append({"sessao": f"s{k}", "variante": var, "evento": "exibicao", "n_itens": 0 if vazia else 5, "nota": np.nan})
            p = 0 if vazia else min(1.0, (P0 * 1.2 if var == "concierge" else P0) / (1 - vazias[var]))  # efeito simulado: +20%
            if rng.random() < p:
                linhas.append({"sessao": f"s{k}", "variante": var, "evento": "clique", "n_itens": np.nan, "nota": np.nan})
            if rng.random() < 0.3 and not vazia:
                linhas.append({"sessao": f"s{k}", "variante": var, "evento": "feedback", "n_itens": np.nan,
                               "nota": float(np.clip(np.round(rng.normal(4.0 if var == "concierge" else 3.8, 0.9)), 1, 5))})
        ev = pd.DataFrame(linhas)

    g, teste = analisar_ab(ev)
    print("\nFonte:", "log do app (tabela eventos_app)" if REAL else "LOG SIMULADO (efeito de +20% colocado de propósito; não é evidência)")
    print(f"SRM: p = {teste['p-valor SRM']:.3f} -> {'divisão ok' if teste['p-valor SRM'] > 0.01 else 'divisão suspeita: checar o sorteio antes de ler o resultado'}")
    print(f"Conversão: {teste['diferença (p.p.)']:+.2f} p.p. (IC 95%: {teste['IC 95% (p.p.)'][0]:+.2f} a {teste['IC 95% (p.p.)'][1]:+.2f}), "
          f"p = {teste['p-valor conversão']:.4f}")
    print(f"Nota do feedback: p = {teste['p-valor nota']:.4f}")
    mostrar(g.round(3))

    # ------------------------------------------------------------------ 6. Decisão (CRISP-DM, Fase 5)
    b_ok = (resumoB.loc[art["nome_B"], "MRR"] > resumoB.loc["Popularidade (check-ins até a data)", "MRR"]
            and resumoB.loc[art["nome_B"], "AUC"] > resumoB.loc["Popularidade (check-ins até a data)", "AUC"])
    a_ok = resumoA.iloc[0]["AUC"] > resumoA.loc["Popularidade (check-ins)", "AUC"]
    fp_ok = bc.loc["concierge", "ruins a cada 1.000 recomendações"] < bc.loc["controle", "ruins a cada 1.000 recomendações"]
    vazia_ok = bc.loc["concierge", "% de listas vazias"] <= 25
    cob_ok = (rai.loc["concierge", "% das indicações na metade de menor movimento"]
              >= rai.loc["controle", "% das indicações na metade de menor movimento"])
    checks = pd.Series({
        "Modelo A supera a popularidade (AUC)": a_ok,
        "Modelo B supera a popularidade (AUC e MRR)": b_ok,
        "Menos indicações de casas mal avaliadas que o controle": fp_ok,
        "Guardrail: até 25% de respostas sem recomendação": vazia_ok,
        "Restaurantes de menor movimento não perdem espaço frente ao controle": cob_ok,
    }, name="ok")
    mostrar(checks.map({True: "sim", False: "não"}).to_frame(), "Checagens da Fase 5")
    if a_ok and b_ok and fp_ok and vazia_ok:
        print("Decisão: seguir para a Fase 6 (deploy do app e piloto A/B)." + ("" if cob_ok else
              " Ponto de atenção: o top 5 concentra mais nos restaurantes movimentados; acompanhar a cobertura no piloto."))
    elif not vazia_ok:
        print("Decisão: voltar à Fase 3/4: a guarda de qualidade deixa respostas demais sem recomendação (rever limiares ou raio).")
    else:
        print("Decisão: voltar à Fase 3 no item que falhou.")
    print(f"\nFiguras e tabelas em {SAIDA}")
    return bc


# %% 7. App · protótipo em Streamlit (Fase 6) =======================================================
"""
Rodar:  python concierge.py app      (ou: streamlit run concierge.py)
Cada sessão é sorteada para controle (lista atual: mesmos filtros, ordenada por movimento e nota) ou concierge
(modelos). Na barra lateral dá para escolher a variante à mão; essas sessões ficam marcadas e não entram no A/B.
Os eventos (lista exibida, clique em pedir/reservar, nota) vão para a tabela eventos_app do banco, que a parte 3 lê.
Deploy no Streamlit Community Cloud: suba este script e o banco num repositório PRIVADO, escolha concierge.py como
arquivo principal e crie um requirements.txt com as bibliotecas do "pip install" do topo.
"""
PRECOS = {1: "$ (barato)", 2: "$$", 3: "$$$", 4: "$$$$ (caro)"}


def _rodando_no_streamlit():
    if "streamlit" not in sys.modules:            # "python concierge.py ..." nunca importa o streamlit aqui
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except TypeError:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _banco_do_app():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--banco", default=None)
    return ap.parse_known_args(sys.argv[1:])[0].banco


def _carregar_app(banco):
    con = conectar(banco)
    conjunto = conjunto_disponivel(con)
    art, tab = carregar(con, conjunto)
    con.close()
    return art, tab, conjunto


def app():
    import hashlib
    import streamlit as st

    banco = _banco_do_app()
    st.set_page_config(page_title="Concierge", page_icon="🍽️", layout="centered")

    def registrar(evento, **campos):
        linha = {c: None for c in CAMPOS_EVENTOS} | {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                      "sessao": st.session_state.sessao, "variante": st.session_state.variante,
                                                      "origem": st.session_state.origem, "evento": evento} | campos
        c = conectar(banco)
        c.execute(f"INSERT INTO eventos_app VALUES ({', '.join('?' * len(CAMPOS_EVENTOS))})",
                  [linha[k] for k in CAMPOS_EVENTOS])
        c.commit(); c.close()

    try:
        art, tab, conjunto = st.cache_resource(_carregar_app)(banco)
    except LookupError as e:
        st.error(f"{e}  Veja as instruções no topo do concierge.py.")
        st.stop()

    # ---------------- sessão e sorteio do teste A/B
    if "sessao" not in st.session_state:
        st.session_state.sessao = uuid.uuid4().hex[:12]
        st.session_state.resultado = None
        st.session_state.escolhidos = set()
        st.session_state.avaliado = False

    with st.sidebar:
        st.header("Sobre")
        st.write(f"Protótipo acadêmico (FGV) com dados do Yelp Open Dataset, Philadelphia, até {art['data_base']}.")
        st.write(f"Modelo de qualidade: {art['nome_A']} · modelo de gosto: {art['nome_B']}.")
        if conjunto != "real":
            st.warning("Modelos treinados na amostra sintética (só para testar o código).")
        st.subheader("Teste A/B")
        escolha = st.radio("Variante", ["sorteio", "concierge", "controle"], index=0,
                           help="Sorteio é o modo do piloto. As outras opções servem para comparar as duas listas.")
        if escolha == "sorteio":
            sorteio = int(hashlib.sha256(st.session_state.sessao.encode()).hexdigest(), 16) % 2
            st.session_state.variante, st.session_state.origem = ("concierge" if sorteio else "controle"), "sorteio"
        else:
            st.session_state.variante, st.session_state.origem = escolha, "manual"
        st.caption(f"Sessão {st.session_state.sessao} · variante **{st.session_state.variante}**")
        c = conectar(banco)
        log = pd.read_sql("SELECT * FROM eventos_app", c); c.close()
        if len(log):
            st.download_button("Baixar log de eventos", log.to_csv(index=False).encode("utf-8"), "eventos.csv", "text/csv")

    # ---------------- questionário
    st.title("🍽️ Concierge")
    st.write("Responda em poucos segundos e receba até 5 lugares para hoje.")
    with st.form("questionario"):
        modo = st.radio("Você quer pedir ou sair para comer?", ["pedir", "sair"], horizontal=True,
                        format_func=lambda m: "Pedir em casa" if m == "pedir" else "Sair para comer")
        bairro = st.selectbox("Onde você está?", list(BAIRROS))
        culinarias = st.multiselect("O que você quer comer hoje? (até 3)", sorted(art["culinarias"]), max_selections=3)
        teto = st.select_slider("Até quanto quer gastar?", options=list(PRECOS), value=2, format_func=PRECOS.get)
        raio = st.slider("Distância máxima (km)", 1.0, 8.0, 3.0, 0.5)
        st.caption("Tem alguma restrição alimentar? Confirme sempre com o restaurante: o Concierge não guarda nem usa essa "
                   "informação.")
        enviar = st.form_submit_button("Ver recomendações", type="primary")

    if enviar:
        if not culinarias:
            st.warning("Escolha pelo menos uma culinária.")
        else:
            out = recomendar(tab, art, culinarias, teto, modo=modo, bairro=bairro, raio_km=raio,
                             variante=st.session_state.variante)
            st.session_state.resultado = {"out": out, "modo": modo, "bairro": bairro, "culinarias": culinarias,
                                          "teto": teto, "raio": raio}
            st.session_state.escolhidos, st.session_state.avaliado = set(), False
            registrar("exibicao", n_itens=len(out), modo=modo, bairro=bairro, culinarias="|".join(culinarias),
                      teto=teto, raio_km=raio)

    # ---------------- resultado
    res = st.session_state.resultado
    if res is not None:
        out = res["out"]
        contexto = {"modo": res["modo"], "bairro": res["bairro"], "culinarias": "|".join(res["culinarias"]),
                    "teto": res["teto"], "raio_km": res["raio"]}
        if out.empty:
            st.info("Nenhum lugar passou nos filtros e na checagem de qualidade. Tente aumentar a distância, o preço ou "
                    "incluir outra culinária.")
        else:
            st.subheader("Para você hoje")
            acao = "Quero pedir" if res["modo"] == "pedir" else "Quero reservar"
            for i, r in out.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{i + 1}. {r['nome']}**" + ("  ·  ✨ descoberta" if r["descoberta"] else ""))
                    st.caption(r["por que"])
                    if r["business_id"] in st.session_state.escolhidos:
                        st.success("Boa escolha! No app real, aqui abriria o pedido ou a reserva.")
                    elif st.button(acao, key=f"btn_{r['business_id']}"):
                        st.session_state.escolhidos.add(r["business_id"])
                        registrar("clique", posicao=i + 1, business_id=r["business_id"], n_itens=len(out), **contexto)
                        st.rerun()
            st.divider()
            if st.session_state.avaliado:
                st.write("Obrigado pela avaliação!")
            else:
                nota = st.radio("Como foi esta recomendação?", [1, 2, 3, 4, 5], index=None, horizontal=True)
                if st.button("Enviar avaliação", disabled=nota is None):
                    registrar("feedback", nota=nota, n_itens=len(out), **contexto)
                    st.session_state.avaliado = True
                    st.rerun()


def abrir_app(banco=None):
    """Sobe o app no navegador (equivale a: streamlit run concierge.py)."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Instalando o streamlit (só na primeira vez)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    tema = ["--theme.base", "light", "--theme.primaryColor", "#B9472A", "--theme.backgroundColor", "#FAF6EF",
            "--theme.secondaryBackgroundColor", "#F0EAE0", "--theme.textColor", "#1E2A2F"]
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *tema]
    if banco:
        cmd += ["--", "--banco", str(banco)]
    subprocess.run(cmd)


# %% 8. Testes do recomendador (tabela pequena feita à mão; não precisa do Yelp nem dos modelos) ====
ATRIBUTOS_B_TESTE = ["user_n_prev_log", "afin_max", "afin_media", "dist_centroide_km", "dist_min_km", "user_preco_medio",
                     "dif_preco", "user_share_delivery", "cand_delivery", "match_delivery", "cand_preco",
                     "cand_chk_prev_log", "cand_chk_365d_log", "cand_tips_prev_log", "cand_idade_anos"]


class _ModeloFalso:
    """Chance de escolha cresce com o movimento: imita o viés para populares do modelo real."""
    def predict_proba(self, X):
        p = np.clip(0.3 + 0.1 * X["cand_chk_prev_log"].to_numpy(), 0, 1)
        return np.c_[1 - p, p]


def _tab_teste():
    lat0, lon0 = BAIRROS["Center City"]

    def r(i, **kw):
        base = {"business_id": f"r{i}", "nome": f"R{i}", "stars": 4.5, "review_count": 100, "is_open": 1,
                "lat": lat0 + 0.001 * i, "lon": lon0, "preco": 2.0, "delivery": 1.0, "reservas": 1.0,
                "culinarias": ["Japanese"], "prob_aprovacao": 0.9, "cand_chk_prev_log": 5.0,
                "cand_chk_365d_log": 2.0, "cand_tips_prev_log": 2.0, "cand_idade_anos": 5.0}
        return base | kw

    linhas = [r(i, cand_chk_prev_log=6.0 + i * 0.1) for i in range(6)]                # populares e bons
    linhas += [
        r(10, stars=3.0),                                                              # estabelecido com nota < 4
        r(11, review_count=5, stars=2.0, prob_aprovacao=0.9, cand_chk_prev_log=0.5),   # novo, A aprova
        r(12, review_count=5, stars=5.0, prob_aprovacao=0.1, cand_chk_prev_log=0.5),   # novo, A reprova
        r(13, preco=4.0),                                                              # acima do teto
        r(14, delivery=0.0),                                                           # não entrega
        r(15, is_open=0),                                                              # fechado
        r(16, culinarias=["Pizza"]),                                                   # outra culinária
        r(17, lat=lat0 + 0.2),                                                         # longe (~22 km)
    ]
    linhas += [r(20 + i, cand_chk_prev_log=0.1, culinarias=["Thai"]) for i in range(6)]   # pouco movimento
    return pd.DataFrame(linhas)


def _art_teste():
    return {"modelo_B": _ModeloFalso(), "limiar_B": 0.5, "limiar_A": 0.5, "atributos_B": ATRIBUTOS_B_TESTE}


def test_atributos_par_tem_as_colunas_do_treino():
    p = perfil_questionario(["Japanese", "Thai"], 2, "pedir", "Center City")
    X = atributos_par(p, _tab_teste())
    assert list(X.columns) == ATRIBUTOS_B_TESTE
    assert math.isclose(X.loc[0, "afin_max"], 0.5)           # 2 culinárias declaradas: metade do perfil em cada
    assert abs(X.loc[0, "dist_centroide_km"]) < 1e-9
    assert X.loc[0, "match_delivery"] == 1.0


def test_filtros_de_negocio():
    cand = candidatos(_tab_teste(), ["Japanese"], 2, "pedir", "Center City", 3.0)
    assert not {"r13", "r14", "r15", "r16", "r17"} & set(cand["business_id"])
    assert (cand["preco"] <= 2).all() and (cand["delivery"] == 1).all()


def test_guarda_de_qualidade():
    out = recomendar(_tab_teste(), _art_teste(), ["Japanese"], 2, top=20, descoberta=False)
    ids = set(out["business_id"])
    assert "r10" not in ids          # estabelecido com nota 3.0
    assert "r12" not in ids          # novo e reprovado pelo modelo A
    assert len(out) <= 20


def test_top5_ordenado_pela_chance_de_escolha():
    out = recomendar(_tab_teste(), _art_teste(), ["Japanese"], 2, descoberta=False)
    assert len(out) == 5
    assert out["prob_escolha"].is_monotonic_decreasing


def test_vaga_de_descoberta():
    out = recomendar(_tab_teste(), _art_teste(), ["Japanese"], 2)
    assert out["descoberta"].sum() == 1 and out.iloc[-1]["business_id"] == "r11"
    assert out.iloc[-1]["por que"].startswith("descoberta")


def test_controle_ordena_por_movimento():
    out = recomendar(_tab_teste(), _art_teste(), ["Japanese"], 2, variante="controle")
    assert out.iloc[0]["business_id"] == "r5" and len(out) == 5


def test_sem_candidatos_devolve_vazio():
    out = recomendar(_tab_teste(), _art_teste(), ["Korean"], 2)
    assert out.empty and list(out.columns) == COLUNAS_SAIDA


def test_culinaria_obrigatoria():
    try:
        recomendar(_tab_teste(), _art_teste(), [], 2)
    except ValueError:
        return
    raise AssertionError("deveria exigir pelo menos uma culinária")


def test_banco_guarda_e_le_tabelas_e_modelos():
    con = sqlite3.connect(":memory:")
    criar_tabelas(con)
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index(["x", "y"]))
    salvar_tabela(con, "t", df)
    assert ler_tabela(con, "t").equals(df)
    salvar_modelo(con, "m", {"limiar_B": 0.4, "sklearn": sklearn.__version__})
    assert carregar_modelo(con, "m")["limiar_B"] == 0.4
    salvar_restaurantes(con, "r", _tab_teste().head(3))
    assert ler_restaurantes(con, "r")["culinarias"].iloc[0] == ["Japanese"]


def test_importacao_filtra_restaurantes_de_philadelphia():
    con = sqlite3.connect(":memory:")
    criar_tabelas(con)
    biz = [{"business_id": "a", "city": " Philadelphia", "categories": "Pizza, Restaurants"},
           {"business_id": "b", "city": "Tampa", "categories": "Restaurants"},
           {"business_id": "c", "city": "Philadelphia", "categories": "Shopping"}]
    tips = [{"user_id": "u", "business_id": "a", "text": "ok", "date": "2020-01-01 10:00:00", "compliment_count": 0},
            {"user_id": "u", "business_id": "b", "text": "ok", "date": "2020-01-01 10:00:00", "compliment_count": 0}]
    n = gravar_yelp(con, biz, tips, [{"business_id": "a", "date": "2020-01-01 10:00:00"}], aviso=lambda *_: None)
    assert n == {"business": 1, "tip": 1, "checkin": 1, "review": 0}
    assert meta(con, "n_negocios_yelp") == "3"


def rodar_testes():
    cab("TESTES do recomendador e do banco")
    testes = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    falhas = 0
    for f in testes:
        try:
            f()
            print(f"  ok      {f.__name__}")
        except Exception as e:                       # noqa: BLE001
            falhas += 1
            print(f"  FALHOU  {f.__name__}: {e!r}")
    print(f"\n{len(testes) - falhas} de {len(testes)} testes passaram.")
    return falhas == 0


# %% 9. Linha de comando ===========================================================================
def status(con, banco):
    cab("Concierge · o que tem no banco")
    print(f"Banco: {banco}")
    fonte = meta(con, "fonte_yelp")
    ny = {t: n_linhas(con, f"yelp_{t}") for t in ("business", "tip", "checkin", "review")}
    if ny["business"]:
        print(f"Yelp ({fonte}): " + " | ".join(f"{t}: {n:,}" for t, n in ny.items()))
    else:
        print("Yelp: tabelas yelp_* vazias -> partes 1 e 2 rodam na amostra sintética "
              "(para usar o Yelp real: python concierge.py importar PASTA)")
    mods = con.execute("SELECT nome, sklearn, criado_em, bytes FROM modelos ORDER BY nome").fetchall()
    for nome, skv, quando, b in mods:
        print(f"Modelo '{nome}': {b / 1e6:.1f} MB, scikit-learn {skv}, gravado em {quando}")
    tabelas = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    res = [t for t in tabelas if t.startswith(("p1_", "p2_", "p3_", "sint_"))]
    print(f"Restaurantes para o app: {n_linhas(con, 'restaurantes'):,} | eventos do app: {n_linhas(con, 'eventos_app'):,}")
    print(f"Tabelas de resultado ({len(res)}): {', '.join(res)}")
    print("\nPartes:  parte1 | parte2 | parte3 | testes | tudo | app | importar PASTA   (ex.: python concierge.py parte3)")


def main(argv=None):
    global MOSTRAR_FIGURAS
    ap = argparse.ArgumentParser(description="Concierge · CRISP-DM em um script só (veja o topo do arquivo).")
    ap.add_argument("comando", nargs="?", default=None,
                    choices=["status", "parte1", "parte2", "parte3", "testes", "tudo", "app", "importar"])
    ap.add_argument("pasta", nargs="?", help="pasta com os JSON do Yelp (só para 'importar')")
    ap.add_argument("--banco", default=None, help="caminho do banco (padrão: concierge_dados.sqlite ao lado do script)")
    ap.add_argument("--saidas", default=None, help="pasta das figuras e tabelas (padrão: saidas/ ao lado do script)")
    ap.add_argument("--sintetico", action="store_true", help="parte 3: usar os modelos da rodada sintética")
    a = ap.parse_args(argv)

    banco = Path(a.banco or os.environ.get("CONCIERGE_BANCO") or BANCO_PADRAO)
    if a.comando is None:                           # rodado sem argumentos (ex.: botão "Run" do VS Code)
        c = conectar(banco); status(c, banco); c.close()
        opcoes = {"1": "parte1", "2": "parte2", "3": "parte3", "t": "testes", "a": "app", "tudo": "tudo"}
        try:
            r = input("\nO que rodar? [1] parte 1  [2] parte 2  [3] parte 3  [t] testes  [a] app  [tudo]  "
                      "(Enter para sair): ").strip().lower()
        except EOFError:
            r = ""
        if r not in opcoes:
            return
        a.comando = opcoes[r]
    if a.comando == "app":
        return abrir_app(a.banco)
    if not banco.exists() and a.comando != "importar":
        print(f"ATENÇÃO: não achei {banco}. Coloque o concierge_dados.sqlite na mesma pasta deste script "
              "(ou passe --banco CAMINHO). Vou criar um banco vazio.")
    plt.switch_backend("Agg")
    MOSTRAR_FIGURAS = False
    con = conectar(banco)
    t0 = time.time()
    if a.comando == "status":
        status(con, banco)
    elif a.comando == "importar":
        if not a.pasta:
            ap.error("informe a pasta com os JSON do Yelp: python concierge.py importar PASTA")
        importar_yelp(con, a.pasta)
    elif a.comando == "parte1":
        parte1(con, a.saidas)
    elif a.comando == "parte2":
        parte2(con, a.saidas)
    elif a.comando == "parte3":
        parte3(con, a.saidas, a.sintetico)
    elif a.comando == "testes":
        if not rodar_testes():
            sys.exit(1)
    elif a.comando == "tudo":
        parte1(con, a.saidas)
        parte2(con, a.saidas)
        parte3(con, a.saidas, a.sintetico)
        rodar_testes()
    con.close()
    if a.comando != "status":
        print(f"\nConcluído em {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    if _rodando_no_streamlit():
        app()
    else:
        main()
