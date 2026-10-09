"""
motor_churn.py — consolida as bases, treina o modelo preditivo e gera as saídas do dashboard.

Uso:
    python motor_churn.py                 # usa a pasta ./dados e grava em ./saida
    python motor_churn.py --dados C:/minha/pasta --saida C:/saida

Entradas esperadas na pasta de dados:
    DADOS_PREDITIVA_*.parquet      vendas (item a item) de 2016 até hoje
    DADOS_PRDITIVA_LIMITE.xlsx     limite disponível / bloqueio financeiro / nomes de rep e supervisor
    base_clientes.xlsx             cadastro (cidade, UF, nomes)
    novos_01.csv                   clientes novos (cadastrados em 2026)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from churn_core import (FEATURES, HORIZONTE_DIAS, calcular_features, classificar_status,
                        cortes_status, sinal_churn)

COLS = ["Codigo Cliente", "Razao Social", "Codigo Grupo Cliente", "Grupo Cliente",
        "Codigo Representante", "Codigo Supervisor", "Numero Pedido", "Data Cadastro",
        "Data Cadastro Cliente", "Codigo Linha", "Linha", "Referencia", "Tipo Pedido",
        "Prazo Medio", "Qtd Venda", "Vlr Venda"]


SUP_CONSUMIDOR_FINAL = 9900  # rep 999: ~17 mil pessoas físicas (2023-2024), canal encerrado -> fora do treino


def corrigir_texto(x):
    """Corrige acentuação quebrada (ex.: 'JosÃ©' -> 'José')."""
    if isinstance(x, str) and ("Ã" in x or "Â" in x):
        try:
            return x.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return x
    return x


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- 0. localizar arquivos
def _todos_arquivos(pasta):
    """Arquivos da pasta de dados (com subpastas) + da pasta do script, sem a pasta saida."""
    aqui = os.path.dirname(os.path.abspath(__file__))
    vistos, lista = set(), []
    for raiz in (pasta, aqui):
        if not os.path.isdir(raiz):
            continue
        for dirpath, dirs, files in os.walk(raiz):
            dirs[:] = [d for d in dirs if d.lower() not in ("saida", "__pycache__", ".venv", "venv")]
            for f in files:
                p = os.path.abspath(os.path.join(dirpath, f))
                if p not in vistos:
                    vistos.add(p)
                    lista.append(p)
    return lista


def achar(pasta, palavras, exts, varios=False):
    """Acha arquivo(s) pelo nome, sem diferenciar maiúsculas/minúsculas."""
    achados = sorted(p for p in _todos_arquivos(pasta)
                     if os.path.splitext(p)[1].lower() in exts
                     and all(w in os.path.basename(p).lower() for w in palavras)
                     and not os.path.basename(p).startswith("~$"))
    if not achados:
        aqui = os.path.dirname(os.path.abspath(__file__))
        conteudo = "\n    ".join(sorted(os.path.relpath(p, aqui) for p in _todos_arquivos(pasta))) or "(vazio)"
        raise FileNotFoundError(
            f"\n\nNão encontrei o arquivo com {palavras} ({'/'.join(exts)}).\n"
            f"Procurei em: {os.path.abspath(pasta)} e {aqui}\n"
            f"Arquivos que existem lá:\n    {conteudo}\n\n"
            "Copie as bases para a pasta 'dados' ao lado do motor_churn.py e rode de novo.\n")
    return achados if varios else achados[0]


# ----------------------------------------------------------------------------- 1. vendas
def carregar_vendas(pasta):
    arqs = [a for a in achar(pasta, ["preditiva"], [".parquet"], varios=True) if "limite" not in a.lower()]
    partes = []
    for a in arqs:
        d = pd.read_parquet(a, columns=COLS)  # os blocos _1.._18 vêm vazios: ignorados
        log(f"  {os.path.basename(a)}: {len(d):,} linhas")
        partes.append(d)
    v = pd.concat(partes, ignore_index=True)
    v = v.drop_duplicates()
    v["data"] = pd.to_datetime(v["Data Cadastro"].astype(str).str[:10], format="%Y/%m/%d")
    v["data_cad_cli"] = pd.to_datetime(v["Data Cadastro Cliente"].astype(str).str[:10],
                                       format="%Y/%m/%d", errors="coerce")
    for c in ["Codigo Cliente", "Codigo Representante", "Codigo Supervisor", "Numero Pedido"]:
        v[c] = pd.to_numeric(v[c], errors="coerce").astype("Int64")
    v = v.rename(columns={"Codigo Cliente": "cod_cliente", "Codigo Representante": "cod_rep",
                          "Codigo Supervisor": "cod_sup", "Numero Pedido": "pedido",
                          "Vlr Venda": "valor", "Qtd Venda": "qtd", "Prazo Medio": "prazo",
                          "Tipo Pedido": "tipo_pedido", "Linha": "linha",
                          "Razao Social": "razao_social", "Grupo Cliente": "grupo",
                          "Codigo Grupo Cliente": "cod_grupo"})
    return v


def nivel_pedido(v):
    """Agrega item -> pedido (reduz ~1,2 mi linhas para ~560 mil)."""
    v = v.assign(pronta_entrega=(v["tipo_pedido"] == "Pronta Entrega").astype(np.int8))
    ped = (v.groupby(["cod_cliente", "pedido"], sort=False)
             .agg(cod_rep=("cod_rep", "first"), cod_sup=("cod_sup", "first"),
                  data=("data", "min"), valor=("valor", "sum"), qtd=("qtd", "sum"),
                  itens=("Referencia", "size"), prazo=("prazo", "mean"),
                  pronta_entrega=("pronta_entrega", "max"))
             .reset_index())
    for c in ["cod_cliente", "cod_rep", "cod_sup", "pedido"]:
        ped[c] = ped[c].astype("int64")
    ped["valor"] = ped["valor"].astype("float64")
    ped["qtd"] = ped["qtd"].astype("float64")
    ped["prazo"] = ped["prazo"].astype("float32")
    ped["itens"] = ped["itens"].astype("int32")
    return ped


# ----------------------------------------------------------------------------- 2. dimensões
def carregar_dimensoes(pasta, v):
    lim = pd.read_excel(achar(pasta, ["limite"], [".xlsx", ".xls"]))
    bc_path = achar(pasta, ["base", "cliente"], [".xlsx", ".xls"])
    bc = pd.read_excel(bc_path, header=2)  # 2 primeiras linhas = cabeçalho do relatório
    bc = bc.loc[:, ~bc.columns.astype(str).str.startswith("Unnamed")]
    bc["Codigo Cliente"] = pd.to_numeric(bc["Codigo Cliente"], errors="coerce")
    bc["Codigo Representante"] = pd.to_numeric(bc["Codigo Representante"], errors="coerce")
    bc = bc.dropna(subset=["Codigo Cliente"])

    # --- representantes: nome mais recente conhecido
    reps = pd.concat([lim[["Codigo Representante", "Representante", "Supervisor"]],
                      bc[["Codigo Representante", "Representante", "Supervisor"]]]).dropna(subset=["Codigo Representante"])
    reps["Codigo Representante"] = reps["Codigo Representante"].astype("int64")
    dim_rep = (reps.groupby("Codigo Representante")["Representante"]
               .agg(lambda s: corrigir_texto(s.mode().iat[0])).rename("nome_rep").reset_index()
               .rename(columns={"Codigo Representante": "cod_rep"}))

    # --- supervisores: nome por código = nome mais frequente entre os reps daquele código
    rs = v[["cod_rep", "cod_sup"]].drop_duplicates().dropna().astype("int64")
    rs = rs.merge(reps.drop_duplicates(["Codigo Representante", "Supervisor"]),
                  left_on="cod_rep", right_on="Codigo Representante", how="left")
    nomes = (rs.dropna(subset=["Supervisor"]).groupby("cod_sup")["Supervisor"]
             .agg(lambda s: s.mode().iat[0]))
    cods = sorted(set(rs["cod_sup"]))
    novos = pd.read_csv(achar(pasta, ["novos"], [".csv"]), sep=None, engine="python", encoding_errors="replace")
    cods = sorted(set(cods) | set(novos["REPSUP"].astype(int)))
    dim_sup = pd.DataFrame({"cod_sup": cods})
    dim_sup["nome_sup"] = [f"{c} - {nomes[c]}" if c in nomes.index else f"{c} - (sem nome)" for c in cods]

    # --- clientes
    ult = v.sort_values("data").groupby("cod_cliente").tail(1)
    dim_cli = ult[["cod_cliente", "razao_social", "cod_grupo", "grupo", "data_cad_cli"]].copy()
    dim_cli["cod_cliente"] = dim_cli["cod_cliente"].astype("int64")
    for c in ["razao_social", "grupo"]:
        dim_cli[c] = dim_cli[c].astype(str).map(corrigir_texto)
    geo = (bc.drop_duplicates("Codigo Cliente")[["Codigo Cliente", "Cidade", "Uf"]]
             .rename(columns={"Codigo Cliente": "cod_cliente", "Cidade": "cidade", "Uf": "uf"}))
    geo["cod_cliente"] = geo["cod_cliente"].astype("int64")
    geo_n = (novos.drop_duplicates("CODCLI")[["CODCLI", "CIDCLI", "SIGUFS"]]
               .rename(columns={"CODCLI": "cod_cliente", "CIDCLI": "cidade", "SIGUFS": "uf"}))
    geo = pd.concat([geo, geo_n]).drop_duplicates("cod_cliente")
    dim_cli = dim_cli.merge(geo, on="cod_cliente", how="left")

    fin = (lim.groupby("Codigo Cliente")
              .agg(limite_disponivel=("Limite Disponivel", "max"),
                   bloqueado=("Bloqueado Financeiro", lambda s: "SIM" if (s == "SIM").any() else "NAO"))
              .reset_index().rename(columns={"Codigo Cliente": "cod_cliente"}))
    dim_cli = dim_cli.merge(fin, on="cod_cliente", how="left")

    novos["DATCAD"] = pd.to_datetime(novos["DATCAD"], dayfirst=True)
    nv = novos.groupby("CODCLI")["DATCAD"].min().rename("data_cad_novo").reset_index().rename(columns={"CODCLI": "cod_cliente"})
    dim_cli = dim_cli.merge(nv, on="cod_cliente", how="left")
    dim_cli["cliente_novo"] = np.where(dim_cli["data_cad_novo"].notna(), "SIM", "NAO")

    # novos que ainda não aparecem nas vendas (cadastrados mas sem faturamento na base)
    sem = novos[~novos["CODCLI"].isin(dim_cli["cod_cliente"])].drop_duplicates("CODCLI")
    novos_sem_venda = sem[["CODCLI", "NOMCLI", "NOM_GRUPO", "CIDCLI", "SIGUFS", "CODREP", "REPSUP", "DATCAD"]].rename(
        columns={"CODCLI": "cod_cliente", "NOMCLI": "razao_social", "NOM_GRUPO": "grupo", "CIDCLI": "cidade",
                 "SIGUFS": "uf", "CODREP": "cod_rep", "REPSUP": "cod_sup", "DATCAD": "data_cadastro"})
    return dim_cli, dim_rep, dim_sup, novos_sem_venda


# ----------------------------------------------------------------------------- 3. modelo
def montar_treino(ped, datas_corte):
    linhas = []
    for t in datas_corte:
        t = pd.Timestamp(t)
        f = calcular_features(ped, t)
        f = f[f["recencia_dias"] < 365]                     # só quem estava ativo no último ano
        fut = ped[(ped["data"] > t) & (ped["data"] <= t + pd.Timedelta(days=HORIZONTE_DIAS))]
        f["churn"] = (~f.index.isin(fut["cod_cliente"].unique())).astype(int)
        f["corte"] = t
        linhas.append(f)
        log(f"  corte {t.date()}: {len(f):,} clientes ativos | churn {f['churn'].mean():.1%}")
    return pd.concat(linhas)


def treinar_modelo(ped, data_max):
    # Cortes trimestrais com dados contínuos (2020-2022 tem base muito reduzida e fica fora)
    ultimo_corte = data_max - pd.Timedelta(days=HORIZONTE_DIAS)
    cortes = [d for d in pd.date_range("2018-03-31", "2019-06-30", freq="QE")] + \
             [d for d in pd.date_range("2024-03-31", ultimo_corte, freq="QE")]
    if cortes[-1] < ultimo_corte - pd.Timedelta(days=30):
        cortes.append(ultimo_corte)
    log(f"Montando base de treino em {len(cortes)} datas de corte…")
    tr = montar_treino(ped, cortes)

    # validação temporal: últimos 2 cortes = teste
    corte_teste = sorted(tr["corte"].unique())[-2]
    a, b = tr[tr["corte"] < corte_teste], tr[tr["corte"] >= corte_teste]
    mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=80, l2_regularization=1.0, random_state=42)
    mdl.fit(a[FEATURES], a["churn"])
    pb = mdl.predict_proba(b[FEATURES])[:, 1]
    metr = {
        "auc_teste": round(float(roc_auc_score(b["churn"], pb)), 3),
        "precisao_teste@0.55": round(float(precision_score(b["churn"], pb >= 0.55)), 3),
        "recall_teste@0.55": round(float(recall_score(b["churn"], pb >= 0.55)), 3),
        "precisao_teste@0.40": round(float(precision_score(b["churn"], pb >= 0.40)), 3),
        "recall_teste@0.40": round(float(recall_score(b["churn"], pb >= 0.40)), 3),
        "taxa_churn_teste": round(float(b["churn"].mean()), 3),
        "linhas_treino": int(len(a)), "linhas_teste": int(len(b)),
        "cortes_treino": [str(pd.Timestamp(x).date()) for x in sorted(a["corte"].unique())],
        "cortes_teste": [str(pd.Timestamp(x).date()) for x in sorted(b["corte"].unique())],
        "horizonte_dias": HORIZONTE_DIAS,
    }
    # backtest do SINAL (modelo ainda sem ver o período de teste)
    ref_bt = pd.Timestamp(corte_teste)
    f = calcular_features(ped, ref_bt)
    ia, ir = cortes_status(ref_bt)
    f["status"] = classificar_status(f["ultima_compra"], ia, ir)
    at = f["recencia_dias"] < 365
    f["prob_churn"] = np.nan
    f.loc[at, "prob_churn"] = mdl.predict_proba(f.loc[at, FEATURES])[:, 1]
    f["sinal"] = sinal_churn(f)
    fut = ped[(ped["data"] > ref_bt) & (ped["data"] <= ref_bt + pd.Timedelta(days=HORIZONTE_DIAS))]["cod_cliente"]
    f["nao_comprou"] = ~f.index.isin(fut.unique())
    bt = f[f["status"] != "VERMELHO"].groupby("sinal")["nao_comprou"].agg(["mean", "size"])
    metr["backtest_sinal"] = {"referencia": str(ref_bt.date()),
                              "pct_nao_comprou": {k: round(float(v), 3) for k, v in bt["mean"].items()},
                              "clientes": {k: int(v) for k, v in bt["size"].items()}}
    log(f"Validação temporal: {metr}")
    mdl.fit(tr[FEATURES], tr["churn"])   # modelo final com todos os cortes
    return mdl, metr


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    aqui = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--dados", default=os.path.join(aqui, "dados"))
    ap.add_argument("--saida", default=os.path.join(aqui, "saida"))
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)

    log(f"Pasta de dados: {os.path.abspath(args.dados)}")
    log("Lendo vendas…")
    v = carregar_vendas(args.dados)
    log(f"Vendas: {len(v):,} itens | {v['data'].min().date()} a {v['data'].max().date()}")

    log("Montando dimensões (clientes, representantes, supervisores)…")
    dim_cli, dim_rep, dim_sup, novos_sem = carregar_dimensoes(args.dados, v)

    log("Agregando no nível de pedido…")
    ped = nivel_pedido(v)
    # linha de produto por cliente x mês (para análise de mix no dashboard)
    mix = (v.assign(mes=v["data"].dt.to_period("M").dt.to_timestamp())
             .groupby(["cod_cliente", "cod_rep", "cod_sup", "mes", "linha"], observed=True)["valor"].sum()
             .reset_index())
    for c in ["cod_cliente", "cod_rep", "cod_sup"]:
        mix[c] = mix[c].astype("int64")
    del v

    mdl, metr = treinar_modelo(ped[ped["cod_sup"] != SUP_CONSUMIDOR_FINAL], ped["data"].max())

    ped.to_parquet(os.path.join(args.saida, "pedidos.parquet"), index=False)
    mix.to_parquet(os.path.join(args.saida, "mix_linhas.parquet"), index=False)
    dim_cli.to_parquet(os.path.join(args.saida, "dim_clientes.parquet"), index=False)
    dim_rep.to_parquet(os.path.join(args.saida, "dim_representantes.parquet"), index=False)
    dim_sup.to_parquet(os.path.join(args.saida, "dim_supervisores.parquet"), index=False)
    novos_sem.to_parquet(os.path.join(args.saida, "novos_sem_venda.parquet"), index=False)
    joblib.dump(mdl, os.path.join(args.saida, "modelo_churn.joblib"))
    with open(os.path.join(args.saida, "metricas_modelo.json"), "w", encoding="utf-8") as f:
        json.dump(metr, f, ensure_ascii=False, indent=2)
    log(f"Pronto. Saídas em {args.saida}. Agora rode:  streamlit run app_churn.py")


if __name__ == "__main__":
    main()
