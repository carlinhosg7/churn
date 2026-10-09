"""
churn_core.py — funções compartilhadas entre o motor (motor_churn.py) e o dashboard (app_churn.py).

Regra de status (padrão, relativa à data de referência):
  AMARELO  -> última compra a partir do 1º dia de (mês da referência - 5)   ex.: ref out/2026 -> desde 01/05/2026
  ROXO     -> última compra a partir do 1º dia de (mês da referência - 11)  ex.: ref out/2026 -> 01/11/2025 a 30/04/2026
  VERMELHO -> última compra antes disso                                    ex.: ref out/2026 -> até 31/10/2025
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATUS_ORDEM = ["AMARELO", "ROXO", "VERMELHO"]
STATUS_DESC = {
    "AMARELO": "Ativo (compra recente)",
    "ROXO": "Inativo recente (risco)",
    "VERMELHO": "Churn (sem compra)",
}
STATUS_COR = {"AMARELO": "#FFD84D", "ROXO": "#9B6BD6", "VERMELHO": "#E0464A"}
STATUS_COR_FUNDO = {"AMARELO": "#FFF4C2", "ROXO": "#E6DAF7", "VERMELHO": "#F9D3D4"}

SINAL_ORDEM = ["Saudável", "Atenção", "Entrando em churn", "Churn"]
SINAL_COR = {"Saudável": "#2E9E5B", "Atenção": "#F0A030", "Entrando em churn": "#E0464A", "Churn": "#7A1F22"}

HORIZONTE_DIAS = 270  # o modelo prevê: "cliente NÃO vai comprar nos próximos 270 dias (~9 meses)"

FEATURES = [
    "recencia_dias", "freq_90", "freq_180", "freq_365", "freq_730",
    "valor_90", "valor_180", "valor_365", "valor_365_ant", "tendencia_valor",
    "tendencia_freq", "ticket_medio_365", "meses_ativos_24", "ciclo_mediano",
    "ciclo_std", "atraso_ciclo", "tempo_casa_dias", "prazo_medio_365",
    "pct_pronta_entrega", "n_reps_365", "itens_por_pedido_365",
]


def cortes_status(ref: pd.Timestamp, meses_amarelo: int = 5, meses_roxo: int = 11):
    """Devolve (inicio_amarelo, inicio_roxo) para uma data de referência."""
    ref = pd.Timestamp(ref).normalize()
    base = ref.replace(day=1)
    ini_amarelo = base - pd.DateOffset(months=meses_amarelo)
    ini_roxo = base - pd.DateOffset(months=meses_roxo)
    return pd.Timestamp(ini_amarelo), pd.Timestamp(ini_roxo)


def classificar_status(ultima: pd.Series, ini_amarelo, ini_roxo) -> pd.Series:
    st = np.where(ultima >= ini_amarelo, "AMARELO", np.where(ultima >= ini_roxo, "ROXO", "VERMELHO"))
    return pd.Series(st, index=ultima.index)


def calcular_features(ped: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    """Features por cliente usando apenas pedidos com data <= ref.

    `ped` é a base no nível de pedido (uma linha por cliente x pedido) com colunas:
    cod_cliente, cod_rep, data, valor, qtd, itens, prazo, pronta_entrega
    """
    ref = pd.Timestamp(ref)
    p = ped[ped["data"] <= ref]
    if p.empty:
        return pd.DataFrame(columns=FEATURES)
    dias = (ref - p["data"]).dt.days.to_numpy()
    p = p.assign(_d=dias)

    g = p.groupby("cod_cliente", sort=False)
    out = pd.DataFrame({
        "ultima_compra": g["data"].max(),
        "primeira_compra": g["data"].min(),
    })
    out["recencia_dias"] = (ref - out["ultima_compra"]).dt.days
    out["tempo_casa_dias"] = (ref - out["primeira_compra"]).dt.days

    for jan in (90, 180, 365, 730):
        m = p["_d"] < jan
        sub = p[m].groupby("cod_cliente")
        out[f"freq_{jan}"] = sub["data"].nunique()
        if jan <= 365:
            out[f"valor_{jan}"] = sub["valor"].sum()
    ant = p[(p["_d"] >= 365) & (p["_d"] < 730)].groupby("cod_cliente")
    out["valor_365_ant"] = ant["valor"].sum()
    out["freq_365_ant"] = ant["data"].nunique()
    out = out.fillna({c: 0 for c in out.columns if c.startswith(("freq_", "valor_"))})

    out["tendencia_valor"] = (out["valor_365"] + 1) / (out["valor_365_ant"] + 1)
    out["tendencia_freq"] = (out["freq_365"] + 1) / (out["freq_365_ant"] + 1)

    u365 = p[p["_d"] < 365].groupby("cod_cliente")
    out["ticket_medio_365"] = (u365["valor"].sum() / u365["data"].nunique()).reindex(out.index).fillna(0)
    out["prazo_medio_365"] = u365["prazo"].mean().reindex(out.index)
    out["pct_pronta_entrega"] = u365["pronta_entrega"].mean().reindex(out.index)
    out["n_reps_365"] = u365["cod_rep"].nunique().reindex(out.index).fillna(0)
    out["itens_por_pedido_365"] = (u365["itens"].sum() / u365["data"].nunique()).reindex(out.index)

    m24 = p[p["_d"] < 730]
    mm = m24["data"].dt.year * 12 + m24["data"].dt.month
    out["meses_ativos_24"] = mm.groupby(m24["cod_cliente"]).nunique().reindex(out.index).fillna(0)

    # ciclo de compra: dias entre "ondas" de compra (pedidos a menos de 15 dias um do outro contam como a mesma
    # onda, pois o cliente costuma fazer vários pedidos por coleção) — últimos 3 anos
    d3 = p[p["_d"] < 1095][["cod_cliente", "data"]].drop_duplicates().sort_values(["cod_cliente", "data"])
    d3["gap"] = d3.groupby("cod_cliente")["data"].diff().dt.days
    ondas = d3[d3["gap"].isna() | (d3["gap"] > 14)].copy()
    ondas["gap"] = ondas.groupby("cod_cliente")["data"].diff().dt.days
    gg = ondas.dropna(subset=["gap"]).groupby("cod_cliente")["gap"]
    out["ciclo_mediano"] = gg.median().reindex(out.index)
    out["ciclo_std"] = gg.std().reindex(out.index)
    out["atraso_ciclo"] = out["recencia_dias"] / out["ciclo_mediano"].clip(lower=30)
    return out


def sinal_churn(df: pd.DataFrame, prob_col: str = "prob_churn") -> pd.Series:
    """Sinalização de entrada em churn (combina regra de ciclo + tendência + modelo).

    - Churn ............ status VERMELHO
    - Entrando em churn  prob >= 0.55 OU (atraso >= 2x o próprio ciclo E 90+ dias sem comprar) OU (ROXO e queda forte)
    - Atenção .......... prob >= 0.40 OU (atraso >= 1,5x ciclo E 60+ dias) OU faturamento 12m caiu > 40%
    - Saudável ......... demais
    """
    prob = df.get(prob_col, pd.Series(0, index=df.index)).fillna(0)
    atraso = df["atraso_ciclo"].fillna(df["recencia_dias"] / 90)
    rec = df["recencia_dias"]
    queda = (df["tendencia_valor"] < 0.6) & (df["valor_365_ant"] > 0)
    entrando = (prob >= 0.55) | ((atraso >= 2.0) & (rec >= 90)) | ((df["status"] == "ROXO") & queda)
    atencao = (prob >= 0.40) | ((atraso >= 1.5) & (rec >= 60)) | queda
    s = np.select([df["status"] == "VERMELHO", entrando, atencao],
                  ["Churn", "Entrando em churn", "Atenção"], "Saudável")
    return pd.Series(s, index=df.index)


def motivo_alerta(r) -> str:
    m = []
    if pd.notna(r.get("prob_churn")) and r["prob_churn"] >= 0.40:
        m.append(f"modelo {r['prob_churn']:.0%}")
    if pd.notna(r.get("atraso_ciclo")) and r["atraso_ciclo"] >= 1.5 and r["recencia_dias"] >= 60:
        m.append(f"atraso {r['atraso_ciclo']:.1f}x do ciclo ({r['ciclo_mediano']:.0f}d)")
    if r.get("tendencia_valor", 1) < 0.6 and r.get("valor_365_ant", 0) > 0:
        m.append(f"faturamento 12m {r['tendencia_valor'] - 1:+.0%}")
    if r.get("bloqueado") == "SIM":
        m.append("bloqueio financeiro")
    return "; ".join(m)
