"""Indicadores de reposição da carteira, calculados sobre todo o histórico filtrado."""
import pandas as pd


def calcular_reposicao(pedidos, referencia, horizonte=270):
    ref = pd.Timestamp(referencia).normalize()
    inicio = pd.Timestamp('2026-01-01')
    corte = inicio - pd.Timedelta(days=1)
    p = pedidos.loc[pedidos['data'].le(ref), ['chave_grupo', 'data', 'valor']].copy()
    if p.empty or ref < inicio:
        return None
    p['data'] = pd.to_datetime(p['data'])
    historico = p.groupby('chave_grupo')['data'].agg(['min', 'max'])
    anteriores = p[p['data'] < inicio].groupby('chave_grupo')['data'].max()
    # Carteira inicial: compradores nos 365 dias anteriores a 2026.
    base_inicial = anteriores[anteriores >= inicio - pd.Timedelta(days=365)].index
    # Perda observada: cliente da base inicial sem comprar nos últimos 270 dias.
    ultimas = historico['max'].reindex(base_inicial)
    perdidos = ultimas[ultimas <= ref - pd.Timedelta(days=horizonte)].index
    # Novo conquistado: primeira compra no ano, não apenas cadastro novo.
    novos = historico[historico['min'].ge(inicio)].index
    novos = novos[historico.loc[novos, 'min'].le(ref)]
    n_base, n_perdidos, n_novos = len(base_inicial), len(perdidos), len(novos)
    receita_novos = p[p['chave_grupo'].isin(novos) & p['data'].ge(inicio)]['valor'].sum()
    return dict(base=n_base, perdidos=n_perdidos, novos=n_novos,
                pct_churn=n_perdidos/n_base if n_base else None,
                pct_novos=n_novos/n_base if n_base else None,
                cobertura=n_novos/n_perdidos if n_perdidos else None,
                saldo=n_novos-n_perdidos, receita_novos=receita_novos,
                data_ref=ref, horizonte=horizonte)
