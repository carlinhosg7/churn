"""Identidade comercial exclusivamente pelo campo textual 'Grupo Cliente'.
Registros sem Grupo Cliente ficam fora das métricas agrupadas.
"""
import pandas as pd


def chave_grupo(cod_cliente, grupo_cliente):
    nomes = pd.Series(grupo_cliente).astype('string').str.strip().str.replace(r'\s+', ' ', regex=True)
    validos = nomes.notna() & ~nomes.str.upper().isin(['', 'NAN', 'NONE', '<NA>', 'NULL', 'NÃO INFORMADO', 'NAO INFORMADO', 'SEM GRUPO', '0'])
    return nomes.str.upper().where(validos, pd.NA)


def adicionar_grupo(pedidos, dimensao):
    p = pedidos.copy()
    nomes = dimensao['grupo'].reindex(p['cod_cliente']).reset_index(drop=True)
    p['chave_grupo'] = chave_grupo(p['cod_cliente'].reset_index(drop=True), nomes).to_numpy()
    return p


def consolidar_grupos(df, referencia, meses_amarelo, meses_roxo):
    from churn_core import classificar_status, cortes_status
    if df.empty:
        return df.copy()
    d = df.dropna(subset=['chave_grupo']).copy()
    if d.empty:
        return d
    ult = d.sort_values(['ultima_compra', 'valor_periodo']).drop_duplicates('chave_grupo', keep='last').set_index('chave_grupo')
    soma = d.groupby('chave_grupo')[['valor_periodo','valor_365','valor_365_ant','pedidos_periodo']].sum()
    for c in soma.columns:
        ult[c] = soma[c]
    ult['ultima_compra'] = d.groupby('chave_grupo')['ultima_compra'].max()
    ult['recencia_dias'] = (pd.Timestamp(referencia) - ult['ultima_compra']).dt.days
    ini_am, ini_rx = cortes_status(referencia, meses_amarelo, meses_roxo)
    ult['status'] = classificar_status(ult['ultima_compra'], ini_am, ini_rx)
    prioridade = {'Saudável':0, 'Atenção':1, 'Entrando em churn':2, 'Churn':3}
    d['_sinal_ord'] = d['sinal'].map(prioridade).fillna(0)
    maior = d.sort_values('_sinal_ord').drop_duplicates('chave_grupo', keep='last').set_index('chave_grupo')
    ult['sinal'] = maior['sinal']
    ult.loc[ult['status']=='VERMELHO','sinal'] = 'Churn'
    ult.loc[(ult['status']=='AMARELO') & (ult['sinal']=='Churn'),'sinal'] = 'Saudável'
    ult['prob_churn'] = d.groupby('chave_grupo')['prob_churn'].max()
    ult['qtd_codigos_cliente'] = d.groupby('chave_grupo').size()
    return ult.reset_index()
