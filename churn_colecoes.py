"""Churn comercial por coleções completas, agregado exclusivamente por Grupo Cliente."""
import pandas as pd
from grupos_cliente import chave_grupo


def colecao_de(data):
    d = pd.Timestamp(data)
    if d.month >= 11:
        return (d.year * 2 + 1, f'Inverno {d.year}/{d.year+1}')
    if d.month <= 4:
        return ((d.year - 1) * 2 + 1, f'Inverno {d.year-1}/{d.year}')
    return (d.year * 2, f'Verão {d.year}')


def ultima_encerrada(referencia):
    d = pd.Timestamp(referencia).normalize()
    if d.month >= 11:
        fim = pd.Timestamp(d.year, 10, 31)
    elif d.month >= 5:
        fim = pd.Timestamp(d.year, 4, 30)
    else:
        fim = pd.Timestamp(d.year-1, 10, 31)
    return colecao_de(fim)[0], fim


def calcular_colecoes(pedidos, clientes, referencia):
    """Grupo elegível = já comprou antes do início da última coleção encerrada.
    Churn: duas coleções completas consecutivas sem compra; risco: uma.
    Novos grupos ficam separados e não inflam o denominador da carteira anterior.
    """
    codigo, fim = ultima_encerrada(referencia)
    inicio_carteira = pd.Timestamp(2024, 1, 1)
    data_ref = pd.Timestamp(referencia).normalize()
    colunas = ['chave_grupo','cod_cliente','data','valor','cod_rep','cod_sup']
    # Carteira: grupos com compra de 01/01/2024 até a referência, inclusive.
    compras_carteira = pedidos.loc[pedidos['data'].between(inicio_carteira, data_ref) & pedidos['chave_grupo'].notna(), colunas].copy()
    carteira = pd.Index(compras_carteira['chave_grupo'].dropna().unique())
    p = pedidos.loc[pedidos['data'].le(data_ref) & pedidos['chave_grupo'].isin(carteira), colunas].copy()
    p_fechado = p.loc[p['data'].le(fim)].copy()
    if p.empty:
        return pd.DataFrame(), dict(fim=fim, colecao=colecao_de(fim)[1], elegiveis=0, grupos_cadastro=len(carteira), grupos_historicos=0, grupos_com_historico=0, inicio_carteira=inicio_carteira)
    p['data'] = pd.to_datetime(p['data'])
    p['colecao_id'] = p['data'].map(lambda x: colecao_de(x)[0])
    # Metadados financeiros vêm do cadastro, mas ele não define a população.
    dim = clientes.reset_index().copy()
    dim['chave_grupo'] = chave_grupo(dim['cod_cliente'], dim['grupo']).values
    grupos_historicos = pedidos.loc[pedidos['data'].le(fim) & pedidos['chave_grupo'].notna(), 'chave_grupo'].nunique()
    if p.empty:
        return pd.DataFrame(), dict(fim=fim, colecao=colecao_de(fim)[1], elegiveis=0, grupos_cadastro=len(carteira), grupos_historicos=grupos_historicos, inicio_carteira=inicio_carteira)
    ult = p.groupby('chave_grupo').agg(primeira=('data','min'),ultima=('data','max'),receita=('valor','sum'))
    # Última compra verdadeira, inclusive durante a coleção em andamento.
    ult['ultima_compra_real'] = ult['ultima']
    ult_fechado = p_fechado.groupby('chave_grupo')['data'].max()
    ult['ultima_colecao_id'] = ult_fechado.map(lambda x: colecao_de(x)[0] if pd.notna(x) else float('nan'))
    # Quem comprou somente na coleção em andamento não tem histórico encerrado.
    ult['compra_atual'] = p.groupby('chave_grupo')['data'].max().gt(fim)
    ult['colecoes_sem_compra'] = (codigo - ult['ultima_colecao_id']).fillna(0).clip(lower=0).astype(int)
    # IDs são semestrais consecutivos (inverno ímpar / verão par).
    ult['status'] = 'Ativo'
    ult.loc[ult['colecoes_sem_compra'] == 1,'status'] = 'Possível churn'
    ult.loc[ult['colecoes_sem_compra'] >= 2,'status'] = 'Churn'
    inicio_ultima = pd.Timestamp(fim.year, 5 if fim.month == 10 else 11, 1) if fim.month == 10 else pd.Timestamp(fim.year-1,11,1)
    ult['elegivel'] = ult['primeira'] < inicio_ultima
    ult.loc[~ult['elegivel'],'status'] = 'Novo na última coleção'
    # Grupo com compra na coleção atual pode ter histórico de churn anterior;
    # preservamos a situação histórica e sinalizamos a reativação.
    ult['reativado_atual'] = ult['compra_atual'] & ult['status'].isin(['Churn','Possível churn'])
    ult['ultima_colecao'] = ult['ultima'].map(lambda x: colecao_de(x)[1])
    # Metadados financeiros são consolidados em TODOS os CNPJs/códigos do grupo.
    dim = dim[dim['chave_grupo'].isin(ult.index)]
    dim['bloq_flag'] = dim['bloqueado'].astype('string').str.strip().str.upper().isin(['SIM','S','TRUE','1','BLOQUEADO'])
    dim['limite_num'] = pd.to_numeric(dim['limite_disponivel'], errors='coerce')
    fin = dim.groupby('chave_grupo').agg(nome=('razao_social','first'),grupo=('grupo','first'),qtd_empresas=('cod_cliente','nunique'),
                                          bloqueadas=('bloq_flag','sum'),empresas_com_limite=('limite_num',lambda x: int(x.gt(0).sum())),
                                          limite_total=('limite_num',lambda x: x.sum(min_count=1)))
    fin['bloqueio'] = fin.apply(lambda r: 'Todas' if r['bloqueadas']==r['qtd_empresas'] else ('Parcial' if r['bloqueadas'] else 'Não'),axis=1)
    fin['situacao_limite'] = fin['empresas_com_limite'].map(lambda x:'Com limite' if x else 'Sem limite')
    ult = ult.join(fin)
    ult['nome'] = ult['grupo'].fillna(ult['nome']).fillna('Sem identificação')
    # Responsável comercial: maior faturamento histórico do grupo.
    reps = p.groupby(['chave_grupo','cod_rep','cod_sup'],dropna=False)['valor'].sum().reset_index().sort_values('valor',ascending=False).drop_duplicates('chave_grupo').set_index('chave_grupo')
    ult = ult.join(reps[['cod_rep','cod_sup']])
    resultado = ult.reset_index().rename(columns={'chave_grupo':'codigo_grupo','ultima':'ultima_compra','receita':'faturamento_historico'})
    # Regra comercial obrigatória: a última compra do grupo é >= 01/01/2024.
    resultado = resultado.loc[resultado['ultima_compra'] >= inicio_carteira].copy()
    elegiveis = resultado[resultado['elegivel']]
    churn = int((elegiveis['status']=='Churn').sum())
    risco = int((elegiveis['status']=='Possível churn').sum())
    total = len(elegiveis)
    return resultado, dict(fim=fim,colecao=colecao_de(fim)[1],elegiveis=total,grupos_cadastro=len(carteira),grupos_historicos=grupos_historicos,inicio_carteira=inicio_carteira,grupos_com_historico=len(resultado),churn=churn,risco=risco,
                           pct_churn=churn/total if total else 0,pct_risco=risco/total if total else 0,
                           ativos=int((elegiveis['status']=='Ativo').sum()),novos=int((~resultado['elegivel']).sum()))
