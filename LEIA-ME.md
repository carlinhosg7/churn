# Máquina de Churn — Kidy

## Instalação (uma vez)
    pip install -r requirements.txt

## Uso
1. Coloque as bases na pasta `dados/` (DADOS_PREDITIVA_*.parquet, DADOS_PRDITIVA_LIMITE.xlsx, base_clientes.xlsx, novos_01.csv).
2. `python motor_churn.py`  → consolida tudo, treina o modelo e grava em `saida/` (~40 s).
   Rode de novo sempre que atualizar as bases (ou dê dois cliques em `atualizar_bases.bat`).
3. `streamlit run app_churn.py`  → abre o dashboard no navegador (ou `rodar_dashboard.bat`).

## Arquivos
- `churn_core.py`   regras de cor, cálculo de ciclo/tendência, sinal de churn (edite aqui para mudar limiares)
- `motor_churn.py`  ETL + modelo preditivo (gradient boosting) + validação temporal
- `app_churn.py`    dashboard (filtros de período, supervisor, representante, UF, cor, sinal; export Excel tarjado)

## Regras
- Amarelo: compra desde o 1º dia de 5 meses antes do mês de referência (out/2026 → desde 01/05/2026)
- Roxo: entre o 1º dia de 11 meses antes e o corte amarelo (01/11/2025 a 30/04/2026)
- Vermelho: sem compra desde 31/10/2025
- A data de referência é a data final do filtro de período: muda a data e a régua anda junto (cenários).

## NOVO — Painel de churn por coleção (outubro/2026)
- No topo do `app_churn.py`: painel de retenção por Código Grupo Cliente.
- Verão: maio a outubro; inverno: novembro a abril.
- Churn confirmado: sem compra nas duas últimas coleções COMPLETAS.
- Possível churn: sem compra na última coleção COMPLETA, mas com compra na anterior.
- Ativo: compra na última coleção completa.
- Grupos cuja primeira compra foi na última coleção são exibidos à parte e não entram no denominador da carteira pré-existente.
- Crédito/bloqueio são consolidados dos códigos individuais cadastrados no grupo; um bloqueio parcial aparece como Parcial.
- As abas antigas foram preservadas como legado e podem exibir classificações diferentes.
- Recalcule a pasta `saida` com `python motor_churn.py` quando atualizar as bases.
- Atenção: faturamento histórico não é projeção de perda; limite informado pode não ser saldo de crédito efetivamente disponível.
