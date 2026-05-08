import streamlit as st
import pandas as pd
from supabase import create_client
import plotly.express as px

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Magi@Melchior",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Magi@Melchior")
st.caption("Painel gerencial e operacional de manutenção")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================================
# HELPERS
# =========================================================
def safe_json_get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "created_at",
        "updated_at",
        "operational_born_at",
        "execution_started_at",
        "execution_finished_at",
        "started_real_at",
        "finished_real_at",
        "forecast_finish_at",
    ]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def load_table_paginated(table_name: str, page_size: int = 1000) -> pd.DataFrame:
    all_rows = []
    offset = 0

    while True:
        resp = (
            supabase
            .table(table_name)
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )

        rows = resp.data or []
        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        offset += page_size

    return pd.DataFrame(all_rows)


@st.cache_data(ttl=60)
def load_data():
    os_df = load_table_paginated("os")
    actives_df = load_table_paginated("actives")
    notes_df = load_table_paginated("os_notes")
    return os_df, actives_df, notes_df


def normalize_os(os_df: pd.DataFrame) -> pd.DataFrame:
    if "equipamento" not in os_df.columns:
        os_df["equipamento"] = None

    os_df["equip_id"] = os_df["equipamento"].apply(lambda x: safe_json_get(x, "id"))
    os_df["equip_descr"] = os_df["equipamento"].apply(
        lambda x: safe_json_get(x, "descr", safe_json_get(x, "desc"))
    )
    os_df["equip_cc"] = os_df["equipamento"].apply(lambda x: safe_json_get(x, "CC"))
    os_df["equip_setor"] = os_df["equipamento"].apply(lambda x: safe_json_get(x, "setor"))
    os_df["equip_area"] = os_df["equipamento"].apply(lambda x: safe_json_get(x, "area"))
    os_df["equip_secao"] = os_df["equipamento"].apply(lambda x: safe_json_get(x, "secao"))
    os_df["equip_tipo"] = os_df["equipamento"].apply(lambda x: safe_json_get(x, "tipo"))

    return os_df


def merge_actives_fallback(os_df: pd.DataFrame, actives_df: pd.DataFrame) -> pd.DataFrame:
    if actives_df.empty or "id" not in actives_df.columns:
        return os_df

    actives = actives_df.copy()

    for col in ["descr", "CC", "setor", "area", "secao", "tipo"]:
        if col not in actives.columns:
            actives[col] = None

    actives = actives.rename(columns={
        "id": "active_id",
        "descr": "active_descr",
        "CC": "active_cc",
        "setor": "active_setor",
        "area": "active_area",
        "secao": "active_secao",
        "tipo": "active_tipo",
    })

    os_df = os_df.merge(
        actives[[
            "active_id",
            "active_descr",
            "active_cc",
            "active_setor",
            "active_area",
            "active_secao",
            "active_tipo",
        ]],
        left_on="equip_id",
        right_on="active_id",
        how="left",
    )

    os_df["equip_descr"] = os_df["equip_descr"].fillna(os_df["active_descr"])
    os_df["equip_cc"] = os_df["equip_cc"].fillna(os_df["active_cc"])
    os_df["equip_setor"] = os_df["equip_setor"].fillna(os_df["active_setor"])
    os_df["equip_area"] = os_df["equip_area"].fillna(os_df["active_area"])
    os_df["equip_secao"] = os_df["equip_secao"].fillna(os_df["active_secao"])
    os_df["equip_tipo"] = os_df["equip_tipo"].fillna(os_df["active_tipo"])

    os_df = os_df.drop(columns=[
        "active_id",
        "active_descr",
        "active_cc",
        "active_setor",
        "active_area",
        "active_secao",
        "active_tipo",
    ], errors="ignore")

    return os_df


def merge_latest_notes(os_df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
    if notes_df.empty or "os_id" not in notes_df.columns:
        os_df["forecast_finish_at"] = pd.NaT
        return os_df

    notes_df = notes_df.copy()
    notes_df = parse_dates(notes_df)

    if "created_at" in notes_df.columns:
        notes_df = notes_df.sort_values("created_at")

    keep_cols = ["os_id"]
    for col in [
        "forecast_finish_at",
        "technical_recommendation",
        "suggested_action",
        "note",
        "risk_severity",
        "risk_probability",
    ]:
        if col in notes_df.columns:
            keep_cols.append(col)

    latest = notes_df[keep_cols].drop_duplicates(subset=["os_id"], keep="last")

    os_df = os_df.merge(
        latest,
        left_on="id",
        right_on="os_id",
        how="left",
    )

    if "forecast_finish_at" not in os_df.columns:
        os_df["forecast_finish_at"] = pd.NaT

    return os_df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if {"started_real_at", "finished_real_at"}.issubset(df.columns):
        df["horas_real"] = (
            (df["finished_real_at"] - df["started_real_at"]).dt.total_seconds() / 3600
        )
    else:
        df["horas_real"] = None

    if {"execution_started_at", "execution_finished_at"}.issubset(df.columns):
        df["horas_sistema"] = (
            (df["execution_finished_at"] - df["execution_started_at"]).dt.total_seconds() / 3600
        )
    else:
        df["horas_sistema"] = None

    df["horas_real"] = df["horas_real"].where(
        (df["horas_real"] > 0) & (df["horas_real"] < 500),
        None
    )
    df["horas_sistema"] = df["horas_sistema"].where(
        (df["horas_sistema"] > 0) & (df["horas_sistema"] < 5000),
        None
    )

    df["duracao_horas"] = df["horas_real"]

    if {"execution_finished_at", "finished_real_at"}.issubset(df.columns):
        df["atraso_baixa_horas"] = (
            (df["execution_finished_at"] - df["finished_real_at"]).dt.total_seconds() / 3600
        )
        df["atraso_baixa_horas"] = df["atraso_baixa_horas"].where(
            df["atraso_baixa_horas"] >= 0,
            None
        )
    else:
        df["atraso_baixa_horas"] = None

    if {"created_at", "started_real_at"}.issubset(df.columns):
        df["lead_inicio_real_horas"] = (
            (df["started_real_at"] - df["created_at"]).dt.total_seconds() / 3600
        )
        df["lead_inicio_real_horas"] = df["lead_inicio_real_horas"].where(
            df["lead_inicio_real_horas"] >= 0,
            None
        )
    else:
        df["lead_inicio_real_horas"] = None

    if {"created_at", "finished_real_at"}.issubset(df.columns):
        df["lead_conclusao_real_horas"] = (
            (df["finished_real_at"] - df["created_at"]).dt.total_seconds() / 3600
        )
        df["lead_conclusao_real_horas"] = df["lead_conclusao_real_horas"].where(
            df["lead_conclusao_real_horas"] >= 0,
            None
        )
    else:
        df["lead_conclusao_real_horas"] = None

    return df


def parse_execution_people_count(value):
    if isinstance(value, list):
        return len(value)
    return 0


def parse_execution_people_names(value):
    if not isinstance(value, list):
        return ""
    names = []
    for item in value:
        if isinstance(item, dict):
            nome = item.get("nome") or item.get("name")
            if nome:
                names.append(str(nome))
        elif item:
            names.append(str(item))
    return ", ".join(names)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros globais")

    periodo = None
    if "created_at" in df.columns:
        created_valid = df["created_at"].dropna()
        if not created_valid.empty:
            min_date = created_valid.min().date()
            max_date = created_valid.max().date()

            periodo = st.sidebar.date_input(
                "Período de abertura",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

    status_sel = st.sidebar.multiselect(
        "Status",
        sorted([x for x in df.get("status", pd.Series(dtype=str)).dropna().unique()])
    )
    prioridade_sel = st.sidebar.multiselect(
        "Prioridade",
        sorted([x for x in df.get("prioridade", pd.Series(dtype=str)).dropna().unique()])
    )
    classe_sel = st.sidebar.multiselect(
        "Classe",
        sorted([x for x in df.get("classe", pd.Series(dtype=str)).dropna().unique()])
    )
    setor_sel = st.sidebar.multiselect(
        "Setor",
        sorted([x for x in df.get("equip_setor", pd.Series(dtype=str)).dropna().unique()])
    )
    area_sel = st.sidebar.multiselect(
        "Área",
        sorted([x for x in df.get("equip_area", pd.Series(dtype=str)).dropna().unique()])
    )
    secao_sel = st.sidebar.multiselect(
        "Seção",
        sorted([x for x in df.get("equip_secao", pd.Series(dtype=str)).dropna().unique()])
    )
    cc_sel = st.sidebar.multiselect(
        "CC",
        sorted([x for x in df.get("equip_cc", pd.Series(dtype=str)).dropna().unique()])
    )
    impacto_opts = st.sidebar.multiselect(
        "Impacto produtivo",
        ["Sim", "Não"]
    )

    only_concluidas = st.sidebar.checkbox("Somente concluídas", value=False)
    excluir_animal = st.sidebar.checkbox("Excluir área animal", value=True)

    filtered = df.copy()

    if periodo and len(periodo) == 2 and "created_at" in filtered.columns:
        d0 = pd.Timestamp(periodo[0], tz="UTC")
        d1 = pd.Timestamp(periodo[1], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        filtered = filtered[
            (filtered["created_at"].notna()) &
            (filtered["created_at"] >= d0) &
            (filtered["created_at"] <= d1)
        ]

    if status_sel:
        filtered = filtered[filtered["status"].isin(status_sel)]
    if prioridade_sel:
        filtered = filtered[filtered["prioridade"].isin(prioridade_sel)]
    if classe_sel:
        filtered = filtered[filtered["classe"].isin(classe_sel)]
    if setor_sel:
        filtered = filtered[filtered["equip_setor"].isin(setor_sel)]
    if area_sel:
        filtered = filtered[filtered["equip_area"].isin(area_sel)]
    if secao_sel:
        filtered = filtered[filtered["equip_secao"].isin(secao_sel)]
    if cc_sel:
        filtered = filtered[filtered["equip_cc"].isin(cc_sel)]

    if impacto_opts:
        mask = pd.Series(False, index=filtered.index)
        if "Sim" in impacto_opts:
            mask = mask | (filtered["impacto_produtivo"].fillna(False) == True)
        if "Não" in impacto_opts:
            mask = mask | (filtered["impacto_produtivo"].fillna(False) == False)
        filtered = filtered[mask]

    if only_concluidas:
        filtered = filtered[filtered["status"] == "CONCLUIDA"]

    if excluir_animal:
        filtered = filtered[
            filtered["equip_area"].fillna("").str.lower() != "animal"
        ]

    return filtered


# =========================================================
# LOAD + PREP
# =========================================================
os_df, actives_df, notes_df = load_data()

if os_df.empty:
    st.warning("A tabela 'os' está vazia.")
    st.stop()

os_df = parse_dates(os_df)
notes_df = parse_dates(notes_df) if not notes_df.empty else notes_df

os_df = normalize_os(os_df)
os_df = merge_actives_fallback(os_df, actives_df)
os_df = merge_latest_notes(os_df, notes_df)
os_df = compute_metrics(os_df)

os_df["manutentores_qtd"] = os_df["execution_people"].apply(parse_execution_people_count)
os_df["manutentores_nomes"] = os_df["execution_people"].apply(parse_execution_people_names)

if "forecast_finish_at" in os_df.columns:
    os_df["previsao_realizacao"] = os_df["forecast_finish_at"].dt.strftime("%d/%m/%Y %H:%M")
    os_df["previsao_realizacao"] = os_df["previsao_realizacao"].fillna("Indefinido")
else:
    os_df["previsao_realizacao"] = "Indefinido"

df = apply_filters(os_df)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Tela 01 · Visão gerencial",
    "Tela 02 · Manutentores",
    "Tela 03 · Ativos em destaque",
    "Tela 04 · Ordens de serviço",
])

# =========================================================
# TELA 01
# =========================================================
with tab1:
    st.subheader("Visão gerencial")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("OS abertas", int(df[df["status"] != "CONCLUIDA"].shape[0]))
    c2.metric("OS concluídas", int(df[df["status"] == "CONCLUIDA"].shape[0]))
    c3.metric("OS em execução", int(df[df["status"] == "EM_EXECUCAO"].shape[0]))
    c4.metric("Impacto produtivo", int(df["impacto_produtivo"].fillna(False).sum()))
    c5.metric("Horas reais", round(df["duracao_horas"].fillna(0).sum(), 2))

    gc1, gc2 = st.columns(2)

    with gc1:
        st.markdown("**Quantidade de OS por setor**")
        os_por_setor = (
            df.groupby("equip_setor", dropna=False)["id"]
            .count()
            .reset_index(name="qtd_os")
            .fillna({"equip_setor": "Não informado"})
            .sort_values("qtd_os", ascending=False)
        )
        if not os_por_setor.empty:
            fig = px.bar(
                os_por_setor,
                x="equip_setor",
                y="qtd_os",
                labels={"equip_setor": "Setor", "qtd_os": "Quantidade de OS"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados para o gráfico.")

    with gc2:
        st.markdown("**Horas reais por ativo (OS concluídas)**")
        tempo_ativo = (
            df[(df["status"] == "CONCLUIDA") & (df["duracao_horas"].notna())]
            .groupby(["equip_id", "equip_descr"], dropna=False)["duracao_horas"]
            .sum()
            .reset_index()
            .fillna({"equip_id": "Sem TAG", "equip_descr": "Sem descrição"})
            .sort_values("duracao_horas", ascending=False)
            .head(15)
        )
        if not tempo_ativo.empty:
            tempo_ativo["ativo"] = tempo_ativo["equip_id"] + " • " + tempo_ativo["equip_descr"]
            fig = px.bar(
                tempo_ativo,
                x="ativo",
                y="duracao_horas",
                labels={"ativo": "Ativo", "duracao_horas": "Horas reais"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados para o gráfico.")

    st.markdown("**Ordens de serviço — visão simplificada**")
    gerencial_cols = [
        "id",
        "status",
        "blocked_by",
        "prioridade",
        "manutentores_nomes",
        "impacto_produtivo",
        "previsao_realizacao",
        "equip_id",
        "equip_descr",
        "equip_setor",
        "equip_area",
        "equip_secao",
        "equip_cc",
        "duracao_horas",
    ]
    gerencial_cols = [c for c in gerencial_cols if c in df.columns]

    tbl = df[gerencial_cols].copy()
    if "duracao_horas" in tbl.columns:
        tbl["duracao_horas"] = tbl["duracao_horas"].round(2)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# =========================================================
# TELA 02
# =========================================================
with tab2:
    st.subheader("Visão sistemática dos manutentores por OS")

    maint_rows = []

    for _, row in df.iterrows():
        people = row.get("execution_people")
        if not isinstance(people, list) or len(people) == 0:
            continue

        for person in people:
            nome = None
            papel = None

            if isinstance(person, dict):
                nome = person.get("nome") or person.get("name")
                papel = person.get("papel") or person.get("role")
            elif person:
                nome = str(person)

            if nome:
                maint_rows.append({
                    "manutentor": nome,
                    "papel": papel,
                    "os_id": row.get("id"),
                    "status": row.get("status"),
                    "classe": row.get("classe"),
                    "prioridade": row.get("prioridade"),
                    "equip_id": row.get("equip_id"),
                    "equip_descr": row.get("equip_descr"),
                    "equip_setor": row.get("equip_setor"),
                    "equip_area": row.get("equip_area"),
                    "duracao_horas": row.get("duracao_horas"),
                    "impacto_produtivo": row.get("impacto_produtivo"),
                })

    maint_df = pd.DataFrame(maint_rows)

    if maint_df.empty:
        st.info("Não há dados de manutentores vinculados nas OS filtradas.")
    else:
        mc1, mc2 = st.columns(2)

        with mc1:
            horas_por_manut = (
                maint_df.groupby("manutentor", dropna=False)["duracao_horas"]
                .sum()
                .reset_index()
                .sort_values("duracao_horas", ascending=False)
            )
            st.markdown("**Horas trabalhadas por manutentor**")
            fig = px.bar(
                horas_por_manut.head(20),
                x="manutentor",
                y="duracao_horas",
                labels={"manutentor": "Manutentor", "duracao_horas": "Horas reais"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with mc2:
            os_por_manut = (
                maint_df.groupby("manutentor", dropna=False)["os_id"]
                .nunique()
                .reset_index(name="qtd_os")
                .sort_values("qtd_os", ascending=False)
            )
            st.markdown("**Quantidade de OS por manutentor**")
            fig = px.bar(
                os_por_manut.head(20),
                x="manutentor",
                y="qtd_os",
                labels={"manutentor": "Manutentor", "qtd_os": "Quantidade de OS"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**OS abertas vinculadas por manutentor**")
        abertas_manut = maint_df[maint_df["status"] != "CONCLUIDA"].copy()
        st.dataframe(
            abertas_manut[[
                "manutentor",
                "os_id",
                "status",
                "prioridade",
                "equip_id",
                "equip_descr",
                "equip_setor",
                "equip_area",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Tabela detalhada Manutentor × OS**")
        det = maint_df.copy()
        if "duracao_horas" in det.columns:
            det["duracao_horas"] = det["duracao_horas"].round(2)
        st.dataframe(det, use_container_width=True, hide_index=True)

# =========================================================
# TELA 03
# =========================================================
with tab3:
    st.subheader("Ativos em destaque")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("**Campeões de quebra corretiva**")
        corr = (
            df[df["classe"] == "CORRETIVA"]
            .groupby(["equip_id", "equip_descr"], dropna=False)["id"]
            .count()
            .reset_index(name="qtd_os")
            .fillna({"equip_id": "Sem TAG", "equip_descr": "Sem descrição"})
            .sort_values("qtd_os", ascending=False)
            .head(15)
        )
        if not corr.empty:
            corr["ativo"] = corr["equip_id"] + " • " + corr["equip_descr"]
            fig = px.bar(corr, x="ativo", y="qtd_os")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados.")

    with a2:
        st.markdown("**Campeões de preventiva**")
        prev = (
            df[df["classe"] == "PREVENTIVA"]
            .groupby(["equip_id", "equip_descr"], dropna=False)["id"]
            .count()
            .reset_index(name="qtd_os")
            .fillna({"equip_id": "Sem TAG", "equip_descr": "Sem descrição"})
            .sort_values("qtd_os", ascending=False)
            .head(15)
        )
        if not prev.empty:
            prev["ativo"] = prev["equip_id"] + " • " + prev["equip_descr"]
            fig = px.bar(prev, x="ativo", y="qtd_os")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados.")

    with a3:
        st.markdown("**Ativos com manutenção em aberto**")
        aberto = (
            df[df["status"] != "CONCLUIDA"]
            .groupby(["equip_id", "equip_descr"], dropna=False)["id"]
            .count()
            .reset_index(name="qtd_os")
            .fillna({"equip_id": "Sem TAG", "equip_descr": "Sem descrição"})
            .sort_values("qtd_os", ascending=False)
            .head(15)
        )
        if not aberto.empty:
            aberto["ativo"] = aberto["equip_id"] + " • " + aberto["equip_descr"]
            fig = px.bar(aberto, x="ativo", y="qtd_os")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados.")

# =========================================================
# TELA 04
# =========================================================
with tab4:
    st.subheader("Tabela de Ordens de Serviço")

    show_cols = [
        "id",
        "status",
        "prioridade",
        "classe",
        "motivo",
        "solicitante",
        "descricao",
        "equip_id",
        "equip_descr",
        "equip_setor",
        "equip_area",
        "equip_secao",
        "equip_cc",
        "equip_tipo",
        "impacto_produtivo",
        "created_at",
        "operational_born_at",
        "execution_started_at",
        "execution_finished_at",
        "started_real_at",
        "finished_real_at",
        "duracao_horas",
        "horas_sistema",
        "atraso_baixa_horas",
        "lead_inicio_real_horas",
        "lead_conclusao_real_horas",
        "previsao_realizacao",
        "manutentores_nomes",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    table_df = df[show_cols].copy()

    for col in [
        "duracao_horas",
        "horas_sistema",
        "atraso_baixa_horas",
        "lead_inicio_real_horas",
        "lead_conclusao_real_horas",
    ]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(2)

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown("### Diagnóstico de qualidade dos tempos")
    d1, d2, d3 = st.columns(3)
    d1.metric("OS com horas reais válidas", int(df["duracao_horas"].notna().sum()))
    d2.metric("OS com horas sistema válidas", int(df["horas_sistema"].notna().sum()))
    d3.metric("OS com atraso de baixa > 24h", int(df["atraso_baixa_horas"].fillna(0).gt(24).sum()))

    with st.expander("Ver OS com possível fechamento tardio no sistema"):
        tardias = df[df["atraso_baixa_horas"].fillna(0) > 24].copy()
        tardias_cols = [
            "id",
            "status",
            "equip_id",
            "equip_descr",
            "finished_real_at",
            "execution_finished_at",
            "atraso_baixa_horas",
        ]
        tardias_cols = [c for c in tardias_cols if c in tardias.columns]
        if "atraso_baixa_horas" in tardias.columns:
            tardias["atraso_baixa_horas"] = tardias["atraso_baixa_horas"].round(2)
        st.dataframe(tardias[tardias_cols], use_container_width=True, hide_index=True)