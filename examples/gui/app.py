"""
gui/app.py — Trade Manager Streamlit UI
3-panel: Tree | Grid+Results | Detail
"""
from __future__ import annotations
import json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE=Path(__file__).resolve().parent; _EXAMPLES=_HERE.parent; _PROJECT=_EXAMPLES.parent
for _p in [str(_EXAMPLES),str(_PROJECT)]:
    if _p not in sys.path: sys.path.insert(0,_p)

import requests, streamlit as st, pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
from db.repository import TradeRepository
from models.enums import TradeDirection

# ── constants ─────────────────────────────────────────────────────────────────
DB_PATH   = str(_PROJECT/"trades.db")
REST_BASE = "http://localhost:8000"
DIRECTIONS= [d.value for d in TradeDirection]
MAX_TREE  = 50   # trades shown per book in tree

LEG_ICONS  = {"FIXED":"🔒","FLOAT":"🌊","BOND":"📄","OPTION":"⚙️",
               "EQUITY":"📈","CREDIT":"🛡️","EQUITY_OPTION":"🎯"}
INST_ICONS = {
    # Keys match the trade_type registry strings stored in DB / returned by REST
    "InterestRateSwap": "🔄",
    "CrossCurrencySwap": "💱",
    "Bond": "📄",
    "AssetSwap": "🔗",
    "OptionableBond": "📊",
    "Option": "⚙️",          # OptionTrade registry key
    "IRSwaption": "📐",       # InterestRateSwaption registry key
    "EquitySwap": "📈",
    "CDS": "🛡️",              # CreditDefaultSwap registry key
    "EquityOption": "🎯",     # EquityOptionTrade registry key
    "CapFloor": "🏦",         # CapFloor registry key
}

# leg_type -> (rate_field, display_multiplier, unit_label)
RATE_MAP: Dict[str,tuple] = {
    "FIXED":        ("coupon_rate",   100,    "Cpn%"),
    "BOND":         ("coupon_rate",   100,    "Cpn%"),
    "FLOAT":        ("spread",        10000,  "Sprd bps"),
    "OPTION":       ("vol",           100,    "Vol%"),
    "EQUITY_OPTION":("vol",           100,    "Vol%"),
    "CREDIT":       ("credit_spread", 10000,  "Sprd bps"),
    "EQUITY":       ("dividend_yield",100,    "DivYld%"),
    "CAP_FLOOR":    ("strike",        100,    "Strike%"),
}

FREQ_OPTS = ["ANNUAL","SEMIANNUAL","QUARTERLY","MONTHLY"]
DC_OPTS   = ["30/360","ACT/360","ACT/ACT","ACT/365"]

# Uniform result columns shown in results table
RESULT_COLS = [
    ("NPV",         "npv",             "${:,.0f}"),
    ("DV01",        "dv01",            "${:,.0f}"),
    ("CR01",        "cr01",            "${:,.0f}"),
    ("Duration",    "duration",        "{:.4f}"),
    ("Delta",       "delta",           "{:.4f}"),
    ("Gamma",       "gamma",           "{:.6f}"),
    ("Vega",        "vega",            "${:,.0f}"),
    ("Theta",       "theta",           "${:,.0f}"),
    ("Clean Px",    "clean_price",     "{:.4f}"),
    ("Par Rate",    "par_rate",        "{:.4f}"),
]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trade Manager", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
/* ── Hide header, reduce padding (from python-crud) ──────────────────── */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
.block-container { padding-top: 0.3rem !important; padding-bottom: 0.3rem !important; }

/* ── Column borders ───────────────────────────────────────────────────── */
div[data-testid="column"]:not(:last-child) {
    border-right: 1px solid #ddd; padding-right: 6px; }

/* ── All buttons: compact, left-aligned (from python-crud) ───────────── */
.stButton > button {
    font-size: 0.75rem; padding: 2px 5px; text-align: left;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 100%; }

/* ── Category label (Books / Trades sub-headers) ─────────────────────── */
.cat { font-size: 0.70rem; color: #888; font-weight: 600;
       padding: 1px 0; margin: 0; line-height: 1.5; }

/* ── Panel header ─────────────────────────────────────────────────────── */
.ph { font-size: 0.85rem; font-weight: 700; margin: 0 0 4px 0; }

/* ── leg/section headers ─────────────────────────────────────────────── */
.leg-hdr{background:#f0f4ff;padding:5px 10px;border-radius:4px;
         border-left:3px solid #1a73e8;margin:8px 0 4px;
         font-size:.80rem;font-weight:700;color:#1a3a6b;}
.sec-hdr{background:#f8f9fa;padding:4px 8px;border-radius:3px;
         font-size:.78rem;font-weight:700;color:#333;margin:6px 0 2px;
         border-bottom:1px solid #dee2e6;}

/* ── panel header bar ────────────────────────────────────────────────── */
.ph2{font-size:.90rem;font-weight:700;padding:3px 0 6px;
    border-bottom:2px solid #1a73e8;margin-bottom:7px;color:#1a73e8;}

/* ── AG-Grid: bold headers ───────────────────────────────────────────── */
.ag-header-cell-text{font-weight:700!important;font-size:.78rem!important;}
.ag-header-cell-label{overflow:visible!important;}
</style>""", unsafe_allow_html=True)

# ── data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _load_all() -> List[Dict]:
    repo=TradeRepository(DB_PATH)
    try: return [t._to_dict() for t in repo.list_all()]
    finally: repo.close()

def _build_tree(trades:List[Dict]) -> Dict[str,Dict[str,List[Dict]]]:
    tree:Dict[str,Dict[str,List]]={}
    for t in trades:
        tr=t.get("trader") or "(no trader)"; bk=t.get("book") or "(no book)"
        tree.setdefault(tr,{}).setdefault(bk,[]).append(t)
    return tree

def _fetch_one(tid:str) -> Optional[Dict]:
    repo=TradeRepository(DB_PATH)
    try: t=repo.get(tid); return t._to_dict() if t else None
    finally: repo.close()

def _save_dict(d:Dict) -> bool:
    from models.trade_base import TradeBase
    try:
        trade=TradeBase.fromJson(json.dumps(d))
        repo=TradeRepository(DB_PATH); repo.upsert(trade); repo.close(); return True
    except Exception as e: st.error(f"Save failed: {e}"); return False

def _leg_detail(leg:Dict) -> str:
    lt=leg.get("leg_type","")
    ccy=leg.get("currency","USD") or "USD"
    ccy_tag=f" [{ccy}]" if ccy and ccy!="USD" else ""
    if lt=="FIXED":
        return f"{leg.get('coupon_rate',0)*100:.3f}% {leg.get('day_count','')} {leg.get('frequency','')}{ccy_tag}"
    if lt=="FLOAT":
        return f"{leg.get('index_name','SOFR')}+{leg.get('spread',0)*10000:.1f}bps {leg.get('frequency','')}{ccy_tag}"
    if lt=="BOND":
        return f"Cpn {leg.get('coupon_rate',0)*100:.3f}% Rdem {leg.get('redemption',100):.0f}{ccy_tag}"
    if lt=="OPTION":
        return f"{leg.get('option_type','OPT')} K={leg.get('strike',0)*100:.2f}% vol={leg.get('vol',0)*100:.1f}%"
    if lt=="EQUITY":
        return f"{leg.get('underlying_ticker','')} S0={leg.get('initial_price',0):.1f} dy={leg.get('dividend_yield',0)*100:.2f}%"
    if lt=="CREDIT":
        return f"{leg.get('reference_entity','')} {leg.get('credit_spread',0)*10000:.0f}bps RR={leg.get('recovery_rate',0)*100:.0f}%"
    if lt=="EQUITY_OPTION":
        return f"{leg.get('option_type','C')} {leg.get('underlying_ticker','')} K={leg.get('strike',0):.1f} vol={leg.get('vol',0)*100:.1f}%"
    return lt

def _trades_to_df(trades:List[Dict]) -> pd.DataFrame:
    rows=[]
    for t in trades:
        legs=t.get("legs",[])
        inst=t.get("trade_type","")
        # Map trade_type to display instrument code
        if inst == "AssetSwap":
            display_inst = "ASSWAP"
        else:
            display_inst = inst
        row:Dict[str,Any]={
            "trade_id":        t.get("trade_id",""),
            "instrument":      display_inst,
            "book":            t.get("book",""),
            "tenor_y":         int(t.get("tenor_y") or 0),
            "direction":       t.get("direction",""),
            "counterparty":    t.get("counterparty",""),
            "trader":          t.get("trader",""),
            "valuation_date":  t.get("valuation_date",""),
            "n_legs":          len(legs)}
        # FX rate column for IRS/XCCY — only populate when non-unity
        if inst in ("InterestRateSwap","CrossCurrencySwap"):
            fx = float(t.get("fx_rate", 1.0) or 1.0)
            row["fx_rate"] = fx
            # Show pay/recv ccy for multi-currency IRS
            ccys = {lg.get("currency","USD") for lg in legs if lg.get("currency")}
            if len(ccys) > 1 or inst == "CrossCurrencySwap":
                row["pay_ccy"]  = legs[0].get("currency","USD") if legs else ""
                row["recv_ccy"] = legs[1].get("currency","USD") if len(legs)>1 else ""
        for i,leg in enumerate(legs[:3],1):
            lt=leg.get("leg_type","")
            row[f"leg{i}_type"]    = lt
            row[f"leg{i}_notional"]= float(leg.get("notional",0))
            row[f"leg{i}_detail"]  = _leg_detail(leg)
            if lt in RATE_MAP:
                field,mult,_=RATE_MAP[lt]
                row[f"leg{i}_rate"]=float(leg.get(field,0))*mult
            else:
                row[f"leg{i}_rate"]=0.0
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ── pricing ───────────────────────────────────────────────────────────────────
def _price_rest(tid:str) -> Dict:
    try:
        r=requests.get(f"{REST_BASE}/trades/{tid}/price",timeout=60)
        d=r.json()
        d["trade_id"]=tid
        # normalise: always expose both "npv" and "swap_npv"
        if "npv" not in d or d["npv"] is None:
            d["npv"]=d.get("swap_npv")
        return d
    except Exception as e: return {"trade_id":tid,"error":str(e),"npv":None}

def _price_book_parallel(trades:List[Dict]) -> List[Dict]:
    ids=[t.get("trade_id","") for t in trades if t.get("trade_id")]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_price_rest,tid):tid for tid in ids}
        results=[f.result() for f in as_completed(futs)]
    return sorted(results,key=lambda x:x.get("trade_id",""))

def _fmt_result_val(v:Any, fmt:str) -> str:
    if v is None or v=="": return "—"
    try:
        fv=float(v)
        import math
        if math.isnan(fv) or math.isinf(fv): return "—"
        return fmt.format(fv)
    except: return "—"

def _results_to_df(results:List[Dict], book_trades:List[Dict]) -> pd.DataFrame:
    # Build lookup for instrument/direction from trades
    info={t.get("trade_id",""):t for t in book_trades}
    rows=[]
    for r in results:
        tid=r.get("trade_id","")
        tr=info.get(tid,{})
        row:Dict[str,Any]={
            "trade_id": tid,
            "instrument": tr.get("trade_type",""),
            "direction":  tr.get("direction",""),
            "status":     "✅" if not r.get("error") else f"❌ {r['error']}"
        }
        for label,field,fmt in RESULT_COLS:
            row[label]=_fmt_result_val(r.get(field),fmt)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ── session state ─────────────────────────────────────────────────────────────
def _init():
    defaults={"sel_trader":None,"sel_book":None,"sel_trade_id":None,
              "price_result":None,"results_df":None,"results_book":None}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
_init()

# ── LEFT: portfolio tree (modelled after python-crud tree_view.py) ────────────
def render_tree(tree: Dict):
    """Trader → Book → Trade tree.
    Pattern from python-crud: one root expander, type=primary/secondary buttons,
    st.columns offsets for indentation. No nested expanders."""
    n_total = sum(len(v) for bks in tree.values() for v in bks.values())

    st.markdown(f'<p class="ph">Traders ({len(tree)})</p>', unsafe_allow_html=True)

    sel_trader = st.session_state.sel_trader
    sel_book   = st.session_state.sel_book
    sel_tid    = st.session_state.sel_trade_id

    with st.expander(f"Traders ({len(tree)})  ·  {n_total:,} trades", expanded=True):

        for trader in sorted(tree):
            books = tree[trader]
            n_trd = sum(len(v) for v in books.values())
            tact  = (trader == sel_trader)

            # ── Depth 0: Trader button (re-click collapses) ───────────────
            if st.button(
                f"{'▶ ' if tact else ''}👤 {trader}  ·  {n_trd}",
                key=f"tr_{trader}",
                use_container_width=True,
                type="primary" if tact else "secondary",
            ):
                # Toggle: clicking selected trader collapses it
                if tact:
                    st.session_state.sel_trader   = None
                    st.session_state.sel_book     = None
                    st.session_state.sel_trade_id = None
                else:
                    st.session_state.sel_trader   = trader
                    st.session_state.sel_book     = None
                    st.session_state.sel_trade_id = None
                st.rerun()

            if not tact:
                continue

            # ── Depth 1: Books sub-header + book buttons ──────────────────
            _, cx = st.columns([0.06, 0.94])
            with cx:
                st.markdown('<p class="cat">📚 Books</p>', unsafe_allow_html=True)

            for book in sorted(books):
                trades = books[book]
                bact   = (book == sel_book)
                _, bc  = st.columns([0.08, 0.92])
                with bc:
                    if st.button(
                        f"{'▶ ' if bact else ''}📖 {book}  ({len(trades)})",
                        key=f"bk_{trader}_{book}",
                        use_container_width=True,
                        type="primary" if bact else "secondary",
                    ):
                        # Toggle: clicking selected book collapses it
                        if bact:
                            st.session_state.sel_book     = None
                            st.session_state.sel_trade_id = None
                        else:
                            st.session_state.sel_trader   = trader
                            st.session_state.sel_book     = book
                            st.session_state.sel_trade_id = None
                        st.rerun()

                if not bact:
                    continue

                # ── Depth 2: Trades sub-header + trade buttons ─────────────
                _, cx2 = st.columns([0.16, 0.84])
                with cx2:
                    st.markdown('<p class="cat">📊 Trades</p>', unsafe_allow_html=True)

                shown = sorted(trades, key=lambda x: x.get("trade_id", ""))[:MAX_TREE]
                for td in shown:
                    tid   = td.get("trade_id", "")
                    inst  = td.get("trade_type", "")
                    ico   = INST_ICONS.get(inst, "📋")
                    iact  = (tid == sel_tid)
                    _, tc = st.columns([0.18, 0.82])
                    with tc:
                        if st.button(
                            f"{'▶ ' if iact else ''}{ico} {tid}",
                            key=f"td_{tid}",
                            use_container_width=True,
                            type="primary" if iact else "secondary",
                        ):
                            st.session_state.sel_trader   = trader
                            st.session_state.sel_book     = book
                            st.session_state.sel_trade_id = tid
                            st.rerun()

                if len(trades) > MAX_TREE:
                    _, tc = st.columns([0.18, 0.82])
                    tc.caption(f"…+{len(trades)-MAX_TREE} more")

# ── MIDDLE: grid + buttons + results ──────────────────────────────────────────
def render_grid(book_trades:List[Dict]):
    trader=st.session_state.sel_trader; book=st.session_state.sel_book

    if not trader or not book:
        st.info("← Select a book in the tree"); return

    if not book_trades: st.warning("No trades in this book."); return

    df=_trades_to_df(book_trades)

    # ── AgGrid ────────────────────────────────────────────────────────────────
    # widths = ceil(chars * 8px) + 56px overhead (sort icon + padding)
    LEG_TYPES=["FIXED","FLOAT","BOND","OPTION","EQUITY","CREDIT","EQUITY_OPTION"]
    gb=GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True,resizable=True,sortable=True,filter=True,minWidth=100)
    gb.configure_column("trade_id",      editable=False, width=160, pinned="left", headerName="Trade ID")
    gb.configure_column("instrument",    editable=False, width=150, headerName="Instrument")
    gb.configure_column("book",          width=140,                 headerName="Book")
    gb.configure_column("tenor_y",       width=150, type=["numericColumn"], headerName="Tenor (yrs)")
    gb.configure_column("n_legs",        editable=False, width=110, headerName="# Legs")
    gb.configure_column("direction",     width=135,                 headerName="Direction",
                        cellEditor="agSelectCellEditor",cellEditorParams={"values":DIRECTIONS})
    gb.configure_column("counterparty",  width=160,                 headerName="Counterparty")
    gb.configure_column("trader",        width=110,                 headerName="Trader")
    gb.configure_column("valuation_date",width=135,                 headerName="Val. Date")

    for col in df.columns:
        if not col.startswith("leg"): continue
        n=col[3]
        if col.endswith("_type"):
            gb.configure_column(col, width=150, headerName=f"Leg {n} Type",
                cellEditor="agSelectCellEditor", cellEditorParams={"values":LEG_TYPES})
        elif col.endswith("_notional"):
            gb.configure_column(col, width=180, type=["numericColumn"],
                valueFormatter="'$'+Math.round(value).toLocaleString()",
                headerName=f"Leg {n} Notional")
        elif col.endswith("_rate"):
            gb.configure_column(col, width=150, type=["numericColumn"],
                valueFormatter="value.toFixed(3)",
                headerName=f"Leg {n} Rate")
        elif col.endswith("_detail"):
            gb.configure_column(col, editable=False, width=280, headerName=f"Leg {n} Detail")

    # FX Rate and pay/recv currency columns (multi-currency IRS / XCCY)
    if "fx_rate" in df.columns:
        gb.configure_column("fx_rate",  width=145, type=["numericColumn"],
            valueFormatter="value.toFixed(4)", headerName="FX Rate (rcv/pay)")
    if "pay_ccy" in df.columns:
        CCYS=["USD","EUR","GBP","JPY","CHF","AUD","CAD","CNH","SGD","HKD"]
        gb.configure_column("pay_ccy",  width=130, headerName="Pay Currency",
            cellEditor="agSelectCellEditor", cellEditorParams={"values":CCYS})
        gb.configure_column("recv_ccy", width=140, headerName="Recv Currency",
            cellEditor="agSelectCellEditor", cellEditorParams={"values":CCYS})

    sel_tid=st.session_state.sel_trade_id or ""
    gb.configure_selection("single",use_checkbox=False)
    gb.configure_grid_options(
        rowHeight=24, headerHeight=30,
        onGridReady=JsCode(f"""
            function(params){{
                params.api.forEachNode(function(node){{
                    if(node.data && node.data.trade_id==='{sel_tid}'){{
                        node.setSelected(true);
                        params.api.ensureIndexVisible(node.rowIndex,'middle');
                    }}
                }});
            }}
        """))

    # Key includes sel_tid so grid remounts (triggering onGridReady) when tree selection changes
    resp=AgGrid(df,gridOptions=gb.build(),update_mode=GridUpdateMode.SELECTION_CHANGED,
                data_return_mode=DataReturnMode.AS_INPUT,fit_columns_on_grid_load=False,
                theme="streamlit",height=min(420,50+len(df)*25),
                key=f"grid_{trader}_{book}_{sel_tid}",allow_unsafe_jscode=True,reload_data=False)

    # Sync grid row click → all three panels
    # When sel_trade_id changes here, tree re-renders (● highlight) and detail panel refetches
    sel=resp.get("selected_rows")
    sel_list=(sel.to_dict("records") if hasattr(sel,"to_dict") else list(sel)) if sel is not None else []
    if sel_list:
        sid=sel_list[0].get("trade_id")
        if sid and sid!=st.session_state.sel_trade_id:
            st.session_state.sel_trade_id=sid
            st.session_state.price_result=None
            st.rerun()

    # ── Button row: Clone · Refresh · Save · Price Trade · Price Book ─────────
    has_sel=bool(st.session_state.sel_trade_id)
    b1,b2,b3,b4,b5=st.columns([1,1,1,1.4,1.4])
    clone_clicked       =b1.button("📋 Clone",      key=f"btn_cln_{trader}_{book}",disabled=not has_sel)
    refresh_clicked     =b2.button("🔄 Refresh",    key=f"btn_ref_{trader}_{book}")
    save_clicked        =b3.button("💾 Save",        key=f"btn_sav_{trader}_{book}")
    price_trade_clicked =b4.button("💰 Price Trade", key=f"btn_pt_{trader}_{book}",disabled=not has_sel)
    price_book_clicked  =b5.button("📊 Price Book",  key=f"btn_pb_{trader}_{book}")

    # Clone
    if clone_clicked and has_sel:
        import copy
        src=_fetch_one(st.session_state.sel_trade_id)
        if src:
            cloned=copy.deepcopy(src); base=st.session_state.sel_trade_id
            existing={t.get("trade_id","") for t in book_trades}
            n=1
            while f"{base}-CLN-{n}" in existing: n+=1
            cloned["trade_id"]=f"{base}-CLN-{n}"
            if _save_dict(cloned):
                _load_all.clear(); st.session_state.sel_trade_id=cloned["trade_id"]
                st.success(f"✅ Cloned → {cloned['trade_id']}"); st.rerun()

    # Refresh
    if refresh_clicked:
        _load_all.clear(); st.session_state.results_df=None
        st.session_state.price_result=None; st.rerun()

    # Save
    if save_clicked:
        upd_df=resp["data"]
        if hasattr(upd_df,"iterrows"):
            ok=fail=0
            for _,row in upd_df.iterrows():
                t=_fetch_one(row["trade_id"])
                if not t: continue
                for k in ["direction","counterparty","trader","valuation_date","book"]:
                    if k in row: t[k]=str(row[k])
                try: t["tenor_y"]=int(float(row.get("tenor_y",t.get("tenor_y",0))))
                except: pass
                # FX rate for multi-currency IRS / XCCY
                if "fx_rate" in row:
                    try: t["fx_rate"]=float(row["fx_rate"])
                    except: pass
                # Pay/recv currency update → write back to leg currency fields
                if "pay_ccy" in row and row["pay_ccy"] and t.get("legs"):
                    t["legs"][0]["currency"] = str(row["pay_ccy"])
                if "recv_ccy" in row and row["recv_ccy"] and len(t.get("legs",[])) > 1:
                    t["legs"][1]["currency"] = str(row["recv_ccy"])
                for i in range(1,4):
                    if i>len(t.get("legs",[])): break
                    leg=t["legs"][i-1]; lt=leg.get("leg_type","")
                    # leg type
                    new_lt=str(row.get(f"leg{i}_type",lt)).strip()
                    if new_lt: leg["leg_type"]=new_lt; lt=new_lt
                    # notional
                    nc=f"leg{i}_notional"; rc=f"leg{i}_rate"
                    if nc in row:
                        try: leg["notional"]=float(row[nc])
                        except: pass
                    if rc in row and lt in RATE_MAP:
                        field,mult,_=RATE_MAP[lt]
                        try: leg[field]=float(row[rc])/mult
                        except: pass
                if _save_dict(t): ok+=1
                else: fail+=1
            _load_all.clear()
            st.success(f"✅ Saved {ok}"+( f", {fail} failed" if fail else "")); st.rerun()

    # Price Trade
    if price_trade_clicked and st.session_state.sel_trade_id:
        tid=st.session_state.sel_trade_id
        with st.spinner(f"Pricing {tid}…"):
            res=_price_rest(tid)
        # Show as single-row results table
        st.session_state.results_df=_results_to_df([res],book_trades)
        st.session_state.results_book=book

    # Price Book
    if price_book_clicked:
        with st.spinner(f"Pricing {len(book_trades)} trades in {book}…"):
            results=_price_book_parallel(book_trades)
        st.session_state.results_df=_results_to_df(results,book_trades)
        st.session_state.results_book=book

    # ── Results panel (below buttons) ─────────────────────────────────────────
    if st.session_state.results_df is not None and st.session_state.results_book==book:
        rdf=st.session_state.results_df
        st.markdown("---")
        st.markdown("**📋 Pricing Results**")

        # Summary metrics
        npv_vals=[]
        for v in rdf["NPV"]:
            try:
                s=str(v).replace("$","").replace(",","")
                if s!="—": npv_vals.append(float(s))
            except: pass

        if npv_vals:
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Total NPV",  f"${sum(npv_vals):,.0f}")
            m2.metric("Avg NPV",    f"${sum(npv_vals)/len(npv_vals):,.0f}")
            m3.metric("Priced",     f"{len(npv_vals)}/{len(rdf)}")
            ok_n=len(rdf[rdf["status"]=="✅"])
            m4.metric("Success",    f"{ok_n}/{len(rdf)}")

        # Results AgGrid — pre-select row matching sel_trade_id
        sel_tid=st.session_state.sel_trade_id
        sel_row_idx=-1
        if sel_tid and "trade_id" in rdf.columns:
            matches=rdf.index[rdf["trade_id"]==sel_tid].tolist()
            if matches: sel_row_idx=int(matches[0])

        gb_r=GridOptionsBuilder.from_dataframe(rdf)
        gb_r.configure_default_column(editable=False,resizable=True,sortable=True,minWidth=100)
        gb_r.configure_column("trade_id",  width=160, pinned="left", headerName="Trade ID")
        gb_r.configure_column("instrument",width=155, headerName="Instrument")
        gb_r.configure_column("direction", width=130, headerName="Direction")
        gb_r.configure_column("status",    width=100, headerName="Status")
        for _lbl,_field,_fmt in RESULT_COLS:
            if _lbl in rdf.columns:
                # width = max(ceil(len(header)*8)+56, 110)
                _w = max(int(len(_lbl)*8)+56, 110)
                gb_r.configure_column(_lbl, width=_w)
        gb_r.configure_selection("single",use_checkbox=False)
        gb_r.configure_grid_options(rowHeight=24,headerHeight=30,
            onGridReady=JsCode(f"""
                function(params){{
                    params.api.forEachNode(function(node){{
                        if(node.data && node.data.trade_id==='{sel_tid or ""}'){{
                            node.setSelected(true);
                            params.api.ensureIndexVisible(node.rowIndex,'middle');
                        }}
                    }});
                }}
            """))

        resp_r=AgGrid(rdf, gridOptions=gb_r.build(),
                      update_mode=GridUpdateMode.SELECTION_CHANGED,
                      data_return_mode=DataReturnMode.AS_INPUT,
                      fit_columns_on_grid_load=False, theme="streamlit",
                      height=min(380,52+len(rdf)*26),
                      # key includes sel_tid so grid re-mounts when selection changes
                      key=f"rgrid_{trader}_{book}_{sel_tid}",
                      allow_unsafe_jscode=True, reload_data=False)

        # Selection in results grid → sync sel_trade_id
        rsel=resp_r.get("selected_rows")
        rsel_list=(rsel.to_dict("records") if hasattr(rsel,"to_dict") else list(rsel)) if rsel is not None else []
        if rsel_list:
            rid=rsel_list[0].get("trade_id")
            if rid and rid!=st.session_state.sel_trade_id:
                st.session_state.sel_trade_id=rid; st.rerun()

        if st.button("✖ Clear Results", key=f"clr_{trader}_{book}"):
            st.session_state.results_df=None; st.rerun()

# ── RIGHT: trade detail ────────────────────────────────────────────────────────
def _parse_date(v:Any)->date:
    try: return date.fromisoformat(str(v))
    except: return date.today()

def _fi(v, default=0.0)->float:
    try: return float(v)
    except: return default

def _ii(v, default=0)->int:
    try: return int(v)
    except: return default

def _render_leg(leg:Dict, idx:int, tid:str, sfx:str="p") -> Dict:
    """Render editable fields for one leg. Returns updated leg dict."""
    out = dict(leg)
    lt  = leg.get("leg_type","FIXED")
    pk  = f"{tid}_L{idx}_{sfx}"

    # Common fields (all leg types)
    a1,a2,a3 = st.columns(3)
    out["notional"]   = a1.number_input("Notional", value=_fi(leg.get("notional",0)),
                            min_value=0.0, step=100_000.0, format="%.0f", key=f"ntl_{pk}")
    out["start_date"] = str(a2.date_input("Start Date", value=_parse_date(leg.get("start_date")), key=f"sd_{pk}"))
    out["end_date"]   = str(a3.date_input("End Date",   value=_parse_date(leg.get("end_date")),   key=f"ed_{pk}"))

    b1,b2,b3 = st.columns(3)
    out["currency"] = b1.text_input("Currency", value=leg.get("currency","USD"), key=f"ccy_{pk}")
    freq = leg.get("frequency","SEMIANNUAL")
    out["frequency"] = b2.selectbox("Frequency", FREQ_OPTS,
                           index=FREQ_OPTS.index(freq) if freq in FREQ_OPTS else 1, key=f"frq_{pk}")
    dc = leg.get("day_count","30/360")
    out["day_count"] = b3.selectbox("Day Count", DC_OPTS,
                           index=DC_OPTS.index(dc) if dc in DC_OPTS else 0, key=f"dc_{pk}")

    # Type-specific fields
    if lt == "FIXED":
        out["coupon_rate"] = st.number_input("Coupon Rate (%)",
            value=_fi(leg.get("coupon_rate",0))*100, min_value=0.0, max_value=50.0,
            step=0.01, format="%.4f", key=f"cpn_{pk}") / 100

    elif lt == "FLOAT":
        f1,f2,f3 = st.columns(3)
        IDX = ["SOFR3M","SOFR1M","LIBOR3M","EURIBOR3M"]
        ix  = leg.get("index_name","SOFR3M")
        out["spread"]        = f1.number_input("Spread (bps)",
            value=_fi(leg.get("spread",0))*10000, min_value=-500.0, max_value=500.0,
            step=0.5, format="%.2f", key=f"sprd_{pk}") / 10000
        out["index_name"]    = f2.selectbox("Index", IDX,
            index=IDX.index(ix) if ix in IDX else 0, key=f"idx_{pk}")
        out["index_tenor_m"] = _ii(f3.number_input("Index Tenor (m)",
            value=_ii(leg.get("index_tenor_m",3)), min_value=1, max_value=12, key=f"itm_{pk}"))

    elif lt == "BOND":
        c1,c2,c3 = st.columns(3)
        out["coupon_rate"]     = c1.number_input("Coupon Rate (%)",
            value=_fi(leg.get("coupon_rate",0))*100, min_value=0.0, max_value=50.0,
            step=0.01, format="%.4f", key=f"bcpn_{pk}") / 100
        out["redemption"]      = c2.number_input("Redemption (%)",
            value=_fi(leg.get("redemption",100)), min_value=0.0, max_value=200.0,
            step=0.1, key=f"rdem_{pk}")
        out["settlement_days"] = _ii(c3.number_input("Settle Days",
            value=_ii(leg.get("settlement_days",2)), min_value=0, max_value=10, key=f"sett_{pk}"))

    elif lt == "OPTION":
        o1,o2,o3 = st.columns(3)
        OPT = ["PAYER_SWAPTION","RECEIVER_SWAPTION","CAP","FLOOR"]
        ot  = leg.get("option_type","PAYER_SWAPTION")
        out["strike"]      = o1.number_input("Strike (%)",
            value=_fi(leg.get("strike",0.05))*100, min_value=0.0, max_value=50.0,
            step=0.01, format="%.4f", key=f"stk_{pk}") / 100
        out["option_type"] = o2.selectbox("Option Type", OPT,
            index=OPT.index(ot) if ot in OPT else 0, key=f"ot_{pk}")
        out["vol"]         = o3.number_input("Vol (%)",
            value=_fi(leg.get("vol",0.4))*100, min_value=0.0, max_value=200.0,
            step=0.5, format="%.2f", key=f"vol_{pk}") / 100
        p1,p2,p3 = st.columns(3)
        EX = ["EUROPEAN","AMERICAN","BERMUDAN"]
        ex = leg.get("exercise_type","EUROPEAN")
        out["exercise_type"]      = p1.selectbox("Exercise", EX,
            index=EX.index(ex) if ex in EX else 0, key=f"ext_{pk}")
        out["underlying_tenor_m"] = _ii(p2.number_input("Und. Tenor (m)",
            value=_ii(leg.get("underlying_tenor_m",60)), min_value=1, key=f"utm_{pk}"))
        VT = ["LOGNORMAL","NORMAL"]; vt = leg.get("vol_type","LOGNORMAL")
        out["vol_type"] = p3.selectbox("Vol Type", VT,
            index=VT.index(vt) if vt in VT else 0, key=f"vt_{pk}")

    elif lt == "EQUITY":
        e1,e2,e3 = st.columns(3)
        out["underlying_ticker"]  = e1.text_input("Ticker",
            value=leg.get("underlying_ticker","SPY"), key=f"tick_{pk}")
        out["initial_price"]      = e2.number_input("Spot (S0)",
            value=_fi(leg.get("initial_price",100)), min_value=0.01, step=1.0, key=f"ip_{pk}")
        RT = ["TOTAL","PRICE","FUNDED"]; rt = leg.get("equity_return_type","TOTAL")
        out["equity_return_type"] = e3.selectbox("Return Type", RT,
            index=RT.index(rt) if rt in RT else 0, key=f"rt_{pk}")
        g1,g2 = st.columns(2)
        out["dividend_yield"]     = g1.number_input("Div Yield (%)",
            value=_fi(leg.get("dividend_yield",0.015))*100, min_value=0.0, max_value=50.0,
            step=0.01, format="%.4f", key=f"dy_{pk}") / 100
        out["participation_rate"] = g2.number_input("Participation",
            value=_fi(leg.get("participation_rate",1.0)), min_value=0.0, step=0.01, key=f"pr_{pk}")

    elif lt == "CREDIT":
        c1,c2 = st.columns(2)
        out["reference_entity"] = c1.text_input("Ref Entity",
            value=leg.get("reference_entity","CORP"), key=f"re_{pk}")
        out["credit_spread"]    = c2.number_input("Spread (bps)",
            value=_fi(leg.get("credit_spread",0.015))*10000, min_value=0.0, max_value=5000.0,
            step=1.0, format="%.1f", key=f"cs_{pk}") / 10000
        d1,d2 = st.columns(2)
        SEN = ["SENIOR_UNSECURED","SUBORDINATED","SENIOR_SECURED"]
        sen = leg.get("seniority","SENIOR_UNSECURED")
        out["recovery_rate"] = d1.number_input("Recovery (%)",
            value=_fi(leg.get("recovery_rate",0.40))*100, min_value=0.0, max_value=100.0,
            step=1.0, key=f"rr_{pk}") / 100
        out["seniority"]     = d2.selectbox("Seniority", SEN,
            index=SEN.index(sen) if sen in SEN else 0, key=f"sen_{pk}")

    elif lt == "EQUITY_OPTION":
        g1,g2,g3 = st.columns(3)
        out["underlying_ticker"] = g1.text_input("Ticker",
            value=leg.get("underlying_ticker","SPY"), key=f"eotick_{pk}")
        out["initial_price"]     = g2.number_input("Spot (S0)",
            value=_fi(leg.get("initial_price",100)), min_value=0.01, step=1.0, key=f"eoip_{pk}")
        EO_OPT = ["CALL","PUT"]; ot2 = leg.get("option_type","CALL")
        out["option_type"]  = g3.selectbox("Option Type", EO_OPT,
            index=EO_OPT.index(ot2) if ot2 in EO_OPT else 0, key=f"eoot_{pk}")
        h1,h2,h3 = st.columns(3)
        out["strike"] = h1.number_input("Strike",
            value=_fi(leg.get("strike",100)), min_value=0.01, step=1.0, key=f"eostk_{pk}")
        out["vol"]    = h2.number_input("Vol (%)",
            value=_fi(leg.get("vol",0.25))*100, min_value=0.0, max_value=200.0,
            step=0.5, format="%.2f", key=f"eovol_{pk}") / 100
        EX2 = ["EUROPEAN","AMERICAN"]; ex2 = leg.get("exercise_type","EUROPEAN")
        out["exercise_type"]  = h3.selectbox("Exercise", EX2,
            index=EX2.index(ex2) if ex2 in EX2 else 0, key=f"eoext_{pk}")
        i1,i2 = st.columns(2)
        out["dividend_yield"] = i1.number_input("Div Yield (%)",
            value=_fi(leg.get("dividend_yield",0.013))*100, min_value=0.0, max_value=50.0,
            step=0.01, format="%.4f", key=f"eody_{pk}") / 100
        out["risk_free_rate"] = i2.number_input("Risk-Free Rate (%)",
            value=_fi(leg.get("risk_free_rate",0.0))*100, min_value=0.0, max_value=30.0,
            step=0.01, format="%.4f", key=f"eorfr_{pk}") / 100

    return out


def render_detail(td:Optional[Dict], *, in_dialog:bool=False):
    if not td:
        st.info("← Click a trade in the tree or grid"); return

    tid  = td.get("trade_id","")
    inst = td.get("trade_type","")
    disp = inst

    extra: Dict[str,Any] = {}

    with st.form(key=f"form_{tid}_{'d' if in_dialog else 'p'}"):
        # ── Trade Header ──────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">📋 Trade Header</div>', unsafe_allow_html=True)
        h1,h2,h3 = st.columns(3)
        dv        = td.get("direction",DIRECTIONS[0])
        direction = h1.selectbox("Direction", DIRECTIONS,
                        index=DIRECTIONS.index(dv) if dv in DIRECTIONS else 0,
                        key=f"dir_{tid}_{'d' if in_dialog else 'p'}")
        cpty = h2.text_input("Counterparty", value=td.get("counterparty",""),
                             key=f"cpty_{tid}_{'d' if in_dialog else 'p'}")
        trd  = h3.text_input("Trader", value=td.get("trader",""),
                             key=f"trd_{tid}_{'d' if in_dialog else 'p'}")
        h4,h5,h6 = st.columns(3)
        bk = h4.text_input("Book", value=td.get("book",""),
                           key=f"bk_{tid}_{'d' if in_dialog else 'p'}")
        vd = h5.date_input("Val. Date", value=_parse_date(td.get("valuation_date")),
                           key=f"vd_{tid}_{'d' if in_dialog else 'p'}")
        h6.text_input("Trade ID", value=tid, disabled=True,
                      key=f"id_{tid}_{'d' if in_dialog else 'p'}")

        # ── Instrument Parameters ─────────────────────────────────────────────
        st.divider()
        st.markdown(f'<div class="sec-hdr">⚙️ {disp} Parameters</div>', unsafe_allow_html=True)
        p1,p2,p3 = st.columns(3)
        sfx = 'd' if in_dialog else 'p'
        extra["tenor_y"] = _ii(p1.number_input("Tenor (yrs)",
            value=_ii(td.get("tenor_y",0)), min_value=0, max_value=50, step=1,
            key=f"tny_{tid}_{sfx}"))

        STYPES = ["FIXED_FLOAT","FLOAT_FIXED","FIXED_FIXED","FLOAT_FLOAT"]

        if inst == "InterestRateSwap":
            ss = td.get("swap_subtype","FIXED_FLOAT")
            extra["swap_subtype"] = p2.selectbox("Swap Subtype", STYPES,
                index=STYPES.index(ss) if ss in STYPES else 0, key=f"sst_{tid}_{sfx}")
            # Detect multi-currency (XCCY) IRS
            legs = td.get("legs",[])
            ccys = {l.get("currency","USD") for l in legs if l.get("currency")}
            is_xccy = len(ccys) > 1 or float(td.get("fx_rate", 1.0) or 1.0) != 1.0
            fx_val = float(td.get("fx_rate", 1.0) or 1.0)
            extra["fx_rate"] = float(p3.number_input(
                "FX Rate (rcv/pay)", value=fx_val, min_value=0.0001,
                step=0.01, format="%.4f", key=f"fx_{tid}_{sfx}"))
            if is_xccy:
                st.markdown(
                    f'<div class="sec-hdr">💱 Cross-Currency: '
                    f'{legs[0].get("currency","USD") if legs else "USD"} / '
                    f'{legs[1].get("currency","USD") if len(legs)>1 else "USD"}'
                    f'</div>', unsafe_allow_html=True)

        elif inst == "CrossCurrencySwap":
            # Legacy CrossCurrencySwap — read-only display (superseded by IRS)
            CCYS = ["USD","EUR","GBP","JPY","CHF","AUD","CAD","CNH","SGD","HKD"]
            pc = td.get("pay_currency","USD")
            rc = td.get("receive_currency","EUR")
            extra["pay_currency"]     = p2.selectbox("Pay Currency", CCYS,
                index=CCYS.index(pc) if pc in CCYS else 0, key=f"pc_{tid}_{sfx}")
            extra["receive_currency"] = p3.selectbox("Receive Currency", CCYS,
                index=CCYS.index(rc) if rc in CCYS else 0, key=f"rc_{tid}_{sfx}")
            q1,q2,q3 = st.columns(3)
            extra["fx_rate"] = float(q1.number_input("FX Rate (rcv/pay)",
                value=float(td.get("fx_rate",1.0) or 1.0), min_value=0.0001, step=0.01,
                format="%.4f", key=f"fx_{tid}_{sfx}"))
            ss2 = td.get("swap_subtype","FIXED_FLOAT")
            extra["swap_subtype"] = q2.selectbox("Swap Subtype", STYPES,
                index=STYPES.index(ss2) if ss2 in STYPES else 0, key=f"sst2_{tid}_{sfx}")
            extra["initial_notional_exchange"] = bool(q3.checkbox(
                "Initial Notional Exch.", value=bool(td.get("initial_notional_exchange",True)),
                key=f"ine_{tid}_{sfx}"))
            r1,_ = st.columns(2)
            extra["final_notional_exchange"] = bool(r1.checkbox(
                "Final Notional Exch.", value=bool(td.get("final_notional_exchange",True)),
                key=f"fne_{tid}_{sfx}"))

        elif inst == "Bond":
            extra["isin"] = p2.text_input("ISIN", value=td.get("isin","") or "",
                                          key=f"isin_{tid}_{sfx}")
            p3.markdown("")

        elif inst == "OptionableBond":
            BOND_SUBTYPES = ["CALLABLE", "PUTABLE", "CONVERTIBLE", "EXTENDABLE", "SINKING_FUND"]
            cur_sub = td.get("bond_subtype", "CALLABLE")
            extra["bond_subtype"] = p2.selectbox("Bond Subtype", BOND_SUBTYPES,
                index=BOND_SUBTYPES.index(cur_sub) if cur_sub in BOND_SUBTYPES else 0,
                key=f"bsub_{tid}_{sfx}")
            extra["isin"] = p3.text_input("ISIN", value=td.get("isin","") or "",
                                          key=f"isin_ob_{tid}_{sfx}")
            if extra["bond_subtype"] == "SINKING_FUND":
                q1, q2, _ = st.columns(3)
                extra["sinking_pct_per_period"] = float(q1.number_input(
                    "Sinking % per Period", value=float(td.get("sinking_pct_per_period", 0.10) or 0.10),
                    min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                    key=f"spct_{tid}_{sfx}"))
            elif extra["bond_subtype"] == "CONVERTIBLE":
                q1, q2, _ = st.columns(3)
                extra["conversion_premium"] = float(q1.number_input(
                    "Conversion Premium", value=float(td.get("conversion_premium", 0.25) or 0.25),
                    min_value=0.0, max_value=2.0, step=0.01, format="%.3f",
                    key=f"cprem_{tid}_{sfx}"))

        elif inst == "AssetSwap":
            q1, q2, q3 = st.columns(3)
            extra["tenor_y"]   = int(q1.number_input("Tenor (years)", value=int(td.get("tenor_y",5) or 5),
                                  min_value=1, max_value=30, step=1, key=f"asy_{tid}_{sfx}"))
            extra["isin"]      = q2.text_input("ISIN", value=td.get("isin","") or "",
                                                key=f"asn_{tid}_{sfx}")
            extra["par_price"] = float(q3.number_input("Par Price", value=float(td.get("par_price",100.0) or 100.0),
                                    min_value=50.0, max_value=150.0, step=0.25, format="%.2f",
                                    key=f"asp_{tid}_{sfx}"))

        elif inst == "Option":           # OptionTrade
            extra["underlying_tenor_y"] = _ii(p2.number_input("Underlying Tenor (yrs)",
                value=_ii(td.get("underlying_tenor_y",0)), min_value=0, max_value=50,
                step=1, key=f"utny_{tid}_{sfx}"))
            p3.markdown("")

        elif inst == "EquitySwap":
            extra["underlying_ticker"] = p2.text_input("Underlying Ticker",
                value=td.get("underlying_ticker",""), key=f"utk_{tid}_{sfx}")
            p3.markdown("")

        elif inst == "EquityOption":      # EquityOptionTrade
            extra["underlying_tenor_y"] = _ii(p2.number_input("Underlying Tenor (yrs)",
                value=_ii(td.get("underlying_tenor_y",0)), min_value=0, max_value=50,
                step=1, key=f"eutny_{tid}_{sfx}"))
            extra["underlying_ticker"] = p3.text_input("Underlying Ticker",
                value=td.get("underlying_ticker",""), key=f"eutk_{tid}_{sfx}")

        elif inst == "CDS":               # CreditDefaultSwap
            cl = td.get("legs", [{}])[0] if td.get("legs") else {}
            extra["tenor_y"] = _ii(p1.number_input("Tenor (yrs)",
                value=_ii(td.get("tenor_y", 5)), min_value=1, max_value=10, step=1,
                key=f"cdsy_{tid}_{sfx}"))
            ref_ent = cl.get("reference_entity", td.get("reference_entity", ""))
            extra["reference_entity"] = p2.text_input("Reference Entity",
                value=ref_ent, key=f"cdse_{tid}_{sfx}")
            raw_spd = cl.get("credit_spread", td.get("credit_spread", 0.01))
            try:
                spd_bps = float(raw_spd) * 10000
            except (TypeError, ValueError):
                spd_bps = 100.0
            spd_val = p3.number_input("Credit Spread (bps)",
                value=round(spd_bps, 2), min_value=0.0, max_value=5000.0,
                step=1.0, format="%.1f", key=f"cdss_{tid}_{sfx}")
            extra["credit_spread"] = spd_val / 10000.0
            q1, q2, _ = st.columns(3)
            raw_rec = cl.get("recovery_rate", td.get("recovery_rate", 0.4))
            try:
                rec_pct = float(raw_rec) * 100
            except (TypeError, ValueError):
                rec_pct = 40.0
            rec_val = q1.number_input("Recovery Rate (%)",
                value=round(rec_pct, 1), min_value=0.0, max_value=100.0,
                step=1.0, format="%.1f", key=f"cdsr_{tid}_{sfx}")
            extra["recovery_rate"] = rec_val / 100.0

        elif inst == "IRSwaption":        # InterestRateSwaption
            SWPN_TYPES = ["FIXED_FLOAT", "FLOAT_FIXED"]
            ss3 = td.get("swap_subtype", "FIXED_FLOAT")
            extra["swap_subtype"] = p2.selectbox(
                "Swaption Type (FIXED_FLOAT=Payer, FLOAT_FIXED=Receiver)",
                SWPN_TYPES,
                index=SWPN_TYPES.index(ss3) if ss3 in SWPN_TYPES else 0,
                key=f"swpnst_{tid}_{sfx}")
            ol = next((l for l in td.get("legs", []) if l.get("leg_type") == "OPTION"), {})
            vol_val = float(ol.get("vol", td.get("vol", 0.20)) or 0.20)
            extra["vol"] = float(p3.number_input("Implied Vol (%)",
                value=round(vol_val * 100, 2), min_value=0.0, max_value=200.0,
                step=0.1, format="%.2f", key=f"swpnv_{tid}_{sfx}")) / 100.0

        # ── Legs ──────────────────────────────────────────────────────────────
        legs = td.get("legs",[])
        if legs:
            st.divider()
            st.markdown('<div class="sec-hdr">🦵 Legs</div>', unsafe_allow_html=True)
        new_legs = []
        for i, leg in enumerate(legs):
            lt   = leg.get("leg_type","FIXED")
            icon2 = LEG_ICONS.get(lt,"📌")
            ccy_tag = f" · {leg.get('currency','USD')}" if inst=="CrossCurrencySwap" else ""
            st.markdown(
                f'<div class="leg-hdr">{icon2} Leg {i+1} &nbsp;·&nbsp; {lt}{ccy_tag}</div>',
                unsafe_allow_html=True)
            new_legs.append(_render_leg(leg, i, tid, sfx=sfx))
            if i < len(legs)-1:
                st.markdown("<hr style='margin:6px 0;border-color:#dee2e6'>",
                            unsafe_allow_html=True)

        sub = st.form_submit_button("💾 Save Trade", type="primary", use_container_width=True)

    if sub:
        u = dict(td)
        u.update(direction=direction, counterparty=cpty, trader=trd,
                 book=bk, valuation_date=vd.isoformat(), legs=new_legs)
        u.update(extra)
        if _save_dict(u):
            _load_all.clear(); st.success("✅ Saved!"); st.rerun()


@st.dialog("📋 Trade Detail — Pop-Out", width="large")
def _popout_detail(td:Dict):
    render_detail(td, in_dialog=True)


# ── main ──────────────────────────────────────────────────────────────────────
@st.dialog("📊 Trade Grid — Pop-Out", width="large")
def _popout_grid(book_trades:List[Dict]):
    render_grid(book_trades)

def _panel_header(title:str, popout_key:str, popout_cb) -> None:
    """Render a panel header with an inline tiny pop-out button on the right."""
    left, right = st.columns([6, 1])
    left.markdown(f'<div class="ph2">{title}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="popout-wrap">', unsafe_allow_html=True)
        clicked = st.button("↗", key=popout_key, help="Pop out to larger view")
        st.markdown('</div>', unsafe_allow_html=True)
    if clicked:
        popout_cb()

def main():
    st.title("📊 Trade Manager")
    all_trades=_load_all(); tree=_build_tree(all_trades)
    lc,mc,rc=st.columns([1,2,2],gap="small")

    with lc:
        render_tree(tree)

    with mc:
        tr=st.session_state.sel_trader; bk=st.session_state.sel_book
        book_trades=tree.get(tr,{}).get(bk,[]) if tr and bk else []
        title = f"📊 {bk} — {tr}" if tr and bk else "📊 Trades"
        _panel_header(title, "po_grid",
                      lambda: _popout_grid(book_trades) if book_trades else None)
        render_grid(book_trades)

    with rc:
        tid=st.session_state.sel_trade_id
        td=_fetch_one(tid) if tid else None
        inst=td.get("trade_type","") if td else ""
        disp=inst
        icon=INST_ICONS.get(inst,"📋")
        title = f"{icon} {tid} — {disp}" if td else "📋 Trade Detail"
        _panel_header(title, "po_detail",
                      lambda: _popout_detail(td) if td else None)
        render_detail(td)

if __name__=="__main__": main()
