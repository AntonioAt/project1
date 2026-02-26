import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. CORE ENGINEERING MODULES (API 13D PHYSICS) - ENGINE LOCKED
# =============================================================================
class SolidControlAnalyzer:
    def __init__(self, target_lgs, water_content=0.90, mud_price_bbl=75.0):
        self.Ks = target_lgs; self.Kw = water_content; self.mud_price = mud_price_bbl
    def calculate_sre(self, hole_in, length_ft, washout, water_added_bbl):
        Vm = water_added_bbl / self.Kw
        Vc = 0.000971 * (hole_in**2) * length_ft * washout
        Vd = Vc / self.Ks
        return max(0.0, (1.0 - (Vm / Vd)) * 100.0), (Vm * self.mud_price)

class DrillingPhysicsEngine:
    def __init__(self, base_pv, base_yp): 
        self.base_pv = base_pv; self.base_yp = base_yp
    def simulate_depth(self, lgs_pct, mw_ppg, depth_ft, hole, dp, gpm, pp, rop_max):
        sim_pv = self.base_pv + (lgs_pct * 1.4) + (lgs_pct**1.3 * 0.4)
        sim_yp = self.base_yp + (lgs_pct * 1.6) + (lgs_pct**1.2 * 0.25)
        annular_cap = (hole**2 - dp**2) / 1029.4
        vel = gpm / (annular_cap * 42.0)
        d_hyd = hole - dp
        fric = (((sim_pv * vel) / (1500 * d_hyd**2)) + (sim_yp / (225 * d_hyd))) * depth_ft
        ecd = mw_ppg + (fric / (0.052 * depth_ft))
        rop = rop_max * np.exp(-0.3 * max(0, ecd - pp))
        return round(sim_pv, 1), round(sim_yp, 1), round(ecd, 2), round(rop, 1)

class EconomicsAnalyzer:
    def __init__(self, rig_rate, bit_cost=25000.0):
        self.rig_rate = rig_rate; self.bit_cost = bit_cost
    def calculate_cost(self, avg_rop, length_ft, avg_lgs, mud_cost, daily_equip_cost):
        d_hrs = (length_ft / avg_rop) if avg_rop > 0 else 9999.0
        t_days = (d_hrs + (length_ft / 1000.0) + (10.0 if avg_lgs > 8.0 else 0)) / 24.0
        t_cost = (self.rig_rate * t_days) + self.bit_cost + mud_cost + (daily_equip_cost * t_days)
        return t_days, t_cost, (self.rig_rate * t_days)

# =============================================================================
# 2. SRE EQUIPMENT DICTIONARY (PHYSICS LOGIC)
# =============================================================================
SRE_TYPES = {
    "No Solid Control (Bypass)": {"lgs_multiplier": 2.5, "base_cost": 0.0},
    "Shale Shaker Only": {"lgs_multiplier": 1.5, "base_cost": 500.0},
    "Shaker + Desander/Desilter": {"lgs_multiplier": 1.1, "base_cost": 1200.0},
    "Full System (Shaker + Centrifuge)": {"lgs_multiplier": 0.85, "base_cost": 2500.0}
}

# Function to Auto-Generate Depth Logs
def generate_dynamic_log(start_d, end_d, sre_type, base_target_lgs):
    mult = SRE_TYPES[sre_type]["lgs_multiplier"]
    log = []
    points = np.linspace(start_d + 100, end_d, 3) 
    for i, d in enumerate(points):
        mw = 9.0 + (d / 2000.0) 
        lgs = (base_target_lgs * mult) + (i * 1.5 * mult) 
        log.append((round(d, 1), round(mw, 2), round(lgs, 2)))
    return log

# =============================================================================
# 3. STREAMLIT WEB APP FRONT-END (DYNAMIC UI)
# =============================================================================
st.set_page_config(page_title="Ultimate Drilling Simulator", layout="wide", page_icon="🛢️")

st.title("🛢️ Dynamic Drilling & Solid Control Simulator")
st.markdown("An interactive platform to design well trajectories and compare the real-time performance of two Solid Control systems.")

# --- SIDEBAR: DYNAMIC INPUT PANEL ---
st.sidebar.header("⚙️ Main Configuration")

# 1. Global Parameters 
with st.sidebar.expander("🌍 Base Rig & Fluid Parameters", expanded=True):
    rig_rate = st.number_input("Rig Lease Rate (USD/Day)", value=35000.0, step=1000.0, format="%.2f")
    base_pv = st.number_input("Base PV (cP)", value=14.0, step=0.5, format="%.2f")
    base_yp = st.number_input("Base YP (lb/100ft2)", value=10.0, step=0.5, format="%.2f")
    target_lgs = st.slider("Target LGS (%)", 2.0, 10.0, 6.0, step=0.5)

# 2. Dynamic Well Design
with st.sidebar.expander("📐 Well Trajectory Design (Dynamic Depth)", expanded=False):
    st.markdown("*Input the length of each section (ft)*")
    len_sec1 = st.number_input("17.5\" Surface Length", value=1250.0, step=100.0)
    len_sec2 = st.number_input("12.25\" Intermediate Length", value=3500.0, step=100.0)
    len_sec3 = st.number_input("8.5\" Production Length", value=3350.0, step=100.0)

# 3. System A Configuration (Old)
with st.sidebar.expander("🔴 Well A Configuration (Old System)", expanded=False):
    sre_A = st.selectbox("SRE Type (Well A)", list(SRE_TYPES.keys()), index=1)
    eq_cost_A = st.number_input("Equipment Lease A (USD/Day)", value=SRE_TYPES[sre_A]["base_cost"])
    st.markdown("*Dilution Water Requirement (bbl)*")
    wa1 = st.number_input("Section 1 Dilution (Well A)", value=1800.0, step=100.0)
    wa2 = st.number_input("Section 2 Dilution (Well A)", value=3200.0, step=100.0)
    wa3 = st.number_input("Section 3 Dilution (Well A)", value=2100.0, step=100.0)

# 4. System B Configuration (New)
with st.sidebar.expander("🔵 Well B Configuration (New System)", expanded=False):
    sre_B = st.selectbox("SRE Type (Well B)", list(SRE_TYPES.keys()), index=3)
    eq_cost_B = st.number_input("Equipment Lease B (USD/Day)", value=SRE_TYPES[sre_B]["base_cost"])
    st.markdown("*Dilution Water Requirement (bbl)*")
    wb1 = st.number_input("Section 1 Dilution (Well B)", value=800.0, step=100.0)
    wb2 = st.number_input("Section 2 Dilution (Well B)", value=1400.0, step=100.0)
    wb3 = st.number_input("Section 3 Dilution (Well B)", value=900.0, step=100.0)

# --- ENGINE EXECUTION ---
if st.sidebar.button("🚀 RUN SIMULATOR", type="primary", use_container_width=True):
    with st.spinner("Calculating drilling physics and equipment dynamics..."):
        
        # Calculate Dynamic Depths
        d1 = len_sec1
        d2 = d1 + len_sec2
        d3 = d2 + len_sec3
        
        # Auto-Generate Scenarios based on UI input
        scenarios = {
            "Well A (Old System)": { "daily_eq_cost": eq_cost_A, "sections": [
                {"hole": 17.5, "dp": 5.0, "len": len_sec1, "wash": 1.15, "gpm": 1050, "pp": 8.6, "fg": 11.5, "rop_max": 90, "wat_add": wa1, "log": generate_dynamic_log(0, d1, sre_A, target_lgs)},
                {"hole": 12.25, "dp": 5.0, "len": len_sec2, "wash": 1.10, "gpm": 850, "pp": 9.2, "fg": 12.8, "rop_max": 65, "wat_add": wa2, "log": generate_dynamic_log(d1, d2, sre_A, target_lgs)},
                {"hole": 8.5, "dp": 4.0, "len": len_sec3, "wash": 1.05, "gpm": 450, "pp": 10.4, "fg": 14.8, "rop_max": 45, "wat_add": wa3, "log": generate_dynamic_log(d2, d3, sre_A, target_lgs)}
            ]},
            "Well B (New System)": { "daily_eq_cost": eq_cost_B, "sections": [
                {"hole": 17.5, "dp": 5.0, "len": len_sec1, "wash": 1.15, "gpm": 1050, "pp": 8.6, "fg": 11.5, "rop_max": 90, "wat_add": wb1, "log": generate_dynamic_log(0, d1, sre_B, target_lgs)},
                {"hole": 12.25, "dp": 5.0, "len": len_sec2, "wash": 1.10, "gpm": 850, "pp": 9.2, "fg": 12.8, "rop_max": 65, "wat_add": wb2, "log": generate_dynamic_log(d1, d2, sre_B, target_lgs)},
                {"hole": 8.5, "dp": 4.0, "len": len_sec3, "wash": 1.05, "gpm": 450, "pp": 10.4, "fg": 14.8, "rop_max": 45, "wat_add": wb3, "log": generate_dynamic_log(d2, d3, sre_B, target_lgs)}
            ]}
        }
        
        # Run Physics Engine
        engine = DrillingPhysicsEngine(base_pv, base_yp)
        econ = EconomicsAnalyzer(rig_rate)
        sim_res = {k: {"depth":[],"rop":[],"ecd":[],"pp":[],"fg":[],"lgs":[],"pv":[],"yp":[], "cost":0,"days":0,"equip_invest":0, "mud_cost":0, "rig_cost":0} for k in scenarios}

        for sc_name, sc_data in scenarios.items():
            t_cost = 0; t_days = 0; t_invest = 0; t_mud = 0; t_rig = 0
            # Dynamic effective LGS based on selected SRE
            eff_lgs = target_lgs * SRE_TYPES[sre_A if "Old" in sc_name else sre_B]["lgs_multiplier"] / 100.0
            sc_ana = SolidControlAnalyzer(target_lgs=eff_lgs)
            
            for sec in sc_data["sections"]:
                sre, mud_cost = sc_ana.calculate_sre(sec['hole'], sec['len'], sec['wash'], sec['wat_add'])
                avg_rop = 0; lgs_sum = 0
                for d, mw, lgs in sec['log']:
                    pv, yp, ecd, rop = engine.simulate_depth(lgs, mw, d, sec['hole'], sec['dp'], sec['gpm'], sec['pp'], sec['rop_max'])
                    sim_res[sc_name]["depth"].append(d); sim_res[sc_name]["rop"].append(rop)
                    sim_res[sc_name]["ecd"].append(ecd); sim_res[sc_name]["pp"].append(sec['pp']); sim_res[sc_name]["fg"].append(sec['fg'])
                    sim_res[sc_name]["lgs"].append(lgs); sim_res[sc_name]["pv"].append(pv); sim_res[sc_name]["yp"].append(yp)
                    avg_rop += rop; lgs_sum += lgs
                
                days, cost, rig_c = econ.calculate_cost(avg_rop/len(sec['log']), sec['len'], lgs_sum/len(sec['log']), mud_cost, sc_data["daily_eq_cost"])
                t_days += days; t_cost += cost; t_invest += (sc_data["daily_eq_cost"] * days); t_mud += mud_cost; t_rig += rig_c
                
            sim_res[sc_name].update({"cost": t_cost, "days": t_days, "equip_invest": t_invest, "mud_cost": t_mud, "rig_cost": t_rig})

        # --- DASHBOARD RENDERING ---
        old = sim_res["Well A (Old System)"]; new = sim_res["Well B (New System)"]
        net_save = old["cost"] - new["cost"]; time_save = old["days"] - new["days"]
        roi = (net_save / new["equip_invest"]) * 100 if new["equip_invest"] > 0 else 0
        
        tab1, tab2 = st.tabs(["📊 Visual Dashboard", "💰 Financial Report"])
        
        with tab1:
            st.subheader(f"Comparison: [{sre_A}]  vs  [{sre_B}]")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Well A Total Cost", f"${old['cost']:,.0f}")
            c2.metric("Well B Total Cost", f"${new['cost']:,.0f}", f"{net_save:,.0f} Saved", delta_color="inverse")
            c3.metric("Well B Drilling Time", f"{new['days']:.1f} Days", f"{time_save:.1f} Days Faster", delta_color="inverse")
            c4.metric("ROI (SRE Upgrade)", f"{roi:,.0f} %")
            
            # MATPLOTLIB (Auto-scale based on dynamic depth)
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'DYNAMIC DRILLING PERFORMANCE: TD {d3:,.0f} ft', fontsize=20, fontweight='bold', y=0.98)
            d_old, d_new = old["depth"], new["depth"]; max_d = max(d_old) + 500
            s_old = {'color':'#e74c3c','marker':'o','lw':2.5,'label':'Well A'}
            s_new = {'color':'#3498db','marker':'o','lw':2.5,'label':'Well B'}

            axs[0,0].plot(old["rop"], d_old, **s_old); axs[0,0].plot(new["rop"], d_new, **s_new); axs[0,0].set(ylim=(max_d, 0), title='ROP (ft/hr)'); axs[0,0].grid(True, ls='--'); axs[0,0].legend()
            axs[0,1].plot(old["ecd"], d_old, **s_old); axs[0,1].plot(new["ecd"], d_new, **s_new); axs[0,1].plot(new["pp"], d_old, 'k--', label='Pore Pressure'); axs[0,1].plot(new["fg"], d_old, 'k-', label='Frac Gradient'); axs[0,1].fill_betweenx(d_old, new["pp"], new["fg"], color='#2ecc71', alpha=0.1); axs[0,1].set(ylim=(max_d, 0), xlim=(8, 15), title='Mud Window (ppg)'); axs[0,1].grid(True, ls='--'); axs[0,1].legend()
            
            bars = axs[0,2].bar(['Well A', 'Well B'], [old["cost"]/1e6, new["cost"]/1e6], color=['#e74c3c','#3498db']); axs[0,2].set(title='Total Cost (Million USD)'); 
            for b in bars: axs[0,2].text(b.get_x() + b.get_width()/2, b.get_height(), f'${b.get_height():.2f}M', ha='center', va='bottom', fontweight='bold')

            axs[1,0].plot(old["lgs"], d_old, **s_old); axs[1,0].plot(new["lgs"], d_new, **s_new); axs[1,0].axvline(target_lgs, color='k', ls='--', label='Target LGS'); axs[1,0].set(ylim=(max_d, 0), xlim=(0, max(max(old["lgs"])+5, 15)), title='Solid Accumulation (LGS %)'); axs[1,0].grid(True, ls='--'); axs[1,0].legend()
            axs[1,1].plot(old["pv"], d_old, **s_old); axs[1,1].plot(new["pv"], d_new, **s_new); axs[1,1].set(ylim=(max_d, 0), title='Plastic Viscosity (cP)'); axs[1,1].grid(True, ls='--')
            axs[1,2].plot(old["yp"], d_old, **s_old); axs[1,2].plot(new["yp"], d_new, **s_new); axs[1,2].set(ylim=(max_d, 0), title='Yield Point (lb/100ft2)'); axs[1,2].grid(True, ls='--')

            plt.tight_layout(rect=[0, 0.03, 1, 0.96]); st.pyplot(fig)

        with tab2:
            st.subheader("Dynamic Authorization for Expenditure (AFE)")
            st.table({
                "Cost Component": ["Total Operating Days", "Rig Lease Cost", "Mud & Chemical Cost", "SRE Equipment Investment", "TOTAL WELL COST"],
                "Well A (Old System)": [f"{old['days']:.1f}", f"${old['rig_cost']:,.0f}", f"${old['mud_cost']:,.0f}", f"${old['equip_invest']:,.0f}", f"${old['cost']:,.0f}"],
                "Well B (New System)": [f"{new['days']:.1f}", f"${new['rig_cost']:,.0f}", f"${new['mud_cost']:,.0f}", f"${new['equip_invest']:,.0f}", f"${new['cost']:,.0f}"]
            })
else:
    st.info("👈 Open the sidebar to adjust well depths or select Solid Control equipment, then click 'RUN SIMULATOR'!")
