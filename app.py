import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. ADVANCED PHYSICS MODULES (KRIEGER-DOUGHERTY & CUTTINGS GENERATOR)
# =============================================================================
class SolidControlAnalyzer:
    def __init__(self, water_content=0.90, mud_price_bbl=75.0):
        self.Kw = water_content; self.mud_price = mud_price_bbl
    def calculate_sre(self, hole_in, length_ft, washout, water_added_bbl, avg_lgs):
        if avg_lgs <= 0: avg_lgs = 0.1
        Vm = water_added_bbl / self.Kw
        Vc = 0.000971 * (hole_in**2) * length_ft * washout
        Vd = Vc / (avg_lgs / 100.0)
        return max(0.0, (1.0 - (Vm / Vd)) * 100.0), (Vm * self.mud_price)

class Fann35Machine:
    def __init__(self):
        self.k_gamma = 1.703  # RPM -> Shear Rate
        self.k_tau = 0.511    # Dial Reading -> Shear Stress (lb/100ft2)
    def get_shear_rate(self, rpm): 
        return rpm * self.k_gamma
    def reading_from_stress(self, stress_lb): 
        return stress_lb / self.k_tau

class AdvancedDrillingPhysics:
    def __init__(self, base_pv, base_yp):
        self.machine = Fann35Machine()
        # Calibration to True Physical Stress
        self.polymer_visc_factor = base_pv / 3.0
        self.polymer_yield_strength = base_yp * 0.8
        
        # Geothermal Gradient
        self.surface_temp = 80.0 
        self.geo_gradient = 1.6  

    def get_temp_at_depth(self, tvd_ft):
        return self.surface_temp + (self.geo_gradient * (tvd_ft / 100.0))

    def calculate_generated_lgs(self, hole_diam_in, rop_fph, gpm, sre_multiplier):
        # 1. Cuttings Volume
        hole_cap = (hole_diam_in ** 2) / 1029.4
        cuttings_bbl_hr = hole_cap * rop_fph
        # 2. Mud Volume
        mud_bbl_hr = (gpm * 60) / 42.0
        # 3. LGS Concentration (Fresh)
        transport_efficiency = 0.8
        lgs_concentration = (cuttings_bbl_hr / (mud_bbl_hr * transport_efficiency)) * 100.0
        
        background_lgs = 3.0
        total_lgs = (lgs_concentration + background_lgs) * sre_multiplier
        return min(total_lgs, 25.0) # Safety cap at 25%

    def calculate_rheology(self, lgs_pct, temp_f):
        # A. Thermal Thinning
        temp_diff = temp_f - 120.0
        thermal_factor = max(0.6, 1.0 - (0.005 * temp_diff))
        
        # B. Krieger-Dougherty (Solid Effects)
        phi = lgs_pct / 100.0
        phi_max = 0.60; eta = 2.5
        
        mu_medium = 1.0 * self.polymer_visc_factor * thermal_factor
        rel_visc = (1 - (phi / phi_max)) ** (-eta * phi_max) if phi < phi_max else 50.0
        true_visc = mu_medium * rel_visc
        
        # C. Yield Stress Thermal Agitation
        thermal_agitation = 1.0 + (0.001 * temp_diff)
        true_yield = (self.polymer_yield_strength + (0.8 * lgs_pct)) * thermal_agitation
        
        # D. Fann 35 Simulation
        r600 = self.machine.reading_from_stress(true_yield + (true_visc * (self.machine.get_shear_rate(600)/300.0)))
        r300 = self.machine.reading_from_stress(true_yield + (true_visc * (self.machine.get_shear_rate(300)/300.0)))
        
        sim_pv = r600 - r300
        sim_yp = r300 - sim_pv
        return round(sim_pv, 1), round(sim_yp, 1), round(r600, 1), round(r300, 1)

    def calculate_hydraulics(self, pv, yp, mw_ppg, depth_ft, hole, dp, gpm, pp, rop_max):
        annular_cap = (hole**2 - dp**2) / 1029.4
        vel = gpm / (annular_cap * 42.0)
        d_hyd = hole - dp
        fric = (((pv * vel) / (1500 * d_hyd**2)) + (yp / (225 * d_hyd))) * depth_ft
        ecd = mw_ppg + (fric / (0.052 * depth_ft))
        rop = rop_max * np.exp(-0.3 * max(0, ecd - pp))
        return round(ecd, 2), round(rop, 1)

class EconomicsAnalyzer:
    def __init__(self, rig_rate, bit_cost=25000.0):
        self.rig_rate = rig_rate; self.bit_cost = bit_cost
    def calculate_cost(self, avg_rop, length_ft, avg_lgs, mud_cost, daily_equip_cost):
        d_hrs = (length_ft / avg_rop) if avg_rop > 0 else 9999.0
        t_days = (d_hrs + (length_ft / 1000.0) + (10.0 if avg_lgs > 8.0 else 0)) / 24.0
        t_cost = (self.rig_rate * t_days) + self.bit_cost + mud_cost + (daily_equip_cost * t_days)
        return t_days, t_cost, (self.rig_rate * t_days)

# =============================================================================
# 2. SRE EQUIPMENT LOGIC
# =============================================================================
SRE_TYPES = {
    "No Solid Control (Bypass)": {"lgs_multiplier": 2.5, "base_cost": 0.0},
    "Shale Shaker Only": {"lgs_multiplier": 1.5, "base_cost": 500.0},
    "Shaker + Desander/Desilter": {"lgs_multiplier": 1.1, "base_cost": 1200.0},
    "Full System (Shaker + Centrifuge)": {"lgs_multiplier": 0.5, "base_cost": 2500.0}
}

def generate_dynamic_log(start_d, end_d, pp_base, fg_base, rop_base):
    log = []
    points = np.linspace(start_d + 100, end_d, 3) 
    for d in points:
        mw = 9.0 + (d / 2000.0) 
        pp = pp_base + (d / 3000.0)
        fg = fg_base + (d / 2000.0)
        log.append((round(d, 1), round(mw, 2), round(pp, 2), round(fg, 2), rop_base))
    return log

# =============================================================================
# 3. STREAMLIT WEB APP FRONT-END 
# =============================================================================
st.set_page_config(page_title="Ultimate Drilling Simulator", layout="wide", page_icon="🛢️")

st.title("🛢️ First-Principles Drilling Simulator")
st.markdown("Powered by Krieger-Dougherty fluid mechanics and virtual Fann 35 viscometer physics. LGS is dynamically generated based on ROP and Flow Rate (GPM).")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Main Configuration")

with st.sidebar.expander("🌍 Base Rig & Fluid Parameters", expanded=True):
    rig_rate = st.number_input("Rig Lease Rate (USD/Day)", value=35000.0, step=1000.0, format="%.2f")
    base_pv = st.number_input("Base Mud PV (cP)", value=14.0, step=0.5, format="%.2f")
    base_yp = st.number_input("Base Mud YP (lb/100ft2)", value=10.0, step=0.5, format="%.2f")

with st.sidebar.expander("📐 Well Trajectory & Hydraulics", expanded=False):
    st.markdown("*Surface Section (17.5\")*")
    len_sec1 = st.number_input("Length (ft) - Sec 1", value=1250.0, step=100.0)
    gpm1 = st.number_input("GPM - Sec 1", value=1050.0, step=50.0)
    
    st.markdown("*Intermediate Section (12.25\")*")
    len_sec2 = st.number_input("Length (ft) - Sec 2", value=3500.0, step=100.0)
    gpm2 = st.number_input("GPM - Sec 2", value=850.0, step=50.0)
    
    st.markdown("*Production Section (8.5\")*")
    len_sec3 = st.number_input("Length (ft) - Sec 3", value=3350.0, step=100.0)
    gpm3 = st.number_input("GPM - Sec 3", value=450.0, step=50.0)

with st.sidebar.expander("🔴 Well A Configuration (Old System)", expanded=False):
    sre_A = st.selectbox("SRE Type (Well A)", list(SRE_TYPES.keys()), index=1)
    wa1 = st.number_input("Sec 1 Dilution (Well A)", value=1800.0, step=100.0)
    wa2 = st.number_input("Sec 2 Dilution (Well A)", value=3200.0, step=100.0)
    wa3 = st.number_input("Sec 3 Dilution (Well A)", value=2100.0, step=100.0)

with st.sidebar.expander("🔵 Well B Configuration (New System)", expanded=False):
    sre_B = st.selectbox("SRE Type (Well B)", list(SRE_TYPES.keys()), index=3)
    wb1 = st.number_input("Sec 1 Dilution (Well B)", value=800.0, step=100.0)
    wb2 = st.number_input("Sec 2 Dilution (Well B)", value=1400.0, step=100.0)
    wb3 = st.number_input("Sec 3 Dilution (Well B)", value=900.0, step=100.0)

# --- ENGINE EXECUTION ---
if st.sidebar.button("🚀 RUN PHYSICS SIMULATION", type="primary", use_container_width=True):
    with st.spinner("Processing cuttings generation and Krieger-Dougherty fluid mechanics..."):
        
        d1 = len_sec1; d2 = d1 + len_sec2; d3 = d2 + len_sec3
        
        scenarios = {
            "Well A (Old System)": { "daily_eq_cost": SRE_TYPES[sre_A]["base_cost"], "sections": [
                {"hole": 17.5, "dp": 5.0, "len": len_sec1, "wash": 1.15, "gpm": gpm1, "wat_add": wa1, "log": generate_dynamic_log(0, d1, 8.6, 11.5, 90)},
                {"hole": 12.25, "dp": 5.0, "len": len_sec2, "wash": 1.10, "gpm": gpm2, "wat_add": wa2, "log": generate_dynamic_log(d1, d2, 9.2, 12.8, 65)},
                {"hole": 8.5, "dp": 4.0, "len": len_sec3, "wash": 1.05, "gpm": gpm3, "wat_add": wa3, "log": generate_dynamic_log(d2, d3, 10.4, 14.8, 45)}
            ]},
            "Well B (New System)": { "daily_eq_cost": SRE_TYPES[sre_B]["base_cost"], "sections": [
                {"hole": 17.5, "dp": 5.0, "len": len_sec1, "wash": 1.15, "gpm": gpm1, "wat_add": wb1, "log": generate_dynamic_log(0, d1, 8.6, 11.5, 90)},
                {"hole": 12.25, "dp": 5.0, "len": len_sec2, "wash": 1.10, "gpm": gpm2, "wat_add": wb2, "log": generate_dynamic_log(d1, d2, 9.2, 12.8, 65)},
                {"hole": 8.5, "dp": 4.0, "len": len_sec3, "wash": 1.05, "gpm": gpm3, "wat_add": wb3, "log": generate_dynamic_log(d2, d3, 10.4, 14.8, 45)}
            ]}
        }
        
        engine = AdvancedDrillingPhysics(base_pv, base_yp)
        econ = EconomicsAnalyzer(rig_rate)
        sc_ana = SolidControlAnalyzer()
        
        sim_res = {k: {"depth":[],"rop":[],"ecd":[],"pp":[],"fg":[],"lgs":[],"pv":[],"yp":[],"r600":[],"r300":[], "cost":0,"days":0,"equip_invest":0, "mud_cost":0, "rig_cost":0} for k in scenarios}

        for sc_name, sc_data in scenarios.items():
            t_cost = 0; t_days = 0; t_invest = 0; t_mud = 0; t_rig = 0
            sre_mult = SRE_TYPES[sre_A if "Old" in sc_name else sre_B]["lgs_multiplier"]
            
            for sec in sc_data["sections"]:
                avg_rop = 0; lgs_sum = 0
                for d, mw, pp, fg, rop_max in sec['log']:
                    # PHYSICS CORE LOOP
                    temp = engine.get_temp_at_depth(d)
                    lgs = engine.calculate_generated_lgs(sec['hole'], rop_max, sec['gpm'], sre_mult)
                    pv, yp, r600, r300 = engine.calculate_rheology(lgs, temp)
                    ecd, rop = engine.calculate_hydraulics(pv, yp, mw, d, sec['hole'], sec['dp'], sec['gpm'], pp, rop_max)
                    
                    sim_res[sc_name]["depth"].append(d); sim_res[sc_name]["rop"].append(rop); sim_res[sc_name]["ecd"].append(ecd)
                    sim_res[sc_name]["pp"].append(pp); sim_res[sc_name]["fg"].append(fg); sim_res[sc_name]["lgs"].append(lgs)
                    sim_res[sc_name]["pv"].append(pv); sim_res[sc_name]["yp"].append(yp); sim_res[sc_name]["r600"].append(r600); sim_res[sc_name]["r300"].append(r300)
                    avg_rop += rop; lgs_sum += lgs
                
                # SRE Dilution based on dynamic generated LGS
                sec_avg_lgs = lgs_sum / len(sec['log'])
                _, mud_cost = sc_ana.calculate_sre(sec['hole'], sec['len'], sec['wash'], sec['wat_add'], sec_avg_lgs)
                
                days, cost, rig_c = econ.calculate_cost(avg_rop/len(sec['log']), sec['len'], sec_avg_lgs, mud_cost, sc_data["daily_eq_cost"])
                t_days += days; t_cost += cost; t_invest += (sc_data["daily_eq_cost"] * days); t_mud += mud_cost; t_rig += rig_c
                
            sim_res[sc_name].update({"cost": t_cost, "days": t_days, "equip_invest": t_invest, "mud_cost": t_mud, "rig_cost": t_rig})

        # --- DASHBOARD RENDERING ---
        old = sim_res["Well A (Old System)"]; new = sim_res["Well B (New System)"]
        net_save = old["cost"] - new["cost"]; time_save = old["days"] - new["days"]
        roi = (net_save / new["equip_invest"]) * 100 if new["equip_invest"] > 0 else 0
        
        tab1, tab2, tab3 = st.tabs(["📊 Visual Dashboard", "💰 Financial Report", "🔬 Real-Time Fann 35 Report"])
        
        with tab1:
            st.subheader(f"Comparison: [{sre_A}]  vs  [{sre_B}]")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Well A Total Cost", f"${old['cost']:,.0f}")
            c2.metric("Well B Total Cost", f"${new['cost']:,.0f}", f"{net_save:,.0f} Saved", delta_color="inverse")
            c3.metric("Well B Drilling Time", f"{new['days']:.1f} Days", f"{time_save:.1f} Days Faster", delta_color="inverse")
            c4.metric("ROI (SRE Upgrade)", f"{roi:,.0f} %")
            
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'TRUE PHYSICS DRILLING PERFORMANCE: TD {d3:,.0f} ft', fontsize=20, fontweight='bold', y=0.98)
            d_old, d_new = old["depth"], new["depth"]; max_d = max(d_old) + 500
            s_old = {'color':'#e74c3c','marker':'o','lw':2.5,'label':'Well A'}
            s_new = {'color':'#3498db','marker':'o','lw':2.5,'label':'Well B'}

            axs[0,0].plot(old["rop"], d_old, **s_old); axs[0,0].plot(new["rop"], d_new, **s_new); axs[0,0].set(ylim=(max_d, 0), title='ROP (ft/hr)'); axs[0,0].grid(True, ls='--'); axs[0,0].legend()
            axs[0,1].plot(old["ecd"], d_old, **s_old); axs[0,1].plot(new["ecd"], d_new, **s_new); axs[0,1].plot(new["pp"], d_old, 'k--', label='Pore Pressure'); axs[0,1].plot(new["fg"], d_old, 'k-', label='Frac Gradient'); axs[0,1].fill_betweenx(d_old, new["pp"], new["fg"], color='#2ecc71', alpha=0.1); axs[0,1].set(ylim=(max_d, 0), xlim=(8, 15), title='Mud Window (ppg)'); axs[0,1].grid(True, ls='--'); axs[0,1].legend()
            
            bars = axs[0,2].bar(['Well A', 'Well B'], [old["cost"]/1e6, new["cost"]/1e6], color=['#e74c3c','#3498db']); axs[0,2].set(title='Total Cost (Million USD)'); 
            for b in bars: axs[0,2].text(b.get_x() + b.get_width()/2, b.get_height(), f'${b.get_height():.2f}M', ha='center', va='bottom', fontweight='bold')

            axs[1,0].plot(old["lgs"], d_old, **s_old); axs[1,0].plot(new["lgs"], d_new, **s_new); axs[1,0].set(ylim=(max_d, 0), xlim=(0, max(max(old["lgs"])+5, 15)), title='Generated Cuttings (LGS %)'); axs[1,0].grid(True, ls='--'); axs[1,0].legend()
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

        with tab3:
            st.subheader(f"Mud Rheology & Fann 35 Viscometer Dial Reading (at TD: {d3:,.0f} ft)")
            st.markdown("*Dial readings are generated via simulation of Krieger-Dougherty fluid mechanics converted into True Shear Stress components.*")
            lgs_old = old["lgs"][-1]; pv_old = old["pv"][-1]; yp_old = old["yp"][-1]; r600_old = old["r600"][-1]; r300_old = old["r300"][-1]
            lgs_new = new["lgs"][-1]; pv_new = new["pv"][-1]; yp_new = new["yp"][-1]; r600_new = new["r600"][-1]; r300_new = new["r300"][-1]
            
            rheo_data = {
                "Parameter": ["Actual Generated LGS (%)", "Plastic Viscosity (cP)", "Yield Point (lb/100ft2)", "Dial Reading 300 RPM", "Dial Reading 600 RPM"],
                "Well A (Old System)": [f"{lgs_old:.1f}", f"{pv_old:.1f}", f"{yp_old:.1f}", f"{r300_old:.1f}", f"{r600_old:.1f}"],
                "Well B (New System)": [f"{lgs_new:.1f}", f"{pv_new:.1f}", f"{yp_new:.1f}", f"{r300_new:.1f}", f"{r600_new:.1f}"]
            }
            st.table(rheo_data)
else:
    st.info("👈 Set drilling parameters (GPM, ROP) in the sidebar to simulate generated LGS, then click 'RUN PHYSICS SIMULATION'!")
