import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# 1. ADVANCED PHYSICS MODULES (API MASS BALANCE, KRIEGER-DOUGHERTY, DENSITY)
# =============================================================================
class API_MassBalanceAnalyzer:
    def __init__(self, mud_price_bbl=85.0, disposal_price_bbl=18.0):
        self.mud_price = mud_price_bbl
        self.disposal_price = disposal_price_bbl
        self.liquid_on_cuttings_ratio = 1.0  

    def calculate_interval(self, hole_in, length_ft, washout, mech_sre_X, target_lgs_frac):
        v_c = 0.000971 * (hole_in**2) * length_ft * washout
        v_h = v_c  
        
        v_sw = mech_sre_X * v_c * (1.0 + self.liquid_on_cuttings_ratio)
        mud_lost_on_cuttings = mech_sre_X * v_c * self.liquid_on_cuttings_ratio
        v_c_retained = v_c * (1.0 - mech_sre_X)
        
        v_m_volumetric = v_h + mud_lost_on_cuttings
        v_m_dilution = v_c_retained / target_lgs_frac if target_lgs_frac > 0 else v_m_volumetric
        v_m_actual = max(v_m_volumetric, v_m_dilution)
        v_lw = max(0.0, v_m_actual - v_m_volumetric)
        
        v_d_theoretical = v_c / target_lgs_frac if target_lgs_frac > 0 else 0
        df = v_m_actual / v_d_theoretical if v_d_theoretical > 0 else 1.0
        e_t = (1.0 - df) * 100.0
        e_t = max(0.0, min(e_t, 100.0))
        
        cost_mud = v_m_actual * self.mud_price
        cost_disposal = (v_sw + v_lw) * self.disposal_price
        
        return v_m_actual, v_sw, v_lw, e_t, cost_mud, cost_disposal

class Fann35Machine:
    def __init__(self):
        self.k_gamma = 1.703  
        self.k_tau = 0.511    
    def get_shear_rate(self, rpm): return rpm * self.k_gamma
    def reading_from_stress(self, stress_lb): return stress_lb / self.k_tau

class AdvancedDrillingPhysics:
    def __init__(self, t600, t300, t200, t100, t6, t3):
        self.machine = Fann35Machine()
        
        # --- HERSCHEL-BULKLEY RHEOLOGY EXTRACTION ---
        self.base_tau_y = t3  # Yield Stress approximation
        
        # Calculate Flow Behavior Index (n)
        try:
            ratio = (t600 - self.base_tau_y) / (t300 - self.base_tau_y)
            if ratio <= 0: ratio = 0.001
            self.base_n = 3.32 * np.log10(ratio)
        except:
            self.base_n = 1.0
            
        self.base_n = max(0.1, min(self.base_n, 1.2)) # Physical bounds limitation
        
        # Calculate Consistency Index (K)
        self.base_K = (t300 - self.base_tau_y) / (511 ** self.base_n)
        
        self.surface_temp = 80.0 
        self.geo_gradient = 1.6  

    def get_temp_at_depth(self, tvd_ft):
        return self.surface_temp + (self.geo_gradient * (tvd_ft / 100.0))

    def calculate_generated_lgs(self, hole_diam_in, rop_fph, gpm, sre_multiplier):
        hole_cap = (hole_diam_in ** 2) / 1029.4
        cuttings_bbl_hr = hole_cap * rop_fph
        mud_bbl_hr = (gpm * 60) / 42.0
        transport_efficiency = 0.8
        lgs_concentration = (cuttings_bbl_hr / (mud_bbl_hr * transport_efficiency)) * 100.0
        
        background_lgs = 3.0
        total_lgs = (lgs_concentration + background_lgs) * sre_multiplier
        return min(total_lgs, 25.0) 

    def calculate_actual_density(self, base_mw, lgs_pct):
        lgs_sg_ppg = 21.7
        lgs_frac = lgs_pct / 100.0
        actual_mw = (base_mw * (1.0 - lgs_frac)) + (lgs_sg_ppg * lgs_frac)
        return round(actual_mw, 2)

    def calculate_rheology(self, lgs_pct, temp_f):
        temp_diff = temp_f - 120.0
        thermal_factor = max(0.6, 1.0 - (0.005 * temp_diff))
        
        phi = lgs_pct / 100.0
        phi_max = 0.60; eta = 2.5
        
        # Krieger-Dougherty effect applied to Consistency Index (K) and Yield Stress
        rel_visc = (1 - (phi / phi_max)) ** (-eta * phi_max) if phi < phi_max else 50.0
        
        actual_K = self.base_K * thermal_factor * rel_visc
        
        thermal_agitation = 1.0 + (0.001 * temp_diff)
        actual_tau_y = (self.base_tau_y + (0.8 * lgs_pct)) * thermal_agitation
        actual_n = self.base_n 
        
        # Simulate Dial Readings using Herschel-Bulkley model back-calculation
        r600 = actual_tau_y + actual_K * (1022 ** actual_n)
        r300 = actual_tau_y + actual_K * (511 ** actual_n)
        
        sim_pv = r600 - r300
        sim_yp = r300 - sim_pv
        return round(actual_n, 3), round(actual_K, 3), round(actual_tau_y, 1), round(sim_pv, 1), round(sim_yp, 1), round(r600, 1), round(r300, 1)

    def calculate_hydraulics(self, n, K, tau_y, actual_mw_ppg, depth_ft, hole, dp, gpm, pp, rop_max):
        # --- HERSCHEL-BULKLEY ANNULAR FRICTION (API RP 13D) ---
        annular_cap = (hole**2 - dp**2) / 1029.4
        vel_ft_min = gpm / annular_cap
        
        # Effective shear rate in annulus
        if hole > dp and n > 0:
            gamma_a = (2.4 * vel_ft_min / (hole - dp)) * ((2 * n + 1) / (3 * n))
        else:
            gamma_a = 10.0
            
        # Wall shear stress in dial units equivalent
        tau_w_dial = tau_y + K * (gamma_a ** n)
        
        # Convert dial units to lb/100ft2
        tau_w_lb100 = 1.066 * tau_w_dial
        
        # Annular Frictional Pressure Drop converted directly to ECD increment
        ecd_inc = tau_w_lb100 / (15.6 * (hole - dp)) if hole > dp else 0
        ecd = actual_mw_ppg + ecd_inc
        
        rop = rop_max * np.exp(-0.3 * max(0, ecd - pp))
        return round(ecd, 2), round(rop, 1)

class EconomicsAnalyzer:
    def __init__(self, rig_rate, bit_cost=25000.0):
        self.rig_rate = rig_rate; self.bit_cost = bit_cost
    def calculate_time_cost(self, avg_rop, length_ft, avg_lgs, daily_equip_cost, daily_chem_penalty):
        d_hrs = (length_ft / avg_rop) if avg_rop > 0 else 9999.0
        t_days = (d_hrs + (length_ft / 1000.0) + (10.0 if avg_lgs > 8.0 else 0)) / 24.0
        rig_c = self.rig_rate * t_days
        eq_c = daily_equip_cost * t_days
        chem_c = daily_chem_penalty * t_days
        t_cost = rig_c + self.bit_cost + eq_c + chem_c
        return t_days, t_cost, rig_c, chem_c

def generate_dynamic_log(start_d, end_d, pp_base, fg_base, rop_base, step_ft=500):
    log = []
    points = np.arange(start_d + step_ft, end_d, step_ft)
    if len(points) == 0 or points[-1] != end_d:
        points = np.append(points, end_d)
    for d in points:
        base_mw = 9.0 + (d / 2000.0) 
        pp = pp_base + (d / 3000.0)
        fg = fg_base + (d / 2000.0)
        log.append((round(d, 1), round(base_mw, 2), round(pp, 2), round(fg, 2), rop_base))
    return log

# =============================================================================
# 2. MODULAR EQUIPMENT CATALOG 
# =============================================================================
EQUIPMENT_CATALOG = {
    "Shale Shaker": {"mech_efficiency": 0.45, "cost": 500.0, "chem_penalty": 0.0},
    "Desander": {"mech_efficiency": 0.15, "cost": 300.0, "chem_penalty": 200.0},
    "Desilter": {"mech_efficiency": 0.20, "cost": 400.0, "chem_penalty": 300.0},
    "Mud Cleaner": {"mech_efficiency": 0.30, "cost": 800.0, "chem_penalty": 150.0},
    "Centrifuge": {"mech_efficiency": 0.60, "cost": 1500.0, "chem_penalty": 1200.0}
}

def calculate_modular_system(selected_equipments):
    if not selected_equipments:
        return 0.0, 0.0, 0.0 
    pass_factor = 1.0
    total_cost = 0.0
    total_chem_penalty = 0.0
    for eq in selected_equipments:
        eff = EQUIPMENT_CATALOG[eq]["mech_efficiency"]
        pass_factor *= (1.0 - eff)
        total_cost += EQUIPMENT_CATALOG[eq]["cost"]
        total_chem_penalty += EQUIPMENT_CATALOG[eq]["chem_penalty"]
    system_efficiency_X = 1.0 - pass_factor
    return system_efficiency_X, total_cost, total_chem_penalty

# =============================================================================
# 3. STREAMLIT WEB APP FRONT-END 
# =============================================================================
st.set_page_config(page_title="Drilling & Solid Control Simulator", layout="wide")

if "num_scenarios" not in st.session_state:
    st.session_state.num_scenarios = 2

st.title("API First-Principles Drilling Simulator")
st.markdown("Dynamic evaluation of Modular SRE, Herschel-Bulkley Rheology, and API Mass Balance.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Global Configurations")

with st.sidebar.expander("Base Rig & Fluid Economics", expanded=False):
    rig_rate = st.number_input("Rig Lease Rate ($/Day)", value=35000.0, step=1000.0)
    mud_price = st.number_input("Fresh Mud Cost ($/bbl)", value=85.0, step=5.0)
    disp_price = st.number_input("Waste Disposal Cost ($/bbl)", value=18.0, step=2.0)
    target_lgs_des = st.number_input("Target LGS (Max Limit %)", value=6.0, step=0.5)
    
    st.markdown("**Fann 35 Viscometer Dial Readings**")
    c1, c2 = st.columns(2)
    t600 = c1.number_input("600 RPM", value=55.0, step=1.0)
    t300 = c2.number_input("300 RPM", value=35.0, step=1.0)
    t200 = c1.number_input("200 RPM", value=28.0, step=1.0)
    t100 = c2.number_input("100 RPM", value=20.0, step=1.0)
    t6 = c1.number_input("6 RPM", value=8.0, step=1.0)
    t3 = c2.number_input("3 RPM", value=6.0, step=1.0)

with st.sidebar.expander("Well Trajectory & Hydraulics", expanded=False):
    len_sec1 = st.number_input("Length (ft) - Sec 1 (17.5\")", value=1250.0, step=100.0)
    gpm1 = st.number_input("GPM - Sec 1", value=1050.0, step=50.0)
    len_sec2 = st.number_input("Length (ft) - Sec 2 (12.25\")", value=3500.0, step=100.0)
    gpm2 = st.number_input("GPM - Sec 2", value=850.0, step=50.0)
    len_sec3 = st.number_input("Length (ft) - Sec 3 (8.5\")", value=3350.0, step=100.0)
    gpm3 = st.number_input("GPM - Sec 3", value=450.0, step=50.0)

st.sidebar.divider()
st.sidebar.header("Modular Scenario Builder")

col1, col2 = st.sidebar.columns(2)
if col1.button("➕ Add Scenario"): st.session_state.num_scenarios += 1
if col2.button("➖ Remove Scenario"): 
    if st.session_state.num_scenarios > 1: st.session_state.num_scenarios -= 1

scenario_configs = {}
for i in range(st.session_state.num_scenarios):
    sc_name = f"Scenario {chr(65+i)}" 
    with st.sidebar.expander(f"🛠️ {sc_name} Equipment", expanded=True):
        selected_eq = st.multiselect(
            "Select Solid Control Equipment", 
            options=list(EQUIPMENT_CATALOG.keys()), 
            default=["Shale Shaker"] if i==0 else ["Shale Shaker", "Mud Cleaner", "Centrifuge"],
            key=f"eq_{i}"
        )
        eff_X, cost, chem_pen = calculate_modular_system(selected_eq)
        st.caption(f"Mech. SRE (X): {eff_X*100:.1f}% | Rent: ${cost:,.0f}/d | Chem Loss: ${chem_pen:,.0f}/d")
        scenario_configs[sc_name] = {"equipments": selected_eq, "mech_X": eff_X, "cost": cost, "chem": chem_pen}

SCENARIO_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#1abc9c']

# --- EXECUTION ENGINE ---
if st.sidebar.button("Run Physics & Mass Balance", type="primary", use_container_width=True):
    with st.spinner("Processing HB fluid mechanics and API mass balance..."):
        
        d1 = len_sec1; d2 = d1 + len_sec2; d3 = d2 + len_sec3
        target_lgs_frac = target_lgs_des / 100.0
        
        scenarios = {}
        for sc_name, config in scenario_configs.items():
            scenarios[sc_name] = {
                "daily_eq_cost": config["cost"], "chem_penalty": config["chem"], "mech_X": config["mech_X"],
                "sections": [
                    {"hole": 17.5, "dp": 5.0, "len": len_sec1, "gpm": gpm1, "wash": 1.15, "log": generate_dynamic_log(0, d1, 8.6, 11.5, 90, step_ft=500)},
                    {"hole": 12.25, "dp": 5.0, "len": len_sec2, "gpm": gpm2, "wash": 1.10, "log": generate_dynamic_log(d1, d2, 9.2, 12.8, 65, step_ft=500)},
                    {"hole": 8.5, "dp": 4.0, "len": len_sec3, "gpm": gpm3, "wash": 1.05, "log": generate_dynamic_log(d2, d3, 10.4, 14.8, 45, step_ft=500)}
                ]
            }
        
        engine = AdvancedDrillingPhysics(t600, t300, t200, t100, t6, t3)
        econ = EconomicsAnalyzer(rig_rate)
        mass_bal = API_MassBalanceAnalyzer(mud_price, disp_price)
        
        sim_res = {k: {"depth":[],"hole":[], "rop":[],"ecd":[],"pp":[],"fg":[],"lgs":[],"base_mw":[],"actual_mw":[],"pv":[],"yp":[],"r600":[],"r300":[], "hb_n":[], "hb_k":[], "hb_tau":[],
                       "cost":0,"days":0,"equip_invest":0, "mud_cost":0, "disp_cost":0, "chem_cost":0, "total_vm":0, "total_waste":0, "api_et_avg":0} for k in scenarios}

        for sc_name, sc_data in scenarios.items():
            t_cost = 0; t_days = 0; t_invest = 0; t_mud_c = 0; t_disp_c = 0; t_chem_c = 0
            t_vm = 0; t_waste = 0; et_sum = 0
            mech_X = sc_data["mech_X"]
            sre_mult = 1.0 - mech_X  
            
            for sec in sc_data["sections"]:
                avg_rop = 0; lgs_sum = 0
                
                vm, vsw, vlw, api_et, c_mud, c_disp = mass_bal.calculate_interval(sec['hole'], sec['len'], sec['wash'], mech_X, target_lgs_frac)
                t_vm += vm; t_waste += (vsw + vlw); et_sum += api_et
                t_mud_c += c_mud; t_disp_c += c_disp
                
                for d, base_mw, pp, fg, rop_max in sec['log']:
                    temp = engine.get_temp_at_depth(d)
                    
                    lgs = engine.calculate_generated_lgs(sec['hole'], rop_max, sec['gpm'], sre_mult)
                    actual_mw = engine.calculate_actual_density(base_mw, lgs)
                    
                    hb_n, hb_k, hb_tau, pv, yp, r600, r300 = engine.calculate_rheology(lgs, temp)
                    ecd, rop = engine.calculate_hydraulics(hb_n, hb_k, hb_tau, actual_mw, d, sec['hole'], sec['dp'], sec['gpm'], pp, rop_max)
                    
                    sim_res[sc_name]["hole"].append(sec['hole']); sim_res[sc_name]["depth"].append(d)
                    sim_res[sc_name]["rop"].append(rop); sim_res[sc_name]["ecd"].append(ecd)
                    sim_res[sc_name]["pp"].append(pp); sim_res[sc_name]["fg"].append(fg)
                    sim_res[sc_name]["lgs"].append(lgs); sim_res[sc_name]["base_mw"].append(base_mw); sim_res[sc_name]["actual_mw"].append(actual_mw)
                    sim_res[sc_name]["hb_n"].append(hb_n); sim_res[sc_name]["hb_k"].append(hb_k); sim_res[sc_name]["hb_tau"].append(hb_tau)
                    sim_res[sc_name]["pv"].append(pv); sim_res[sc_name]["yp"].append(yp)
                    sim_res[sc_name]["r600"].append(r600); sim_res[sc_name]["r300"].append(r300)
                    avg_rop += rop; lgs_sum += lgs
                
                sec_avg_lgs = lgs_sum / len(sec['log'])
                
                days, base_cost, rig_c, chem_c = econ.calculate_time_cost(avg_rop/len(sec['log']), sec['len'], sec_avg_lgs, sc_data["daily_eq_cost"], sc_data["chem_penalty"])
                t_days += days; t_invest += (sc_data["daily_eq_cost"] * days); t_chem_c += chem_c
                t_cost += base_cost + c_mud + c_disp
                
            sim_res[sc_name].update({
                "cost": t_cost, "days": t_days, "equip_invest": t_invest, 
                "mud_cost": t_mud_c, "disp_cost": t_disp_c, "chem_cost": t_chem_c,
                "total_vm": t_vm, "total_waste": t_waste, "api_et_avg": et_sum / len(sc_data["sections"])
            })

        # --- DASHBOARD RENDERING ---
        tab1, tab2, tab3, tab4 = st.tabs(["Interactive Dashboard", "API Mass Balance & AFE", "Rheology Report (at TD)", "Detailed Data Logs"])
        
        with tab1:
            st.subheader("Multi-Scenario Performance Comparison")
            
            fig = make_subplots(
                rows=2, cols=3, 
                subplot_titles=('Drilling Speed (ft/hr)', 'Hydraulics & Actual MW (ppg)', 'Total Project Cost (USD)', 
                                'Solid Accumulation (%)', 'Plastic Viscosity (cP)', 'Yield Point (lb/100ft2)'),
                horizontal_spacing=0.08, vertical_spacing=0.15
            )

            base_sc = list(sim_res.keys())[0]
            fig.add_trace(go.Scatter(x=sim_res[base_sc]["pp"], y=sim_res[base_sc]["depth"], name='Pore Pressure', line=dict(color='black', dash='dash'), hovertemplate="%{x:.2f} ppg"), row=1, col=2)
            fig.add_trace(go.Scatter(x=sim_res[base_sc]["fg"], y=sim_res[base_sc]["depth"], name='Frac Gradient', line=dict(color='black'), hovertemplate="%{x:.2f} ppg"), row=1, col=2)
            fig.add_vline(x=target_lgs_des, line_dash="dash", line_color="black", line_width=2, annotation_text="Target LGS", row=2, col=1)

            costs_x = []; costs_y = []; bar_colors = []; bar_texts = []
            
            for idx, (sc_name, data) in enumerate(sim_res.items()):
                c = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
                
                fig.add_trace(go.Scatter(x=data["rop"], y=data["depth"], name=sc_name, line=dict(color=c, width=2.5), mode='lines+markers', hovertemplate="%{x:.2f} | Depth: %{y:.0f} ft"), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=data["ecd"], y=data["depth"], name=f"{sc_name} (ECD)", line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False, hovertemplate="ECD: %{x:.2f} ppg"), row=1, col=2)
                fig.add_trace(go.Scatter(x=data["actual_mw"], y=data["depth"], name=f"{sc_name} (MW)", line=dict(color=c, dash='dot', width=1.5), mode='lines', showlegend=False, hovertemplate="Actual MW: %{x:.2f} ppg"), row=1, col=2)
                
                costs_x.append(sc_name); costs_y.append(data["cost"]); bar_colors.append(c); bar_texts.append(f'${data["cost"]/1e6:.2f}M')

                fig.add_trace(go.Scatter(x=data["lgs"], y=data["depth"], line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False, hovertemplate="%{x:.2f} %"), row=2, col=1)
                fig.add_trace(go.Scatter(x=data["pv"], y=data["depth"], line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False, hovertemplate="%{x:.1f} cP"), row=2, col=2)
                fig.add_trace(go.Scatter(x=data["yp"], y=data["depth"], line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False, hovertemplate="%{x:.1f} lb/100ft2"), row=2, col=3)

                fig.update_yaxes(autorange="reversed", row=1, col=1)
                fig.update_yaxes(autorange="reversed", row=1, col=2)
                fig.update_yaxes(autorange="reversed", row=2, col=1)
                fig.update_yaxes(autorange="reversed", row=2, col=2)
                fig.update_yaxes(autorange="reversed", row=2, col=3)

            fig.add_trace(go.Bar(x=costs_x, y=costs_y, marker_color=bar_colors, text=bar_texts, textposition='auto', showlegend=False, hovertemplate="Cost: $%{y:,.0f}"), row=1, col=3)

            fig.update_layout(height=750, hovermode="y unified", margin=dict(t=50, l=20, r=20, b=20), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='lightgrey')
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='lightgrey')
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with tab2:
            st.subheader("Mass Balance & AFE Economics (API Standard)")
            summary_data = {
                "Metric": ["Equipment Selected", "Mud Built (Vm) bbls", "Waste Disposed (Vt) bbls", "API Efficiency (Et)", "1. Mud Cost ($)", "2. Disposal Cost ($)", "3. Barite/Chem Pen. ($)", "4. SRE Capex ($)", "TOTAL AFE COST ($)"]
            }
            for sc_name, data in sim_res.items():
                eq_str = " + ".join(scenario_configs[sc_name]["equipments"]) if scenario_configs[sc_name]["equipments"] else "None (Bypass)"
                summary_data[sc_name] = [
                    eq_str, 
                    f"{data['total_vm']:,.0f}", 
                    f"{data['total_waste']:,.0f}", 
                    f"{data['api_et_avg']:.1f}%", 
                    f"${data['mud_cost']:,.0f}", 
                    f"${data['disp_cost']:,.0f}", 
                    f"${data['chem_cost']:,.0f}", 
                    f"${data['equip_invest']:,.0f}", 
                    f"${data['cost']:,.0f}"
                ]
            st.table(summary_data)

        with tab3:
            st.subheader(f"Mud Rheology & Fann 35 Viscometer Data (at TD: {d3:,.0f} ft)")
            st.markdown("*Simulation now uses Herschel-Bulkley flow behavior modeling for non-Newtonian fluids.*")
            
            rheo_data = {
                "Parameter": ["Actual Generated LGS (%)", "Plastic Viscosity (cP)", "Yield Point (lb/100ft2)", "Dial Reading 300 RPM", "Dial Reading 600 RPM"]
            }
            for sc_name, data in sim_res.items():
                rheo_data[sc_name] = [
                    f"{data['lgs'][-1]:.1f}", 
                    f"{data['pv'][-1]:.1f}", 
                    f"{data['yp'][-1]:.1f}", 
                    f"{data['r300'][-1]:.1f}", 
                    f"{data['r600'][-1]:.1f}"
                ]
            st.table(rheo_data)

        with tab4:
            st.subheader("Comprehensive Section & Depth Logs")
            for sc_name, data in sim_res.items():
                st.markdown(f"**{sc_name} Data**")
                df = pd.DataFrame({
                    "Depth (ft)": data["depth"], 
                    "Hole (\")": data["hole"], 
                    "Generated LGS (%)": data["lgs"], 
                    "Base MW (ppg)": data["base_mw"], 
                    "Actual MW (ppg)": data["actual_mw"],
                    "HB 'n'": data["hb_n"],
                    "HB 'K'": data["hb_k"],
                    "HB 'Tau_y'": data["hb_tau"],
                    "PV (cP)": data["pv"], 
                    "YP (lb/100ft2)": data["yp"], 
                    "Fann 600": data["r600"], 
                    "Fann 300": data["r300"]
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("Configure your modular scenarios and Fann 35 readings in the sidebar, then click 'Run Physics & Mass Balance'.")
