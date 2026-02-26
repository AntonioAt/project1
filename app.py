import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# 1. ADVANCED PHYSICS MODULES (API MASS BALANCE & RHEOLOGY)
# =============================================================================
class API_MassBalanceAnalyzer:
    def __init__(self, mud_price_bbl=75.0, disposal_price_bbl=15.0):
        self.mud_price = mud_price_bbl
        self.disposal_price = disposal_price_bbl
        self.liquid_on_cuttings_ratio = 1.0  # (Y parameter) 1 bbl liquid per 1 bbl rock

    def calculate_interval(self, hole_in, length_ft, washout, mech_sre_X, target_lgs_frac):
        # Eq 2: Volume of drilled solids (Vc)
        v_c = 0.000971 * (hole_in**2) * length_ft * washout
        
        # Hole volume created (Vh)
        v_h = v_c  # Assuming volume of rock removed equals hole volume created
        
        # Eq 11: Wet solids volume to be disposed (Vsw)
        v_sw = mech_sre_X * v_c * (1.0 + self.liquid_on_cuttings_ratio)
        mud_lost_on_cuttings = mech_sre_X * v_c * self.liquid_on_cuttings_ratio
        
        # Solids retained in the active system
        v_c_retained = v_c * (1.0 - mech_sre_X)
        
        # 1. Minimum Mud Built required to fill hole and replace losses (Volume Building)
        v_m_volumetric = v_h + mud_lost_on_cuttings
        
        # 2. Mud Built required to maintain target LGS (Dilution Requirement)
        if target_lgs_frac > 0:
            v_m_dilution = v_c_retained / target_lgs_frac
        else:
            v_m_dilution = v_m_volumetric
            
        # Actual Mud Built (Vm) is the maximum of the two requirements
        v_m_actual = max(v_m_volumetric, v_m_dilution)
        
        # Liquid Waste (Vlw) occurs if dilution volume exceeds hole filling needs
        v_lw = max(0.0, v_m_actual - v_m_volumetric)
        
        # Eq 3: Dilution volume required if NO solid is removed (Vd theoretical)
        v_d_theoretical = v_c / target_lgs_frac if target_lgs_frac > 0 else 0
        
        # Eq 4 & 5: Total Efficiency of Solid System (Et)
        df = v_m_actual / v_d_theoretical if v_d_theoretical > 0 else 1.0
        e_t = (1.0 - df) * 100.0
        e_t = max(0.0, min(e_t, 100.0))
        
        # Costs
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
    def __init__(self, base_pv, base_yp):
        self.machine = Fann35Machine()
        self.polymer_visc_factor = base_pv / 3.0
        self.polymer_yield_strength = base_yp * 0.8
        self.surface_temp = 80.0 
        self.geo_gradient = 1.6  

    def get_temp_at_depth(self, tvd_ft):
        return self.surface_temp + (self.geo_gradient * (tvd_ft / 100.0))

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
        
        mu_medium = 1.0 * self.polymer_visc_factor * thermal_factor
        rel_visc = (1 - (phi / phi_max)) ** (-eta * phi_max) if phi < phi_max else 50.0
        true_visc = mu_medium * rel_visc
        
        thermal_agitation = 1.0 + (0.001 * temp_diff)
        true_yield = (self.polymer_yield_strength + (0.8 * lgs_pct)) * thermal_agitation
        
        r600 = self.machine.reading_from_stress(true_yield + (true_visc * (self.machine.get_shear_rate(600)/300.0)))
        r300 = self.machine.reading_from_stress(true_yield + (true_visc * (self.machine.get_shear_rate(300)/300.0)))
        
        sim_pv = r600 - r300
        sim_yp = r300 - sim_pv
        return round(sim_pv, 1), round(sim_yp, 1), round(r600, 1), round(r300, 1)

    def calculate_hydraulics(self, pv, yp, actual_mw_ppg, depth_ft, hole, dp, gpm, pp, rop_max):
        annular_cap = (hole**2 - dp**2) / 1029.4
        vel = gpm / (annular_cap * 42.0)
        d_hyd = hole - dp
        fric = (((pv * vel) / (1500 * d_hyd**2)) + (yp / (225 * d_hyd))) * depth_ft
        ecd = actual_mw_ppg + (fric / (0.052 * depth_ft))
        rop = rop_max * np.exp(-0.3 * max(0, ecd - pp))
        return round(ecd, 2), round(rop, 1)

class EconomicsAnalyzer:
    def __init__(self, rig_rate, bit_cost=25000.0):
        self.rig_rate = rig_rate; self.bit_cost = bit_cost
    def calculate_time_cost(self, avg_rop, length_ft, avg_lgs, daily_equip_cost, chem_penalty_daily):
        d_hrs = (length_ft / avg_rop) if avg_rop > 0 else 9999.0
        # Time penalty for poor LGS control
        t_days = (d_hrs + (length_ft / 1000.0) + (10.0 if avg_lgs > 8.0 else 0)) / 24.0
        t_cost = (self.rig_rate * t_days) + self.bit_cost + (daily_equip_cost * t_days) + (chem_penalty_daily * t_days)
        return t_days, t_cost, (self.rig_rate * t_days), (chem_penalty_daily * t_days)

def generate_dynamic_log(start_d, end_d, pp_base, fg_base, rop_base):
    log = []
    points = np.linspace(start_d + 100, end_d, 3) 
    for d in points:
        base_mw = 9.0 + (d / 2000.0) 
        pp = pp_base + (d / 3000.0)
        fg = fg_base + (d / 2000.0)
        log.append((round(d, 1), round(base_mw, 2), round(pp, 2), round(fg, 2), rop_base))
    return log

# =============================================================================
# 2. MODULAR EQUIPMENT CATALOG
# =============================================================================
# Mech_Efficiency: % of particles removed (X in API formula)
# Chem_Penalty: Daily cost of replacing Barite/Polymers thrown away by the unit
EQUIPMENT_CATALOG = {
    "Shale Shaker": {"mech_efficiency": 0.45, "cost": 500.0, "chem_penalty": 0.0},
    "Desander": {"mech_efficiency": 0.15, "cost": 300.0, "chem_penalty": 200.0},
    "Desilter": {"mech_efficiency": 0.20, "cost": 400.0, "chem_penalty": 300.0},
    "Mud Cleaner": {"mech_efficiency": 0.30, "cost": 800.0, "chem_penalty": 150.0},
    "Centrifuge": {"mech_efficiency": 0.60, "cost": 1500.0, "chem_penalty": 1200.0}
}

def calculate_modular_system(selected_equipments):
    if not selected_equipments:
        return 0.0, 0.0, 0.0 # X=0 (no separation), Cost=0
    
    pass_factor = 1.0
    total_cost = 0.0
    total_chem_penalty = 0.0
    
    for eq in selected_equipments:
        # Probabilistic sequential removal: If Shaker removes 45%, 55% passes to next unit.
        eff = EQUIPMENT_CATALOG[eq]["mech_efficiency"]
        pass_factor *= (1.0 - eff)
        total_cost += EQUIPMENT_CATALOG[eq]["cost"]
        total_chem_penalty += EQUIPMENT_CATALOG[eq]["chem_penalty"]
        
    system_efficiency_X = 1.0 - pass_factor
    return system_efficiency_X, total_cost, total_chem_penalty

# =============================================================================
# 3. STREAMLIT WEB APP FRONT-END 
# =============================================================================
st.set_page_config(page_title="Advanced API Drilling Simulator", layout="wide")

if "num_scenarios" not in st.session_state:
    st.session_state.num_scenarios = 2

st.title("API Mass-Balance Drilling Simulator")
st.markdown("Engineering-grade evaluation using API volumetric mass balance, dynamic modular SRE, and Barite loss penalties.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Global Configurations")

with st.sidebar.expander("Base Rig & Fluid Economics", expanded=False):
    rig_rate = st.number_input("Rig Lease Rate ($/Day)", value=35000.0, step=1000.0)
    mud_price = st.number_input("Fresh Mud Base Cost ($/bbl)", value=85.0, step=5.0)
    disp_price = st.number_input("Waste Disposal Cost ($/bbl)", value=18.0, step=2.0)
    base_pv = st.number_input("Base Mud PV (cP)", value=14.0, step=0.5)
    base_yp = st.number_input("Base Mud YP (lb/100ft2)", value=10.0, step=0.5)
    target_lgs_des = st.number_input("Target LGS (ks fraction)", value=0.06, step=0.01, format="%.2f")
    st.caption("Total circulating volume is now dynamically built based on depth progression.")

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
            "Select Processing Units", 
            options=list(EQUIPMENT_CATALOG.keys()), 
            default=["Shale Shaker"] if i==0 else ["Shale Shaker", "Mud Cleaner", "Centrifuge"],
            key=f"eq_{i}"
        )
        eff_X, cost, chem_pen = calculate_modular_system(selected_eq)
        st.caption(f"Mechanical SRE (X): {eff_X*100:.1f}%")
        st.caption(f"Rental: ${cost:,.0f}/d | Chem Loss: ${chem_pen:,.0f}/d")
        scenario_configs[sc_name] = {"equipments": selected_eq, "mech_X": eff_X, "cost": cost, "chem": chem_pen}

SCENARIO_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#1abc9c']

# --- EXECUTION ENGINE ---
if st.sidebar.button("Run API Mass Balance Simulation", type="primary", use_container_width=True):
    with st.spinner("Processing API equations, barite loss kinetics, and rheology..."):
        
        d1 = len_sec1; d2 = d1 + len_sec2; d3 = d2 + len_sec3
        
        scenarios = {}
        for sc_name, config in scenario_configs.items():
            scenarios[sc_name] = {
                "daily_eq_cost": config["cost"], "chem_penalty": config["chem"], "mech_X": config["mech_X"],
                "sections": [
                    {"hole": 17.5, "dp": 5.0, "len": len_sec1, "gpm": gpm1, "log": generate_dynamic_log(0, d1, 8.6, 11.5, 90)},
                    {"hole": 12.25, "dp": 5.0, "len": len_sec2, "gpm": gpm2, "log": generate_dynamic_log(d1, d2, 9.2, 12.8, 65)},
                    {"hole": 8.5, "dp": 4.0, "len": len_sec3, "gpm": gpm3, "log": generate_dynamic_log(d2, d3, 10.4, 14.8, 45)}
                ]
            }
        
        engine = AdvancedDrillingPhysics(base_pv, base_yp)
        econ = EconomicsAnalyzer(rig_rate)
        mass_bal = API_MassBalanceAnalyzer(mud_price, disp_price)
        
        sim_res = {k: {"depth":[],"rop":[],"ecd":[],"actual_mw":[],"pv":[],"yp":[], 
                       "cost":0,"days":0,"equip_invest":0, "mud_cost":0, "disp_cost":0, "chem_cost":0, 
                       "total_vm":0, "total_waste":0, "api_et_avg":0} for k in scenarios}

        for sc_name, sc_data in scenarios.items():
            t_cost = 0; t_days = 0; t_invest = 0; t_mud_c = 0; t_disp_c = 0; t_chem_c = 0
            t_vm = 0; t_waste = 0; et_sum = 0
            mech_X = sc_data["mech_X"]
            
            for sec in sc_data["sections"]:
                avg_rop = 0
                
                # 1. API Mass Balance per section
                vm, vsw, vlw, api_et, c_mud, c_disp = mass_bal.calculate_interval(
                    sec['hole'], sec['len'], 1.0, mech_X, target_lgs_des
                )
                
                t_vm += vm; t_waste += (vsw + vlw); et_sum += api_et
                t_mud_c += c_mud; t_disp_c += c_disp
                
                # 2. Physics Logging
                sim_lgs_pct = target_lgs_des * 100.0 if vm > 0 else (1.0 - mech_X) * 15.0 # simplified steady state LGS
                
                for d, base_mw, pp, fg, rop_max in sec['log']:
                    temp = engine.get_temp_at_depth(d)
                    actual_mw = engine.calculate_actual_density(base_mw, sim_lgs_pct)
                    pv, yp, r600, r300 = engine.calculate_rheology(sim_lgs_pct, temp)
                    ecd, rop = engine.calculate_hydraulics(pv, yp, actual_mw, d, sec['hole'], sec['dp'], sec['gpm'], pp, rop_max)
                    
                    sim_res[sc_name]["depth"].append(d); sim_res[sc_name]["rop"].append(rop); sim_res[sc_name]["ecd"].append(ecd)
                    sim_res[sc_name]["actual_mw"].append(actual_mw); sim_res[sc_name]["pv"].append(pv); sim_res[sc_name]["yp"].append(yp)
                    avg_rop += rop
                
                # 3. Time & Economics
                days, base_cost, rig_c, chem_c = econ.calculate_time_cost(
                    avg_rop/len(sec['log']), sec['len'], sim_lgs_pct, sc_data["daily_eq_cost"], sc_data["chem_penalty"]
                )
                
                t_days += days; t_invest += (sc_data["daily_eq_cost"] * days); t_chem_c += chem_c
                t_cost += base_cost + c_mud + c_disp
                
            sim_res[sc_name].update({
                "cost": t_cost, "days": t_days, "equip_invest": t_invest, 
                "mud_cost": t_mud_c, "disp_cost": t_disp_c, "chem_cost": t_chem_c,
                "total_vm": t_vm, "total_waste": t_waste, "api_et_avg": et_sum / len(sc_data["sections"])
            })

        # --- DASHBOARD RENDERING ---
        tab1, tab2 = st.tabs(["Interactive Physics Dashboard", "API AFE & Mass Balance"])
        
        with tab1:
            st.subheader("Multi-Scenario Performance Comparison")
            
            fig = make_subplots(
                rows=2, cols=3, 
                subplot_titles=('Drilling Speed (ft/hr)', 'Hydraulics & Actual MW (ppg)', 'Total Project Cost (USD)', 
                                'API Total Efficiency (Et %)', 'Plastic Viscosity (cP)', 'Yield Point (lb/100ft2)'),
                horizontal_spacing=0.08, vertical_spacing=0.15
            )

            # Draw Cost Bar Chart (Col 3) and Efficiency Bar Chart (Col 4)
            costs_x = []; costs_y = []; eff_y = []; bar_colors = []; bar_texts = []; eff_texts = []
            
            for idx, (sc_name, data) in enumerate(sim_res.items()):
                c = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
                
                fig.add_trace(go.Scatter(x=data["rop"], y=data["depth"], name=sc_name, line=dict(color=c, width=2.5), mode='lines+markers', hovertemplate="%{x:.2f} ft/hr"), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=data["ecd"], y=data["depth"], line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False, hovertemplate="ECD: %{x:.2f} ppg"), row=1, col=2)
                fig.add_trace(go.Scatter(x=data["actual_mw"], y=data["depth"], line=dict(color=c, dash='dot', width=1.5), mode='lines', showlegend=False), row=1, col=2)
                
                costs_x.append(sc_name); costs_y.append(data["cost"])
                eff_y.append(data["api_et_avg"]); bar_colors.append(c)
                bar_texts.append(f'${data["cost"]/1e6:.2f}M')
                eff_texts.append(f'{data["api_et_avg"]:.1f}%')

                fig.add_trace(go.Scatter(x=data["pv"], y=data["depth"], line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False), row=2, col=2)
                fig.add_trace(go.Scatter(x=data["yp"], y=data["depth"], line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False), row=2, col=3)

                fig.update_yaxes(autorange="reversed", row=1, col=1)
                fig.update_yaxes(autorange="reversed", row=1, col=2)
                fig.update_yaxes(autorange="reversed", row=2, col=2)
                fig.update_yaxes(autorange="reversed", row=2, col=3)

            fig.add_trace(go.Bar(x=costs_x, y=costs_y, marker_color=bar_colors, text=bar_texts, textposition='auto', showlegend=False), row=1, col=3)
            fig.add_trace(go.Bar(x=costs_x, y=eff_y, marker_color=bar_colors, text=eff_texts, textposition='auto', showlegend=False), row=2, col=1)

            fig.update_layout(height=750, hovermode="y unified", margin=dict(t=50, l=20, r=20, b=20), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
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

else:
    st.info("Configure your modular scenarios and equipment in the sidebar, then click 'Run API Mass Balance Simulation'.")
