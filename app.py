import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# 1. ADVANCED PHYSICS MODULES 
# =============================================================================
class SolidControlAnalyzer:
    def __init__(self, mud_price_bbl=75.0):
        self.mud_price = mud_price_bbl

    def calculate_dilution_and_sre(self, lgs_in, target_lgs, circ_vol_bbl):
        if lgs_in <= target_lgs:
            return 0.0, 100.0, 0.0
        v_d = circ_vol_bbl * ((lgs_in - target_lgs) / target_lgs)
        sre = (1.0 - (v_d / circ_vol_bbl)) * 100.0
        sre = max(0.0, min(sre, 100.0))
        mud_cost = v_d * self.mud_price
        return v_d, sre, mud_cost

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
    def calculate_cost(self, avg_rop, length_ft, avg_lgs, mud_cost, daily_equip_cost):
        d_hrs = (length_ft / avg_rop) if avg_rop > 0 else 9999.0
        t_days = (d_hrs + (length_ft / 1000.0) + (10.0 if avg_lgs > 8.0 else 0)) / 24.0
        t_cost = (self.rig_rate * t_days) + self.bit_cost + mud_cost + (daily_equip_cost * t_days)
        return t_days, t_cost, (self.rig_rate * t_days)

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
EQUIPMENT_CATALOG = {
    "Shale Shaker": {"pass_factor": 0.50, "cost": 500.0},
    "Desander": {"pass_factor": 0.80, "cost": 300.0},
    "Desilter": {"pass_factor": 0.80, "cost": 400.0},
    "Mud Cleaner": {"pass_factor": 0.70, "cost": 800.0},
    "Centrifuge": {"pass_factor": 0.40, "cost": 1500.0}
}

def calculate_modular_sre(selected_equipments):
    if not selected_equipments:
        return 2.5, 0.0 # No equipment: high bypass factor, 0 cost
    
    total_multiplier = 2.5
    total_cost = 0.0
    for eq in selected_equipments:
        total_multiplier *= EQUIPMENT_CATALOG[eq]["pass_factor"]
        total_cost += EQUIPMENT_CATALOG[eq]["cost"]
    
    return total_multiplier, total_cost

# =============================================================================
# 3. STREAMLIT WEB APP FRONT-END (DYNAMIC SCENARIOS)
# =============================================================================
st.set_page_config(page_title="Drilling & Solid Control Simulator", layout="wide")

# Initialize Dynamic Scenarios in Session State
if "num_scenarios" not in st.session_state:
    st.session_state.num_scenarios = 2

st.title("First-Principles Drilling Simulator")
st.markdown("Dynamic scenario modeling of Modular Solid Control Equipment, Dilution Kinetics, and Rheology.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Global Configurations")

with st.sidebar.expander("Base Rig & Fluid Parameters", expanded=False):
    rig_rate = st.number_input("Rig Lease Rate (USD/Day)", value=35000.0, step=1000.0, format="%.2f")
    base_pv = st.number_input("Base Mud PV (cP)", value=14.0, step=0.5, format="%.2f")
    base_yp = st.number_input("Base Mud YP (lb/100ft2)", value=10.0, step=0.5, format="%.2f")
    target_lgs_des = st.number_input("Target LGS after Dilution (%)", value=6.0, step=0.5, format="%.2f")
    circulating_volume = st.number_input("Total Circulating Volume (bbls)", value=2000.0, step=100.0)

with st.sidebar.expander("Well Trajectory & Hydraulics", expanded=False):
    len_sec1 = st.number_input("Length (ft) - Sec 1 (17.5\")", value=1250.0, step=100.0)
    gpm1 = st.number_input("GPM - Sec 1", value=1050.0, step=50.0)
    len_sec2 = st.number_input("Length (ft) - Sec 2 (12.25\")", value=3500.0, step=100.0)
    gpm2 = st.number_input("GPM - Sec 2", value=850.0, step=50.0)
    len_sec3 = st.number_input("Length (ft) - Sec 3 (8.5\")", value=3350.0, step=100.0)
    gpm3 = st.number_input("GPM - Sec 3", value=450.0, step=50.0)

st.sidebar.divider()
st.sidebar.header("Scenario Builder")

# Dynamic Scenario Management
col1, col2 = st.sidebar.columns(2)
if col1.button("➕ Add Scenario"):
    st.session_state.num_scenarios += 1
if col2.button("➖ Remove Scenario"):
    if st.session_state.num_scenarios > 1:
        st.session_state.num_scenarios -= 1

# Dictionary to hold user configurations
scenario_configs = {}

for i in range(st.session_state.num_scenarios):
    sc_name = f"Scenario {chr(65+i)}" # Generates Scenario A, B, C...
    with st.sidebar.expander(f"🛠️ {sc_name} Equipment", expanded=(i<2)):
        selected_eq = st.multiselect(
            "Select Solid Control Equipment", 
            options=list(EQUIPMENT_CATALOG.keys()), 
            default=["Shale Shaker"] if i==0 else ["Shale Shaker", "Mud Cleaner", "Centrifuge"],
            key=f"eq_{i}"
        )
        mult, cost = calculate_modular_sre(selected_eq)
        st.caption(f"Daily Eq. Cost: ${cost:,.0f} | Extracted SRE Multiplier: {mult:.2f}")
        scenario_configs[sc_name] = {"equipments": selected_eq, "multiplier": mult, "cost": cost}

# PLOTLY COLORS DEFINITION
SCENARIO_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#1abc9c']

# --- EXECUTION ENGINE ---
if st.sidebar.button("Run Physics Simulation", type="primary", use_container_width=True):
    with st.spinner("Processing cuttings generation, modular SRE dynamics, and rheology..."):
        
        d1 = len_sec1; d2 = d1 + len_sec2; d3 = d2 + len_sec3
        
        # Build scenario inputs dynamically
        scenarios = {}
        for sc_name, config in scenario_configs.items():
            scenarios[sc_name] = {
                "daily_eq_cost": config["cost"],
                "lgs_multiplier": config["multiplier"],
                "sections": [
                    {"hole": 17.5, "dp": 5.0, "len": len_sec1, "gpm": gpm1, "log": generate_dynamic_log(0, d1, 8.6, 11.5, 90)},
                    {"hole": 12.25, "dp": 5.0, "len": len_sec2, "gpm": gpm2, "log": generate_dynamic_log(d1, d2, 9.2, 12.8, 65)},
                    {"hole": 8.5, "dp": 4.0, "len": len_sec3, "gpm": gpm3, "log": generate_dynamic_log(d2, d3, 10.4, 14.8, 45)}
                ]
            }
        
        engine = AdvancedDrillingPhysics(base_pv, base_yp)
        econ = EconomicsAnalyzer(rig_rate)
        sc_ana = SolidControlAnalyzer()
        
        sim_res = {k: {"depth":[],"hole":[], "rop":[],"ecd":[],"pp":[],"fg":[],"lgs":[],"base_mw":[],"actual_mw":[],"pv":[],"yp":[],"r600":[],"r300":[], 
                       "cost":0,"days":0,"equip_invest":0, "mud_cost":0, "rig_cost":0, "dilution_vol":0, "sre_avg":0} for k in scenarios}

        for sc_name, sc_data in scenarios.items():
            t_cost = 0; t_days = 0; t_invest = 0; t_mud = 0; t_rig = 0; t_dilution = 0; t_sre = 0
            sre_mult = sc_data["lgs_multiplier"]
            
            for sec in sc_data["sections"]:
                avg_rop = 0; lgs_sum = 0
                for d, base_mw, pp, fg, rop_max in sec['log']:
                    temp = engine.get_temp_at_depth(d)
                    
                    lgs = engine.calculate_generated_lgs(sec['hole'], rop_max, sec['gpm'], sre_mult)
                    actual_mw = engine.calculate_actual_density(base_mw, lgs)
                    
                    pv, yp, r600, r300 = engine.calculate_rheology(lgs, temp)
                    ecd, rop = engine.calculate_hydraulics(pv, yp, actual_mw, d, sec['hole'], sec['dp'], sec['gpm'], pp, rop_max)
                    
                    sim_res[sc_name]["hole"].append(sec['hole']); sim_res[sc_name]["depth"].append(d)
                    sim_res[sc_name]["rop"].append(rop); sim_res[sc_name]["ecd"].append(ecd)
                    sim_res[sc_name]["pp"].append(pp); sim_res[sc_name]["fg"].append(fg)
                    sim_res[sc_name]["lgs"].append(lgs); sim_res[sc_name]["base_mw"].append(base_mw); sim_res[sc_name]["actual_mw"].append(actual_mw)
                    sim_res[sc_name]["pv"].append(pv); sim_res[sc_name]["yp"].append(yp)
                    sim_res[sc_name]["r600"].append(r600); sim_res[sc_name]["r300"].append(r300)
                    avg_rop += rop; lgs_sum += lgs
                
                sec_avg_lgs = lgs_sum / len(sec['log'])
                sec_vd, sec_sre, sec_mud_cost = sc_ana.calculate_dilution_and_sre(sec_avg_lgs, target_lgs_des, circulating_volume)
                
                days, cost, rig_c = econ.calculate_cost(avg_rop/len(sec['log']), sec['len'], sec_avg_lgs, sec_mud_cost, sc_data["daily_eq_cost"])
                
                t_days += days; t_cost += cost; t_invest += (sc_data["daily_eq_cost"] * days)
                t_mud += sec_mud_cost; t_rig += rig_c; t_dilution += sec_vd; t_sre += sec_sre
                
            sim_res[sc_name].update({"cost": t_cost, "days": t_days, "equip_invest": t_invest, "mud_cost": t_mud, "rig_cost": t_rig, "dilution_vol": t_dilution, "sre_avg": t_sre / len(sc_data["sections"])})

        # --- DASHBOARD RENDERING ---
        tab1, tab2, tab3 = st.tabs(["Interactive Dashboard", "Financial & Dilution Report", "Detailed Data Logs"])
        
        with tab1:
            st.subheader("Multi-Scenario Performance Comparison")
            
            # PLOTLY INTERACTIVE CHARTS
            fig = make_subplots(
                rows=2, cols=3, 
                subplot_titles=('Drilling Speed (ft/hr)', 'Hydraulics & Actual MW (ppg)', 'Total Cost (USD)', 
                                'Solid Accumulation (%)', 'Plastic Viscosity (cP)', 'Yield Point (lb/100ft2)'),
                horizontal_spacing=0.08, vertical_spacing=0.15
            )

            # Common background traces (PP and FG) using Scenario A's depth geometry
            base_sc = list(sim_res.keys())[0]
            fig.add_trace(go.Scatter(x=sim_res[base_sc]["pp"], y=sim_res[base_sc]["depth"], name='Pore Pressure', line=dict(color='black', dash='dash'), hovertemplate="%{x:.2f} ppg"), row=1, col=2)
            fig.add_trace(go.Scatter(x=sim_res[base_sc]["fg"], y=sim_res[base_sc]["depth"], name='Frac Gradient', line=dict(color='black'), hovertemplate="%{x:.2f} ppg"), row=1, col=2)
            fig.add_vline(x=target_lgs_des, line_dash="dash", line_color="black", line_width=2, annotation_text="Target LGS", row=2, col=1)

            # Loop through all dynamic scenarios to plot
            costs_x = []; costs_y = []; bar_colors = []; bar_texts = []
            
            for idx, (sc_name, data) in enumerate(sim_res.items()):
                c = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
                
                # Plot ROP
                fig.add_trace(go.Scatter(x=data["rop"], y=data["depth"], name=sc_name, line=dict(color=c, width=2.5), mode='lines+markers', hovertemplate="%{x:.2f} | Depth: %{y:.0f} ft"), row=1, col=1)
                
                # Plot Hydraulics (ECD and Actual MW)
                fig.add_trace(go.Scatter(x=data["ecd"], y=data["depth"], name=f"{sc_name} (ECD)", line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False, hovertemplate="ECD: %{x:.2f} ppg"), row=1, col=2)
                fig.add_trace(go.Scatter(x=data["actual_mw"], y=data["depth"], name=f"{sc_name} (MW)", line=dict(color=c, dash='dot', width=1.5), mode='lines', showlegend=False, hovertemplate="Actual MW: %{x:.2f} ppg"), row=1, col=2)
                
                # Prepare Bar Chart Data
                costs_x.append(sc_name)
                costs_y.append(data["cost"])
                bar_colors.append(c)
                bar_texts.append(f'${data["cost"]/1e6:.2f}M')

                # Plot LGS, PV, YP
                fig.add_trace(go.Scatter(x=data["lgs"], y=data["depth"], line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False, hovertemplate="%{x:.2f} %"), row=2, col=1)
                fig.add_trace(go.Scatter(x=data["pv"], y=data["depth"], line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False, hovertemplate="%{x:.1f} cP"), row=2, col=2)
                fig.add_trace(go.Scatter(x=data["yp"], y=data["depth"], line=dict(color=c, width=2.5), mode='lines+markers', showlegend=False, hovertemplate="%{x:.1f} lb/100ft2"), row=2, col=3)

                # Fix Y axes
                fig.update_yaxes(autorange="reversed", row=1, col=1)
                fig.update_yaxes(autorange="reversed", row=1, col=2)
                fig.update_yaxes(autorange="reversed", row=2, col=1)
                fig.update_yaxes(autorange="reversed", row=2, col=2)
                fig.update_yaxes(autorange="reversed", row=2, col=3)

            # Draw Dynamic Bar Chart
            fig.add_trace(go.Bar(x=costs_x, y=costs_y, marker_color=bar_colors, text=bar_texts, textposition='auto', showlegend=False, hovertemplate="Cost: $%{y:,.0f}"), row=1, col=3)

            fig.update_layout(
                height=750, 
                hovermode="y unified", 
                margin=dict(t=50, l=20, r=20, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='lightgrey')
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='lightgrey')
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with tab2:
            st.subheader("Multi-Scenario Economics & Dilution Report")
            
            summary_data = {
                "Metric": ["Equipment Selected", "Operating Days", "Avg SRE (%)", "Dilution Vol (bbls)", "Rig Lease ($)", "Mud Cost ($)", "SRE Capex ($)", "TOTAL COST ($)"]
            }
            
            for sc_name, data in sim_res.items():
                eq_str = " + ".join(scenario_configs[sc_name]["equipments"]) if scenario_configs[sc_name]["equipments"] else "None (Bypass)"
                summary_data[sc_name] = [
                    eq_str,
                    f"{data['days']:.1f}", 
                    f"{data['sre_avg']:.1f}%", 
                    f"{data['dilution_vol']:,.0f}", 
                    f"${data['rig_cost']:,.0f}", 
                    f"${data['mud_cost']:,.0f}", 
                    f"${data['equip_invest']:,.0f}", 
                    f"${data['cost']:,.0f}"
                ]
                
            st.table(summary_data)

        with tab3:
            st.subheader("Comprehensive Data Logs")
            
            for sc_name, data in sim_res.items():
                st.markdown(f"**{sc_name} Data**")
                df = pd.DataFrame({
                    "Depth (ft)": data["depth"], "Hole (\")": data["hole"], "Generated LGS (%)": data["lgs"],
                    "Base MW (ppg)": data["base_mw"], "Actual MW (ppg)": data["actual_mw"], 
                    "PV (cP)": data["pv"], "YP (lb/100ft2)": data["yp"], "Fann 600": data["r600"], "Fann 300": data["r300"]
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("Configure your custom scenarios and equipment in the sidebar, then click 'Run Physics Simulation'.")
