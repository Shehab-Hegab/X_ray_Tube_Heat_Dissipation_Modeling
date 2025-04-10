# xray_thermal_sim.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import gradio as gr # Make sure gradio is installed: pip install gradio

# Define physical constants
STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann constant (W/m²K⁴)
AMBIENT_TEMP = 22  # Room temperature (°C)
KELVIN_OFFSET = 273.15  # Offset to convert °C to K

class XRayTubeThermalModel:
    """
    Simulation model for X-ray tube thermal dynamics.
    Models heat generation and dissipation through multiple mechanisms.
    """
    def __init__(self,
                 anode_material="tungsten",
                 anode_diameter=0.12,  # m
                 housing_area=0.2,     # m²
                 tube_housing_material="aluminum",
                 max_safe_temp=1200,   # °C
                 anode_thickness=0.01): # m (Added thickness parameter)

        # Tube specifications
        self.anode_diameter = anode_diameter
        self.anode_thickness = anode_thickness # Store thickness
        self.anode_area = np.pi * (anode_diameter/2)**2
        self.housing_area = housing_area

        # Material properties lookup
        self.material_properties = {
            "tungsten": {
                "thermal_conductivity": 173,  # W/m·K
                "specific_heat": 132,         # J/kg·K
                "density": 19300,             # kg/m³
                "emissivity": 0.35
            },
            "aluminum": {
                "thermal_conductivity": 237,  # W/m·K
                "specific_heat": 903,         # J/kg·K
                "density": 2700,              # kg/m³
                "emissivity": 0.09
            },
            "copper": {
                "thermal_conductivity": 401,  # W/m·K
                "specific_heat": 385,         # J/kg·K
                "density": 8960,              # kg/m³
                "emissivity": 0.07
            },
            "molybdenum": {
                "thermal_conductivity": 138,  # W/m·K
                "specific_heat": 251,         # J/kg·K
                "density": 10280,             # kg/m³
                "emissivity": 0.25
            }
        }

        # Apply material properties
        self.anode_material = anode_material.lower() # Ensure lowercase
        if self.anode_material not in self.material_properties:
            raise ValueError(f"Anode Material '{anode_material}' not supported. Available materials: {list(self.material_properties.keys())}")

        self.anode_properties = self.material_properties[self.anode_material]

        self.housing_material = tube_housing_material.lower() # Ensure lowercase
        if self.housing_material not in self.material_properties:
            raise ValueError(f"Housing Material '{tube_housing_material}' not supported. Available materials: {list(self.material_properties.keys())}")

        self.housing_properties = self.material_properties[self.housing_material]

        # Heat transfer coefficients
        # Using housing properties for external convection might be more realistic
        # but keeping original logic for now. Could be adjusted.
        self.convection_coeff = 25  # W/m²·K (forced convection with cooling)

        # Operating conditions
        self.initial_anode_temp = 40  # °C
        self.max_safe_temp = max_safe_temp     # °C

        # Add temperature alert indicator
        self.temperature_alert = False
        self.alert_message = ""

        # Calculate anode thermal mass (moved here as it depends on init params)
        anode_volume = self.anode_area * self.anode_thickness
        self.anode_mass = anode_volume * self.anode_properties["density"]
        self.thermal_capacity = self.anode_mass * self.anode_properties["specific_heat"]
        if self.thermal_capacity <= 0:
             raise ValueError("Calculated thermal capacity is zero or negative. Check anode dimensions and material properties.")


    def calculate_heat_input(self, kvp, ma, time_step):
        """
        Calculate heat power generated in an X-ray tube based on exposure parameters
        during a specific time step.

        Args:
            kvp: Kilovoltage peak (kV)
            ma: Tube current (mA)
            time_step: Duration of the time step (s)

        Returns:
            Heat power generated (W)
        """
        # Standard equation for heat generation in X-ray tubes
        # Typically ~99% of electron energy becomes heat
        efficiency_factor = 0.01  # Only ~1% of energy becomes X-rays
        heat_efficiency = 1 - efficiency_factor

        # Convert kV and mA to V and A
        # Power (W) = V * A
        power = heat_efficiency * (kvp * 1000) * (ma / 1000)
        # Heat energy over timestep (J) = Power * time_step
        # We return POWER (W) here, temperature change depends on energy (Power * dt) / capacity
        return power # Return Power in Watts

    def calculate_conduction_loss(self, temp):
        """Calculate heat loss power due to conduction (W)."""
        # Simplified model: assumes heat conducts away from anode
        # Actual path depends heavily on tube design (e.g., stem, bearings)
        temp_diff = temp - AMBIENT_TEMP # Simplified: conduction to ambient via structure
        if temp_diff <= 0: return 0 # No conduction loss if colder than ambient

        # Using thermal conductivity with a geometric factor
        # This factor is highly empirical and depends on the specific tube design
        # Represents (Area / Length) * complexity_factor
        geometric_factor = 0.01 # Needs calibration for a specific tube type
        # A smaller factor means less effective conduction path
        conduction_power = self.anode_properties["thermal_conductivity"] * geometric_factor * temp_diff
        return conduction_power

    def calculate_convection_loss(self, temp):
        """Calculate heat loss power due to convection from housing (W)."""
        # Assumes housing temp is related to anode temp, simplified here
        # A more complex model would track housing temp separately
        # Using anode temp for calculation here is a strong simplification.
        temp_diff = temp - AMBIENT_TEMP
        if temp_diff <= 0: return 0 # No convection loss if colder than ambient

        # Uses housing area and convection coefficient
        convection_power = self.convection_coeff * self.housing_area * temp_diff
        return convection_power

    def calculate_radiation_loss(self, temp):
        """Calculate heat loss power due to radiation from anode (W)."""
        # Radiation from anode surface itself (e.g., to tube envelope/oil)
        temp_k = temp + KELVIN_OFFSET
        ambient_k = AMBIENT_TEMP + KELVIN_OFFSET # Assuming radiation eventually goes to ambient

        if temp_k <= ambient_k: return 0 # No net radiation loss if colder

        # Stefan-Boltzmann law for anode surface
        radiation_power = (self.anode_properties["emissivity"] * STEFAN_BOLTZMANN *
                           self.anode_area * (temp_k**4 - ambient_k**4))
        return radiation_power

    def simulate(self, kvp, ma, exposure_time, total_time=300, time_step=1.0, cooling_time=3.0, stop_on_overheat=True):
        """
        Simulate X-ray tube temperature over time with discrete exposures.

        Args:
            kvp: Kilovoltage peak (kV)
            ma: Tube current (mA)
            exposure_time: Duration of each exposure (s)
            total_time: Total simulation time (s)
            time_step: Simulation time step (s)
            cooling_time: Duration of cooling period between exposures (s)
            stop_on_overheat: Whether to stop simulation if max temp exceeded

        Returns:
            Dictionary containing simulation results
        """
        # Input validation
        if time_step <= 0: raise ValueError("time_step must be positive.")
        if exposure_time < time_step: print(f"Warning: exposure_time ({exposure_time}s) is less than time_step ({time_step}s). Exposure heat will be averaged over the time step.")
        if cooling_time < 0: raise ValueError("cooling_time cannot be negative.")

        # Reset alert indicator
        self.temperature_alert = False
        self.alert_message = ""

        # Prepare time array
        time_points = np.arange(0, total_time + time_step, time_step)
        num_steps = len(time_points)

        # Initialize arrays
        temperatures = np.zeros(num_steps)
        temperatures[0] = self.initial_anode_temp

        conduction_losses = np.zeros(num_steps) # Power (W)
        convection_losses = np.zeros(num_steps) # Power (W)
        radiation_losses = np.zeros(num_steps)  # Power (W)
        heat_input_power = np.zeros(num_steps)  # Power (W)

        # Simulate exposure patterns
        # Pattern: exposure_time seconds on, cooling_time seconds off
        exposure_pattern = []
        current_sim_time = 0
        is_exposure_phase = True

        while current_sim_time < total_time:
            if is_exposure_phase:
                phase_duration = exposure_time
                pattern_value = 1
            else:
                phase_duration = cooling_time
                pattern_value = 0

            # Determine how many full time steps fit in the current phase
            num_steps_in_phase = int(np.floor(phase_duration / time_step))

            for _ in range(num_steps_in_phase):
                if current_sim_time < total_time:
                    exposure_pattern.append(pattern_value)
                    current_sim_time += time_step
                else:
                    break

            # Handle potential partial time step at the end of the phase
            remaining_phase_time = phase_duration - num_steps_in_phase * time_step
            if remaining_phase_time > 1e-6 and current_sim_time < total_time: # Use tolerance for float comparison
                 # Apply partial exposure/cooling in the next step if needed
                 # For simplicity here, we'll just assign the pattern value for the full next step
                 # A more accurate approach would average the heat input/loss over the partial step
                 exposure_pattern.append(pattern_value)
                 current_sim_time += time_step


            is_exposure_phase = not is_exposure_phase # Switch phase

            # Ensure we don't run over total_time accidentally in loop logic
            if current_sim_time >= total_time:
                break

        # Extend or trim pattern to match time points array size exactly
        pattern_len = len(exposure_pattern)
        if pattern_len < num_steps:
            exposure_pattern.extend([0] * (num_steps - pattern_len)) # Assume cooling if simulation ends
        elif pattern_len > num_steps:
            exposure_pattern = exposure_pattern[:num_steps]


        # Run simulation loop
        time_exceeded = None
        for i in range(1, num_steps):
            current_temp = temperatures[i-1]

            # Check for overheating
            if current_temp >= self.max_safe_temp and not self.temperature_alert:
                self.temperature_alert = True
                time_exceeded = time_points[i-1]
                self.alert_message = f"⚠️ Warning: Anode temperature exceeded safe limit of {self.max_safe_temp}°C at t={time_exceeded:.1f}s!"
                print(self.alert_message) # Also print to console

                if stop_on_overheat:
                    # Fill remaining arrays with the last value for plotting continuity
                    temperatures[i:] = current_temp
                    conduction_losses[i:] = conduction_losses[i-1]
                    convection_losses[i:] = convection_losses[i-1]
                    radiation_losses[i:] = radiation_losses[i-1]
                    heat_input_power[i:] = heat_input_power[i-1]
                    # Trim results to the point where limit was exceeded + 1 step
                    results = {
                        "time": time_points[:i],
                        "temperature": temperatures[:i],
                        "conduction_loss": conduction_losses[:i],
                        "convection_loss": convection_losses[:i],
                        "radiation_loss": radiation_losses[:i],
                        "heat_input_power": heat_input_power[:i],
                        "exposure_pattern": exposure_pattern[:i],
                        "temperature_alert": self.temperature_alert,
                        "alert_message": self.alert_message,
                        "time_exceeded": time_exceeded
                    }
                    # Close any open matplotlib figures before returning
                    plt.close('all')
                    return results


            # Calculate heat loss powers (W) at the beginning of the time step
            conduction_loss_power = self.calculate_conduction_loss(current_temp)
            convection_loss_power = self.calculate_convection_loss(current_temp)
            radiation_loss_power = self.calculate_radiation_loss(current_temp)

            # Record loss powers
            conduction_losses[i] = conduction_loss_power
            convection_losses[i] = convection_loss_power
            radiation_losses[i] = radiation_loss_power

            # Calculate heat input power (W) for this time step
            current_heat_input_power = 0
            if exposure_pattern[i-1] == 1: # Check pattern of previous step to determine input during interval i-1 to i
                # Heat is being generated during exposure
                current_heat_input_power = self.calculate_heat_input(kvp, ma, time_step) # Pass time_step just for consistency, power is instantaneous
            heat_input_power[i] = current_heat_input_power

            # Calculate net power (W) = Power In - Power Out
            total_heat_loss_power = conduction_loss_power + convection_loss_power + radiation_loss_power
            net_power = current_heat_input_power - total_heat_loss_power

            # Calculate temperature change over the time step using Energy = Power * time
            # delta_Temp = Energy / HeatCapacity = (net_power * time_step) / self.thermal_capacity
            if self.thermal_capacity > 0:
                 temp_change = (net_power * time_step) / self.thermal_capacity
            else:
                 temp_change = 0 # Avoid division by zero
                 print("Warning: Thermal capacity is zero, temperature cannot change.")


            # Update temperature for the next step
            temperatures[i] = current_temp + temp_change

            # Ensure temperature doesn't drop below ambient physically (simplification)
            if temperatures[i] < AMBIENT_TEMP:
                temperatures[i] = AMBIENT_TEMP


        # Simulation completed without stopping early
        results = {
            "time": time_points,
            "temperature": temperatures,
            "conduction_loss": conduction_losses, # Power (W)
            "convection_loss": convection_losses, # Power (W)
            "radiation_loss": radiation_losses,   # Power (W)
            "heat_input_power": heat_input_power, # Power (W) during exposure steps
            "exposure_pattern": exposure_pattern,
            "temperature_alert": self.temperature_alert,
            "alert_message": self.alert_message
        }
        if time_exceeded is not None:
             results["time_exceeded"] = time_exceeded

        # Close any open matplotlib figures before returning
        plt.close('all')
        return results

    def plot_results(self, results):
        """
        Create professional visualizations of the simulation results.
        Returns the figure objects instead of showing them.
        """
        # Close previous figures to prevent memory leaks in interactive environments
        plt.close('all')

        # Create figure with subplots
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.4) # Added hspace

        # --- Temperature plot ---
        ax1 = plt.subplot(gs[0])
        ax1.plot(results["time"], results["temperature"], 'r-', linewidth=2, label="Anode Temperature")
        ax1.set_ylabel('Temperature (°C)', fontsize=12)

        # Set title with material information
        title = (f'X-ray Tube Anode Temperature Over Time\n'
                 f'({self.anode_material.capitalize()} Anode, {self.housing_material.capitalize()} Housing)')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(labelbottom=False) # Hide x-axis labels for this subplot

        # Add exposure indicators (visual guide, might get crowded for short steps)
        exposure_indices = np.where(np.array(results["exposure_pattern"]) == 1)[0]
        if len(exposure_indices) > 0 and len(exposure_indices) < 500: # Limit number of lines plotted
            # Find blocks of exposure
            starts = exposure_indices[np.where(np.diff(exposure_indices, prepend=-1) != 1)[0]]
            ends = exposure_indices[np.where(np.diff(exposure_indices, append=-1) != 1)[0]]
            for start_idx, end_idx in zip(starts, ends):
                 if start_idx < len(results["time"]) and end_idx < len(results["time"]):
                    start_time = results["time"][start_idx]
                    # Ensure end_time corresponds to the *end* of the step
                    end_time = results["time"][end_idx] + (results["time"][1]-results["time"][0] if len(results["time"]) > 1 else 0)
                    ax1.axvspan(start_time, end_time, color='gray', alpha=0.15, lw=0, label='_nolegend_')


        # Add maximum safe temperature line
        ax1.axhline(y=self.max_safe_temp, color='orangered', linestyle='--', alpha=0.8,
                    label=f'Max Safe Temp ({self.max_safe_temp}°C)')

        # Add temperature alert annotation if needed
        if results.get("temperature_alert", False) and "time_exceeded" in results:
            alert_time = results["time_exceeded"]
            alert_temp = self.max_safe_temp
            # Adjust text position dynamically
            text_x = alert_time + 0.05 * results["time"][-1]
            text_y = alert_temp * 0.85
            if text_x > results["time"][-1] * 0.8: # If alert is late, move text left
                 text_x = alert_time - 0.3 * results["time"][-1]
            ax1.annotate(f'Limit Exceeded!\n{self.max_safe_temp}°C at t={alert_time:.1f}s',
                         xy=(alert_time, alert_temp),
                         xytext=(text_x, text_y),
                         arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8),
                         bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                         fontsize=10, ha='center')

        ax1.legend(loc='best')
        ax1.set_ylim(bottom=0) # Start y-axis at 0

        # --- Heat loss breakdown ---
        ax2 = plt.subplot(gs[1], sharex=ax1)
        # Ensure loss values are non-negative for plotting
        ax2.plot(results["time"], np.maximum(0, results["conduction_loss"]), 'b-',
                 label='Conduction Loss', linewidth=2)
        ax2.plot(results["time"], np.maximum(0, results["convection_loss"]), 'g-',
                 label='Convection Loss', linewidth=2)
        ax2.plot(results["time"], np.maximum(0, results["radiation_loss"]), 'm-', # Changed color
                 label='Radiation Loss', linewidth=2)
        ax2.set_ylabel('Heat Loss (W)', fontsize=12)
        # ax2.set_title('Heat Dissipation by Mechanism', fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='best')
        ax2.tick_params(labelbottom=False) # Hide x-axis labels for this subplot
        ax2.set_ylim(bottom=0)

        # --- Heat input and total heat balance ---
        ax3 = plt.subplot(gs[2], sharex=ax1)
        total_loss = (results["conduction_loss"] +
                      results["convection_loss"] +
                      results["radiation_loss"])
        # Use step plot for heat input as it's constant during exposure steps
        ax3.step(results["time"], results["heat_input_power"], 'darkorange', where='post',
                 label='Heat Input Power', linewidth=2)
        ax3.plot(results["time"], np.maximum(0, total_loss), 'purple',
                 label='Total Heat Loss Power', linewidth=2)
        ax3.set_xlabel('Time (seconds)', fontsize=12)
        ax3.set_ylabel('Power (W)', fontsize=12)
        # ax3.set_title('Heat Generation vs. Dissipation Power', fontsize=14, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='best')
        ax3.set_ylim(bottom=0)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect to prevent title overlap

        # --- Create a separate figure for pie chart of average heat dissipation ---
        fig2, ax4 = plt.subplots(figsize=(7, 7)) # Smaller pie chart

        # Calculate average heat losses *during the simulation* (excluding initial zero values if any)
        valid_indices = results["temperature"] > self.initial_anode_temp # Consider loss only when heated up
        if not np.any(valid_indices): # If temperature never increased, use last values
             valid_indices = slice(-1, None) # Use the last index

        avg_conduction = np.mean(results["conduction_loss"][valid_indices]) if np.any(results["conduction_loss"][valid_indices]) else 0
        avg_convection = np.mean(results["convection_loss"][valid_indices]) if np.any(results["convection_loss"][valid_indices]) else 0
        avg_radiation = np.mean(results["radiation_loss"][valid_indices]) if np.any(results["radiation_loss"][valid_indices]) else 0

        total_avg_loss = avg_conduction + avg_convection + avg_radiation

        if total_avg_loss > 1e-6: # Avoid division by zero if no loss occurred
            labels = ['Conduction', 'Convection', 'Radiation']
            # Ensure sizes are non-negative
            sizes = [max(0, avg_conduction), max(0, avg_convection), max(0, avg_radiation)]
            colors = ['lightblue', 'lightgreen', 'lightcoral'] # Adjusted colors
            explode = (0.05, 0.05, 0.05) # explode slices slightly

            # Filter out zero-sized slices for cleaner pie chart
            non_zero_sizes = [s for s in sizes if s > 1e-6]
            non_zero_labels = [l for s, l in zip(sizes, labels) if s > 1e-6]
            non_zero_colors = [c for s, c in zip(sizes, colors) if s > 1e-6]
            non_zero_explode = [e for s, e in zip(sizes, explode) if s > 1e-6]

            if non_zero_sizes: # Only plot if there's something to show
                 wedges, texts, autotexts = ax4.pie(non_zero_sizes,
                                                    explode=non_zero_explode,
                                                    labels=non_zero_labels,
                                                    colors=non_zero_colors,
                                                    autopct='%1.1f%%',
                                                    shadow=False, # Cleaner look
                                                    startangle=90,
                                                    pctdistance=0.85) # Position percentage inside wedge
                 # Improve text visibility
                 plt.setp(autotexts, size=10, weight="bold", color="white")
                 plt.setp(texts, size=11)

            ax4.set_title('Average Heat Dissipation Breakdown', fontsize=14, fontweight='bold', pad=20)
            # Equal aspect ratio ensures that pie is drawn as a circle.
            ax4.axis('equal')
        else:
            ax4.text(0.5, 0.5, "No significant heat loss detected\nduring simulation period.",
                     horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
            ax4.set_title('Average Heat Dissipation Breakdown', fontsize=14, fontweight='bold', pad=20)
            ax4.axis('off') # Hide axes if no pie

        plt.tight_layout()

        # IMPORTANT: Return the figures, DO NOT call plt.show() here for Gradio
        return fig, fig2

#--------------------------------------------------------------------------
# Function to be called by Gradio Interface
#--------------------------------------------------------------------------
def simulate_xray_tube_for_gradio(anode_material_name, housing_material_name, voltage, current, exposure_time, max_temp, total_sim_time=200, sim_time_step=0.5, cooling_time=3.0):
    """
    Runs the XRayTubeThermalModel simulation and prepares outputs for Gradio.
    """
    try:
        # Create the model instance
        model = XRayTubeThermalModel(
            anode_material=anode_material_name,
            tube_housing_material=housing_material_name,
            max_safe_temp=max_temp,
            # Can add other params like diameter, thickness as inputs if needed
            anode_diameter=0.12,
            anode_thickness=0.01,
            housing_area=0.2
        )

        # Run the detailed simulation
        results = model.simulate(
            kvp=voltage,
            ma=current,
            exposure_time=exposure_time,
            total_time=total_sim_time,
            time_step=sim_time_step,
            cooling_time=cooling_time,
            stop_on_overheat=True # Stop simulation if limit reached
        )

        # Generate plots using the model's method (returns figures)
        fig_temp_profile, fig_dissipation_pie = model.plot_results(results)

        # Prepare summary text
        max_temp_reached = np.max(results['temperature']) if len(results['temperature']) > 0 else model.initial_anode_temp
        safety_margin = max_temp - max_temp_reached

        # Calculate average powers from results (W)
        avg_heat_in_power = np.mean(results["heat_input_power"][results["heat_input_power"] > 0]) if np.any(results["heat_input_power"] > 0) else 0
        avg_total_loss_power = np.mean(results["conduction_loss"] + results["convection_loss"] + results["radiation_loss"])

        summary = f"""📊 **Simulation Results Summary** 📊

        **Configuration:**
           • **Anode Material:** {model.anode_material.capitalize()}
           • **Housing Material:** {model.housing_material.capitalize()}
           • **Tube Voltage (kVp):** {voltage:.1f}
           • **Tube Current (mA):** {current:.1f}
           • **Exposure Time (s):** {exposure_time:.2f}
           • **Cooling Time (s):** {cooling_time:.2f}
           • **Simulation Duration (s):** {results['time'][-1]:.1f} / {total_sim_time:.1f}
           • **Time Step (s):** {sim_time_step:.2f}

        **Temperature Analysis:**
           • **Max Anode Temp Reached:** {max_temp_reached:.1f}°C
           • **Max Safe Temperature Set:** {max_temp}°C
           • **Temperature Safety Margin:** {safety_margin:.1f}°C

        **Heat Management (Average Power):**
           • **Avg. Heat Input Power (during exposure):** {avg_heat_in_power:.1f} W
           • **Avg. Total Heat Dissipation Power:** {avg_total_loss_power:.1f} W

        **Status:**
        """
        if results.get("temperature_alert", False):
            summary += f"   🚨 {results.get('alert_message', 'Temperature exceeded safe limits!')}"
            if "time_exceeded" in results:
                 summary += f" Simulation stopped early at {results['time_exceeded']:.1f}s."
            else:
                 summary += " Simulation completed but limit was exceeded."
        else:
            summary += f"   ✅ Temperature remained within the safe limit ({max_temp}°C)."


        # Return summary text and the two plot figures
        return summary, fig_temp_profile, fig_dissipation_pie

    except ValueError as ve:
        print(f"Value Error during simulation: {ve}")
        # Return error message and blank plots
        plt.close('all') # Ensure no plots linger
        blank_fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Error:\n{ve}", ha='center', va='center', color='red')
        ax.axis('off')
        return f"Error: {ve}", blank_fig, blank_fig # Return error in text, and blank figs
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to console for debugging
        # Return error message and blank plots
        plt.close('all')
        blank_fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"An unexpected error occurred:\n{e}", ha='center', va='center', color='red')
        ax.axis('off')
        return f"An unexpected error occurred: {e}", blank_fig, blank_fig


#--------------------------------------------------------------------------
# Gradio Interface Definition
#--------------------------------------------------------------------------
def create_gradio_interface():
    """Creates and returns the Gradio interface."""

    available_materials = list(XRayTubeThermalModel().material_properties.keys())

    interface = gr.Interface(
        fn=simulate_xray_tube_for_gradio,
        inputs=[
            gr.Dropdown(choices=[m.capitalize() for m in available_materials], value="Tungsten", label="Anode Material"),
            gr.Dropdown(choices=[m.capitalize() for m in available_materials], value="Aluminum", label="Housing Material"),
            gr.Slider(minimum=40, maximum=150, value=100, step=1, label="Tube Voltage (kVp)"),
            gr.Slider(minimum=10, maximum=800, value=200, step=10, label="Tube Current (mA)"),
            gr.Slider(minimum=0.1, maximum=5.0, value=0.5, step=0.1, label="Exposure Time per Shot (seconds)"),
            gr.Slider(minimum=500, maximum=1800, value=1200, step=50, label="Max Safe Anode Temperature (°C)"),
            gr.Slider(minimum=10, maximum=1000, value=200, step=10, label="Total Simulation Time (seconds)"),
            gr.Slider(minimum=0.05, maximum=5.0, value=0.5, step=0.05, label="Simulation Time Step (seconds)"),
            gr.Slider(minimum=0.1, maximum=60.0, value=3.0, step=0.1, label="Cooling Time Between Shots (seconds)")
        ],
        outputs=[
            gr.Markdown(label="Simulation Results Summary"), # Use Markdown for better formatting
            gr.Plot(label="Temperature Profile & Power Balance"),
            gr.Plot(label="Average Heat Dissipation Breakdown")
        ],
        title="🔬 X-ray Tube Thermal Simulation 🔬",
        description="Simulate the anode temperature of an X-ray tube under repeated exposures. Adjust parameters to see their effect on heating and cooling dynamics.",
        allow_flagging='never',
        examples=[
             ["Tungsten", "Copper", 80, 150, 0.2, 1300, 100, 0.2, 5.0],
             ["Molybdenum", "Aluminum", 120, 300, 1.0, 1100, 300, 0.5, 10.0],
             ["Tungsten", "Tungsten", 140, 50, 2.0, 1500, 60, 1.0, 2.0],
        ]
    )
    return interface

#--------------------------------------------------------------------------
# Main execution block
#--------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Option 1: Run the Gradio Web Interface (Default) ---
    print("Launching Gradio interface...")
    app = create_gradio_interface()
    app.launch()

    # --- Option 2: Run the original command-line interactive simulation ---
    # print("Starting command-line interactive simulation...")
    # # Needs the original run_interactive_simulation function defined
    # # (You would need to copy/paste that function from your original code)
    # # run_interactive_simulation()

    # --- Option 3: Run with predefined parameters (for testing) ---
    # print("Running simulation with predefined parameters...")
    # test_model = XRayTubeThermalModel(
    #     anode_material="tungsten",
    #     tube_housing_material="copper",
    #     max_safe_temp=1200
    # )
    # test_results = test_model.simulate(
    #     kvp=120,      # 120 kV
    #     ma=250,       # 250 mA
    #     exposure_time=0.5,  # 0.5 second exposure
    #     total_time=200,     # 200 seconds of simulation
    #     time_step=0.5,      # 0.5 second time steps
    #     cooling_time=3.0,
    #     stop_on_overheat=False # Let it run even if overheating
    # )
    # print("Simulation finished. Plotting results...")
    # fig1, fig2 = test_model.plot_results(test_results)
    # plt.show() # Need plt.show() here when not using Gradio
    # print("Max temp reached:", np.max(test_results['temperature']))
    # if test_results.get("temperature_alert"):
    #      print(test_results.get("alert_message"))