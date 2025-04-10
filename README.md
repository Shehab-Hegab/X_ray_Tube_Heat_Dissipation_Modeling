# X-ray Tube Thermal Simulation 🔬

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with Gradio](https://img.shields.io/badge/Made%20with-Gradio-orange)](https://www.gradio.app/)

An interactive web application to simulate the thermal behavior and heat dissipation dynamics of an X-ray tube anode under various operating conditions. Built with Python using NumPy, Matplotlib, and Gradio.

## Overview

X-ray tubes generate significant heat during operation, primarily at the anode target. Managing this heat is crucial for tube longevity and performance. This simulation tool allows users to:

*   Model the temperature evolution of the anode over time under pulsed exposure sequences.
*   Explore the impact of different anode/housing materials, tube voltage (kVp), current (mA), exposure times, and cooling periods.
*   Visualize the contributions of different heat dissipation mechanisms (conduction, convection, radiation).
*   Estimate whether operating parameters stay within safe thermal limits.

This tool is useful for educational purposes, understanding thermal constraints in X-ray imaging protocols, and exploring the effects of design parameter changes.

## Features

*   **Physics-Based Modeling:** Simulates heat generation based on kVp, mA, and efficiency, and heat loss via conduction, convection (simplified), and radiation (Stefan-Boltzmann law).
*   **Material Selection:** Choose from common X-ray tube materials (Tungsten, Molybdenum, Copper, Aluminum) for both anode and housing, using their respective thermal properties.
*   **Parameter Customization:** Interactively adjust key operating parameters:
    *   Tube Voltage (kVp)
    *   Tube Current (mA)
    *   Exposure Time per Shot
    *   Cooling Time Between Shots
    *   Maximum Safe Anode Temperature
    *   Total Simulation Duration & Time Step
*   **Interactive Web Interface:** User-friendly interface powered by Gradio for easy parameter input and immediate visualization of results.
*   **Comprehensive Visualization:** Generates plots for:
    *   Anode Temperature vs. Time (with safety limit)
    *   Instantaneous Power Balance (Heat Input vs. Heat Loss)
    *   Average Heat Dissipation Breakdown (Pie Chart)
*   **Real-time Feedback:** Provides a summary of results and alerts if the simulated temperature exceeds the defined safe limit.

## Demo Interface

The interactive simulation is accessible via a web interface:

![Gradio Interface](https://github.com/user-attachments/assets/0d68ea0e-f06a-4954-a6b6-fe58c487b341)
*(Screenshot showing the main Gradio interface with input sliders and output areas)*

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Shehab-Hegab/X_ray_Tube_Heat_Dissipation_Modeling.git
    cd X_ray_Tube_Heat_Dissipation_Modeling
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.8 or newer installed. Then, install the required libraries:
    ```bash
    pip install numpy matplotlib gradio
    ```
    *(Alternatively, if a `requirements.txt` file is present: `pip install -r requirements.txt`)*

## Usage

1.  **Run the simulation script:**
    ```bash
    python xray_thermal_sim.py
    ```

2.  **Access the interface:**
    The script will output a local URL (usually `http://127.0.0.1:7860` or similar). Open this URL in your web browser.

3.  **Interact:**
    *   Adjust the sliders and dropdown menus to set the desired simulation parameters.
    *   The simulation will run automatically, and the results (summary text and plots) will update in the output sections.

## Simulation Model Details

*   **Heat Input:** Calculated as `Power = kVp * mA * (1 - Efficiency)`, where efficiency (X-ray production) is assumed to be ~1%. Heat energy added per time step depends on this power and the step duration.
*   **Heat Dissipation:**
    *   **Conduction:** Modeled using a simplified geometric factor and the anode material's thermal conductivity. Represents heat flow away from the focal spot through the anode structure.
    *   **Convection:** Calculated based on the housing surface area, a convection coefficient (representing airflow/oil cooling), and the temperature difference between the (simplified) housing/anode and ambient temperature.
    *   **Radiation:** Modeled using the Stefan-Boltzmann law, considering the anode's surface area, emissivity, and the temperature difference (in K⁴) between the anode and ambient.
*   **Thermal Capacity:** Calculated based on the anode's volume (diameter, thickness), density, and specific heat capacity (`ΔT = Energy / Thermal Capacity`).
*   **Assumptions:** The model uses simplifications, such as uniform anode temperature, fixed geometric factors for conduction, and direct coupling of anode temperature to housing convection/radiation losses. Actual thermal behavior can be more complex.

## Example Outputs

The simulation provides the following visualizations:

1.  **Temperature Profile & Power Balance:** Shows anode temperature evolution and instantaneous heat input/output powers over time.

    ![Temperature and Power Plot](https://github.com/user-attachments/assets/41244277-792e-46c3-a42d-35456eb30a85)
    *(Example plot showing Temperature (°C) and Power (W) vs. Time (s))*

2.  **Average Heat Dissipation Breakdown:** A pie chart illustrating the relative contributions of conduction, convection, and radiation to the *average* heat loss during the simulation period.

    ![Dissipation Pie Chart](https://github.com/user-attachments/assets/c5d965a8-8efa-475c-995e-24aaf48dab0f)
    *(Example pie chart showing the percentage contribution of each heat loss mechanism)*

3.  **Results Summary:** A text box summarizing the input parameters, key results (peak temperature, safety margin), and status alerts.

    ![Summary Text Example](https://github.com/user-attachments/assets/2d801cff-97b5-4e7c-82b7-1dbc5e25ac5f)
    *(Example screenshot of the summary text output)*

*(Additional plots showing different material behaviors or parameter effects, like those provided, can be generated using the interface.)*

## Technology Stack

*   **Python:** Core programming language.
*   **NumPy:** For numerical calculations and array manipulation.
*   **Matplotlib:** For generating plots.
*   **Gradio:** For creating the interactive web interface.

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features (e.g., more complex thermal models, different geometries, specific tube types):

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature` or `bugfix/YourBugfix`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

Please ensure your code adheres to basic Python style guidelines and includes comments where necessary.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you add one, otherwise state it here).

## Author

*   **Shehab Hegab** - [GitHub Profile](https://github.com/Shehab-Hegab)
