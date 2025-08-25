# 2D-material-heterostructure-plotter-
A Python script for generating visualizations (side-view) of 2D material heterostructures, including simulation of bubbles, and alloying.

A versatile Python script for generating high-quality, publication-ready visualizations of 2D material heterostructures. This tool, built with Matplotlib and NumPy, allows for detailed customization of atomic layers, including the simulation of bubbles, various alloy configurations, and layer deflation effects.

![Summary of Heterostructure Plotter Capabilities](https://raw.githubusercontent.com/nikodemsokolowski/2D-material-heterostructure-plotter-/main/fig1_jpeg.jpg)
*A summary of the various heterostructure configurations that can be generated with this script.*

---

## Features

-   **Customizable Stacks:** Easily define complex heterostructures layer-by-layer, specifying material type (`TMD`, `hBN`, `graphene`) and interlayer spacing.
-   **Bubble Simulation:** Introduce realistic bubbles with adjustable height, width, and center position.
-   **Layer Deflation:** Model the mechanical response of encapsulating layers over a bubble, from perfect conformity to gradual or full flattening.
-   **Advanced Alloying:** Create TMD layers with different chalcogen configurations:
    -   **`random`**: Randomly distributed atoms based on a specified ratio.
    -   **`janus`**: Different atomic species on the top and bottom surfaces.
    -   **`regular`**: An ordered, alternating pattern of atoms.
-   **High-Quality Output:** Save figures in PNG and SVG formats with transparent backgrounds, perfect for presentations and publications.
-   **Modular Control:** A simple control panel at the top of the script allows you to switch between generating a full set of documentation figures or single, specific device plots.

---

## Prerequisites

Before running the script, ensure you have the following Python libraries installed:

-   NumPy
-   Matplotlib

You can install them using pip:
```bash
pip install numpy matplotlib
```

---

## Usage

All primary configurations are located in the **Master Configuration Parameters** section at the top of the `heterostructure_plotter.py` script.

### 1. Select Plotting Mode

In the `Plotting Control` section, set **one** of the following boolean variables to `True` to choose your desired output:

-   `plot_documentation_figures = True`: Generates a full set of individual figures showcasing the script's capabilities. These are saved to the `plots/documentation_figures/` subfolder.
-   `plot_bubble_device = True`: Generates and displays a single plot of the example bubble device.
-   `plot_flat_device = True`: Generates and displays a single plot of the example flat device.

### 2. Define Your Heterostructure

You can define your own structures by modifying the `bubble_device_layers` and `flat_device_layers` lists. Each dictionary in the list represents one atomic layer.

**Key Parameters for a Layer:**

-   `'type'`: `'tmd'` or `'hex'`.
-   `'material'`: The chemical formula (e.g., `'WSe2'`, `'hBN'`, `'graphene'`).
-   `'spacing'`: The vertical distance (in angstroms) to the *next* layer.
-   `'deformation'`: (Optional) A dictionary to control bubble effects.
    -   `'h': True`: Enables deformation for this layer.
    -   `'deflate'`: A factor from `1.0` (no deflation) to `0.0` (full flattening) that scales the bubble height.
-   `'alloy_type'`: (For TMDs) `'random'`, `'janus'`, or `'regular'`.
-   `'alloy'`: (For TMDs) A dictionary defining the chalcogen atoms.
    -   For `random`/`regular`: `{'S': 0.5, 'Se': 0.5}`
    -   For `janus`: `{'top': 'S', 'bottom': 'Se'}`

### 3. Customize Output

-   **Filename:** For single plots, you can set the base output filename using the `single_plot_filename` variable.
-   **Physical Parameters:** Adjust atom sizes, bond widths, and the number of atoms (`num_x`) in the `Physical Parameters` section.

---

## Description of Documentation Figures

When `plot_documentation_figures` is enabled, the script will generate and save the following images to the `plots/documentation_figures/` directory:

-   **A\_Device\_with\_Bubble.png**: A complete heterostructure (Graphene/hBN/TMD/TMD/hBN/Graphene) featuring a prominent bubble in the center. The encapsulating layers show a gradual deflation effect.
-   **B\_Flat\_Device.png**: The same heterostructure as Figure A, but rendered as a perfectly flat stack, demonstrating the structure without any deformations.
-   **C\_Alloying\_Configurations**: A series of four plots showing a single TMD layer with different alloy types:
    -   `C1_20_80_Random`: A random alloy with a 20% / 80% composition.
    -   `C2_Janus`: A Janus TMD with different chalcogens on the top and bottom.
    -   `C3_50_50_Random`: A 50/50 random alloy.
    -   `C4_50_50_Regular`: A 50/50 ordered alloy with an alternating pattern.
-   **D\_Bubble\_Geometry\_Variations**: Three plots of a TMD bilayer showcasing bubbles with different aspect ratios:
    -   `D1_Low_Wide_Bubble`: A short and wide bubble.
    -   `D2_Standard_Bubble`: A bubble with standard proportions.
    -   `D3_High_Narrow_Bubble`: A tall and narrow bubble.
-   **E\_Encapsulation\_Layer\_Deflation**: Three plots illustrating the deflation effect of four hBN layers over a TMD bubble:
    -   `E1_No_Deflation`: The encapsulating layers perfectly conform to the shape of the bubble.
    -   `E2_Gradual_Deflation`: The layers progressively flatten with increasing distance from the bubble.
    -   `E3_Full_Deflation`: The top encapsulating layers are completely flat, fully relaxed from the strain induced by the bubble.
