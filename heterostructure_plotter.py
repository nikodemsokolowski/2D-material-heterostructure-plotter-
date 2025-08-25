import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import to_rgb
import os
import random

# =============================================================================
# --- Master Configuration Parameters ---
# =============================================================================

# --- Plotting Control ---
# Set ONE of these to True to select which plot to generate.
plot_documentation_figures = False
plot_bubble_device = True
plot_flat_device = False
# Specify the base filename for single plots
single_plot_filename = "heterostructure_plot"


# --- General Settings ---
output_directory = "plots_2"
documentation_directory = os.path.join(output_directory, "documentation_figures")
save_png = True
save_svg = True
dpi = 300
face_color = 'none' # Set to 'none' for transparency

# --- Physical Parameters ---
num_x = 40
metal_atom_radius = 0.40
chalc_atom_radius = 0.30
hex_atom_radius = 0.35
bond_linewidth = 2.0

# --- Color Palette ---
colors = {
    'W': "#0074D9", 'Mo': "#00D32A", 'S': '#BB0000', 'Se': "#FFDC00",
    'B': "#1697BE", 'N': "#E00099", 'graphene': "#242424", 'bond': "#757373"
}

# --- Example 1: Heterostructure with a Bubble (with custom spacing) ---
bubble_height = 10
bubble_width = 10
bubble_center_x = 0
bubble_device_params = {'center_x': bubble_center_x, 'height': bubble_height, 'width': bubble_width}

bubble_device_layers = [
    {'type': 'hex', 'material': 'graphene', 'spacing': 3.2},
    {'type': 'hex', 'material': 'hBN', 'spacing': 3.2},
    {'type': 'hex', 'material': 'hBN', 'spacing': 3.2},
    {'type': 'tmd', 'material': 'MoSe2', 'alloy_type': None, 'alloy': {'top': 'S', 'bottom': 'S'}, 'spacing': 3.2},
    {'type': 'tmd', 'material': 'WS2', 'alloy_type': None, 'alloy': {'top': 'Se', 'bottom': 'Se'}, 'deformation': {'h': True, 'deflate': 1.0}, 'spacing': 3.2},
    {'type': 'hex', 'material': 'hBN', 'deformation': {'h': True, 'deflate': 1.0}, 'spacing': 3.2},
    {'type': 'hex', 'material': 'hBN', 'deformation': {'h': True, 'deflate': 1.0}, 'spacing': 3.2},
    {'type': 'hex', 'material': 'graphene', 'deformation': {'h': True, 'deflate': 1.0}, 'spacing': 0},
]

# --- Example 2: Flat Heterostructure (with custom spacing) ---
flat_device_layers = [
    {'material': 'graphene', 'type': 'hex', 'spacing': 3.2},
    {'material': 'hBN', 'type': 'hex', 'spacing': 3.2},
    {'material': 'hBN', 'type': 'hex', 'spacing': 3.2},
    {'material': 'Moe2', 'type': 'tmd', 'alloy_type': None, 'alloy': {'top': 'S', 'bottom': 'S'}, 'spacing': 3.2},
    {'material': 'WS2', 'type': 'tmd', 'alloy_type': None, 'alloy': {'top': 'Se', 'bottom': 'Se'}, 'spacing': 3.2},
    {'material': 'hBN', 'type': 'hex', 'spacing': 3.2},
    {'material': 'hBN', 'type': 'hex', 'spacing': 3.2},
    {'material': 'graphene', 'type': 'hex', 'spacing': 0},
]

# =============================================================================
# --- Core Functions ---
# =============================================================================

def draw_atom_radial_gradient(ax, x, y, radius, color, alpha=1.0, zorder=1,
                              dark_factor=0.5, bright_factor=1.5, power=0.5):
    """Draws an atom with a configurable radial gradient."""
    base_color = np.array(to_rgb(color))
    num_layers = 10
    for i in range(num_layers):
        interp_factor = i / (num_layers - 1)
        bright_color = np.clip(base_color * bright_factor, 0, 1)
        dark_color = base_color * dark_factor
        current_color = dark_color + (bright_color - dark_color) * (interp_factor ** power)
        current_color = np.clip(current_color, 0, 1)
        ax.add_patch(Circle((x, y), radius * (1 - i/num_layers),
                            facecolor=tuple(current_color),
                            alpha=alpha, zorder=zorder+i, lw=0))

def create_tmd_layer(y_offset, num_x, metal_color, alpha, deformation, bubble_params,
                     chalc_config=None, alloy_type=None):
    """Creates a TMD layer with various alloying and deformation options."""
    x_coords = np.linspace(-(num_x - 1) * 1.5 / 2, (num_x - 1) * 1.5 / 2, num_x) + 0.75
    metal_atoms_initial = [{'pos': np.array([x, y_offset]), 'color': metal_color, 'alpha': alpha} for x in x_coords]

    deformed_metal_atoms = []
    for atom in metal_atoms_initial:
        x, y = atom['pos']
        if deformation and deformation.get('h', 0) > 0:
            dist_x = x - bubble_params['center_x']
            deformation_y = deformation['h'] * np.exp(-(dist_x**2) / (2 * bubble_params['width']**2))
            atom['pos'][1] += deformation_y
        deformed_metal_atoms.append(atom)

    deformed_chalc_bottom, deformed_chalc_top = [], []
    for i, atom in enumerate(deformed_metal_atoms):
        p_metal = atom['pos']
        p_left = deformed_metal_atoms[i-1]['pos'] if i > 0 else p_metal + np.array([-1.5, 0])
        midpoint_left = (p_metal + p_left) / 2
        direction_left = p_metal - p_left
        perp_direction_left = np.array([-direction_left[1], direction_left[0]])
        norm = np.linalg.norm(perp_direction_left)
        if norm != 0: perp_direction_left /= norm

        color_top, color_bottom = None, None
        if alloy_type == 'janus':
            color_top, color_bottom = colors[chalc_config['top']], colors[chalc_config['bottom']]
        elif alloy_type == 'random':
            elements, weights = list(chalc_config.keys()), list(chalc_config.values())
            color_top = colors[random.choices(elements, weights=weights, k=1)[0]]
            color_bottom = colors[random.choices(elements, weights=weights, k=1)[0]]
        elif alloy_type == 'regular' and len(chalc_config) == 2:
            el1, el2 = list(chalc_config.keys())
            color_top, color_bottom = (colors[el1], colors[el2]) if i % 2 == 0 else (colors[el2], colors[el1])
        else:
            color_top, color_bottom = colors[chalc_config['top']], colors[chalc_config['bottom']]

        deformed_chalc_bottom.append({'pos': midpoint_left - perp_direction_left * 0.77, 'color': color_bottom, 'alpha': alpha})
        deformed_chalc_top.append({'pos': midpoint_left + perp_direction_left * 0.77, 'color': color_top, 'alpha': alpha})

    all_atoms = deformed_chalc_bottom + deformed_metal_atoms + deformed_chalc_top
    bonds = []
    metal_offset = len(deformed_chalc_bottom)
    for i in range(len(deformed_metal_atoms)):
        metal_idx = metal_offset + i
        bonds.append((metal_idx, i))
        if i < len(deformed_metal_atoms) - 1: bonds.append((metal_idx, i + 1))
        bonds.append((metal_idx, metal_offset + len(deformed_metal_atoms) + i))
        if i < len(deformed_metal_atoms) - 1: bonds.append((metal_idx, metal_offset + len(deformed_metal_atoms) + i + 1))
    return all_atoms, bonds

def create_hex_layer(y_offset, num_x, atom_colors, alpha, deformation, bubble_params):
    """Creates a hexagonal layer, applying deformation if specified."""
    x_coords = np.linspace(-(num_x - 1) * 1.5 / 2, (num_x - 1) * 1.5 / 2, num_x)
    atoms = []
    for i, x in enumerate(x_coords):
        y = y_offset
        if deformation and deformation.get('h', 0) > 0:
            dist_x = x - bubble_params['center_x']
            y += deformation['h'] * np.exp(-(dist_x**2) / (2 * bubble_params['width']**2))
        atoms.append({'pos': [x, y], 'color': atom_colors[i % len(atom_colors)], 'alpha': alpha})
    return atoms, [(i, i + 1) for i in range(num_x - 1)]

def plot_heterostructure(ax, layer_definitions, bubble_params, use_manual_y=True):
    """Main function to generate and save the plot."""
    all_atoms, all_bonds = [], []
    base_idx, current_y = 0, 0
    for i, layer in enumerate(layer_definitions):
        y = layer.get('y', current_y) if use_manual_y else current_y
        if i > 0 and not use_manual_y:
            current_y += layer_definitions[i-1].get('spacing', 3.2)
            y = current_y
        
        deformation = layer.get('deformation', {})
        if 'h' in deformation and deformation.get('h', False):
            deformation['h'] = bubble_params['height'] * deformation.get('deflate', 1.0)
        else:
            deformation['h'] = 0

        if layer['type'] == 'tmd':
            metal = layer['material'][0:2] if len(layer['material']) > 1 and layer['material'][1].islower() else layer['material'][0]
            atoms, bonds = create_tmd_layer(y, num_x, colors[metal], layer.get('alpha', 1.0), deformation, bubble_params,
                                            chalc_config=layer.get('alloy'), alloy_type=layer.get('alloy_type'))
        elif layer['type'] == 'hex':
            atom_colors = [colors['graphene']] if layer['material'] == 'graphene' else [colors['B'], colors['N']]
            atoms, bonds = create_hex_layer(y, num_x, atom_colors, layer.get('alpha', 1.0), deformation, bubble_params)

        all_atoms.extend(atoms)
        all_bonds.extend([(b[0] + base_idx, b[1] + base_idx) for b in bonds])
        base_idx = len(all_atoms)

    for atom in all_atoms:
        x, y_pos = atom['pos']
        radius = hex_atom_radius if atom['color'] in [colors['graphene'], colors['B'], colors['N']] else \
                 (metal_atom_radius if atom['color'] in [colors['W'], colors['Mo']] else chalc_atom_radius)
        draw_atom_radial_gradient(ax, x, y_pos, radius, atom['color'], atom.get('alpha', 1.0), zorder=y_pos)

    for start_idx, end_idx in all_bonds:
        pos1, pos2 = all_atoms[start_idx]['pos'], all_atoms[end_idx]['pos']
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color=colors['bond'], linewidth=bond_linewidth,
                alpha=all_atoms[start_idx].get('alpha', 1.0)*0.9, zorder=min(pos1[1], pos2[1])-0.1)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-num_x * 1.5 / 2 - 2, num_x * 1.5 / 2 + 2)
    if all_atoms:
        ax.set_ylim(min(a['pos'][1] for a in all_atoms) - 2, max(a['pos'][1] for a in all_atoms) + 2)
    ax.axis('off')

# =============================================================================
# --- Documentation Figure Generation ---
# =============================================================================

def generate_documentation_files():
    """Creates and saves a series of individual figures for documentation."""
    if not os.path.exists(documentation_directory):
        os.makedirs(documentation_directory)

    # --- Figure A: Bubble Device ---
    fig, ax = plt.subplots(figsize=(16, 12))
    plot_heterostructure(ax, bubble_device_layers, bubble_device_params, use_manual_y=False)
    plt.savefig(os.path.join(documentation_directory, "A_Device_with_Bubble.png"), dpi=dpi, transparent=True)
    plt.close(fig)

    # --- Figure B: Flat Device ---
    fig, ax = plt.subplots(figsize=(16, 12))
    plot_heterostructure(ax, flat_device_layers, {'height':0, 'center_x':0, 'width':0}, use_manual_y=False)
    plt.savefig(os.path.join(documentation_directory, "B_Flat_Device.png"), dpi=dpi, transparent=True)
    plt.close(fig)

    # --- Figure C: Alloying Configurations ---
    alloy_layers = [
        {'type': 'tmd', 'material': 'WSSe', 'alloy_type': 'random', 'alloy': {'S': 0.2, 'Se': 0.8}},
        {'type': 'tmd', 'material': 'WSSe', 'alloy_type': 'janus', 'alloy': {'top': 'S', 'bottom': 'Se'}},
        {'type': 'tmd', 'material': 'WSSe', 'alloy_type': 'random', 'alloy': {'S': 0.5, 'Se': 0.5}},
        {'type': 'tmd', 'material': 'WSSe', 'alloy_type': 'regular', 'alloy': {'S': 0.5, 'Se': 0.5}},
    ]
    alloy_titles = ["C1_20_80_Random", "C2_Janus", "C3_50_50_Random", "C4_50_50_Regular"]
    for layer, title in zip(alloy_layers, alloy_titles):
        fig, ax = plt.subplots(figsize=(16, 4))
        plot_heterostructure(ax, [layer], {'height':0, 'center_x':0, 'width':0})
        plt.savefig(os.path.join(documentation_directory, f"{title}.png"), dpi=dpi, transparent=True)
        plt.close(fig)

    # --- Figure D: Bubble Geometry Variations ---
    geo_layers = [
        {'type': 'tmd', 'material': 'WSe2', 'y': 0, 'alloy_type': None, 'alloy': {'top': 'Se', 'bottom': 'Se'}},
        {'type': 'tmd', 'material': 'WSe2', 'y': 3.5, 'alloy_type': None, 'alloy': {'top': 'Se', 'bottom': 'Se'}, 'deformation': {'h': True, 'deflate': 1.0}},
    ]
    bubble_geometries = [
        {'center_x': 0, 'height': 4, 'width': 18}, {'center_x': 0, 'height': 8, 'width': 12}, {'center_x': 0, 'height': 12, 'width': 8},
    ]
    geo_titles = ["D1_Low_Wide_Bubble", "D2_Standard_Bubble", "D3_High_Narrow_Bubble"]
    for bubble, title in zip(bubble_geometries, geo_titles):
        fig, ax = plt.subplots(figsize=(16, 6))
        plot_heterostructure(ax, geo_layers, bubble, use_manual_y=True)
        plt.savefig(os.path.join(documentation_directory, f"{title}.png"), dpi=dpi, transparent=True)
        plt.close(fig)

    # --- Figure E: Encapsulation Layer Deflation ---
    bubble_D = {'center_x': 0, 'height': 10, 'width': 12}
    deflate_configs = [
        {'title': "E1_No_Deflation", 'deflations': [1.0, 1.0, 1.0, 1.0], 'y_coords': [0, 3.2, 6.4, 9.6]},
        {'title': "E2_Gradual_Deflation", 'deflations': [1.0, 0.75, 0.5, 0.25], 'y_coords': [0, 4.2, 8.4, 12.6]},
        {'title': "E3_Full_Deflation", 'deflations': [1.0, 0.0, 0.0, 0.0], 'y_coords': [0, 13.2, 16.4, 19.6]},
    ]
    for config in deflate_configs:
        fig, ax = plt.subplots(figsize=(16, 8))
        layers_D = [
            {'type': 'tmd', 'material': 'WS2', 'y': config['y_coords'][0], 'alloy_type': None, 'alloy': {'top': 'Se', 'bottom': 'Se'}, 'deformation': {'h': True, 'deflate': config['deflations'][0]}},
            {'type': 'hex', 'material': 'hBN', 'y': config['y_coords'][1], 'deformation': {'h': True, 'deflate': config['deflations'][1]}},
            {'type': 'hex', 'material': 'hBN', 'y': config['y_coords'][2], 'deformation': {'h': True, 'deflate': config['deflations'][2]}},
            {'type': 'hex', 'material': 'hBN', 'y': config['y_coords'][3], 'deformation': {'h': True, 'deflate': config['deflations'][3]}},
        ]
        plot_heterostructure(ax, layers_D, bubble_D, use_manual_y=True)
        plt.savefig(os.path.join(documentation_directory, f"{config['title']}.png"), dpi=dpi, transparent=True)
        plt.close(fig)

    print(f"Documentation figures successfully saved to '{documentation_directory}'.")


if __name__ == "__main__":
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if plot_documentation_figures:
        generate_documentation_files()
    elif plot_bubble_device:
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_heterostructure(ax, bubble_device_layers, bubble_device_params, use_manual_y=False)
        if save_png:
            plt.savefig(os.path.join(output_directory, f"{single_plot_filename}_bubble.png"), dpi=dpi, transparent=True)
        if save_svg:
            plt.savefig(os.path.join(output_directory, f"{single_plot_filename}_bubble.svg"), transparent=True)
        plt.show()
        print(f"Bubble device figure saved and displayed.")
    elif plot_flat_device:
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_heterostructure(ax, flat_device_layers, {'height':0, 'center_x':0, 'width':0}, use_manual_y=False)
        if save_png:
            plt.savefig(os.path.join(output_directory, f"{single_plot_filename}_flat.png"), dpi=dpi, transparent=True)
        if save_svg:
            plt.savefig(os.path.join(output_directory, f"{single_plot_filename}_flat.svg"), transparent=True)
        plt.show()
        print(f"Flat device figure saved and displayed.")
