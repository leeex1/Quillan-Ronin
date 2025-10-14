#!/usr/bin/env python3
"""
Advanced 3D Modeling & Visualization Tool (visualizer.py)
A professional, general-purpose visualization toolkit for creating high-quality 2D/3D plots and models.
Leverages Matplotlib, Plotly, NetworkX, and PyVista.
NOTE: For 3D modeling, PyVista is used. You may need to install it:
pip install pyvista
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pyvista as pv
import os

class DataVisualizer:
    """
    A versatile and comprehensive visualization class for general data analysis and 3D modeling.
    """
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        pv.set_plot_theme("document")
        print("DataVisualizer initialized. Ready for advanced 2D/3D visualization and modeling.")

    # --- 2D PLOTTING METHODS ---
    def plot_2d_scatter(self, x, y, title="2D Scatter Plot", xlabel="X-axis", ylabel="Y-axis"):
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.7, edgecolors='w', s=50)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    def plot_line(self, x, y, title="Line Plot", xlabel="X-axis", ylabel="Y-axis"):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='b')
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    def plot_histogram(self, data, bins=30, title="Histogram", xlabel="Value", ylabel="Frequency"):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis='y')
        plt.show()
        
    def plot_bar_chart(self, x_data, y_data, title="Bar Chart", xlabel="Category", ylabel="Value"):
        plt.figure(figsize=(10, 6))
        plt.bar(x_data, y_data, color='teal')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title, fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_dataframe(self, df, kind="bar", title="DataFrame Plot"):
        """
        Quick visualization of a DataFrame.
        """
        ax = df.plot(kind=kind, figsize=(10, 6), legend=True)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- NETWORK/GRAPH VISUALIZATION ---
    def plot_network_graph(self, G, layout="spring", node_color='#ff6f69', node_size=450, with_labels=True, title="NetworkX Graph"):
        """
        Visualize a NetworkX graph.
        """
        plt.figure(figsize=(8, 6))
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.random_layout(G)
        nx.draw(G, pos, node_color=node_color, node_size=node_size, with_labels=with_labels, edge_color='gray')
        plt.title(title)
        plt.show()
    
    # --- 3D DATA PLOTTING METHODS ---
    def plot_3d_scatter(self, x, y, z, colors=None, sizes=None, title="3D Scatter Plot"):
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=sizes if sizes is not None else 8, color=colors, colorscale='Viridis', opacity=0.8)
        )])
        fig.update_layout(title=title, scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis'))
        fig.show()

    def plot_3d_surface(self, x, y, z, title="3D SurfQuillan Plot"):
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='cividis')])
        fig.update_layout(title=title, autosize=True, margin=dict(l=65, r=50, b=65, t=90))
        fig.show()

    # --- ADVANCED 3D MODELING & VISUALIZATION (PYVISTA) ---
    def create_3d_scene(self, models, title="3D Scene"):
        """
        Creates and displays a 3D scene with multiple models.
        'models' should be a list of PyVista mesh objects.
        """
        plotter = pv.Plotter(window_size=[1000, 800])
        plotter.set_background('white')
        cmap = ["red", "green", "blue", "orange", "purple", "cyan", "yellow"]
        for i, model in enumerate(models):
            color = cmap[i % len(cmap)]
            plotter.add_mesh(model, color=color, show_edges=True)
        plotter.add_text(title, position='upper_edge', font_size=12)
        plotter.camera_position = 'xy'
        plotter.enable_zoom_scaling()
        print("Showing interactive 3D scene. Close the window to continue.")
        plotter.show()
        
    def load_3d_model(self, file_path):
        """
        Loads a 3D model from a file (e.g., .stl, .obj, .vtk).
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
        try:
            mesh = pv.read(file_path)
            print(f"Successfully loaded model from {file_path}")
            return mesh
        except Exception as e:
            print(f"Failed to load model from {file_path}: {e}")
            return None

    def save_mesh(self, mesh, file_path):
        """
        Save a PyVista mesh to STL or OBJ file.
        """
        try:
            mesh.save(file_path)
            print(f"Mesh saved to {file_path}")
        except Exception as e:
            print(f"Failed to save mesh: {e}")

    def create_sphere(self, center=(0, 0, 0), radius=1.0):
        return pv.Sphere(center=center, radius=radius)
    
    def create_cube(self, center=(0, 0, 0), x_length=1.0, y_length=1.0, z_length=1.0):
        return pv.Cube(center=center, x_length=x_length, y_length=y_length, z_length=z_length)
    
    def create_cylinder(self, center=(0, 0, 0), direction=(0, 0, 1), radius=1.0, height=2.0):
        return pv.Cylinder(center=center, direction=direction, radius=radius, height=height)

    def create_cone(self, center=(0, 0, 0), direction=(0, 0, 1), radius=1.0, height=2.0):
        return pv.Cone(center=center, direction=direction, radius=radius, height=height)
    
    def create_torus(self, center=(0,0,0), ring_radius=2.0, tube_radius=0.5, n_theta=60, n_phi=30):
        """Creates a true torus as a surfQuillan mesh."""
        # Torus parameterization
        theta = np.linspace(0, 2 * np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta, phi = np.meshgrid(theta, phi)
        x = (ring_radius + tube_radius * np.cos(phi)) * np.cos(theta) + center[0]
        y = (ring_radius + tube_radius * np.cos(phi)) * np.sin(theta) + center[1]
        z = tube_radius * np.sin(phi) + center[2]
        torus = pv.StructuredGrid(x, y, z)
        return torus

if __name__ == '__main__':
    print("--- Running Data Visualizer Demonstration ---")
    vis = DataVisualizer()

    # --- Section 1: 2D and 3D Data Plotting ---
    print("\n--- 2D/3D Data Plotting Demonstrations ---")
    # 1. Line Plot (uncomment to display)
    x_line = np.linspace(0, 10, 100)
    y_line = np.sin(x_line) + np.random.normal(0, 0.1, 100)
    # vis.plot_line(x_line, y_line, title="Sine Wave with Noise") # Uncomment to run

    # 2. Histogram (uncomment to display)
    hist_data = np.random.randn(1000)
    # vis.plot_histogram(hist_data, bins=50, title="Distribution of a Normal Dataset") # Uncomment to run

    # 3. 3D Scatter Plot (uncomment to display)
    x3d = np.random.rand(100)
    y3d = np.random.rand(100)
    z3d = np.random.rand(100)
    # vis.plot_3d_scatter(x3d, y3d, z3d, colors=np.random.rand(100), title="Interactive 3D Scatter Plot") # Uncomment to run

    # 4. Quick DataFrame Visualization
    print("\n4. DataFrame visualization example...")
    df = pd.DataFrame({
        'A': np.random.randint(1, 10, 5),
        'B': np.random.randint(1, 10, 5)
    }, index=['X', 'Y', 'Z', 'W', 'V'])
    vis.plot_dataframe(df, kind="bar", title="Bar Plot of Sample DataFrame")

    # 5. Network/Graph Visualization
    print("\n5. Graph/network visualization example...")
    G = nx.erdos_renyi_graph(8, 0.3)
    vis.plot_network_graph(G, title="Random Graph Example")

    # --- Section 2: Advanced 3D Modeling ---
    print("\n--- Advanced 3D Modeling Demonstrations (using PyVista) ---")
    # 6. Primitive shapes and torus
    print("\n6. Generating and displaying primitive 3D shapes + torus...")
    sphere = vis.create_sphere(center=(-3, 0, 0), radius=1)
    cube = vis.create_cube(center=(0, 0, 0))
    cylinder = vis.create_cylinder(center=(3, 0, 0), direction=(0, 1, 0), radius=0.8, height=2.5)
    cone = vis.create_cone(center=(0, -3, 0), direction=(1,0,0))
    torus = vis.create_torus(center=(0,3,0))
    vis.create_3d_scene([sphere, cube, cylinder, cone, torus], title="Primitive Shapes and Torus")
    
    # 7. Save and reload model
    print("\n7. Saving and loading a 3D model example...")
    out_model = "example_cube.stl"
    vis.save_mesh(cube, out_model)
    loaded_cube = vis.load_3d_model(out_model)
    if loaded_cube:
        vis.create_3d_scene([loaded_cube], title="Loaded 3D Model")

    print("\n--- Data Visualizer Demonstration Complete. ---")