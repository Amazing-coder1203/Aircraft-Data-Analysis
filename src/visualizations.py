import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class AircraftVisualizer:
    def __init__(self, data_path):
        """Initialize with the aircraft dataset"""
        self.df = pd.read_csv(data_path)
        self.clean_data()
    
    def clean_data(self):
        """Basic data cleaning for visualization"""
        # Fill missing values for key columns
        self.df['SHP'] = pd.to_numeric(self.df['SHP'], errors='coerce').fillna(0)
        self.df['THR'] = pd.to_numeric(self.df['THR'], errors='coerce').fillna(0)
        
        # Create a combined power metric
        self.df['Power'] = self.df['SHP'] + self.df['THR']
        
        # Remove rows with missing critical data
        self.df = self.df.dropna(subset=['Engine Type', 'Company'])
    
    def create_engine_type_distribution(self):
        """
        VISUALIZATION 1: Animated Bar Chart - Engine Type Distribution
        This shows how many aircraft use each engine type with growing bars
        """
        # Count aircraft by engine type
        engine_counts = self.df['Engine Type'].value_counts()
        
        # Custom red colorscale - no yellows, visible red at both ends
        custom_red_scale = [
            [0.0, '#ffdddd'],  # Light red (visible red, not white)
            [0.3, '#ff9999'],  # Medium light red
            [0.6, '#ff4444'],  # Medium red
            [0.8, '#cc0000'],  # Dark red
            [1.0, '#990000']   # Very dark red
        ]
        
        # Create animation frames - bars grow from 0 to full height
        frames = []
        steps = 10  # Number of animation steps
        
        for i in range(steps + 1):
            frame_data = []
            progress = i / steps
            
            # Create animated values that grow from 0 to full value
            animated_values = engine_counts.values * progress
            
            frame_data.append(go.Bar(
                x=engine_counts.index,
                y=animated_values,
                marker=dict(
                    color=animated_values,
                    colorscale=custom_red_scale,
                    showscale=False
                ),
                name=f'Frame {i}'
            ))
            
            frames.append(go.Frame(data=frame_data, name=f'Frame {i}'))
        
        # Create the initial figure
        fig = go.Figure(
            data=[go.Bar(
                x=engine_counts.index,
                y=[0] * len(engine_counts),
                marker=dict(
                    color=[0] * len(engine_counts),
                    colorscale=custom_red_scale,
                    showscale=False
                )
            )],
            frames=frames
        )
        
        # Update layout with animation settings
        fig.update_layout(
            title="Aircraft Distribution by Engine Type",
            xaxis_title="Engine Type",
            yaxis_title="Number of Aircraft",
            title_font_size=20,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            showlegend=False,
            plot_bgcolor='#fff5f0',
            paper_bgcolor='#fff5f0',
            font_color='#b30000',
            title_font_color='#b30000'
        )
        
        return fig
    
    def create_power_vs_range_scatter(self):
        """
        VISUALIZATION 2: Animated Scatter Plot - Power vs Range
        Shows relationship between aircraft power and range.
        Now, all points are small and no longer scaled by AUW.
        Points are displayed as small dots for a clean look.
        """
        # Custom red colorscale for scatter points
        custom_red_scale = [
            [0.0, '#ffdddd'],  # Light red
            [0.3, '#ff9999'],  # Medium light red
            [0.6, '#ff4444'],  # Medium red
            [0.8, '#cc0000'],  # Dark red
            [1.0, '#990000']   # Very dark red
        ]
        
        # Remove rows with missing power, range, AUW, engine type, model
        clean_df = self.df.dropna(subset=['Power', 'Range', 'AUW', 'Engine Type', 'Model'])

        # Sort for smooth animation
        clean_df = clean_df.sort_values('Power').reset_index(drop=True)
        total_points = len(clean_df)

        # Calculate axis ranges with padding
        padding_x = (clean_df['Power'].max() - clean_df['Power'].min()) * 0.05
        padding_y = (clean_df['Range'].max() - clean_df['Range'].min()) * 0.05
        x_range = [clean_df['Power'].min() - padding_x, clean_df['Power'].max() + padding_x]
        y_range = [clean_df['Range'].min() - padding_y, clean_df['Range'].max() + padding_y]

        # Prepare color code
        color_codes = clean_df['Engine Type'].astype('category').cat.codes

        # Create animation frames: reveal points in increments
        frames = []
        batch_size = max(1, total_points // 30)
        for i in range(batch_size, total_points + batch_size, batch_size):
            frame_data = clean_df.iloc[:min(i, total_points)]
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=frame_data['Power'],
                    y=frame_data['Range'],
                    mode='markers',
                    marker=dict(
                        size=6,  # Small, fixed size for all points
                        color=color_codes[:min(i, total_points)],
                        colorscale=custom_red_scale,
                        showscale=False,
                        symbol='circle'  # Ensures marker is a point
                    ),
                    text=frame_data['Model'],
                    hovertemplate='<b>%{text}</b><br>Power: %{x} HP<br>Range: %{y} nm<extra></extra>'
                )],
                name=f'Frame {i}'
            ))

        # Initial empty frame
        fig = go.Figure(
            data=[go.Scatter(x=[], y=[], mode='markers')],
            frames=frames
        )

        # Set axis limits and layout for a polished look
        fig.update_layout(
            title="Aircraft Power vs Range Analysis",
            xaxis_title="Power (HP)",
            yaxis_title="Range (nm)",
            xaxis=dict(range=x_range, autorange=False, gridcolor="#eee"),
            yaxis=dict(range=y_range, autorange=False, gridcolor="#eee"),
            plot_bgcolor='#fff5f0',
            paper_bgcolor='#fff5f0',
            font_color='#b30000',
            title_font_color='#b30000',
            title_font_size=20,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [
                        None,
                        {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 60},
                            'mode': 'immediate'
                        }
                    ]
                }]
            }]
        )

        return fig

    def create_top_manufacturers_chart(self):
        """
        VISUALIZATION 3: Animated Horizontal Bar Chart - Top Manufacturers
        Shows which companies make the most aircraft models with growing bars
        """
        # Custom red colorscale
        custom_red_scale = [
            [0.0, '#ffdddd'],  # Light red
            [0.3, '#ff9999'],  # Medium light red
            [0.6, '#ff4444'],  # Medium red
            [0.8, '#cc0000'],  # Dark red
            [1.0, '#990000']   # Very dark red
        ]
        
        # Count aircraft by manufacturer
        manufacturer_counts = self.df['Company'].value_counts().head(15)
        
        # Create animation frames
        frames = []
        steps = 10
        
        for i in range(steps + 1):
            progress = i / steps
            animated_values = manufacturer_counts.values * progress
            
            frames.append(go.Frame(
                data=[go.Bar(
                    x=animated_values,
                    y=manufacturer_counts.index,
                    orientation='h',
                    marker=dict(
                        color=animated_values,
                        colorscale=custom_red_scale,
                        showscale=False
                    )
                )],
                name=f'Frame {i}'
            ))
        
        # Create initial figure
        fig = go.Figure(
            data=[go.Bar(
                x=[0] * len(manufacturer_counts),
                y=manufacturer_counts.index,
                orientation='h',
                marker=dict(
                    color=[0] * len(manufacturer_counts),
                    colorscale=custom_red_scale,
                    showscale=False
                )
            )],
            frames=frames
        )
        
        fig.update_layout(
            title="Top 15 Aircraft Manufacturers by Number of Models",
            xaxis_title="Number of Models",
            yaxis_title="Manufacturer",
            yaxis=dict(autorange="reversed"),
            title_font_size=20,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            height=600,
            plot_bgcolor='#fff5f0',
            paper_bgcolor='#fff5f0',
            font_color='#b30000',
            title_font_color='#b30000'
        )
        
        return fig

    def create_size_vs_weight_scatter(self, axis='Wing Span'):
        """
        VISUALIZATION: Animated Size vs Weight
        Scatter plot with animated points appearing by engine type
        """
        # Custom red colorscale
        custom_red_scale = [
            [0.0, '#ffdddd'],  # Light red
            [0.3, '#ff9999'],  # Medium light red
            [0.6, '#ff4444'],  # Medium red
            [0.8, '#cc0000'],  # Dark red
            [1.0, '#990000']   # Very dark red
        ]
        
        df = self.df.dropna(subset=[axis, 'MEW', 'Engine Type'])
        
        # Sort by engine type for smooth animation
        df = df.sort_values('Engine Type').reset_index(drop=True)
        
        # Create animation frames by engine type
        frames = []
        engine_types = df['Engine Type'].unique()
        
        for i, engine_type in enumerate(engine_types):
            frame_data = df[df['Engine Type'].isin(engine_types[:i+1])]
            
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=frame_data[axis],
                    y=frame_data['MEW'],
                    mode='markers',
                    marker=dict(
                        color=frame_data['Engine Type'].astype('category').cat.codes,
                        colorscale=custom_red_scale,
                        showscale=False
                    ),
                    text=frame_data['Model'],
                    hovertemplate='<b>%{text}</b><br>' + axis + ': %{x} ft<br>MEW: %{y} lbs<extra></extra>'
                )],
                name=f'Frame {i}'
            ))
        
        # Create initial figure
        fig = go.Figure(
            data=[go.Scatter(x=[], y=[], mode='markers')],
            frames=frames
        )
        
        fig.update_layout(
            title=f"{axis} vs. Empty Weight (MEW) by Engine Type",
            xaxis_title=f"{axis} (ft)",
            yaxis_title="Empty Weight (lbs)",
            plot_bgcolor='#fff5f0',
            paper_bgcolor='#fff5f0',
            font_color='#b30000',
            title_font_color='#b30000',
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        
        return fig

    def create_power_vs_climb_scatter(self):
        """
        VISUALIZATION: Animated Power vs Rate of Climb
        Scatter plot with animated points by engine type
        """
        # Custom red colorscale
        custom_red_scale = [
            [0.0, '#ffdddd'],  # Light red
            [0.3, '#ff9999'],  # Medium light red
            [0.6, '#ff4444'],  # Medium red
            [0.8, '#cc0000'],  # Dark red
            [1.0, '#990000']   # Very dark red
        ]
        
        df = self.df.copy()
        # Coerce to numeric and fill missing values
        df['SHP'] = pd.to_numeric(df['SHP'], errors='coerce').fillna(0)
        df['THR'] = pd.to_numeric(df['THR'], errors='coerce').fillna(0)
        df['Power'] = df['SHP'] + df['THR']
        df['ROC'] = pd.to_numeric(df['ROC'], errors='coerce')
        # Drop missing
        df = df.dropna(subset=['Power', 'ROC', 'Engine Type'])
        # Only plot where power and ROC are both positive
        df = df[(df['Power'] > 0) & (df['ROC'] > 0)]
        
        # Sort by power for smooth animation
        df = df.sort_values('Power').reset_index(drop=True)
        
        # Create animation frames
        frames = []
        batch_size = max(1, len(df) // 50)
        
        for i in range(0, len(df), batch_size):
            end_idx = min(i + batch_size, len(df))
            frame_data = df.iloc[:end_idx]
            
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=frame_data['Power'],
                    y=frame_data['ROC'],
                    mode='markers',
                    marker=dict(
                        color=frame_data['Engine Type'].astype('category').cat.codes,
                        colorscale=custom_red_scale,
                        showscale=False
                    ),
                    text=frame_data['Model'],
                    hovertemplate='<b>%{text}</b><br>Power: %{x} HP<br>ROC: %{y} ft/min<extra></extra>'
                )],
                name=f'Frame {i}'
            ))
        
        # Create initial figure
        fig = go.Figure(
            data=[go.Scatter(x=[], y=[], mode='markers')],
            frames=frames
        )
        
        fig.update_layout(
            title="Total Power vs. Rate of Climb by Engine Type",
            xaxis_title="Total Power (SHP + THR)",
            yaxis_title="Rate of Climb (ft/min)",
            plot_bgcolor='#fff5f0',
            paper_bgcolor='#fff5f0',
            font_color='#b30000',
            title_font_color='#b30000',
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        
        return fig

    def create_payload_fraction_vs_range_heatmap(self):
        """
        VISUALIZATION: Payload Fraction vs Range Heatmap
        2D density heatmap with improved red theme
        """
        df = self.df.copy()
        # Convert to numeric and drop missing or zero values
        df['MEW'] = pd.to_numeric(df['MEW'], errors='coerce')
        df['AUW'] = pd.to_numeric(df['AUW'], errors='coerce')
        df['Range'] = pd.to_numeric(df['Range'], errors='coerce')
        df = df.dropna(subset=['MEW', 'AUW', 'Range'])
        df = df[(df['MEW'] > 0) & (df['AUW'] > 0) & (df['Range'] > 0)]
        # Calculate payload fraction sensibly
        df['Payload Fraction'] = 1 - (df['MEW'] / df['AUW'])
        df = df[(df['Payload Fraction'] > 0.15) & (df['Payload Fraction'] < 0.6) & (df['Range'] < 8000)]

        # Improved red color scale - no yellows, visible red at both ends
        fig = px.density_heatmap(
            df,
            x='Payload Fraction',
            y='Range',
            nbinsx=35,
            nbinsy=40,
            marginal_x='histogram',
            marginal_y='histogram',
            color_continuous_scale=['#ffdddd', '#ff9999', '#ff4444', '#cc0000', '#990000'],
            title="Payload Fraction vs. Range (Density Heatmap)",
            labels={'Payload Fraction': 'Payload Fraction (1 - MEW/AUW)', 'Range': 'Range (NM)'}
        )

        # Color the marginal histograms with consistent red
        fig.update_traces(marker_color='#cc0000', selector=dict(type="histogram"), opacity=0.7)

        fig.update_layout(
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#b30000',
            title_font_color='#b30000',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            legend_title_font_color='#b30000',
            xaxis=dict(range=[0.15, 0.6], showgrid=False),
            yaxis=dict(range=[0, 8000], showgrid=False),
            coloraxis_colorbar=dict(title='Aircraft Density')
        )

        return fig

    def create_dimension_histograms(self):
        """
        VISUALIZATION: Dimension Histograms
        Histograms with improved red color scheme
        """
        df = self.df.copy()
        # Ensure numeric and drop NaNs
        df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
        df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
        df['Wing Span'] = pd.to_numeric(df['Wing Span'], errors='coerce')
        
        # Create melted dataframe
        melted_df = df.melt(
            value_vars=['Length', 'Height', 'Wing Span'], 
            var_name='Dimension', 
            value_name='Feet'
        ).dropna()
        
        # Create histogram with improved red color scheme
        fig = px.histogram(
            melted_df,
            x='Feet',
            color='Dimension',
            facet_col='Dimension',
            color_discrete_sequence=['#cc0000', '#ff4444', '#990000'],  # Pure red tones
            nbins=30,
            title="Aircraft Size Distribution: Length, Height, Wing Span"
        )
        
        fig.update_layout(
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#b30000',
            title_font_color='#b30000',
            showlegend=False
        )
        
        fig.for_each_annotation(lambda a: a.update(font=dict(color='#b30000')))
        fig.update_xaxes(title_font=dict(size=14, color='#b30000'))
        fig.update_yaxes(title_font=dict(size=14, color='#b30000'))
        
        return fig

# HOW TO USE THIS CODE:
if __name__ == "__main__":
    # Initialize the visualizer with your data
    viz = AircraftVisualizer('../data/Airplane_Cleaned.csv')
    
    # Create and export all charts
    fig1 = viz.create_engine_type_distribution()
    fig1.write_html("charts/engine_type_distribution.html")

    fig2 = viz.create_power_vs_range_scatter()
    fig2.write_html("charts/power_vs_range_scatter.html")

    fig3 = viz.create_top_manufacturers_chart()
    fig3.write_html("charts/top_manufacturers_chart.html")

    fig_span = viz.create_size_vs_weight_scatter(axis='Wing Span')
    fig_span.write_html("charts/size_vs_weight_wingspan.html")

    fig_length = viz.create_size_vs_weight_scatter(axis='Length')
    fig_length.write_html("charts/size_vs_weight_length.html")

    fig_climb = viz.create_power_vs_climb_scatter()
    fig_climb.write_html("charts/power_vs_climb_scatter.html")

    fig_payload = viz.create_payload_fraction_vs_range_heatmap()
    fig_payload.write_html("charts/payload_fraction_vs_range_heatmap.html")

    fig_dim = viz.create_dimension_histograms()
    fig_dim.write_html("charts/dimension_histograms.html")
    
    print("All visualizations have been created and saved!")
