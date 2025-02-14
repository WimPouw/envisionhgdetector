import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback_context, State
import os
from moviepy import VideoFileClip
import dash
from dash.dependencies import ALL

def convert_videos_for_web(data_folder: str, assets_folder: str) -> None:
    """Convert videos to web-compatible format using MoviePy."""
    converted_folder = os.path.join(assets_folder, 'videos_rerendered')
    os.makedirs(converted_folder, exist_ok=True)

    root_dir = os.path.dirname(data_folder)  
    source_folder = os.path.join(root_dir, 'retracked', 'tracked_videos')

    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    video_files = [f for f in os.listdir(source_folder) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos to convert...")

    for video_file in video_files:
        input_path = os.path.join(source_folder, video_file)
        output_path = os.path.join(converted_folder, video_file)

        if os.path.exists(output_path):
            print(f"Skipping existing video: {video_file}")
            continue

        try:
            print(f"Converting: {video_file}")
            video = VideoFileClip(input_path)
            video.write_videofile(output_path, codec='libx264', audio=False, preset='ultrafast', fps=video.fps)
            video.close()
        except Exception as e:
            print(f"Error converting {video_file}: {e}")

    print("Video conversion complete.")

def create_kinematic_plot(df, feature, selected_gesture=None):
    """Create a violin plot with overlaid box plot and jittered points for a kinematic feature."""
    fig = go.Figure()
    

    # Generate jitter values
    np.random.seed(42)  # For consistent jitter
    jitter_values = [0.2 * (np.random.random() - 0.5) for _ in range(len(df))]
    
    # Convert to numpy arrays for easier masking
    jitter_array = np.array(jitter_values)
    feature_array = df[feature].to_numpy()
    gesture_array = df['gesture_id'].to_numpy()

    if selected_gesture:
        # Non-selected points
        mask_non_selected = gesture_array != selected_gesture
        fig.add_trace(go.Scatter(
            x=jitter_array[mask_non_selected],
            y=feature_array[mask_non_selected],
            mode='markers',
            name='Other Gestures',
            marker=dict(
                color='rgba(200, 200, 200, 0.5)',
                size=6,
                symbol='circle'
            ),
            customdata=gesture_array[mask_non_selected],
            hovertemplate=f"{feature}: %{{y}}<br>Gesture: %{{customdata}}<extra></extra>",
        ))
        
        # Selected points
        mask_selected = gesture_array == selected_gesture
        fig.add_trace(go.Scatter(
            x=jitter_array[mask_selected],
            y=feature_array[mask_selected],
            mode='markers',
            name='Selected Gesture',
            marker=dict(
                color='rgba(255, 0, 0, 1)',
                size=8,
                line=dict(color='white', width=1),
                symbol='circle'
            ),
            customdata=gesture_array[mask_selected],
            hovertemplate=f"{feature}: %{{y}}<br>Gesture: %{{customdata}}<extra></extra>",
        ))
    else:
        # All points without selection
        fig.add_trace(go.Scatter(
            x=jitter_array,
            y=feature_array,
            mode='markers',
            name='All Gestures',
            marker=dict(
                color='rgba(200, 200, 200, 0.7)',
                size=6,
                symbol='circle'
            ),
            customdata=gesture_array,
            hovertemplate=f"{feature}: %{{y}}<br>Gesture: %{{customdata}}<extra></extra>",
        ))
# Add violin plot first (as background)
    fig.add_trace(go.Violin(
        y=df[feature],
        name='Distribution',
        side='both',
        line_color='rgba(100, 100, 100, 0.5)',
        fillcolor='rgba(100, 100, 100, 0.2)',
        meanline_visible=True,
        points=False,
        width=0.8,
        showlegend=False
    ))

    # Add box plot overlay
    fig.add_trace(go.Box(
        y=df[feature],
        name='Statistics',
        boxpoints=False,  # no individual points
        line_color='rgba(255, 255, 255, 0.3)',
        fillcolor='rgba(0, 0, 0, 0)',  # transparent fill
        width=0.2,
        showlegend=False
    ))
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        height=200,
        title=dict(
            text=feature.replace('_', ' ').title(),
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, 0.5],
            fixedrange=True
        ),
        yaxis=dict(
            title=None,
            titlefont=dict(size=10),
            tickfont=dict(size=9),
            gridcolor='rgba(128, 128, 128, 0.1)',
            showgrid=True,
            fixedrange=True
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def run_dashboard_server(data_folder: str, assets_folder: str = "./assets", debug: bool = True) -> None:
    """Run the dashboard server with gesture similarity space and kinematic feature plots."""
    os.makedirs(assets_folder, exist_ok=True)
    
    # Load datasets
    gesture_viz_df = pd.read_csv(os.path.join(data_folder, "gesture_visualization.csv"))
    kinematic_df = pd.read_csv(os.path.join(data_folder, "kinematic_features.csv"))
    
    # Select all kinematic features to display (excluding gesture_id and video_id)
    key_features = [col for col in kinematic_df.columns 
                   if col not in ['gesture_id', 'video_id']]
    
    convert_videos_for_web(data_folder, assets_folder)
    
    app = Dash(__name__, suppress_callback_exceptions=True)
    
    # Define the layout
    layout = html.Div([
        # Styling injection
        dcc.Store(id='theme-store'),
        dcc.Store(id='selected-gesture-store', data=None),
        
        # Main container
        html.Div([
            html.H1("EnvisionHGdetector Kinematic Visualizer"),
            
            # Top section with gesture space and video
            html.Div([
                # Left: Gesture space
                html.Div([
                    html.Div([
                        html.H2("Gesture DTW Similarity Space"),
                        dcc.Graph(
                            id='MY_XY_Map',
                            className='interactive-element',
                            style={'height': '600px', 'width': '80%'},
                        )
                    ], className='visualization-section')
                ], style={'flex': '2', 'marginRight': '2rem'}),
                
                # Right: Video container
                html.Div([
                    html.Div([
                        html.H2("Gesture Video"),
                        html.Video(
                            id='videoplayer',
                            controls=True,
                            autoPlay=True,
                            loop=True,
                            className='video-container',
                            style={'width': '100%', 'height': '300px'}
                        )
                    ], className='visualization-section')
                ], style={
                    'position': 'fixed',
                    'top': '100px',
                    'right': '20px',
                    'width': '380px',
                    'zIndex': '1000'
                })
            ], style={'display': 'flex', 'marginBottom': '2rem'}),
            
            # Feature Selection Section
            html.Div([
                html.H2("Select Kinematic Features"),
                html.Div([
                    # Add selection control buttons
                    html.Div([
                        html.Button(
                            "Select All",
                            id='select-all-button',
                            n_clicks=0,
                            style={
                                'marginRight': '10px',
                                'backgroundColor': '#444444',
                                'color': 'white',
                                'border': 'none',
                                'padding': '5px 10px',
                                'cursor': 'pointer'
                            }
                        ),
                        html.Button(
                            "Deselect All",
                            id='deselect-all-button',
                            n_clicks=0,
                            style={
                                'backgroundColor': '#444444',
                                'color': 'white',
                                'border': 'none',
                                'padding': '5px 10px',
                                'cursor': 'pointer'
                            }
                        ),
                    ], style={'marginBottom': '10px'}),
                    dcc.Checklist(
                        id='feature-checklist',
                        options=[{'label': feature.replace('_', ' ').title(), 'value': feature} 
                                for feature in key_features],
                        value=key_features,  # All features selected by default
                        inline=True,
                        className='feature-checklist',
                        labelStyle={'marginRight': '20px', 'color': 'white'}
                    )
                ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#222222'}),
            ]),
            
            # Kinematic features section
            html.Div([
                html.H2("Kinematic Features"),
                html.Div(id='kinematic-plots-container', className='kinematic-grid')
            ]),
            
            # Debug and output info
            html.Div([
                html.Div(id='debug-info'),
                html.Div(id='output_container')
            ], style={'margin': '1rem 0'})
            
        ], className='dashboard-container')
    ], style={'backgroundColor': '#111111'})
    
    # Set the layout
    app.layout = layout

    @app.callback(
        Output('feature-checklist', 'value'),
        [Input('select-all-button', 'n_clicks'),
         Input('deselect-all-button', 'n_clicks')],
        [State('feature-checklist', 'options'),
         State('feature-checklist', 'value')]
    )
    def update_feature_selection(select_clicks, deselect_clicks, options, current_value):
        ctx = callback_context
        if not ctx.triggered:
            return current_value
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'select-all-button':
            return [option['value'] for option in options]
        elif button_id == 'deselect-all-button':
            return []
        return current_value

    @app.callback(
        Output('selected-gesture-store', 'data'),
        [Input('MY_XY_Map', 'clickData'),
         Input({'type': 'kinematic-plot', 'index': ALL}, 'clickData')]
    )
    def update_selected_gesture(xy_click, kinematic_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update

        trigger_id = ctx.triggered[0]['prop_id']
        
        # Handle scatter plot clicks
        if 'MY_XY_Map' in trigger_id:
            if xy_click:
                return xy_click['points'][0]['customdata'][0]
                
        # Handle kinematic plot clicks
        elif 'kinematic-plot' in trigger_id:
            for click_data in kinematic_clicks:
                if click_data:
                    return click_data['points'][0]['customdata']
                    
        return dash.no_update

    @app.callback(
        Output('kinematic-plots-container', 'children'),
        [Input('feature-checklist', 'value'),
         Input('selected-gesture-store', 'data')]
    )
    def update_kinematic_plots(selected_features, selected_gesture):
        if not selected_features:
            return []
            
        return [
            html.Div([
                dcc.Graph(
                    id={'type': 'kinematic-plot', 'index': i},
                    figure=create_kinematic_plot(kinematic_df, feature, selected_gesture),
                    className='interactive-element',
                    config={'displayModeBar': False}
                )
            ], className='visualization-section')
            for i, feature in enumerate(selected_features)
        ]

    @app.callback(
        [Output('output_container', 'children'),
         Output('MY_XY_Map', 'figure'),
         Output('videoplayer', 'src')],
        [Input('selected-gesture-store', 'data')]
    )
    def update_visualization(selected_gesture):
        # Update XY scatter plot
        fig_xy = px.scatter(gesture_viz_df, x='x', y='y', 
                           hover_data=['gesture'],
                           template='plotly_dark',
                           opacity=0.75)
        fig_xy.update_traces(marker_size=15)
        
        fig_xy.update_layout(
            title=None,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            showlegend=False
        )
        
        if selected_gesture:
            mask = gesture_viz_df['gesture'] == selected_gesture
            fig_xy.add_trace(
                go.Scatter(
                    x=gesture_viz_df[mask]['x'],
                    y=gesture_viz_df[mask]['y'],
                    mode='markers',
                    marker=dict(size=20, color='red'),
                    showlegend=False
                )
            )
        
        # Update video source
        video_src = ''
        if selected_gesture:
            video_path = f'assets/videos_rerendered/{selected_gesture}_tracked.mp4'
            if os.path.exists(video_path):
                video_src = video_path
        
        return (
            f"Selected gesture: {selected_gesture}" if selected_gesture else "Click on any point to inspect.",
            fig_xy,
            video_src
        )

    app.run_server(debug=debug)

if __name__ == '__main__':
    OUTPUT_DIR = './'
    ANALYSIS_DIR = os.path.join(OUTPUT_DIR, 'analysis')
    ASSETS_DIR = './assets'
    debug = True
    
    run_dashboard_server(ANALYSIS_DIR, ASSETS_DIR, debug)