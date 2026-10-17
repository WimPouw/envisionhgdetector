css_content = '''
    body, 
    .dash-graph,
    .dash-core-components,
    .dash-html-components { 
        margin: 0; 
        background-color: #111; 
        font-family: sans-serif !important;
        min-height: 100vh;
        width: 100%;
        color: #ffffff;
    }

    /* Modern container styling */
    .dashboard-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
        font-family: sans-serif !important;
    }

    /* Enhanced headings */
    h1, h2, h3, h4, h5, h6 {
        color: rgba(255, 255, 255, 0.95);
        font-weight: 600;
        letter-spacing: -0.02em;
        font-family: sans-serif !important;
    }

    h1 {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #fff, #a8a8a8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255,255,255,0.1);
        font-family: sans-serif !important;
    }

    h2 {
        font-size: 1.5rem;
        margin: 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        font-family: sans-serif !important;
    }

    /* Card-like sections */
    .visualization-section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Grid layout for kinematic features */
    .kinematic-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-right: 120px; /* Space for fixed video */
        grid-auto-rows: minmax(200px, auto); 
        height: 500px; /* Adjust as needed */
    }

    /* Video container styling */
    .video-container {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    /* Interactive elements */
    .interactive-element {
        transition: all 0.2s ease-in-out;
    }

    .interactive-element:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.4);
    }

    /* Loading states */
    .loading {
        opacity: 0.7;
        transition: opacity 0.3s ease;
    }

    /* Tooltip styling */
    .tooltip {
        background: rgba(0, 0, 0, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 0.5rem;
        font-size: 0.875rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        font-family: sans-serif !important;
    }

    /* Force Dash components to use sans-serif */
    .dash-plot-container, 
    .dash-graph-container,
    .js-plotly-plot,
    .plotly {
        font-family: sans-serif !important;
    }
    '''