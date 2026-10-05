"""Write ELAN annotations from gesture segments."""

import os
import time

import pandas as pd

from envisionhgdetector.state import SegmentColumns

def create_elan_file(
    video_path: str, 
    segments_df: pd.DataFrame, 
    output_path: str, 
    fps: float, 
    include_ground_truth: bool = False
) -> None:
    """
    Create ELAN file from segments DataFrame.
    
    Supports combined model output by creating separate tiers for each model
    when a 'model' column is present (CNN, LightGBM).
    
    Args:
        video_path: Path to the source video file
        segments_df: DataFrame containing segments with columns: start_time, end_time, label
                    Optional 'model' column for combined model (values: 'CNN', 'LightGBM')
        output_path: Path to save the ELAN file
        fps: Video frame rate
        include_ground_truth: Whether to include ground truth tier (not implemented)
    """
    # Check if this is combined model output (has 'model' column)
    has_model_column = 'model' in segments_df.columns
    
    if has_model_column:
        # Get unique models
        models = segments_df['model'].unique().tolist()
    else:
        models = ['PREDICTED']  # Single tier for non-combined models
    
    # Create the basic ELAN file structure
    header = f'''<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="" DATE="{time.strftime('%Y-%m-%d-%H-%M-%S')}" FORMAT="3.0" VERSION="3.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR MEDIA_URL="file://{os.path.abspath(video_path)}"
            MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL=""/>
        <PROPERTY NAME="lastUsedAnnotationId">0</PROPERTY>
    </HEADER>
    <TIME_ORDER>
'''

    # Create time slots for ALL segments (across all tiers)
    time_slots = []
    time_slot_id = 1
    time_slot_refs = {}  # Store references: (start_ms, end_ms) -> (start_slot, end_slot)

    for _, segment in segments_df.iterrows():
        # Convert time to milliseconds
        start_ms = int(segment[SegmentColumns.START_TIME] * 1000)
        end_ms = int(segment[SegmentColumns.END_TIME] * 1000)
        
        # Only create new time slots if we haven't seen these times before
        if start_ms not in time_slot_refs:
            time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{start_ms}"/>')
            time_slot_refs[start_ms] = f"ts{time_slot_id}"
            time_slot_id += 1
        
        if end_ms not in time_slot_refs:
            time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{end_ms}"/>')
            time_slot_refs[end_ms] = f"ts{time_slot_id}"
            time_slot_id += 1

    # Sort time slots by time value for cleaner output
    time_slots_sorted = sorted(time_slots, key=lambda x: int(x.split('TIME_VALUE="')[1].split('"')[0]))
    
    # Add time slots to header
    header += '\n'.join(time_slots_sorted) + '\n    </TIME_ORDER>\n'

    # Create tiers for each model
    annotation_id = 1
    tiers_content = ""
    
    for model in models:
        # Filter segments for this model (or use all if no model column)
        if has_model_column:
            model_segments = segments_df[segments_df['model'] == model]
            tier_id = model  # Use model name as tier ID (e.g., "CNN", "LightGBM")
        else:
            model_segments = segments_df
            tier_id = "PREDICTED"
        
        if model_segments.empty:
            continue
            
        # Start tier
        tiers_content += f'    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="{tier_id}">\n'
        
        annotations = []
        for _, segment in model_segments.iterrows():
            start_ms = int(segment[SegmentColumns.START_TIME] * 1000)
            end_ms = int(segment[SegmentColumns.END_TIME] * 1000)
            start_slot = time_slot_refs[start_ms]
            end_slot = time_slot_refs[end_ms]
            
            annotation = f'''        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{annotation_id}" TIME_SLOT_REF1="{start_slot}" TIME_SLOT_REF2="{end_slot}">
                <ANNOTATION_VALUE>{segment[SegmentColumns.PREDICTION]}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>'''
            
            annotations.append(annotation)
            annotation_id += 1
        
        tiers_content += '\n'.join(annotations) + '\n    </TIER>\n'

    # Combine header and tiers
    header += tiers_content

    # Add linguistic type definitions
    footer = '''    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/>
    <LOCALE LANGUAGE_CODE="en"/>
    <CONSTRAINT DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
    <CONSTRAINT DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations cannot be time-aligned" STEREOTYPE="Symbolic_Subdivision"/>
    <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
    <CONSTRAINT DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
</ANNOTATION_DOCUMENT>'''

    # Write the complete ELAN file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header + footer)
    
    # Print info about tiers created
    if has_model_column:
        print(f"Created ELAN file with {len(models)} tiers: {', '.join(models)}")

