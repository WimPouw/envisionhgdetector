# envisionhgdetector/elan.py
"""
ELAN file generation utilities.
Creates ELAN annotation files (.eaf) from gesture segments.
"""

import os
import time
import pandas as pd


def create_elan_file(
    video_path: str,
    segments_df: pd.DataFrame,
    output_path: str,
    fps: float,
    include_ground_truth: bool = False
) -> None:
    """
    Create ELAN file from segments DataFrame.

    ELAN is a professional tool for annotating video and audio recordings.
    This function generates .eaf files compatible with ELAN for manual
    verification and further annotation.

    Args:
        video_path: Path to the source video file
        segments_df: DataFrame containing segments with columns:
                    - start_time: Segment start time in seconds
                    - end_time: Segment end time in seconds
                    - label: Gesture label
        output_path: Path to save the ELAN file (.eaf extension)
        fps: Video frame rate (used for reference, not for time conversion)
        include_ground_truth: Whether to include ground truth tier (not implemented)

    Note:
        Times are stored in milliseconds in the ELAN format.
    """
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

    # Create time slots
    time_slots = []
    time_slot_id = 1
    time_slot_refs = {}  # Store references for annotations

    for _, segment in segments_df.iterrows():
        # Convert time to milliseconds
        start_ms = int(segment['start_time'] * 1000)
        end_ms = int(segment['end_time'] * 1000)

        # Store start time slot
        time_slots.append(
            f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{start_ms}"/>'
        )
        time_slot_refs[start_ms] = f"ts{time_slot_id}"
        time_slot_id += 1

        # Store end time slot
        time_slots.append(
            f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{end_ms}"/>'
        )
        time_slot_refs[end_ms] = f"ts{time_slot_id}"
        time_slot_id += 1

    # Add time slots to header
    header += '\n'.join(time_slots) + '\n    </TIME_ORDER>\n'

    # Create predicted annotations tier
    annotations = []
    annotation_id = 1

    header += '    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="PREDICTED">\n'

    for _, segment in segments_df.iterrows():
        start_ms = int(segment['start_time'] * 1000)
        end_ms = int(segment['end_time'] * 1000)
        start_slot = time_slot_refs[start_ms]
        end_slot = time_slot_refs[end_ms]

        annotation = f'''        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{annotation_id}" TIME_SLOT_REF1="{start_slot}" TIME_SLOT_REF2="{end_slot}">
                <ANNOTATION_VALUE>{segment['label']}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>'''

        annotations.append(annotation)
        annotation_id += 1

    header += '\n'.join(annotations) + '\n    </TIER>\n'

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
