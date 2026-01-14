# tests/test_elan.py
"""Unit tests for ELAN file generation module."""

import os
import tempfile
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from envisionhgdetector.elan import create_elan_file


class TestCreateElanFile:
    """Tests for create_elan_file function."""

    def test_creates_valid_xml(self):
        """Test that output is valid XML."""
        segments_df = pd.DataFrame({
            'start_time': [1.0, 5.0],
            'end_time': [3.0, 8.0],
            'label': ['Gesture', 'Move']
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.eaf")
            video_path = "/fake/video.mp4"

            create_elan_file(video_path, segments_df, output_path, fps=25.0)

            # Parse as XML to verify validity
            tree = ET.parse(output_path)
            root = tree.getroot()

            assert root.tag == "ANNOTATION_DOCUMENT"

    def test_time_slots_in_milliseconds(self):
        """Test that times are correctly converted to milliseconds."""
        segments_df = pd.DataFrame({
            'start_time': [1.5],  # 1500ms
            'end_time': [2.5],    # 2500ms
            'label': ['Gesture']
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.eaf")
            video_path = "/fake/video.mp4"

            create_elan_file(video_path, segments_df, output_path, fps=25.0)

            tree = ET.parse(output_path)
            root = tree.getroot()

            time_order = root.find("TIME_ORDER")
            time_slots = time_order.findall("TIME_SLOT")

            # Should have 2 time slots (start and end)
            assert len(time_slots) == 2

            # Check values are in milliseconds
            values = [int(ts.get("TIME_VALUE")) for ts in time_slots]
            assert 1500 in values
            assert 2500 in values

    def test_annotations_created(self):
        """Test that annotations are properly created."""
        segments_df = pd.DataFrame({
            'start_time': [1.0, 5.0],
            'end_time': [3.0, 8.0],
            'label': ['Gesture', 'Move']
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.eaf")
            video_path = "/fake/video.mp4"

            create_elan_file(video_path, segments_df, output_path, fps=25.0)

            tree = ET.parse(output_path)
            root = tree.getroot()

            tier = root.find("TIER")
            assert tier is not None
            assert tier.get("TIER_ID") == "PREDICTED"

            annotations = tier.findall("ANNOTATION")
            assert len(annotations) == 2

            # Check annotation values
            values = [
                ann.find("ALIGNABLE_ANNOTATION").find("ANNOTATION_VALUE").text
                for ann in annotations
            ]
            assert "Gesture" in values
            assert "Move" in values

    def test_empty_segments(self):
        """Test handling of empty segments DataFrame."""
        segments_df = pd.DataFrame({
            'start_time': [],
            'end_time': [],
            'label': []
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.eaf")
            video_path = "/fake/video.mp4"

            create_elan_file(video_path, segments_df, output_path, fps=25.0)

            # Should still create valid XML
            tree = ET.parse(output_path)
            root = tree.getroot()
            assert root.tag == "ANNOTATION_DOCUMENT"

    def test_media_descriptor_path(self):
        """Test that video path is correctly embedded."""
        segments_df = pd.DataFrame({
            'start_time': [1.0],
            'end_time': [2.0],
            'label': ['Gesture']
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.eaf")
            video_path = os.path.join(tmpdir, "video.mp4")

            create_elan_file(video_path, segments_df, output_path, fps=25.0)

            tree = ET.parse(output_path)
            root = tree.getroot()

            header = root.find("HEADER")
            media_desc = header.find("MEDIA_DESCRIPTOR")

            assert media_desc is not None
            media_url = media_desc.get("MEDIA_URL")
            assert "video.mp4" in media_url

    def test_linguistic_type_definition(self):
        """Test that required linguistic type is defined."""
        segments_df = pd.DataFrame({
            'start_time': [1.0],
            'end_time': [2.0],
            'label': ['Gesture']
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.eaf")
            video_path = "/fake/video.mp4"

            create_elan_file(video_path, segments_df, output_path, fps=25.0)

            tree = ET.parse(output_path)
            root = tree.getroot()

            ling_type = root.find("LINGUISTIC_TYPE")
            assert ling_type is not None
            assert ling_type.get("LINGUISTIC_TYPE_ID") == "default"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
