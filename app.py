import cv2
import streamlit as st

from estimators.pose import estimate_pose_video


"""
# Openpose estimation tool
"""
vid_file = st.file_uploader('Select the video you want to process', type=['mp4', 'mkv', 'mov', 'avi'])
mode = st.radio('Mode', options=['COCO', 'MPI'], index=1)

if vid_file:
    with st.spinner('Processing video...'):
        process_bar = st.progress(0)
        info_holder = st.empty()
        frame_holder = st.empty()
        estimate_pose_video(vid_file, mode=mode, st_bar=process_bar, frame_holder=frame_holder, info_holder=info_holder)
