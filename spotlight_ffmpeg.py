# #!/usr/bin/env python3
# import json
# import subprocess
# import cv2
# import numpy as np
# from pathlib import Path

# # ── CONFIGURATION ──────────────────────────────────────────────────────
# INPUT_VIDEO = r"D:\LLMS\PROJECT_2\modified_zoom\zoom_sample.mp4"
# SEGMENTS_FILE = "updated_segments.json"
# OUTPUT_VIDEO = r"D:\LLMS\PROJECT_2\modified_zoom\Zoom_centered.mp4"
# GRID_ROWS, GRID_COLS = 2, 2  # Zoom grid layout

# # Face detection parameters
# FACE_CONFIDENCE_THRESHOLD = 0.5
# FACE_SIZE_THRESHOLD = 0.1  # min % of frame size

# def load_segments(path):
#     with open(path, "r") as f:
#         return json.load(f)

# def detect_faces(frame):
#     net = cv2.dnn.readNetFromCaffe(
#         "deploy.prototxt",
#         "res10_300x300_ssd_iter_140000.caffemodel"
#     )
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                 (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
    
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > FACE_CONFIDENCE_THRESHOLD:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             width = endX - startX
#             height = endY - startY
            
#             # Filter small faces
#             if width > w * FACE_SIZE_THRESHOLD and height > h * FACE_SIZE_THRESHOLD:
#                 centerX = (startX + endX) // 2
#                 centerY = (startY + endY) // 2
#                 faces.append((startX, startY, endX, endY, centerX, centerY))
#     return faces

# def assign_speaker_to_cell(face_positions, width, height):
#     cell_w = width // GRID_COLS
#     cell_h = height // GRID_ROWS
    
#     speaker_map = {}
#     for (sx, sy, ex, ey, cx, cy) in face_positions:
#         col = min(cx // cell_w, GRID_COLS - 1)
#         row = min(cy // cell_h, GRID_ROWS - 1)
#         cell_index = row * GRID_COLS + col
        
#         # Use largest face in cell
#         face_size = (ex - sx) * (ey - sy)
#         if cell_index not in speaker_map:
#             speaker_map[cell_index] = (face_size, cx, cy)
#         elif face_size > speaker_map[cell_index][0]:
#             speaker_map[cell_index] = (face_size, cx, cy)
            
#     # Return only cell indices with faces
#     return {cell: idx for idx, cell in enumerate(speaker_map.keys())}

# def get_speaker_mapping(segments, width, height):
#     cap = cv2.VideoCapture(INPUT_VIDEO)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open video")
    
#     # Group segments by speaker
#     speaker_segments = {}
#     for start, end, spk in segments:
#         if spk not in speaker_segments:
#             speaker_segments[spk] = []
#         speaker_segments[spk].append((start, end))
    
#     # For each speaker, use their first segment
#     position_map = {}
#     for spk, segs in speaker_segments.items():
#         start, end = segs[0]  # First segment for this speaker
#         cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
#         ret, frame = cap.read()
        
#         if not ret:
#             continue
            
#         # Detect faces and use the largest one
#         faces = detect_faces(frame)
#         if faces:
#             # Sort by face size (largest first)
#             faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
#             position_map[spk] = faces[0]  # Use largest face
    
#     cap.release()
    
#     if not position_map:
#         return {}
        
#     return assign_speaker_to_cell(
#         position_map.values(),
#         width,
#         height
#     )

# def build_filter_complex(segments, speaker_map, width, height):
#     cell_w = width // GRID_COLS
#     cell_h = height // GRID_ROWS
#     filters, labels = [], []
#     segment_count = 0

#     for i, (start, end, spk) in enumerate(segments):
#         if spk not in speaker_map:
#             continue
            
#         cell = speaker_map[spk]
#         col, row = divmod(cell, GRID_COLS)
#         x, y = col * cell_w, row * cell_h

#         vtag = f"[v{segment_count}]"
#         filters.append(
#             f"[0:v]trim=start={start:.3f}:end={end:.3f},"
#             f"setpts=PTS-STARTPTS,"
#             f"crop={cell_w}:{cell_h}:{x}:{y},"
#             f"scale={width}:{height}:flags=lanczos,"
#             f"setsar=1{vtag}"
#         )
#         labels.append(vtag)

#         atag = f"[a{segment_count}]"
#         filters.append(
#             f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
#             f"asetpts=PTS-STARTPTS{atag}"
#         )
#         labels.append(atag)
#         segment_count += 1

#     concat_inputs = "".join(labels)
#     filters.append(
#         f"{concat_inputs}concat=n={segment_count}:v=1:a=1[outv][outa]"
#     )
#     return ";".join(filters), segment_count

# def main():
#     segments = load_segments(SEGMENTS_FILE)
    
#     cap = cv2.VideoCapture(INPUT_VIDEO)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()
    
#     # Automatically generate speaker mapping
#     speaker_map = get_speaker_mapping(segments, width, height)
#     print(f"Generated speaker mapping: {speaker_map}")
    
#     if not speaker_map:
#         print("⚠️ Warning: No speaker mapping generated. Using manual fallback.")
#         # Manual fallback mapping (adjust as needed)
#         speaker_map = {0: 0, 1: 1, 2: 2}

#     fc, segment_count = build_filter_complex(segments, speaker_map, width, height)
#     print(f"Processing {segment_count}/{len(segments)} segments")
    
#     cmd = [
#         "ffmpeg", "-y", "-i", INPUT_VIDEO,
#         "-filter_complex", fc,
#         "-map", "[outv]", "-map", "[outa]",
#         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
#         "-c:a", "aac", "-b:a", "192k",
#         "-vsync", "vfr",
#         OUTPUT_VIDEO
#     ]

#     print("Running FFmpeg command...")
#     result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    
#     if result.returncode != 0:
#         print("❌ FFmpeg failed with error:")
#         print(result.stderr)
#     else:
#         print(f"✔ Output saved to {OUTPUT_VIDEO}")

# if __name__ == "__main__":
#     main()


























# #!/usr/bin/env python3
# import json
# import subprocess
# import cv2
# import numpy as np
# from pathlib import Path

# # ── CONFIGURATION ──────────────────────────────────────────────────────
# INPUT_VIDEO = r"D:\LLMS\PROJECT_2\modified_zoom\zoom_sample.mp4"
# SEGMENTS_FILE = "updated_segments.json"
# OUTPUT_VIDEO = r"D:\LLMS\PROJECT_2\modified_zoom\Zoom_centered.mp4"
# GRID_ROWS, GRID_COLS = 2, 2  # Zoom grid layout

# # Face detection parameters
# FACE_CONFIDENCE_THRESHOLD = 0.5
# FACE_SIZE_THRESHOLD = 0.1  # min % of frame size

# def load_segments(path):
#     with open(path, "r") as f:
#         return json.load(f)

# def detect_faces(frame):
#     net = cv2.dnn.readNetFromCaffe(
#         "deploy.prototxt",
#         "res10_300x300_ssd_iter_140000.caffemodel"
#     )
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                 (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
    
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > FACE_CONFIDENCE_THRESHOLD:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             width = endX - startX
#             height = endY - startY
            
#             # Filter small faces
#             if width > w * FACE_SIZE_THRESHOLD and height > h * FACE_SIZE_THRESHOLD:
#                 centerX = (startX + endX) // 2
#                 centerY = (startY + endY) // 2
#                 faces.append((startX, startY, endX, endY, centerX, centerY))
#     return faces

# def get_speaker_mapping(segments, width, height):
#     cap = cv2.VideoCapture(INPUT_VIDEO)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open video")
    
#     # Group segments by speaker
#     speaker_segments = {}
#     for start, end, spk in segments:
#         if spk not in speaker_segments:
#             speaker_segments[spk] = []
#         speaker_segments[spk].append((start, end))
    
#     # For each speaker, use their first segment to detect faces and map to cells
#     speaker_cell_map = {}
#     for spk, segs in speaker_segments.items():
#         start, end = segs[0]  # First segment for this speaker
#         cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
#         ret, frame = cap.read()
        
#         if not ret:
#             continue
            
#         # Detect faces
#         faces = detect_faces(frame)
#         if faces:
#             # Sort by face size (largest first)
#             faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
#             sx, sy, ex, ey, cx, cy = faces[0]  # Use largest face
            
#             # Calculate cell index
#             cell_w = width // GRID_COLS
#             cell_h = height // GRID_ROWS
#             col = min(cx // cell_w, GRID_COLS - 1)
#             row = min(cy // cell_h, GRID_ROWS - 1)
#             cell_index = row * GRID_COLS + col
            
#             # Map speaker to cell
#             speaker_cell_map[spk] = cell_index
    
#     cap.release()
#     return speaker_cell_map

# def build_filter_complex(segments, speaker_map, width, height):
#     cell_w = width // GRID_COLS
#     cell_h = height // GRID_ROWS
#     filters, labels = [], []
#     segment_count = 0

#     for i, (start, end, spk) in enumerate(segments):
#         if spk not in speaker_map:
#             continue
            
#         cell = speaker_map[spk]
#         col, row = divmod(cell, GRID_COLS)
#         x, y = col * cell_w, row * cell_h

#         vtag = f"[v{segment_count}]"
#         filters.append(
#             f"[0:v]trim=start={start:.3f}:end={end:.3f},"
#             f"setpts=PTS-STARTPTS,"
#             f"crop={cell_w}:{cell_h}:{x}:{y},"
#             f"scale={width}:{height}:flags=lanczos,"
#             f"setsar=1{vtag}"
#         )
#         labels.append(vtag)

#         atag = f"[a{segment_count}]"
#         filters.append(
#             f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
#             f"asetpts=PTS-STARTPTS{atag}"
#         )
#         labels.append(atag)
#         segment_count += 1

#     concat_inputs = "".join(labels)
#     filters.append(
#         f"{concat_inputs}concat=n={segment_count}:v=1:a=1[outv][outa]"
#     )
#     return ";".join(filters), segment_count

# def main():
#     segments = load_segments(SEGMENTS_FILE)
    
#     cap = cv2.VideoCapture(INPUT_VIDEO)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()
    
#     # Automatically generate speaker mapping
#     speaker_map = get_speaker_mapping(segments, width, height)
#     print(f"Generated speaker mapping: {speaker_map}")
    
#     if not speaker_map:
#         print("⚠️ Warning: No speaker mapping generated. Using manual fallback.")
#         # Manual fallback mapping (adjust as needed)
#         speaker_map = {0: 0, 1: 1, 2: 2}

#     fc, segment_count = build_filter_complex(segments, speaker_map, width, height)
#     print(f"Processing {segment_count}/{len(segments)} segments")
    
#     cmd = [
#         "ffmpeg", "-y", "-i", INPUT_VIDEO,
#         "-filter_complex", fc,
#         "-map", "[outv]", "-map", "[outa]",
#         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
#         "-c:a", "aac", "-b:a", "192k",
#         "-vsync", "vfr",
#         OUTPUT_VIDEO
#     ]

#     print("Running FFmpeg command...")
#     result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    
#     if result.returncode != 0:
#         print("❌ FFmpeg failed with error:")
#         print(result.stderr)
#     else:
#         print(f"✔ Output saved to {OUTPUT_VIDEO}")

# if __name__ == "__main__":
#     main()









# #!/usr/bin/env python3
# import json
# import subprocess
# import cv2
# import numpy as np
# from pathlib import Path
# import os

# # ── CONFIGURATION ──────────────────────────────────────────────────────
# INPUT_VIDEO = r"D:\LLMS\PROJECT_2\modified_zoom\zoom_sample.mp4"
# SEGMENTS_FILE = "updated_segments.json"
# OUTPUT_VIDEO = r"D:\LLMS\PROJECT_2\modified_zoom\Zoom_centered.mp4"
# GRID_ROWS, GRID_COLS = 2, 2  # Zoom grid layout
# DEBUG_DIR = "debug_frames"  # Directory to save debug frames

# # Face detection parameters
# FACE_CONFIDENCE_THRESHOLD = 0.4  # Lowered to be more sensitive
# FACE_SIZE_THRESHOLD = 0.05  # Lowered to detect smaller faces

# def load_segments(path):
#     with open(path, "r") as f:
#         return json.load(f)

# def detect_faces(frame):
#     net = cv2.dnn.readNetFromCaffe(
#         "deploy.prototxt",
#         "res10_300x300_ssd_iter_140000.caffemodel"
#     )
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                 (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
    
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > FACE_CONFIDENCE_THRESHOLD:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             width = endX - startX
#             height = endY - startY
            
#             # Filter small faces
#             if width > w * FACE_SIZE_THRESHOLD and height > h * FACE_SIZE_THRESHOLD:
#                 centerX = (startX + endX) // 2
#                 centerY = (startY + endY) // 2
#                 faces.append((startX, startY, endX, endY, centerX, centerY))
#     return faces

# def save_debug_frame(frame, speaker_id, segment_idx, faces, output_dir=DEBUG_DIR):
#     """Save frame with bounding boxes for detected faces for debugging."""
#     os.makedirs(output_dir, exist_ok=True)
#     frame_copy = frame.copy()
#     for (startX, startY, endX, endY, _, _) in faces:
#         cv2.rectangle(frame_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)
#     filename = f"{output_dir}/spk_{speaker_id}_seg_{segment_idx}.jpg"
#     cv2.imwrite(filename, frame_copy)
#     print(f"Saved debug frame: {filename}")

# def get_speaker_mapping(segments, width, height):
#     cap = cv2.VideoCapture(INPUT_VIDEO)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open video")
    
#     # Group segments by speaker
#     speaker_segments = {}
#     for start, end, spk in segments:
#         if spk not in speaker_segments:
#             speaker_segments[spk] = []
#         speaker_segments[spk].append((start, end))
    
#     # For each speaker, try multiple segments to detect faces
#     speaker_cell_map = {}
#     for spk, segs in speaker_segments.items():
#         print(f"Processing speaker {spk} with {len(segs)} segments")
#         for idx, (start, end) in enumerate(segs):
#             cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
#             ret, frame = cap.read()
            
#             if not ret:
#                 print(f"Failed to read frame for speaker {spk} at segment {idx} (start: {start}s)")
#                 continue
                
#             # Detect faces
#             faces = detect_faces(frame)
#             print(f"Speaker {spk}, segment {idx}: Detected {len(faces)} faces")
            
#             # Save debug frame
#             save_debug_frame(frame, spk, idx, faces)
            
#             if faces:
#                 # Sort by face size (largest first)
#                 faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
#                 sx, sy, ex, ey, cx, cy = faces[0]  # Use largest face
                
#                 # Calculate cell index
#                 cell_w = width // GRID_COLS
#                 cell_h = height // GRID_ROWS
#                 col = min(cx // cell_w, GRID_COLS - 1)
#                 row = min(cy // cell_h, GRID_ROWS - 1)
#                 cell_index = row * GRID_COLS + col
                
#                 # Map speaker to cell
#                 speaker_cell_map[spk] = cell_index
#                 print(f"Mapped speaker {spk} to cell {cell_index} (row: {row}, col: {col})")
#                 break  # Stop after finding a face
#             else:
#                 print(f"No faces detected for speaker {spk} in segment {idx}")
    
#     cap.release()
#     return speaker_cell_map

# def build_filter_complex(segments, speaker_map, width, height):
#     cell_w = width // GRID_COLS
#     cell_h = height // GRID_ROWS
#     filters, labels = [], []
#     segment_count = 0

#     for i, (start, end, spk) in enumerate(segments):
#         if spk not in speaker_map:
#             print(f"Skipping segment {i} for speaker {spk}: not in speaker_map")
#             continue
            
#         cell = speaker_map[spk]
#         col, row = divmod(cell, GRID_COLS)
#         x, y = col * cell_w, row * cell_h

#         vtag = f"[v{segment_count}]"
#         filters.append(
#             f"[0:v]trim=start={start:.3f}:end={end:.3f},"
#             f"setpts=PTS-STARTPTS,"
#             f"crop={cell_w}:{cell_h}:{x}:{y},"
#             f"scale={width}:{height}:flags=lanczos,"
#             f"setsar=1{vtag}"
#         )
#         labels.append(vtag)

#         atag = f"[a{segment_count}]"
#         filters.append(
#             f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
#             f"asetpts=PTS-STARTPTS{atag}"
#         )
#         labels.append(atag)
#         segment_count += 1

#     concat_inputs = "".join(labels)
#     filters.append(
#         f"{concat_inputs}concat=n={segment_count}:v=1:a=1[outv][outa]"
#     )
#     return ";".join(filters), segment_count

# def main():
#     segments = load_segments(SEGMENTS_FILE)
    
#     cap = cv2.VideoCapture(INPUT_VIDEO)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()
    
#     # Automatically generate speaker mapping
#     speaker_map = get_speaker_mapping(segments, width, height)
#     print(f"Generated speaker mapping: {speaker_map}")
    
#     if not speaker_map:
#         print("⚠️ Warning: No speaker mapping generated. Using manual fallback.")
#         # Manual fallback mapping (adjust as needed)
#         speaker_map = {0: 0, 1: 1, 2: 2}

#     fc, segment_count = build_filter_complex(segments, speaker_map, width, height)
#     print(f"Processing {segment_count}/{len(segments)} segments")
    
#     cmd = [
#         "ffmpeg", "-y", "-i", INPUT_VIDEO,
#         "-filter_complex", fc,
#         "-map", "[outv]", "-map", "[outa]",
#         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
#         "-c:a", "aac", "-b:a", "192k",
#         "-vsync", "vfr",
#         OUTPUT_VIDEO
#     ]

#     print("Running FFmpeg command...")
#     result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    
#     if result.returncode != 0:
#         print("❌ FFmpeg failed with error:")
#         print(result.stderr)
#     else:
#         print(f"✔ Output saved to {OUTPUT_VIDEO}")

# if __name__ == "__main__":
#     main()







# #!/usr/bin/env python3
# # spotlight_ffmpeg.py
# import os
# import json
# import subprocess
# import argparse
# import cv2
# import numpy as np

# # ── CONFIGURATION ──────────────────────────────────────────────────────
# GRID_ROWS, GRID_COLS = 2, 2  # Zoom grid layout
# FACE_CONFIDENCE_THRESHOLD = 0.5
# FACE_SIZE_THRESHOLD = 0.1  # min % of frame size


# def load_segments(path):
#     with open(path, "r") as f:
#         return json.load(f)


# def detect_faces(frame):
#     net = cv2.dnn.readNetFromCaffe(
#         "deploy.prototxt",
#         "res10_300x300_ssd_iter_140000.caffemodel"
#     )
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(
#         cv2.resize(frame, (300, 300)), 1.0,
#         (300, 300), (104.0, 177.0, 123.0)
#     )
#     net.setInput(blob)
#     detections = net.forward()
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > FACE_CONFIDENCE_THRESHOLD:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (sx, sy, ex, ey) = box.astype("int")
#             width = ex - sx
#             height = ey - sy
#             if width > w * FACE_SIZE_THRESHOLD and height > h * FACE_SIZE_THRESHOLD:
#                 centerX = (sx + ex) // 2
#                 centerY = (sy + ey) // 2
#                 faces.append((sx, sy, ex, ey, centerX, centerY))
#     return faces


# def assign_speaker_to_cell(face_positions, width, height):
#     cell_w = width // GRID_COLS
#     cell_h = height // GRID_ROWS
#     speaker_map = {}
#     for (sx, sy, ex, ey, cx, cy) in face_positions:
#         col = min(cx // cell_w, GRID_COLS - 1)
#         row = min(cy // cell_h, GRID_ROWS - 1)
#         cell_index = row * GRID_COLS + col
#         size = (ex - sx) * (ey - sy)
#         if cell_index not in speaker_map or size > speaker_map[cell_index][0]:
#             speaker_map[cell_index] = (size, cx, cy)
#     return {cell: idx for idx, cell in enumerate(speaker_map.keys())}


# def get_speaker_mapping(segments, width, height, input_video):
#     cap = cv2.VideoCapture(input_video)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open video")
#     speaker_segments = {}
#     for start, end, spk in segments:
#         speaker_segments.setdefault(spk, []).append((start, end))
#     position_map = {}
#     for spk, segs in speaker_segments.items():
#         start, _ = segs[0]
#         cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         faces = detect_faces(frame)
#         if faces:
#             faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
#             position_map[spk] = faces[0]
#     cap.release()
#     if not position_map:
#         return {}
#     return assign_speaker_to_cell(position_map.values(), width, height)


# def build_filter_complex(segments, speaker_map, width, height):
#     cell_w = width // GRID_COLS
#     cell_h = height // GRID_ROWS
#     filters, labels = [], []
#     count = 0
#     for start, end, spk in segments:
#         if spk not in speaker_map:
#             continue
#         cell = speaker_map[spk]
#         col, row = divmod(cell, GRID_COLS)
#         x, y = col * cell_w, row * cell_h
#         vtag = f"[v{count}]"
#         filters.append(
#             f"[0:v]trim=start={start:.3f}:end={end:.3f},"
#             f"setpts=PTS-STARTPTS," \
#             f"crop={cell_w}:{cell_h}:{x}:{y},"
#             f"scale={width}:{height}:flags=lanczos,"
#             f"setsar=1{vtag}"
#         )
#         labels.append(vtag)
#         atag = f"[a{count}]"
#         filters.append(
#             f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
#             f"asetpts=PTS-STARTPTS{atag}"
#         )
#         labels.append(atag)
#         count += 1
#     concat_in = "".join(labels)
#     filters.append(f"{concat_in}concat=n={count}:v=1:a=1[outv][outa]")
#     return ";".join(filters), count


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", "-i", default="zoom_sample.mp4", help="Source video path")
#     parser.add_argument("--segments", "-s", default="updated_segments.json", help="JSON segments file")
#     parser.add_argument("--output", "-o", default="Zoom_centered.mp4", help="Output video path")
#     args = parser.parse_args()

#     segments = load_segments(args.segments)
#     cap = cv2.VideoCapture(args.input)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     speaker_map = get_speaker_mapping(segments, width, height, args.input)
#     print(f"Generated speaker mapping: {speaker_map}")
#     if not speaker_map:
#         print("⚠️ No mapping, falling back to sequential cells.")
#         speaker_map = {spk: idx for idx, (_, _, spk) in enumerate(segments)}

#     fc, cnt = build_filter_complex(segments, speaker_map, width, height)
#     print(f"Processing {cnt}/{len(segments)} segments")

#     # write filter graph
#     with open("filters.txt", "w", encoding="utf-8") as f:
#         f.write(fc)
#     os.makedirs("debug_frames", exist_ok=True)
#     os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
#     print("Wrote filters.txt")

#     cmd = [
#         "ffmpeg", "-y", "-i", args.input,
#         "-filter_complex_script", "filters.txt",
#         "-map", "[outv]", "-map", "[outa]",
#         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
#         "-c:a", "aac", "-b:a", "192k",
#         "-vsync", "vfr", args.output
#     ]
#     print("Running FFmpeg...")
#     res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
#     if res.returncode != 0:
#         print("❌ FFmpeg error:\n", res.stderr)
#     else:
#         print(f"✔ Saved to {args.output}")

# if __name__ == "__main__":
#     main()














# #!/usr/bin/env python3
# # spotlight_ffmpeg.py
# import os
# import json
# import subprocess
# import argparse
# import cv2
# import numpy as np
# from deepface import DeepFace

# # ── CONFIGURATION ──────────────────────────────────────────────────────
# GRID_ROWS, GRID_COLS = 2, 2  # Zoom grid layout
# DEBUG_DIR = "debug_frames"  # Directory to save debug frames

# def load_segments(path):
#     with open(path, "r") as f:
#         return json.load(f)

# def detect_faces_deepface(frame, detector_backend='yunet'):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     try:
#         faces = DeepFace.extract_faces(frame_rgb, detector_backend=detector_backend, enforce_detection=False)
#         detected_faces = []
#         for face in faces:
#             region = face['region']
#             startX, startY = region['x'], region['y']
#             endX = startX + region['w']
#             endY = startY + region['h']
#             centerX = (startX + endX) // 2
#             centerY = (startY + endY) // 2
#             detected_faces.append((startX, startY, endX, endY, centerX, centerY))
#         return detected_faces
#     except Exception as e:
#         print(f"Error in face detection: {e}")
#         return []

# def save_debug_frame(frame, speaker_id, segment_idx, faces, output_dir=DEBUG_DIR):
#     os.makedirs(output_dir, exist_ok=True)
#     frame_copy = frame.copy()
#     for (startX, startY, endX, endY, _, _) in faces:
#         cv2.rectangle(frame_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)
#     filename = f"{output_dir}/spk_{speaker_id}_seg_{segment_idx}.jpg"
#     cv2.imwrite(filename, frame_copy)
#     print(f"Saved debug frame: {filename}")

# def get_speaker_mapping(segments, width, height, input_video):
#     cap = cv2.VideoCapture(input_video)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open video")
#     speaker_segments = {}
#     for start, end, spk in segments:
#         speaker_segments.setdefault(spk, []).append((start, end))
#     position_map = {}
#     for spk, segs in speaker_segments.items():
#         print(f"Processing speaker {spk} with {len(segs)} segments")
#         for idx, (start, _) in enumerate(segs):
#             cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"Failed to read frame for speaker {spk} at segment {idx} (start: {start}s)")
#                 continue
#             faces = detect_faces_deepface(frame)
#             print(f"Speaker {spk}, segment {idx}: Detected {len(faces)} faces")
#             save_debug_frame(frame, spk, idx, faces)
#             if faces:
#                 faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
#                 position_map[spk] = faces[0]
#                 print(f"Mapped speaker {spk} to cell based on segment {idx}")
#                 break
#             else:
#                 print(f"No faces detected for speaker {spk} in segment {idx}")
#     cap.release()
#     if not position_map:
#         return {}
#     return assign_speaker_to_cell(position_map.values(), width, height)

# def assign_speaker_to_cell(face_positions, width, height):
#     cell_w = width // GRID_COLS
#     cell_h = height // GRID_ROWS
#     speaker_map = {}
#     for (sx, sy, ex, ey, cx, cy) in face_positions:
#         col = min(cx // cell_w, GRID_COLS - 1)
#         row = min(cy // cell_h, GRID_ROWS - 1)
#         cell_index = row * GRID_COLS + col
#         size = (ex - sx) * (ey - sy)
#         if cell_index not in speaker_map or size > speaker_map[cell_index][0]:
#             speaker_map[cell_index] = (size, cx, cy)
#     return {cell: idx for idx, cell in enumerate(speaker_map.keys())}

# def build_filter_complex(segments, speaker_map, width, height):
#     cell_w = width // GRID_COLS
#     cell_h = height // GRID_ROWS
#     filters, labels = [], []
#     count = 0
#     for start, end, spk in segments:
#         if spk not in speaker_map:
#             print(f"Skipping segment for speaker {spk}: not in speaker_map")
#             continue
#         cell = speaker_map[spk]
#         col, row = divmod(cell, GRID_COLS)
#         x, y = col * cell_w, row * cell_h
#         vtag = f"[v{count}]"
#         filters.append(
#             f"[0:v]trim=start={start:.3f}:end={end:.3f},"
#             f"setpts=PTS-STARTPTS,"
#             f"crop={cell_w}:{cell_h}:{x}:{y},"
#             f"scale={width}:{height}:flags=lanczos,"
#             f"setsar=1{vtag}"
#         )
#         labels.append(vtag)
#         atag = f"[a{count}]"
#         filters.append(
#             f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
#             f"asetpts=PTS-STARTPTS{atag}"
#         )
#         labels.append(atag)
#         count += 1
#     concat_in = "".join(labels)
#     filters.append(f"{concat_in}concat=n={count}:v=1:a=1[outv][outa]")
#     return ";".join(filters), count

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", "-i", default="zoom_sample.mp4", help="Source video path")
#     parser.add_argument("--segments", "-s", default="updated_segments.json", help="JSON segments file")
#     parser.add_argument("--output", "-o", default="Zoom_centered.mp4", help="Output video path")
#     args = parser.parse_args()

#     segments = load_segments(args.segments)
#     cap = cv2.VideoCapture(args.input)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     speaker_map = get_speaker_mapping(segments, width, height, args.input)
#     print(f"Generated speaker mapping: {speaker_map}")
#     if not speaker_map:
#         print("⚠️ No mapping, falling back to sequential cells.")
#         speaker_map = {spk: idx for idx, (_, _, spk) in enumerate(segments)}

#     fc, cnt = build_filter_complex(segments, speaker_map, width, height)
#     print(f"Processing {cnt}/{len(segments)} segments")

#     os.makedirs("debug_frames", exist_ok=True)
#     os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
#     with open("filters.txt", "w", encoding="utf-8") as f:
#         f.write(fc)
#     print("Wrote filters.txt")

#     cmd = [
#         "ffmpeg", "-y", "-i", args.input,
#         "-filter_complex_script", "filters.txt",
#         "-map", "[outv]", "-map", "[outa]",
#         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
#         "-c:a", "aac", "-b:a", "192k",
#         "-vsync", "vfr", args.output
#     ]
#     print("Running FFmpeg...")
#     res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
#     if res.returncode != 0:
#         print("❌ FFmpeg error:\n", res.stderr)
#     else:
#         print(f"✔ Saved to {args.output}")

# if __name__ == "__main__":
#     main()











# #!/usr/bin/env python3
# """
# spotlight_ffmpeg_dynamic.py

# Automatically centers active speaker using updated_segments.json,
# with dynamic grid sizing and speaker-to-cell mapping.
# """
# import os
# import json
# import subprocess
# import argparse
# import math
# import cv2
# import numpy as np

# # ── FACE DETECTION PARAMETERS ───────────────────────────────────────────
# FACE_CONFIDENCE_THRESHOLD = 0.5
# FACE_SIZE_THRESHOLD = 0.1  # min % of frame size
# DEBUG_DIR = "debug_frames"  # Directory to save debug frames


# def load_segments(path):
#     with open(path, "r") as f:
#         return json.load(f)


# def detect_faces(frame):
#     net = cv2.dnn.readNetFromCaffe(
#         "deploy.prototxt",
#         "res10_300x300_ssd_iter_140000.caffemodel"
#     )
#     h, w = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                  (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > FACE_CONFIDENCE_THRESHOLD:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             sx, sy, ex, ey = box.astype(int)
#             fw, fh = ex - sx, ey - sy
#             if fw > w * FACE_SIZE_THRESHOLD and fh > h * FACE_SIZE_THRESHOLD:
#                 cx, cy = (sx + ex) // 2, (sy + ey) // 2
#                 faces.append((sx, sy, ex, ey, cx, cy))
#     return faces


# def save_debug_frame(frame, spk, idx, faces, out_dir=DEBUG_DIR):
#     os.makedirs(out_dir, exist_ok=True)
#     img = frame.copy()
#     for sx, sy, ex, ey, _, _ in faces:
#         cv2.rectangle(img, (sx, sy), (ex, ey), (0, 255, 0), 2)
#     fname = f"{out_dir}/spk_{spk}_seg_{idx}.jpg"
#     cv2.imwrite(fname, img)


# def compute_grid(n):
#     # square grid or nearest rectangle
#     cols = math.ceil(math.sqrt(n))
#     rows = math.ceil(n / cols)
#     return rows, cols


# def get_speaker_cell_map(segments, width, height, video_path):
#     # determine speakers and dynamic grid
#     speakers = sorted({seg[2] for seg in segments})
#     n = len(speakers)
#     rows, cols = compute_grid(n)

#     cap = cv2.VideoCapture(video_path)
#     speaker_to_face = {}

#     # find first face for each speaker
#     for spk in speakers:
#         segs = [s for s in segments if s[2] == spk]
#         start = segs[0][0]
#         cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         faces = detect_faces(frame)
#         save_debug_frame(frame, spk, 0, faces)
#         if faces:
#             faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
#             speaker_to_face[spk] = faces[0]

#     cap.release()

#     # map each speaker to a grid cell index
#     mapping = {}
#     used = set()
#     for spk, (sx, sy, ex, ey, cx, cy) in speaker_to_face.items():
#         w_cell = width // cols
#         h_cell = height // rows
#         col = min(cx // w_cell, cols - 1)
#         row = min(cy // h_cell, rows - 1)
#         cell = row * cols + col
#         # if conflict, assign next available
#         if cell in used:
#             for alt in range(n):
#                 if alt not in used:
#                     cell = alt
#                     break
#         used.add(cell)
#         mapping[spk] = cell

#     return mapping, rows, cols


# # def build_filter_complex(segments, mapping, width, height, rows, cols):
# #     vlen = width // cols
# #     hlen = height // rows
# #     filters, tags = [], []
# #     cnt = 0
# #     for st, en, spk in segments:
# #         if spk not in mapping:
# #             continue
# #         cell = mapping[spk]
# #         row, col = divmod(cell, cols)
# #         x, y = col * vlen, row * hlen
# #         vtag = f"[v{cnt}]"
# #         filters.append(
# #             f"[0:v]trim=start={st:.3f}:end={en:.3f},"
# #             f"setpts=PTS-STARTPTS,crop={vlen}:{hlen}:{x}:{y},"
# #             f"scale={width}:{height}:flags=lanczos,setsar=1{vtag}"
# #         )
# #         tags.append(vtag)
# #         atag = f"[a{cnt}]"
# #         filters.append(
# #             f"[0:a]atrim=start={st:.3f}:end={en:.3f},asetpts=PTS-STARTPTS{atag}"
# #         )
# #         tags.append(atag)
# #         cnt += 1
# #     inp = "".join(tags)
# #     filters.append(f"{inp}concat=n={cnt}:v=1:a=1[outv][outa]")
# #     return ";".join(filters), cnt

# def build_filter_complex(segments, mapping, width, height, rows, cols):
#     cell_w = width  // cols
#     cell_h = height // rows
#     filters, tags = [], []
#     cnt = 0

#     for st, en, spk in segments:
#         if spk not in mapping:
#             continue

#         cell = mapping[spk]
#         # compute row/col
#         row = cell // cols
#         col = cell %  cols

#         x = col * cell_w
#         y = row * cell_h

#         # DEBUG LOG:
#         print(f"-- segment #{cnt}: speaker={spk}, cell={cell} "
#               f"-> row={row},col={col}, crop=(x={x},y={y},w={cell_w},h={cell_h})")

#         vtag = f"[v{cnt}]"
#         filters.append(
#             f"[0:v]trim=start={st:.3f}:end={en:.3f},"
#             f"setpts=PTS-STARTPTS,"
#             f"crop={cell_w}:{cell_h}:{x}:{y},"
#             f"scale={width}:{height}:flags=lanczos,setsar=1{vtag}"
#         )
#         tags.append(vtag)

#         atag = f"[a{cnt}]"
#         filters.append(
#             f"[0:a]atrim=start={st:.3f}:end={en:.3f},"
#             f"asetpts=PTS-STARTPTS{atag}"
#         )
#         tags.append(atag)

#         cnt += 1

#     concat_in = "".join(tags)
#     filters.append(f"{concat_in}concat=n={cnt}:v=1:a=1[outv][outa]")
#     return ";".join(filters), cnt

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("-i", "--input", default="zoom_sample.mp4")
#     p.add_argument("-s", "--segments", default="updated_segments.json")
#     p.add_argument("-o", "--output", default="Zoom_centered.mp4")
#     args = p.parse_args()

#     segments = load_segments(args.segments)
#     cap = cv2.VideoCapture(args.input)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     mapping, rows, cols = get_speaker_cell_map(segments, width, height, args.input)
#     print(f"Mapping: {mapping}, Grid: {rows}x{cols}")

#     fc, processed = build_filter_complex(segments, mapping, width, height, rows, cols)
#     print(f"Processing {processed}/{len(segments)} segments")

#     os.makedirs(DEBUG_DIR, exist_ok=True)
#     with open("filters.txt", "w") as f:
#         f.write(fc)
#     print("Wrote filter graph to filters.txt")

#     cmd = [
#         "ffmpeg", "-y", "-i", args.input,
#         "-filter_complex_script", "filters.txt",
#         "-map", "[outv]", "-map", "[outa]",
#         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
#         "-c:a", "aac", "-b:a", "192k", "-vsync", "vfr", args.output
#     ]
#     print("Running FFmpeg...")
#     res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
#     if res.returncode:
#         print("❌ FFmpeg error:\n", res.stderr)
#     else:
#         print(f"✔ Saved to {args.output}")

# if __name__ == "__main__":
#     main()









# #!/usr/bin/env python3
# """
# spotlight_ffmpeg_dynamic.py

# Centers each speaker in its own cell based on fixed speaker IDs (no face detection).
# Grid size is computed from number of speakers, and speaker N always maps to cell N.
# """
# import os
# import json
# import subprocess
# import argparse
# import math
# import cv2
# import numpy as np

# # ── FACE DETECTION PARAMETERS (unused) ────────────────────────────────────
# # All mapping now static: speaker ID -> same cell index.

# # ── DEBUG PARAMETERS ─────────────────────────────────────────────────────
# DEBUG_DIR = "debug_frames"


# def load_segments(path):
#     with open(path, "r") as f:
#         return json.load(f)


# def compute_grid(n):
#     # square-ish grid
#     cols = math.ceil(math.sqrt(n))
#     rows = math.ceil(n / cols)
#     return rows, cols


# def get_speaker_cell_map(segments):
#     # Unique speaker IDs
#     speakers = sorted({seg[2] for seg in segments})
#     # Map each speaker to its own index (0->0, 1->1, etc.)
#     return {spk: idx for idx, spk in enumerate(speakers)}, len(speakers)


# def build_filter_complex(segments, mapping, width, height, rows, cols):
#     cell_w = width  // cols
#     cell_h = height // rows
#     filters, tags = [], []
#     cnt = 0

#     for st, en, spk in segments:
#         if spk not in mapping:
#             continue
#         cell = mapping[spk]
#         row = cell // cols
#         col = cell % cols
#         x = col * cell_w
#         y = row * cell_h

#         # Debug:
#         print(f"segment #{cnt}: speaker={spk}, cell={cell}, crop=(x={x},y={y},w={cell_w},h={cell_h})")

#         vtag = f"[v{cnt}]"
#         filters.append(
#             f"[0:v]trim=start={st:.3f}:end={en:.3f},"
#             f"setpts=PTS-STARTPTS,crop={cell_w}:{cell_h}:{x}:{y},"
#             f"scale={width}:{height}:flags=lanczos,setsar=1{vtag}"
#         )
#         tags.append(vtag)

#         atag = f"[a{cnt}]"
#         filters.append(
#             f"[0:a]atrim=start={st:.3f}:end={en:.3f},asetpts=PTS-STARTPTS{atag}"
#         )
#         tags.append(atag)
#         cnt += 1

#     concat_in = "".join(tags)
#     filters.append(f"{concat_in}concat=n={cnt}:v=1:a=1[outv][outa]")
#     return ";".join(filters), cnt


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("-i", "--input", default="zoom_sample.mp4")
#     p.add_argument("-s", "--segments", default="updated_segments.json")
#     p.add_argument("-o", "--output", default="Zoom_centered.mp4")
#     args = p.parse_args()

#     segments = load_segments(args.segments)
#     cap = cv2.VideoCapture(args.input)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     mapping, n_speakers = get_speaker_cell_map(segments)
#     rows, cols = compute_grid(n_speakers)
#     print(f"Static mapping: {mapping}, Grid: {rows}x{cols}")

#     fc, processed = build_filter_complex(segments, mapping, width, height, rows, cols)
#     print(f"Processing {processed}/{len(segments)} segments")

#     # ensure dirs
#     os.makedirs(DEBUG_DIR, exist_ok=True)
#     with open("filters.txt", "w") as f:
#         f.write(fc)

#     cmd = [
#         "ffmpeg", "-y", "-i", args.input,
#         "-filter_complex_script", "filters.txt",
#         "-map", "[outv]", "-map", "[outa]",
#         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
#         "-c:a", "aac", "-b:a", "192k", "-vsync", "vfr", args.output
#     ]
#     print("Running FFmpeg...")
#     res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
#     if res.returncode:
#         print("❌ FFmpeg error:\n", res.stderr)
#     else:
#         print(f"✔ Saved to {args.output}")

# if __name__ == "__main__":
#     main()





#!/usr/bin/env python3
"""
spotlight_ffmpeg_dynamic.py

Centers each speaker in its own cell using column-major ordering:
cell IDs increase down each column, then move right.
Grid size computes to accommodate all speakers.
"""
import os
import json
import subprocess
import argparse
import math
import cv2
import numpy as np

# ── DEBUG PARAMETERS ─────────────────────────────────────────────────────
DEBUG_DIR = "debug_frames"


def load_segments(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_grid(n):
    # Choose smallest grid (rows x cols) s.t. rows*cols >= n
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def get_speaker_cell_map(segments):
    # Unique sorted speaker IDs
    speakers = sorted({seg[2] for seg in segments})
    # Static mapping: speaker N -> cell N
    return {spk: idx for idx, spk in enumerate(speakers)}, len(speakers)


def build_filter_complex(segments, mapping, width, height, rows, cols):
    cell_w = width  // cols
    cell_h = height // rows
    filters, tags = [], []
    cnt = 0

    for st, en, spk in segments:
        if spk not in mapping:
            continue
        cell = mapping[spk]
        # COLUMN-MAJOR: down then right
        col = cell // rows
        row = cell % rows
        x = col * cell_w
        y = row * cell_h

        # Debug output
        print(f"segment #{cnt}: speaker={spk}, cell={cell} -> row={row},col={col},crop=(x={x},y={y},w={cell_w},h={cell_h})")

        vtag = f"[v{cnt}]"
        filters.append(
            f"[0:v]trim=start={st:.3f}:end={en:.3f},"
            f"setpts=PTS-STARTPTS,crop={cell_w}:{cell_h}:{x}:{y},"
            f"scale={width}:{height}:flags=lanczos,setsar=1{vtag}"
        )
        tags.append(vtag)

        atag = f"[a{cnt}]"
        filters.append(
            f"[0:a]atrim=start={st:.3f}:end={en:.3f},asetpts=PTS-STARTPTS{atag}"
        )
        tags.append(atag)
        cnt += 1

    concat_in = "".join(tags)
    filters.append(f"{concat_in}concat=n={cnt}:v=1:a=1[outv][outa]")
    return ";".join(filters), cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default="output_grid_with_audio_1_edited.mp4")
    p.add_argument("-s", "--segments", default="updated_segments2.json")
    p.add_argument("-o", "--output", default="Zoom_centered2.mp4")
    args = p.parse_args()

    segments = load_segments(args.segments)
    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    mapping, n_speakers = get_speaker_cell_map(segments)
    rows, cols = compute_grid(n_speakers)
    print(f"Static column-major mapping: {mapping}, Grid: {rows} rows x {cols} cols")

    fc, processed = build_filter_complex(segments, mapping, width, height, rows, cols)
    print(f"Processing {processed}/{len(segments)} segments")

    os.makedirs(DEBUG_DIR, exist_ok=True)
    with open("filters.txt", "w") as f:
        f.write(fc)
    print("Wrote filter graph to filters.txt")

    cmd = [
        "ffmpeg", "-y", "-i", args.input,
        "-filter_complex_script", "filters.txt",
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k", "-vsync", "vfr", args.output
    ]
    print("Running FFmpeg...")
    res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if res.returncode:
        print("❌ FFmpeg error:\n", res.stderr)
    else:
        print(f"✔ Saved to {args.output}")

if __name__ == "__main__":
    main()
