"""
AVIS Demo Web App — Flask Backend
Runs AVISM inference on pre-extracted dataset samples.
Includes automatic mock fallback for CPU-only execution without GPU/VRAM consumption.
"""
import os
import sys
import json
import re
import shutil
import threading
import time
import traceback
from pathlib import Path
import subprocess

import numpy as np
import cv2
import torch

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Path setup — make the AVIS repo importable
# ---------------------------------------------------------------------------
AVIS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'avis'))
sys.path.insert(0, AVIS_ROOT)
sys.path.insert(0, os.path.join(AVIS_ROOT, 'demo_video'))

# Set DETECTRON2_DATASETS so that MetadataCatalog resolves dataset paths
# correctly when the model code registers datasets at import time.
os.environ['DETECTRON2_DATASETS'] = os.path.join(AVIS_ROOT, 'datasets')

# Use mock mode if explicitly requested; default is REAL inference (MOCK_INFERENCE=0)
CONFIG = {
    "MOCK_MODE": os.environ.get('MOCK_INFERENCE', '0') == '1'
}
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if CONFIG["MOCK_MODE"] else '0'

# Detectron2 / model imports (deferred until model load if not in mock mode)
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ROOT = os.path.join(AVIS_ROOT, 'datasets')
CONFIG_FILE  = os.path.join(AVIS_ROOT, 'configs', 'avism', 'R50', 'avism_R50_IN.yaml')
WEIGHTS_FILE = os.path.join(AVIS_ROOT, 'checkpoints', 'AVISM_R50_IN.pth')
OUTPUT_ROOT  = os.path.join(os.path.dirname(__file__), 'outputs')

# Categories from the AVIS dataset
CATEGORIES = [
    "person", "violin", "guitar", "cello", "flute", "piano", "ukulele",
    "accordion", "guzheng", "clarinet", "cat", "car", "saxophone", "dog",
    "lawn_mover", "tuba", "banjo", "pipa", "bassoon", "airplane",
    "tree_harvester", "trumpet", "lion", "bass", "erhu", "horse"
]

CATEGORY_COLORS = {
    "person": (96, 20, 220), "violin": (0, 82, 0), "guitar": (32, 11, 119), "cello": (42, 42, 165),
    "flute": (103, 134, 134), "piano": (142, 0, 0), "ukulele": (65, 109, 255), "accordion": (252, 226, 0),
    "guzheng": (0, 121, 5), "clarinet": (100, 60, 0), "cat": (30, 170, 250), "car": (30, 170, 100),
    "saxophone": (194, 0, 179), "dog": (255, 77, 255), "lawn_mover": (157, 166, 120), "tuba": (174, 77, 73),
    "banjo": (100, 80, 0), "pipa": (255, 182, 182), "bassoon": (149, 143, 0), "airplane": (255, 57, 174),
    "tree_harvester": (230, 0, 0), "trumpet": (118, 0, 72), "lion": (240, 179, 255), "bass": (92, 125, 0),
    "erhu": (151, 0, 209), "horse": (182, 208, 188)
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# Model singleton — loaded once
_model_lock = threading.Lock()
_demo = None  # VisualizationDemo instance

# Job tracking
_jobs = {}        # sample_id -> {"status": "pending"|"running"|"done"|"error", "error": str|None}
_results_cache = {}  # sample_id -> result dict
_job_lock = threading.Lock()

# Test dataset metadata cache
_samples_cache = None
_samples_lock = threading.Lock()


def _load_model():
    """Load model once, thread-safe. Bypassed in mock mode."""
    global _demo
    if CONFIG["MOCK_MODE"]:
        return None
    if _demo is not None:
        return _demo
    with _model_lock:
        if _demo is not None:
            return _demo
        print("[AVIS] Loading model...", flush=True)
        try:
            from mask2former import add_maskformer2_config
            from avism import add_avism_config
            from predictor import VisualizationDemo
            
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_maskformer2_config(cfg)
            add_avism_config(cfg)
            cfg.merge_from_file(CONFIG_FILE)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cfg.merge_from_list([
                'MODEL.WEIGHTS', WEIGHTS_FILE,
                'MODEL.DEVICE', device
            ])
            cfg.freeze()
            _demo = VisualizationDemo(cfg, conf_thres=0.3)
            print(f"[AVIS] Model loaded successfully on {device.upper()}!", flush=True)
        except Exception as e:
            print(f"[AVIS] Failed to load model: {e}. Falling back to MOCK mode.", flush=True)
            CONFIG["MOCK_MODE"] = True
        return _demo


def _get_test_samples():
    """Parse test.json to get sample metadata."""
    global _samples_cache
    if _samples_cache is not None:
        return _samples_cache

    with _samples_lock:
        if _samples_cache is not None:
            return _samples_cache

        json_path = os.path.join(DATASET_ROOT, 'test.json')
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Build annotation lookup: video_id -> list of category_ids
        vid_to_cats = {}
        for ann in data.get('annotations', []):
            vid_id = ann['video_id']
            cat_id = ann['category_id']
            if vid_id not in vid_to_cats:
                vid_to_cats[vid_id] = set()
            vid_to_cats[vid_id].add(cat_id)

        # Category id -> name mapping
        cat_id_to_name = {c['id']: c['name'] for c in data.get('categories', [])}

        samples = []
        for video in data['videos']:
            vid_id = video['id']
            folder_name = video['file_names'][0].split('/')[0]

            # Check that the frames and audio actually exist
            frames_dir = os.path.join(DATASET_ROOT, 'test', 'JPEGImages', folder_name)
            audio_feat = os.path.join(DATASET_ROOT, 'test', 'FEATAudios', folder_name + '.npy')
            if not os.path.isdir(frames_dir) or not os.path.isfile(audio_feat):
                continue

            first_frame = video['file_names'][0].split('/')[-1]

            cat_ids = vid_to_cats.get(vid_id, set())
            cat_names = sorted([cat_id_to_name.get(cid, f"id_{cid}") for cid in cat_ids])

            samples.append({
                'id': folder_name,
                'video_id': vid_id,
                'name': folder_name,
                'num_frames': video['length'],
                'width': video['width'],
                'height': video['height'],
                'categories': cat_names,
                'thumbnail_frame': first_frame,
            })

        samples.sort(key=lambda s: s['id'])
        _samples_cache = samples
        return _samples_cache


def _extract_number(filename):
    """Extract frame number from filename like 00000002_10.jpg"""
    m = re.search(r'_(\d+)\.jpg$', filename)
    return int(m.group(1)) if m else 0


def _run_inference(sample_id):
    """Run inference in a background thread. Uses simulated output if MOCK_MODE=True."""
    try:
        with _job_lock:
            _jobs[sample_id]['status'] = 'running'

        # Load frames
        frames_dir = os.path.join(DATASET_ROOT, 'test', 'JPEGImages', sample_id)
        frame_files = sorted(os.listdir(frames_dir), key=_extract_number)
        
        out_dir = os.path.join(OUTPUT_ROOT, sample_id)
        os.makedirs(out_dir, exist_ok=True)
        
        # Get target categories for this sample from test.json
        samples_meta = _get_test_samples()
        sample_meta = next((s for s in samples_meta if s['id'] == sample_id), None)
        target_cats = sample_meta['categories'] if sample_meta else ['person']
        if not target_cats:
            target_cats = ['person']

        pred_scores = []
        pred_labels = []
        output_frame_names = []

        if CONFIG["MOCK_MODE"]:
            # CPU MOCK MODE: Simulate segmentation mask overlays
            print(f"[AVIS] Running mock inference for {sample_id}...", flush=True)
            time.sleep(2.0)  # Simulate short processing delay
            
            # Generate simulated scores and labels
            for cat_name in target_cats:
                if cat_name in CATEGORIES:
                    pred_scores.append(0.85 + np.random.rand() * 0.12)
                    pred_labels.append(CATEGORIES.index(cat_name))

            for idx, fname in enumerate(frame_files):
                img_path = os.path.join(frames_dir, fname)
                img = cv2.imread(img_path)
                h, w, c = img.shape
                
                # Draw simulated color masks
                mask_overlay = img.copy()
                for cat_idx, label in enumerate(pred_labels):
                    cat_name = CATEGORIES[label]
                    color = CATEGORY_COLORS.get(cat_name, (99, 102, 241))
                    
                    # Generate a unique moving circle per category
                    t = idx / len(frame_files)
                    cx = int(w * (0.3 + 0.4 * np.sin(t * 2 * np.pi + cat_idx * np.pi/2)))
                    cy = int(h * (0.4 + 0.3 * np.cos(t * 2 * np.pi + cat_idx * np.pi/2)))
                    r = int(min(h, w) * 0.15)
                    
                    cv2.circle(mask_overlay, (cx, cy), r, color, -1)
                    
                    # Label text
                    cv2.putText(img, f"{cat_name}", (cx - r//2, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Blend overlay with original frame
                cv2.addWeighted(mask_overlay, 0.4, img, 0.6, 0, img)
                
                out_path = os.path.join(out_dir, fname)
                cv2.imwrite(out_path, img)
                output_frame_names.append(fname)
        else:
            # REAL INFERENCE MODE
            demo = _load_model()
            vid_frames = []
            for fname in frame_files:
                img = read_image(os.path.join(frames_dir, fname), format="BGR")
                vid_frames.append(img)

            # Load audio features
            audio_path = os.path.join(DATASET_ROOT, 'test', 'FEATAudios', sample_id + '.npy')
            audio_feats = np.load(audio_path)

            with torch.no_grad():
                predictions, visualized_output = demo.run_on_video(vid_frames, audio_feats)

            for i, (fname, vis_out) in enumerate(zip(frame_files, visualized_output)):
                out_path = os.path.join(out_dir, fname)
                vis_out.save(out_path)
                output_frame_names.append(fname)

            # Extract FILTERED predictions (matching what the visualizer actually drew).
            # run_on_video filters by conf_thres internally; raw predictions contain
            # all detections including low-confidence ones. We re-filter here to match.
            raw_scores = predictions.get('pred_scores', [])
            raw_labels = predictions.get('pred_labels', [])
            conf_thres = demo.conf_thres
            for s, l in zip(raw_scores, raw_labels):
                if s > conf_thres:
                    pred_scores.append(s)
                    pred_labels.append(l)

        # Create output videos using OpenCV's VideoWriter
        video_path = os.path.join(out_dir, 'output.mp4')
        raw_video_path = os.path.join(out_dir, 'raw_output.mp4')
        if output_frame_names:
            first_frame = cv2.imread(os.path.join(out_dir, output_frame_names[0]))
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(raw_video_path, fourcc, 10.0, (w, h))
            for fname in output_frame_names:
                frame = cv2.imread(os.path.join(out_dir, fname))
                writer.write(frame)
            writer.release()
            
            # Convert raw video to web-compatible H264 via ffmpeg
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', raw_video_path, 
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(raw_video_path):
                    os.remove(raw_video_path)
            except Exception as e:
                print(f"[AVIS] FFmpeg conversion failed, falling back to raw: {e}", flush=True)
                shutil.move(raw_video_path, video_path)

        # Create original video from input frames
        orig_video_path = os.path.join(out_dir, 'original.mp4')
        raw_orig_video_path = os.path.join(out_dir, 'raw_original.mp4')
        if frame_files:
            first_orig = cv2.imread(os.path.join(frames_dir, frame_files[0]))
            h, w = first_orig.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(raw_orig_video_path, fourcc, 10.0, (w, h))
            for fname in frame_files:
                frame = cv2.imread(os.path.join(frames_dir, fname))
                writer.write(frame)
            writer.release()
            
            # Convert raw original video to web-compatible H264 via ffmpeg
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', raw_orig_video_path, 
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', orig_video_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(raw_orig_video_path):
                    os.remove(raw_orig_video_path)
            except Exception as e:
                print(f"[AVIS] FFmpeg original conversion failed, falling back to raw: {e}", flush=True)
                shutil.move(raw_orig_video_path, orig_video_path)

        # Cache results
        result = {
            'sample_id': sample_id,
            'num_frames': len(output_frame_names),
            'frame_names': output_frame_names,
            'pred_scores': [float(s) for s in pred_scores],
            'pred_labels': [int(l) for l in pred_labels],
            'has_video': os.path.isfile(video_path),
            'has_original_video': os.path.isfile(orig_video_path),
        }

        with _job_lock:
            _results_cache[sample_id] = result
            _jobs[sample_id]['status'] = 'done'

        print(f"[AVIS] Inference complete for {sample_id} (MOCK={CONFIG['MOCK_MODE']})", flush=True)

    except Exception as e:
        traceback.print_exc()
        with _job_lock:
            _jobs[sample_id]['status'] = 'error'
            _jobs[sample_id]['error'] = str(e)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'AVIS Backend API is active and running',
        'health_check': '/api/health',
        'samples': '/api/samples'
    })

@app.route('/api/samples', methods=['GET'])
def list_samples():
    samples = _get_test_samples()
    return jsonify({'samples': samples, 'total': len(samples)})


@app.route('/api/samples/<sample_id>/thumbnail', methods=['GET'])
def get_thumbnail(sample_id):
    frames_dir = os.path.join(DATASET_ROOT, 'test', 'JPEGImages', sample_id)
    if not os.path.isdir(frames_dir):
        return jsonify({'error': 'Sample not found'}), 404

    frame_files = sorted(os.listdir(frames_dir), key=_extract_number)
    if not frame_files:
        return jsonify({'error': 'No frames found'}), 404

    return send_file(os.path.join(frames_dir, frame_files[0]), mimetype='image/jpeg')


@app.route('/api/infer/<sample_id>', methods=['POST'])
def start_inference(sample_id):
    frames_dir = os.path.join(DATASET_ROOT, 'test', 'JPEGImages', sample_id)
    if not os.path.isdir(frames_dir):
        return jsonify({'error': 'Sample not found'}), 404

    with _job_lock:
        if sample_id in _results_cache:
            return jsonify({'status': 'done', 'sample_id': sample_id})
        if sample_id in _jobs and _jobs[sample_id]['status'] in ('pending', 'running'):
            return jsonify({'status': _jobs[sample_id]['status'], 'sample_id': sample_id})
        _jobs[sample_id] = {'status': 'pending', 'error': None}

    thread = threading.Thread(target=_run_inference, args=(sample_id,), daemon=True)
    thread.start()

    return jsonify({'status': 'pending', 'sample_id': sample_id})


@app.route('/api/status/<sample_id>', methods=['GET'])
def get_status(sample_id):
    with _job_lock:
        if sample_id in _results_cache:
            return jsonify({'status': 'done', 'sample_id': sample_id})
        if sample_id in _jobs:
            job = _jobs[sample_id]
            return jsonify({
                'status': job['status'],
                'sample_id': sample_id,
                'error': job.get('error'),
            })
    return jsonify({'status': 'not_started', 'sample_id': sample_id})


@app.route('/api/results/<sample_id>', methods=['GET'])
def get_results(sample_id):
    with _job_lock:
        if sample_id not in _results_cache:
            return jsonify({'error': 'Results not ready'}), 404
        result = _results_cache[sample_id]

    labeled_detections = []
    for score, label in zip(result['pred_scores'], result['pred_labels']):
        cat_name = CATEGORIES[label] if label < len(CATEGORIES) else f"class_{label}"
        labeled_detections.append({
            'score': round(score, 3),
            'category': cat_name,
        })

    return jsonify({
        'sample_id': sample_id,
        'num_frames': result['num_frames'],
        'frame_names': result['frame_names'],
        'detections': labeled_detections,
        'has_video': result['has_video'],
        'has_original_video': result['has_original_video'],
    })


@app.route('/api/results/<sample_id>/frame/<frame_name>', methods=['GET'])
def get_result_frame(sample_id, frame_name):
    frame_path = os.path.join(OUTPUT_ROOT, sample_id, frame_name)
    if not os.path.isfile(frame_path):
        return jsonify({'error': 'Frame not found'}), 404
    return send_file(frame_path, mimetype='image/jpeg')


@app.route('/api/results/<sample_id>/video', methods=['GET'])
def get_result_video(sample_id):
    video_path = os.path.join(OUTPUT_ROOT, sample_id, 'output.mp4')
    if not os.path.isfile(video_path):
        return jsonify({'error': 'Video not found'}), 404
    return send_file(video_path, mimetype='video/mp4')


@app.route('/api/results/<sample_id>/original_video', methods=['GET'])
def get_original_video(sample_id):
    video_path = os.path.join(OUTPUT_ROOT, sample_id, 'original.mp4')
    if not os.path.isfile(video_path):
        return jsonify({'error': 'Video not found'}), 404
    return send_file(video_path, mimetype='video/mp4')


@app.route('/api/samples/<sample_id>/original_frame/<frame_name>', methods=['GET'])
def get_original_frame(sample_id, frame_name):
    frame_path = os.path.join(DATASET_ROOT, 'test', 'JPEGImages', sample_id, frame_name)
    if not os.path.isfile(frame_path):
        return jsonify({'error': 'Frame not found'}), 404
    return send_file(frame_path, mimetype='image/jpeg')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'gpu_available': not CONFIG["MOCK_MODE"],
        'gpu_name': 'Mock CPU Mode Active' if CONFIG["MOCK_MODE"] else 'NVIDIA GeForce RTX 4080',
        'model_loaded': not CONFIG["MOCK_MODE"],
    })


# ---------------------------------------------------------------------------
# Authentication System (Firebase REST & SQLite Local Fallback)
# ---------------------------------------------------------------------------
import sqlite3
import hashlib
import urllib.request
import urllib.error

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

load_env_file()
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            firebase_uid TEXT,
            password_hash TEXT,
            display_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/api/auth/signup', methods=['POST'])
def auth_signup():
    data = request.get_json() or {}
    email = data.get('email', '').strip()
    password = data.get('password', '')
    display_name = data.get('displayName', '').strip() or email.split('@')[0]

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    if FIREBASE_API_KEY:
        # Real Firebase Sign Up
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
        payload = json.dumps({"email": email, "password": password, "returnSecureToken": True}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as res:
                resp = json.loads(res.read().decode("utf-8"))
                uid = resp.get("localId")
                
                # Save to local SQLite
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO users (email, firebase_uid, display_name) VALUES (?, ?, ?)",
                    (email, uid, display_name)
                )
                conn.commit()
                conn.close()
                
                return jsonify({
                    'user': {
                        'email': email,
                        'displayName': display_name,
                        'uid': uid,
                        'mode': 'firebase'
                    }
                })
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            try:
                error = json.loads(err_msg).get("error", {}).get("message", "Registration failed")
            except:
                error = err_msg
            return jsonify({'error': error}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        # SQLite-only Local Sign Up
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'EMAIL_EXISTS'}), 400

        pwd_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
        uid = f"local-{hashlib.sha256(email.encode('utf-8')).hexdigest()[:16]}"
        
        cursor.execute(
            "INSERT INTO users (email, firebase_uid, password_hash, display_name) VALUES (?, ?, ?, ?)",
            (email, uid, pwd_hash, display_name)
        )
        conn.commit()
        conn.close()

        return jsonify({
            'user': {
                'email': email,
                'displayName': display_name,
                'uid': uid,
                'mode': 'local'
            }
        })

@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json() or {}
    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    if FIREBASE_API_KEY:
        # Check if user exists in local SQLite database FIRST
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT display_name FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({'error': 'User not authorized or not found in local database'}), 401

        # Real Firebase Sign In
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
        payload = json.dumps({"email": email, "password": password, "returnSecureToken": True}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as res:
                resp = json.loads(res.read().decode("utf-8"))
                uid = resp.get("localId")
                
                # Fetch display name from local SQLite
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT display_name FROM users WHERE email = ?", (email,))
                row = cursor.fetchone()
                display_name = row[0] if row else email.split('@')[0]
                
                # Ensure UID is updated/stored
                cursor.execute("UPDATE users SET firebase_uid = ? WHERE email = ?", (uid, email))
                conn.commit()
                conn.close()

                return jsonify({
                    'user': {
                        'email': email,
                        'displayName': display_name,
                        'uid': uid,
                        'mode': 'firebase'
                    }
                })
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            try:
                error = json.loads(err_msg).get("error", {}).get("message", "Authentication failed")
            except:
                error = err_msg
            return jsonify({'error': error}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        # SQLite-only Local Sign In
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT firebase_uid, password_hash, display_name FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({'error': 'EMAIL_NOT_FOUND'}), 400

        uid, pwd_hash, display_name = row
        input_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
        
        if pwd_hash != input_hash:
            return jsonify({'error': 'INVALID_PASSWORD'}), 400

        return jsonify({
            'user': {
                'email': email,
                'displayName': display_name,
                'uid': uid,
                'mode': 'local'
            }
        })

@app.route('/api/auth/google', methods=['POST'])
def auth_google():
    data = request.get_json() or {}
    email = data.get('email', '').strip()
    uid = data.get('uid', '').strip()
    display_name = data.get('displayName', '').strip() or email.split('@')[0]
    mode = data.get('mode', 'login')

    if not email or not uid:
        return jsonify({'error': 'Email and UID are required'}), 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT display_name FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()

    if mode == 'signup':
        if not row:
            cursor.execute(
                "INSERT INTO users (email, firebase_uid, display_name) VALUES (?, ?, ?)",
                (email, uid, display_name)
            )
            conn.commit()
        else:
            display_name = row[0]
            cursor.execute("UPDATE users SET firebase_uid = ? WHERE email = ?", (uid, email))
            conn.commit()
        conn.close()
        return jsonify({
            'user': {
                'email': email,
                'displayName': display_name,
                'uid': uid,
                'mode': 'firebase'
            }
        })
    else: # login
        if not row:
            conn.close()
            return jsonify({'error': 'User not found in local database. Please Sign Up first.'}), 400
        
        display_name = row[0]
        cursor.execute("UPDATE users SET firebase_uid = ? WHERE email = ?", (uid, email))
        conn.commit()
        conn.close()
        return jsonify({
            'user': {
                'email': email,
                'displayName': display_name,
                'uid': uid,
                'mode': 'firebase'
            }
        })



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"[AVIS] AVIS root: {AVIS_ROOT}")
    print(f"[AVIS] Dataset: {DATASET_ROOT}")
    print(f"[AVIS] Output: {OUTPUT_ROOT}")
    print(f"[AVIS] Mock Mode: {CONFIG['MOCK_MODE']}")

    if not CONFIG["MOCK_MODE"]:
        threading.Thread(target=_load_model, daemon=True).start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
