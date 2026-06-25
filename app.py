import os
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
import pickle
import time

# --- CONFIGURATION ---
IMG_SIZE = 224
SEQUENCE_LENGTH = 16 
CONFIDENCE_THRESHOLD = 0.55
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'SignSpeak_Model_Final.keras')
LABEL_PATH = os.path.join(BASE_DIR, 'master_label_map.pkl')
CSS_PATH = os.path.join(BASE_DIR, 'static', 'style.css')

# --- LOAD ASSETS ---
model = None
try:
    print("Loading model from:", MODEL_PATH)
    # compile=False lagane se Hugging Face server par weights bina crash kiye smoothly load ho jate hain
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABEL_PATH, 'rb') as f:
        label_map = pickle.load(f)
        inv_label_map = {int(v): k for k, v in label_map.items()}
    print("Model and Labels loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING ASSETS: {e}")

# Labels
URDU_LABELS = { 'aaj': 'آج', 'aath': 'آٹھ', 'ahista': 'آہستہ', 'anywalakal': 'آنے والا کل', 'behtreen': 'بہترین', 'btana': 'بتانا', 'bukhar': 'بخار', 'bus': 'بس', 'car': 'کار', 'char': 'چار', 'chawal': 'چاول', 'chay': 'چھ', 'chaye': 'چائے', 'chini': 'چینی', 'dard': 'درد', 'das': 'دس', 'dawai': 'دوائی', 'dekhna': 'ڈاکٹر', 'do': 'دو', 'dobara': 'دوبارہ', 'doctor': 'ڈاکٹر', 'doodh': 'دودھ', 'dost': 'دوست', 'ek': 'ایک', 'emergency': 'ایمرجنسی', 'ghalat': 'غلط', 'ghanta': 'گھنتہ', 'gosht': 'گوشت', 'hafta': 'ہفتہ', 'intezar': 'انتظار', 'kal': 'کل', 'likhna': 'لکھنا', 'mahina': 'مہینہ', 'mask': 'ماسک', 'minute': 'منٹ', 'no': 'نو','paanch': 'پانچ', 'parhna': 'پڑھنا', 'raasta': 'راستہ', 'roti': 'روٹی', 'saat': 'سات', 'sabzi': 'سبزی', 'sahih': 'صحیح', 'samajhna': 'سمجھنا', 'stop': 'سٹاپ', 'sunna': 'سننا', 'tabdeel': 'تبدیل', 'teen': 'تین', 'tez': 'تیز', 'ticket': 'ٹکٹ' }
ENGLISH_MAP = { 'aaj': 'Today', 'aath': 'Eight (8)', 'ahista': 'Slow', 'anywalakal': 'Tomorrow', 'behtreen': 'Perfect', 'btana': 'To Tell', 'bukhar': 'Fever', 'bus': 'Bus', 'car': 'Car', 'char': 'Four (4)', 'chawal': 'Rice', 'chay': 'Six (6)', 'chaye': 'Tea', 'chini': 'Sugar', 'dard': 'Pain', 'das': 'Ten (10)', 'dawai': 'Medicine', 'dekhna': 'To See', 'do': 'Two (2)', 'dobara': 'Again', 'doctor': 'Doctor', 'doodh': 'Milk', 'dost': 'Friend', 'ek': 'One (1)', 'emergency': 'Emergency', 'ghalat': 'Wrong', 'ghanta': 'Hour', 'gosht': 'Meat', 'hafta': 'Week', 'intezar': 'Wait', 'kal': 'Yesterday', 'likhna': 'To Write', 'mahina': 'Month', 'mask': 'Mask', 'minute': 'Minute', 'no': 'Nine (9)', 'paanch': 'Five (5)', 'parhna': 'To Read', 'raasta': 'Way / Path', 'roti': 'Bread (Roti)', 'saat': 'Seven (7)', 'sabzi': 'Vegetable', 'sahih': 'Correct', 'samajhna': 'To Understand', 'stop': 'Stop', 'sunna': 'To Listen', 'tabdeel': 'Change', 'teen': 'Three (3)', 'tez': 'Fast', 'ticket': 'Ticket' }

# 100% PURE ORIGINAL PREDICTION FUNCTION (No logic changed)
def process_and_predict(video_path, rotate_90, flip_mirror):
    global model
    if model is None:
        return "❌ AI Model is not initialized properly on server. Please restart Space."
        
    if video_path is None:
        return "⚠️ No video recorded yet. Please record and wait 1 sec."
    time.sleep(0.5)
    if not os.path.exists(video_path):
        return "⚠️ File system error. Try again."

    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        return "⚠️ Could not open video stream."

    while True:
        ret, frame = cap.read()
        if not ret: break
        if flip_mirror: frame = cv2.flip(frame, 1)
        if rotate_90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) < SEQUENCE_LENGTH:
        return f"⚠️ Recorded {len(frames)} frames. Need at least {SEQUENCE_LENGTH}."

    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
    sampled = np.array([frames[i] for i in indices], dtype=np.float32)
    
    try:
        predictions = model.predict(np.expand_dims(sampled, axis=0), verbose=0)[0]
        idx = np.argmax(predictions)
        conf = predictions[idx]

        if conf < CONFIDENCE_THRESHOLD:
            return f"Low Confidence ({conf*100:.1f}%)\nMove closer to camera."
        
        label_key = inv_label_map.get(idx, "Unknown")
        return f"{URDU_LABELS.get(label_key, label_key)}\n{ENGLISH_MAP.get(label_key, label_key)}\nConfidence: {conf*100:.1f}%"
    except Exception as e:
        return f"Error: {str(e)}"

def load_css(path):
    if os.path.exists(path):
        with open(path, "r") as f: return f.read()
    return ""

with gr.Blocks(title="SignSpeak AI") as app:
    
    with gr.Column(elem_classes="header-box"):
        gr.Markdown("# SignSpeak 🤟")

    with gr.Row():
        with gr.Column(scale=2, elem_classes="glass-card"):
            # FIX: Android camera flip aur Mobile media playback issues hal karne ke liye properties update ki hain
            video_input = gr.Video(
                label="Input Stream", 
                sources=["webcam", "upload"],
                format="mp4",                 # Strict format taake Android browsers preview crash na karein
                interactive=True              # Mobile browser par playback interface auto render karne ke liye
            )
            with gr.Row():
                mirror_opt = gr.Checkbox(label="Mirror Fix", value=False)
                rotate_opt = gr.Checkbox(label="Rotate 90°", value=False)
            predict_btn = gr.Button(" TRANSLATE", elem_id="predict-btn")
            reset_btn = gr.Button("🔄 RESET")
            
        with gr.Column(scale=1, elem_classes="glass-card"):
            gr.Markdown("###  Prediction Result")
            result_output = gr.Textbox(show_label=False, lines=5, elem_id="result-display", interactive=False)
            gr.Markdown("---")
            gr.Markdown("**Instructions:**\n- Record for at least 3 seconds.\n- Ensure clear lighting.\n- Keep hands fully visible.")

    predict_btn.click(process_and_predict, [video_input, rotate_opt, mirror_opt], result_output)
    reset_btn.click(lambda: [None, False, True, ""], None, [video_input, rotate_opt, mirror_opt, result_output])

if __name__ == "__main__":
    css_data = load_css(CSS_PATH)
    app.launch(css=css_data)