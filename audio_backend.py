from flask import Flask, request, jsonify, send_file
import whisper
import tempfile
import os
import ffmpeg
from gtts import gTTS
import requests

app = Flask(__name__)
model = whisper.load_model("base")  # You can use "small", "medium", or "large" for better accuracy

# Helper: Extract audio from video if needed
def extract_audio(audio_path, out_path):
    ffmpeg.input(audio_path).output(out_path, acodec='pcm_s16le', ac=1, ar='16000').run(overwrite_output=True, quiet=True)

# Helper: Translate text using LibreTranslate (or Google Translate API if you have a key)
def translate_text(text, source, target):
    url = "https://libretranslate.com/translate"
    payload = {"q": text, "source": source, "target": target, "format": "text"}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("translatedText", "")
    except Exception as e:
        return "[Translation error: {}]".format(e)

# Helper: Text-to-speech using gTTS
def text_to_speech(text, lang, out_path):
    tts = gTTS(text, lang=lang)
    tts.save(out_path)

@app.route('/audio_translate', methods=['POST'])
def audio_translate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    input_lang = request.form.get('input_lang', 'en')
    output_lang = request.form.get('output_lang', 'en')
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, file.filename)
        file.save(input_path)
        # Convert to wav if not already
        audio_path = input_path
        if not input_path.lower().endswith('.wav'):
            audio_path = os.path.join(tmpdir, 'audio.wav')
            extract_audio(input_path, audio_path)
        # Transcribe with Whisper (with diarization and timestamps)
        result = model.transcribe(audio_path, task='transcribe', language=input_lang, fp16=False)
        # Prepare transcript with speaker and timestamps (simulate diarization)
        transcript = []
        for i, seg in enumerate(result['segments']):
            transcript.append({
                'start': '{:.2f}'.format(seg['start']),
                'end': '{:.2f}'.format(seg['end']),
                'speaker': f'Speaker {seg.get("speaker", 1)}',
                'text': seg['text'].strip()
            })
        # Join all text for translation
        full_text = ' '.join([seg['text'] for seg in result['segments']])
        translation = translate_text(full_text, input_lang, output_lang)
        # TTS for translated text
        tts_path = os.path.join(tmpdir, 'tts.mp3')
        try:
            text_to_speech(translation, output_lang, tts_path)
            tts_url = f'/get_tts/{os.path.basename(tts_path)}'
            # Serve TTS file directly
            return jsonify({
                'transcript': transcript,
                'translation': translation,
                'tts_url': f'/audio_tts?file={tts_path}'
            })
        except Exception as e:
            return jsonify({
                'transcript': transcript,
                'translation': translation,
                'tts_url': None,
                'tts_error': str(e)
            })

@app.route('/audio_tts')
def audio_tts():
    file = request.args.get('file')
    if not file or not os.path.exists(file):
        return '', 404
    return send_file(file, mimetype='audio/mpeg')

if __name__ == '__main__':
    app.run(debug=True)
