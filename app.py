from flask import Flask, request, send_file, jsonify
import whisper
import tempfile
import os
import ffmpeg

app = Flask(__name__)
model = whisper.load_model("base")  # You can use "small", "medium", or "large" for better accuracy

def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16000').run(overwrite_output=True, quiet=True)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, file.filename)
        file.save(input_path)
        # Extract audio if video
        audio_path = input_path
        if file.filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
            audio_path = os.path.join(tmpdir, 'audio.wav')
            extract_audio(input_path, audio_path)
        result = model.transcribe(audio_path, fp16=False)
        srt_path = os.path.join(tmpdir, 'subtitles.srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(result['srt'])
        return send_file(srt_path, as_attachment=True, download_name='subtitles.srt', mimetype='application/octet-stream')

if __name__ == '__main__':
    app.run(debug=True)
