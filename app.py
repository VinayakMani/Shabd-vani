

import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import langid
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from deep_translator import GoogleTranslator
from gtts import gTTS
import os
from moviepy.editor import VideoFileClip

st.set_page_config(page_title="VOICE-LINGUA", page_icon=":microphone:")

primary_color = "#0072C6"  # Blue
secondary_color = "#F5F5F5"  # Light gray
background_color = "#FFFF00"  # White

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {background_color}
    }}
    h1 {{
        color: {primary_color};
    }}
    .stFileUploader {{
        color: {primary_color};
    }}
    .stHeader {{
        color: {primary_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    # st.image("/content/Logo.jpg")
    st.title("VOICE-LINGUA")
    if "option" not in st.session_state:
        st.session_state.option = "Speech Recognition"
    option = st.radio("Select an option:", ("Speech Recognition", "Translation", " Speech Generation", "Audio Extraction","Summarization"), key="option")

lang_code_mapping = {
    "en": "eng_Latn",   # English
    "hi": "hin_Deva",   # Hindi
    "fr": "fra_Latn",   # French
    "de": "deu_Latn",   # German
    "es": "spa_Latn",   # Spanish
    "it": "ita_Latn",   # Italian
    "pt": "por_Latn",   # Portuguese
    "ru": "rus_Cyrl",   # Russian
    "ja": "jpn_Jpan",   # Japanese
    "ko": "kor_Hang",   # Korean
    "zh": "chi_Hans",   # Simplified Chinese
    "ar": "ara_Arab",   # Arabic
    "tr": "tur_Latn",   # Turkish
    "nl": "nld_Latn",   # Dutch
    "pl": "pol_Latn",   # Polish
    "uk": "ukr_Cyrl",   # Ukrainian
    "vi": "vie_Latn",   # Vietnamese
    "th": "tha_Thai",   # Thai
    "id": "ind_Latn",   # Indonesian
    "ms": "mal_Mlym",   # Malay
    "ta": "tam_Taml",   # Tamil
    "te": "tel_Telu",   # Telugu
    "mr": "mar_Deva",   # Marathi
    "bn": "ben_Beng",   # Bengali
    "gu": "guj_Gujr",   # Gujarati
    "kn": "kan_Knda",   # Kannada
    "pa": "pan_Guru",   # Punjabi
    "ur": "urd_Arab",   # Urdu
    "si": "sin_Sinh",   # Sinhala
    "mt": "mlt_Latn",   # Maltese
    "fi": "fin_Latn",   # Finnish
    "sv": "swe_Latn",   # Swedish
    "da": "dan_Latn",   # Danish
    "no": "nor_Latn",   # Norwegian
    "hu": "hun_Latn",   # Hungarian
    "he": "heb_Hebr",   # Hebrew
    "el": "ell_Grek",   # Greek
    "ro": "rom_Latn",   # Romanian
    "bg": "bul_Cyrl",   # Bulgarian
    "sr": "srp_Cyrl",   # Serbian
    "cs": "ces_Latn",   # Czech
    "sk": "slk_Latn",   # Slovak
    "hr": "hrv_Latn",   # Croatian
    "fa": "pes_Arab",   # Persian
    "lt": "lit_Latn",   # Lithuanian
    "lv": "lav_Latn",   # Latvian
    "et": "est_Latn",   # Estonian
    "sw": "swa_Latn",   # Swahili
    "sl": "slv_Latn"    # Slovenian
}


lang_code_mapping2 = {
    "en": "en",   # English
    "hi": "hi",   # Hindi
    "fr": "fr",   # French
    "de": "de",   # German
    "es": "es",   # Spanish
    "it": "it",   # Italian
    "pt": "pt",   # Portuguese
    "ru": "ru",   # Russian
    "ja": "ja",   # Japanese
    "ko": "ko",   # Korean
    "zh": "zh",   # Simplified Chinese
    "ar": "ar",   # Arabic
    "tr": "tr",   # Turkish
    "nl": "nl",   # Dutch
    "pl": "pl",   # Polish
    "uk": "uk",   # Ukrainian
    "vi": "vi",   # Vietnamese
    "th": "th",   # Thai
    "id": "id",   # Indonesian
    "ms": "ms",   # Malay
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "mr": "mr",   # Marathi
    "bn": "bn",   # Bengali
    "gu": "gu",   # Gujarati
    "kn": "kn",   # Kannada
    "pa": "pa",   # Punjabi
    "ur": "ur",   # Urdu
    "si": "si",   # Sinhala
    "mt": "mt",   # Maltese
    "fi": "fi",   # Finnish
    "sv": "sv",   # Swedish
    "da": "da",   # Danish
    "no": "no",   # Norwegian
    "hu": "hu",   # Hungarian
    "he": "he",   # Hebrew
    "el": "el",   # Greek
    "ro": "ro",   # Romanian
    "bg": "bg",   # Bulgarian
    "sr": "sr",   # Serbian
    "cs": "cs",   # Czech
    "sk": "sk",   # Slovak
    "hr": "hr",   # Croatian
    "fa": "fa",   # Persian
    "lt": "lt",   # Lithuanian
    "lv": "lv",   # Latvian
    "et": "et",   # Estonian
    "sw": "sw",   # Swahili
    "sl": "sl"    # Slovenian
}

def detect_language_nllb(text):
    # Detect language using langid
    lang_code, _ = langid.classify(text)
    print(f"Detected Language Code: {lang_code}")

    nllb_code = lang_code_mapping.get(lang_code, "eng_Latn")  # Default to "eng_Latn" if not found
    return nllb_code

def translate_and_generate_audio(text, target_lang, filename):

    translator = GoogleTranslator(source='auto', target=target_lang)
    translated_text = translator.translate(text)
    tts = gTTS(text=translated_text, lang=target_lang)
    tts.save(filename)
    print(f"Audio file saved as: {filename}")

if option == "Speech Recognition":

    st.title("Speech Recognition")
    st.subheader("Upload an audio file to transcribe:")

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_file = st.file_uploader("", type=["wav", "mp3", "m4a", "mpeg"], key="speech_to_text")

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
    else:
        uploaded_file = st.session_state.get("uploaded_file", None)

    if uploaded_file is not None:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(uploaded_file, format="audio/wav")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

        def transcribe_audio(audio):
            result = pipe(audio)
            transcript = result['text']
            return transcript

        transcript = transcribe_audio("uploaded_audio.wav")

        # Detect language of the transcribed text
        nllb_lang_code = detect_language_nllb(transcript)
        print(f"NLLB-200 Language Code: {nllb_lang_code}")

        st.header("Transcription:")
        st.write(transcript)
        st.subheader("Language Code:")
        st.write(nllb_lang_code)

elif option == "Translation":

    st.title("Translation")
    st.subheader("Translate text from one language to another:")

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    if "src_lang" not in st.session_state:
        st.session_state.src_lang = "en"

    if "target_lang" not in st.session_state:
        st.session_state.target_lang = "en"

    input_text = st.text_area("Enter text to translate:", value=st.session_state.input_text, key="translation_input")
    src_lang = st.selectbox("Select source language:", list(lang_code_mapping.keys()), index=list(lang_code_mapping.keys()).index(st.session_state.src_lang), key="translation_src_lang")
    target_lang = st.selectbox("Select target language:", list(lang_code_mapping.keys()), index=list(lang_code_mapping.keys()).index(st.session_state.target_lang), key="translation_target_lang")

    if st.button("Translate"):
        st.session_state.input_text = input_text
        st.session_state.src_lang = src_lang
        st.session_state.target_lang = target_lang

        src_lang_code = lang_code_mapping[src_lang]
        target_lang_code = lang_code_mapping[target_lang]

        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

        def translate_text(input_text, src_lang_code, target_lang_code):
            translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=src_lang_code, tgt_lang=target_lang_code)
            translated_text = translator(input_text)[0]['translation_text']
            return translated_text

        output_text = translate_text(input_text, src_lang_code, target_lang_code)

        st.header("Translated Text:")
        st.write(output_text)

elif option == " Speech Generation":
    st.title(" Speech Generation")
    st.subheader("Translate text and generate audio in the target language:")

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    if "target_lang" not in st.session_state:
        st.session_state.target_lang = "en"

    input_text = st.text_area("Enter text to translate:", value=st.session_state.input_text, key="translation_input")
    target_lang = st.selectbox("Select target language:", list(lang_code_mapping2.keys()), index=list(lang_code_mapping2.keys()).index(st.session_state.target_lang), key="translation_target_lang2")

    if st.button("Speech Generation"):
        st.session_state.input_text = input_text
        st.session_state.target_lang = target_lang

        if target_lang in lang_code_mapping2:
            target_lang_code2 = lang_code_mapping2[target_lang]
        else:
            st.error("Invalid language code. Please select a valid target language.")


        filename = "output.mp3"
        translate_and_generate_audio(input_text, target_lang_code2, filename)

        st.success("Audio file generated successfully!")
        st.audio(filename, format="audio/mp3")

elif option == "Audio Extraction":

    st.title("Audio Extraction")
    st.subheader("Extract audio from a video file:")

    video_file = st.text_input("Enter the path to the video file:", key="video_file")

    if st.button("Extract Audio"):

        import os
        if not os.path.isfile(video_file):
            st.error("Error: The provided path is not a file.")
        else:
            # Load the video file
            video = VideoFileClip(video_file)

            # Extract the audio from the video
            audio = video.audio

            # Write the audio to a file
            audio_file = "output_audio.mp3"
            audio.write_audiofile(audio_file)

            st.success("Audio extracted successfully!")

            download_audio = st.button("Download the extracted audio file")
            if download_audio:

                with open(audio_file, "rb") as file:

                    audio_data = file.read()

                st.markdown(f"Content-Type: audio/mpeg")
                st.markdown(f"Content-Disposition: attachment; filename={audio_file}")
                st.markdown(f"Content-Length: {len(audio_data)}")
                st.write(audio_data)
            else:
                st.info("Audio file not downloaded.")
elif option == "Summarization":
    st.title("Text Summarizer")

    input_text = st.text_area("Enter the text to summarize:", height=200)
    input_text_words = input_text.split()
    st.subheader("Number of words in the input text:")
    st.write(len(input_text_words))

    min1 = max(10, int(len(input_text_words) / 3))  # Ensure min1 is at least 10
    max1 = int(len(input_text_words) / 2)
    min2 = int(len(input_text_words) / 2)
    max2 = len(input_text_words)

    min_length = st.slider("Choose the minimum summary length", min_value=min1, max_value=max1, value=min1, step=5)
    max_length = st.slider("Choose the maximum summary length", min_value=min2, max_value=max2, value=max2, step=10)

    if st.button("Summarize"):
        summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
        summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        st.subheader("Summary:")
        st.write(summary)



