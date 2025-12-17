# streamlit_app.py
import streamlit as st
import numpy as np
import pickle
import os
import json
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===================== КОНФИГУРАЦИЯ СТРАНИЦЫ =====================
st.set_page_config(
    page_title="🎭 Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== СТИЛИ =====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .emotion-card {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .stTextArea textarea {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ===================== ЭМОДЗИ И ЦВЕТА ДЛЯ ЭМОЦИЙ =====================
EMOTION_CONFIG = {
    "neutral": {"emoji": "😐", "color": "#95a5a6", "bg": "#ecf0f1"},
    "joy": {"emoji": "😄", "color": "#f1c40f", "bg": "#fef9e7"},
    "sadness": {"emoji": "😢", "color": "#3498db", "bg": "#ebf5fb"},
    "anger": {"emoji": "😠", "color": "#e74c3c", "bg": "#fdedec"},
    "fear": {"emoji": "😨", "color": "#9b59b6", "bg": "#f5eef8"},
    "surprise": {"emoji": "😲", "color": "#e67e22", "bg": "#fef5e7"}
}

# ===================== ЗАГРУЗКА МОДЕЛИ =====================
@st.cache_resource
def load_model_and_tokenizer():
    """Загрузка модели и токенизатора (кэшируется)"""
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "ml_service", "models", "cnn_model.h5")
    TOKENIZER_PATH = os.path.join(BASE_DIR, "ml_service", "models", "tokenizer.pickle")
    CONFIG_PATH = os.path.join(BASE_DIR, "ml_service", "models", "model_config.json")
    
    # Дефолтные значения
    config = {
        "max_length": 50,
        "padding": "post",
        "class_names": {0: "neutral", 1: "joy", 2: "sadness", 3: "anger", 4: "fear", 5: "surprise"}
    }
    
    # Загрузка конфигурации
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
            if 'class_names' in loaded_config:
                config['class_names'] = {int(k): v for k, v in loaded_config['class_names'].items()}
    
    # Загрузка модели
    model = keras.models.load_model(MODEL_PATH, compile=False)
    
    # Загрузка токенизатора
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer, config

# ===================== ФУНКЦИЯ ПРЕДСКАЗАНИЯ =====================
def predict_emotion(text: str, model, tokenizer, config) -> dict:
    """Предсказание эмоции для текста"""
    
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=config['max_length'], padding=config['padding'])
    
    pred = model.predict(pad, verbose=0)[0]
    
    predicted_class = int(np.argmax(pred))
    confidence = float(np.max(pred))
    emotion = config['class_names'].get(predicted_class, f"unknown_{predicted_class}")
    
    # Все вероятности
    all_probs = {config['class_names'][i]: float(pred[i]) for i in range(len(pred))}
    
    return {
        "emotion": emotion,
        "confidence": confidence,
        "all_probabilities": all_probs
    }

# ===================== ГЛАВНАЯ ФУНКЦИЯ =====================
def main():
    # Заголовок
    st.markdown('<h1 class="main-header">🎭 Emotion Detector</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Определение эмоций в тексте с помощью нейросети</p>", unsafe_allow_html=True)
    
    # Загрузка модели
    try:
        with st.spinner("🔄 Загрузка модели..."):
            model, tokenizer, config = load_model_and_tokenizer()
        
        # Sidebar с информацией
        with st.sidebar:
            st.header("ℹ️ О приложении")
            st.success("✅ Модель загружена")
            
            st.markdown("---")
            st.subheader("📊 Поддерживаемые эмоции")
            for emotion, cfg in EMOTION_CONFIG.items():
                st.markdown(f"{cfg['emoji']} **{emotion.capitalize()}**")
            
            st.markdown("---")
            st.subheader("⚙️ Конфигурация")
            st.json({
                "max_length": config['max_length'],
                "padding": config['padding'],
                "tensorflow": tf.__version__
            })
            
            st.markdown("---")
            st.subheader("📝 Примеры")
            example_texts = [
                "I'm so happy today!",
                "This makes me really angry",
                "I feel so sad and lonely",
                "Wow, I didn't expect that!",
                "I'm scared of what might happen",
                "It's just a normal day"
            ]
            
            if st.button("🎲 Случайный пример"):
                st.session_state.example_text = np.random.choice(example_texts)
        
        # Основной контент
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("✍️ Введите текст")
            
            # Получаем текст из примера или пустой
            default_text = st.session_state.get('example_text', '')
            
            text_input = st.text_area(
                "Текст для анализа:",
                value=default_text,
                height=150,
                placeholder="Введите текст для определения эмоции...",
                key="text_input"
            )
            
            # Очистка примера после использования
            if 'example_text' in st.session_state:
                del st.session_state.example_text
            
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                analyze_btn = st.button("🔍 Анализировать", type="primary", use_container_width=True)
            
            with col_btn2:
                clear_btn = st.button("🗑️ Очистить", use_container_width=True)
            
            with col_btn3:
                batch_mode = st.checkbox("📦 Batch режим")
        
        # Анализ
        if analyze_btn and text_input.strip():
            with st.spinner("🧠 Анализируем..."):
                time.sleep(0.3)  # Небольшая задержка для эффекта
                result = predict_emotion(text_input, model, tokenizer, config)
            
            with col2:
                st.subheader("🎯 Результат")
                
                emotion = result['emotion']
                confidence = result['confidence']
                cfg = EMOTION_CONFIG.get(emotion, {"emoji": "❓", "color": "#666", "bg": "#f0f0f0"})
                
                # Карточка с результатом
                st.markdown(f"""
                <div style="
                    background: {cfg['bg']};
                    border: 3px solid {cfg['color']};
                    border-radius: 1rem;
                    padding: 2rem;
                    text-align: center;
                    margin: 1rem 0;
                ">
                    <div style="font-size: 4rem;">{cfg['emoji']}</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {cfg['color']}; margin: 0.5rem 0;">
                        {emotion.upper()}
                    </div>
                    <div style="font-size: 1.2rem; color: #666;">
                        Уверенность: {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Прогресс-бар уверенности
                st.progress(confidence)
            
            # График всех вероятностей
            st.subheader("📊 Распределение вероятностей")
            
            probs = result['all_probabilities']
            
            # Сортируем по вероятности
            sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            
            for emotion_name, prob in sorted_probs.items():
                cfg = EMOTION_CONFIG.get(emotion_name, {"emoji": "❓", "color": "#666"})
                
                col_emoji, col_name, col_bar, col_val = st.columns([0.5, 1.5, 6, 1])
                
                with col_emoji:
                    st.markdown(f"<span style='font-size: 1.5rem;'>{cfg['emoji']}</span>", unsafe_allow_html=True)
                
                with col_name:
                    st.markdown(f"**{emotion_name.capitalize()}**")
                
                with col_bar:
                    st.progress(prob)
                
                with col_val:
                    st.markdown(f"`{prob:.1%}`")
        
        elif analyze_btn:
            st.warning("⚠️ Введите текст для анализа!")
        
        # Batch режим
        if batch_mode:
            st.markdown("---")
            st.subheader("📦 Пакетный анализ")
            
            batch_input = st.text_area(
                "Введите несколько текстов (каждый с новой строки):",
                height=200,
                placeholder="Текст 1\nТекст 2\nТекст 3..."
            )
            
            if st.button("🚀 Анализировать все", type="primary"):
                texts = [t.strip() for t in batch_input.split('\n') if t.strip()]
                
                if texts:
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, text in enumerate(texts):
                        result = predict_emotion(text, model, tokenizer, config)
                        results.append({
                            "Текст": text[:50] + "..." if len(text) > 50 else text,
                            "Эмоция": f"{EMOTION_CONFIG.get(result['emotion'], {}).get('emoji', '❓')} {result['emotion']}",
                            "Уверенность": f"{result['confidence']:.1%}"
                        })
                        progress_bar.progress((i + 1) / len(texts))
                    
                    st.dataframe(results, use_container_width=True)
                    
                    # Статистика
                    st.subheader("📈 Статистика")
                    emotion_counts = {}
                    for r in results:
                        em = r['Эмоция'].split()[1]  # Убираем эмодзи
                        emotion_counts[em] = emotion_counts.get(em, 0) + 1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(emotion_counts)
                    with col2:
                        st.json(emotion_counts)
                else:
                    st.warning("⚠️ Введите тексты для анализа!")
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: #999;'>Made with ❤️ using Streamlit & TensorFlow</p>",
            unsafe_allow_html=True
        )
    
    except FileNotFoundError as e:
        st.error(f"❌ Файлы модели не найдены: {e}")
        st.info("📁 Убедитесь, что папка `models/` содержит: `cnn_model.h5`, `tokenizer.pickle`")
    
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()