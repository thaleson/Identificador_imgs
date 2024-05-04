import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Carregar o modelo treinado, ignorando quaisquer argumentos de otimizador
model = tf.keras.models.load_model(r'C:\Users\thale\Identificador_imgs\modelo_cifar10 .h5', custom_objects={'Optimizer': tf.keras.optimizers.Adam})

# Lista de nomes de classe correspondentes aos números de classe
class_names = [
    'Avião', 'Automóvel', 'Pássaro', 'Gato', 'Cervo',
    'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão'
]

# Função para fazer previsões com o modelo
def predict_image(img):
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    try:
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100
        return predicted_class_name, confidence
    except Exception as e:
        st.error("Ocorreu um erro durante a previsão.")
        st.error(e)
        return None, None

# Estrutura do site
def main():
    st.title('Classificador de Imagens CIFAR-10')
    st.write("""
    Faça upload de uma imagem para receber uma previsão de classe.
    """)
    
    st.write("Este é um classificador de imagens treinado no conjunto de dados CIFAR-10, que contém 10 classes diferentes de objetos.")
    st.write("**Classes que o modelo pode prever:**")
    st.write(", ".join(class_names))
    st.write("Por favor, note que este modelo pode não ser 100% preciso e pode cometer erros.")
    
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Abrir a imagem
        img = Image.open(uploaded_file)
        st.image(img, caption='Imagem enviada', channels='RGB', use_column_width=True)
        
        # Redimensionar a imagem para o tamanho esperado pelo modelo
        img_resized = img.resize((32, 32))
        
        # Fazer a previsão com a imagem redimensionada
        predicted_class, confidence = predict_image(img_resized)
        if predicted_class is not None and confidence is not None:
            st.write(f"Esta imagem é provavelmente um(a) {predicted_class} com uma confiança de {confidence:.2f}%.")

# Executar o aplicativo
if __name__ == '__main__':
    main()
