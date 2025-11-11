import os
import numpy as np
import tensorflow as tf
import pytest

# Définir une fixture pour créer un modèle de test simple
@pytest.fixture
def dummy_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(1, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Test pour vérifier la sauvegarde et le chargement du modèle
def test_save_and_load_model(dummy_model):
    model_path = "test_model.h5"
    
    # Sauvegarder le modèle
    dummy_model.save(model_path)
    
    # Vérifier que le fichier a été créé
    assert os.path.exists(model_path)
    
    # Charger le modèle
    loaded_model = tf.keras.models.load_model(model_path)
    
    # Vérifier que le modèle chargé n'est pas None
    assert loaded_model is not None
    
    # Nettoyer le fichier de modèle
    os.remove(model_path)

# Test pour vérifier le format de la prédiction
def test_prediction_format(dummy_model):
    # Créer une image de test (48x48, 1 canal)
    test_image = np.random.rand(1, 48, 48, 1)
    
    # Obtenir la prédiction
    prediction = dummy_model.predict(test_image)
    
    # Vérifier la forme de la prédiction (doit être de taille 7 pour les 7 émotions)
    assert prediction.shape == (1, 7)
    
    # Vérifier que la somme des probabilités est proche de 1
    assert np.isclose(np.sum(prediction), 1.0)
