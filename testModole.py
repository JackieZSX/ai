from model import build_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    model = build_model()

    df = pd.read_csv('./train/word_index.csv')
    vec = df['one_hot_vector']
    df2 = pd.read_csv('./train/clean_cut.csv')
    label = df2['label']
    vec_value = vec.values

    vec_list = [[float(x) for x in vec_value[i].strip('[]').split(',')] for i in range(len(vec_value))]
    vec = np.array(vec_list, dtype=float)
    print(vec)
    label_list = [[float(i)] for i in label]
    label = np.array(label_list)

    X_train, X_test, y_train, y_test = train_test_split(vec, label, test_size=0.25, random_state=42)

    model.fit(X_train, y_train, epochs=100)
    test_loss, test_acc = model.evaluate(X_train, y_train)
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1).reshape(-1, 1)
    accuracy = np.sum(predicted_classes == y_test) / len(y_test)
    print(accuracy)
    model.save('./model.h5')
