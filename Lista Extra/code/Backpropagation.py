import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X, W1, W2):
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    return A2

def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

def train_neural_network(X_train, y_train, learning_rate, iterations, hidden_size):
    N, input_size = X_train.shape
    W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))
    W2 = np.random.normal(scale=0.5, size=(hidden_size, 1))

    for itr in range(iterations):
        # Feedforward propagation
        A1 = sigmoid(np.dot(X_train, W1))
        A2 = sigmoid(np.dot(A1, W2))

        # Backpropagation
        E2 = A2 - y_train.reshape(-1, 1)
        dW2 = np.dot(A1.T, E2) / N
        E1 = np.dot(E2, W2.T) * A1 * (1 - A1)
        dW1 = np.dot(X_train.T, E1) / N

        # Update weights
        W2 = W2 - learning_rate * dW2
        W1 = W1 - learning_rate * dW1

    return W1, W2

def main():
    # Obter entradas do usuário
    input_size = int(input("Número de entradas: "))
    operation_type = input("Tipo de operação (AND, OR, XOR): ").upper()

    # Criar conjunto de treinamento
    X_train = np.random.randint(2, size=(1000, input_size))
    
    if operation_type == "AND":
        y_train = np.logical_and.reduce(X_train, axis=1)
    elif operation_type == "OR":
        y_train = np.logical_or.reduce(X_train, axis=1)
    elif operation_type == "XOR":
        y_train = np.logical_xor.reduce(X_train, axis=1)

    # Adicionar coluna extra para bias
    X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))

    # Treinar a rede neural
    learning_rate = 0.2
    iterations = 20000
    hidden_size = 8  # Aumente o tamanho da camada oculta
    W1, W2 = train_neural_network(X_train, y_train, learning_rate, iterations, hidden_size)

    # Testar a rede neural
    inputs = np.array([int(input(f"Insira a entrada {i + 1} (0 ou 1): ")) for i in range(input_size)])
    input_with_bias = np.concatenate([inputs, [1]])  # Adicionar bias
    result = round(predict(input_with_bias, W1, W2)[0])

    print(f"Resultado da operação {operation_type} para as entradas {inputs}: {result}")

if __name__ == "__main__":
    main()
