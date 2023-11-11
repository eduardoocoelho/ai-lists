import numpy as np

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Classe da rede neural
class NeuralNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = 1
        self.hidden_size = 4  # Número de neurônios na camada oculta

        # Inicialização aleatória dos pesos
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    def forward(self, inputs):
        # Propagação direta
        self.hidden_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_hidden_output))

        return self.output

    def backward(self, inputs, target, learning_rate):
        # Cálculo do erro
        error = target - self.output

        # Cálculo do delta da camada de saída
        delta_output = error * sigmoid_derivative(self.output)

        # Cálculo do erro da camada oculta
        error_hidden = delta_output.dot(self.weights_hidden_output.T)

        # Cálculo do delta da camada oculta
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        # Atualização dos pesos
        self.weights_hidden_output += np.outer(self.hidden_output, delta_output) * learning_rate
        self.weights_input_hidden += np.outer(inputs, delta_hidden) * learning_rate

    def train(self, inputs, targets, learning_rate, epochs):
        for _ in range(epochs):
            for i in range(len(inputs)):
                input_data = inputs[i]
                target = targets[i]
                self.forward(input_data)
                self.backward(input_data, target, learning_rate)

# Função de treinamento da porta XOR
def train_xor(n):
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(n)
    nn.train(input_data, targets, learning_rate=0.1, epochs=10000)

    for i in range(4):
        output = nn.forward(input_data[i])
        print(f'Input: {input_data[i]}, Predicted: {output[0]}, Actual: {targets[i][0]}')

if __name__ == "__main__":
    n = int(input("Número de entradas: "))
    train_xor(n)
