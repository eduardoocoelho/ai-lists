import numpy as np

# Criar objeto Perceptron com específicos pesos
class Perceptron:
    def __init__(self, num_inputs):
        self.weights = np.zeros(num_inputs)
        self.bias = 0.0

    # Treinar o Perceptron, recebendo os dados de treinamento, taxa de aprendizado 
    # e o número de iterações do treinamento
    def train(self, inputs, labels, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = labels[i] - prediction
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error

    # Fazer previsões com base nas entradas.
    def predict(self, inputs):
        activation = np.dot(self.weights, inputs) + self.bias
        return 1 if activation >= 0 else 0

# Treinar o Perceptron para a função AND ou OR
def train_perceptron(num_inputs, function_name):
    if function_name == 'AND':
        inputs = np.array([[0] * num_inputs, [0] * (num_inputs - 1) + [1]])
        labels = np.array([0, 1])
    elif function_name == 'OR':
        inputs = np.array([[0] * num_inputs, [1] * num_inputs])
        labels = np.array([0, 1])
    else:
        print("Função desconhecida")
        return

    perceptron = Perceptron(num_inputs)
    perceptron.train(inputs, labels, learning_rate=0.1, num_epochs=1000)

    return perceptron

# Testar o Perceptron treinado
def test_perceptron(perceptron, inputs):
    for input_data in inputs:
        prediction = perceptron.predict(input_data)
        print(f"Entradas: {input_data}, Saída: {prediction}")

if __name__ == "__main__":
    num_inputs = int(input("Digite o número de entradas (2, 10, etc.): "))
    function_name = input("Digite a função a ser resolvida (AND ou OR): ")

    perceptron = train_perceptron(num_inputs, function_name)

    num_test_cases = int(input("Digite o número de casos de teste: "))
    test_inputs = []
    for i in range(num_test_cases):
        test_input = [int(x) for x in input(f"Digite as {num_inputs} entradas separadas por espaços: ").split()]
        test_inputs.append(test_input)

    test_perceptron(perceptron, test_inputs)
