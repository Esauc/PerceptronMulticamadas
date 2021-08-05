# -*- coding: utf-8 -*-
import numpy as np
import math

class MLP:
    
    def __init__(self, input_values, output_values, layers, learning_rate=0.1, precision=0.000001):
       ones_column = np.ones((len(input_values), 1)) * -1
       self.input_values = np.append(ones_column, input_values, axis=1)
       self.output_values = output_values
       self.learning_rate = learning_rate
       self.precision = precision
       
       self.W = []
       neuron_input = self.input_values.shape[1]

       for i in range(len(layers)):
           self.W.append(np.random.rand(layers[i], neuron_input))
           neuron_input = layers[i] + 1
          
       self.epochs = 0
       self.eqms = []
       #print(len(self.W))
       #print(len(self.W[0]))
       #print(len(self.W[0][0]))
       
    def train(self):

        print(f'Initializing the train process....')
        error = True
        
        while error:
            #print(f'[EPOCH] {self.epochs}')
            error = False
            
            eqm_previous = self.eqm()
          
            for x, d in zip(self.input_values, self.output_values):

                I1 = np.dot(self.W[0], x)
                Y1 = np.zeros(I1.shape)
                for i in range(Y1.shape[0]):
                    Y1[i] = self.g(I1[i])
                Y1 = np.append(-1, Y1)
                
                I2 = np.dot(self.W[1], Y1)
                Y2 = np.zeros(I2.shape)
                for i in range(Y2.shape[0]):
                    Y2[i] = self.g(I2[i])                  

                # BACKPROPAGATION :
    
                # Calcula gradiente local para cada neurônio na camada de saída:
                grad2 = np.zeros(Y2.shape[0])

                for i in range(Y2.shape[0]) :
                  grad2[i] = (d[i] - Y2[i]) * self.dg(I2[i])
                
                # Atualiza os pesos para cada ligação de cada neurônio na saída:
                for i in range(Y2.shape[0]) :

                  for j in range(Y1.shape[0]) : 
                    self.W[1][i][j] += self.learning_rate * grad2[i] * Y1[j]

                # Calcula gradiente local para cada neurônio na camada de entrada:
                grad1 = np.zeros(I1.shape[0])

                # Faz o somatório 
                for i in range(I1.shape[0]) :
                  s = 0 # <-- Variavel auxiliar para somatório

                  for j in range(Y2.shape[0]) :
                    s += grad2[j] * self.W[1][j][i]

                  grad1[i] = -s *self.dg(I1[i])

                # Atualiza os pesos para cada ligação de cada neurônio na entrada
                for i in range(len(x)) : # para cada entrada (x0 ~ x4)

                  for j in range(I1.shape[0]) : 
                    self.W[0][j][i] += self.learning_rate * grad1[j] * x[i]

            
            eqm_actual = self.eqm()
            self.eqms.append(eqm_actual)
            
            self.epochs += 1

            print(eqm_actual)

            if abs(eqm_actual - eqm_previous) > self.precision:
              error = True
                
        print("final do treino")
        print(f'[EPOCH] {self.epochs}')
        

    def eqm(self):
        
        eq = 0
        
        for x, d in zip(self.input_values, self.output_values):
            I1 = np.dot(self.W[0], x)
            Y1 = np.zeros(I1.shape)
            for i in range(Y1.shape[0]):
                Y1[i] = self.g(I1[i])
            Y1 = np.append(-1, Y1)
            
            I2 = np.dot(self.W[1], Y1)
            Y2 = np.zeros(I2.shape)
            for i in range(Y2.shape[0]):
                Y2[i] = self.g(I2[i])
                
            eq += 0.5 * sum((d - Y2) ** 2)
            
        return eq/len(self.output_values)

    def evaluate(self, ix):

        ix = np.append(-1, ix)

        for i in range(ix.shape[0]) : 
            I1 = np.dot(self.W[0], ix)
            Y1 = np.zeros(I1.shape)
            for i in range(Y1.shape[0]):
                Y1[i] = self.g(I1[i])
            Y1 = np.append(-1, Y1)
            
            I2 = np.dot(self.W[1], Y1)
            Y2 = np.zeros(I2.shape)
            for i in range(Y2.shape[0]):
                Y2[i] = self.g(I2[i])

            # Pós-processamento:
            for i in range(Y2.shape[0]) :
              if Y2[i] >= 0.5 :
                Y2[i] = 1
              else :
                Y2[i] = 0

            return Y2
                 
    # Não sei programar python, estava dando erro quando eu ia usar as funções de ativação lá na classe
    # Por isso exclui a classe e fiz aqui mesmo, aí deu certo.
    # O erro era sobre os 'parâmetros posicionais' da função, dizia que estava faltando um.
    # Provavelemente Alguma coisa da sintaxe do python que eu não conheço bem.
    # Peço desculpa pela bagunça, se fosse em Java o código teria ficado mais organizado rs

    # Função Logística
    def g(self, u) :
      #u *= 2
      return (1 / (1 + math.e **(-u)))

    # Derivada da função Logística
    def dg(self, u) :
      #u *= 2
      return (self.g(u) * (1 - self.g(u)))
      

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        