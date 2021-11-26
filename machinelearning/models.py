from numpy.lib.function_base import gradient
import nn
import sys


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        Deberiais obtener el producto escalar (o producto punto) que es "equivalente" a la distancia del coseno
        """
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        Dependiendo del valor del coseno devolvera 1 o -1

        Returns: 1 or -1
        """
        # Obtenemos el valor de la distancia.
        if nn.as_scalar(self.run(x)) >= 0:  # Distancia coseno no negativa
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        Hasta que TODOS los ejemplos del train esten bien clasificados. Es decir, hasta que la clase predicha en se corresponda con la real en TODOS los ejemplos del train
        """
        error = True

        while error:
            pred = []
            for X, Y in dataset.iterate_once(1): #hacemos un pasada por todas las instancias del dataset
                if self.get_prediction(X) == nn.as_scalar(Y): #el valor de clase predicho y el real son los mismos
                    pass
                else:
                    pred.append(False) #flag para avisar de que no ha acertado todas las predicciones y de que debemos volver a iterar el dataset completo
                    nn.Parameter.update(self.w, X, nn.as_scalar(Y)) #actualizamos los pesos

            if not False in pred:
                error = False


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    NO ES CLASIFICACION, ES REGRESION. ES DECIR; APRENDER UNA FUNCION.
    SI ME DAN X TENGO QUE APRENDER A OBTENER LA MISMA Y QUE EN LA FUNCION ORIGINAL DE LA QUE QUIERO APRENDER
    """

    def __init__(self):
        # Initialize your model parameters here
        # For example:

        self.batch_size = 10  # He probado varios valores.
        self.w0 = nn.Parameter(1, 20) #pesos para la entrada
        self.b0 = nn.Parameter(1, 20) #bias para la entrada
        self.w1 = nn.Parameter(20, 1) #tamaño de la capa oculta de 20
        self.b1 = nn.Parameter(1, 1)

        self.lr = -0.01

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1). En este caso cada ejemplo solo est� compuesto por un rasgo
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values.
            Como es un modelo de regresion, cada valor y tambien tendra un unico valor
        """
        "*** YOUR CODE HERE ***"
        
        xW_0 = nn.Linear(x, self.w0) #primera capa sin el bias
        salida_capa0 = nn.AddBias(xW_0, self.b0) #salida de la primera capa
        salida_capa0_relu = nn.ReLU(salida_capa0)  # Salida computando ReLu (No lineal)

        xW_1 = nn.Linear(salida_capa0_relu, self.w1) #segunda capa sin el bias
        salida_capa1 = nn.AddBias(xW_1, self.b1) #salida de la segunda capa

        return salida_capa1

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
                ----> ES FACIL COPIA Y PEGA ESTO Y ANNADE LA VARIABLE QUE HACE FALTA PARA CALCULAR EL ERROR 
                return nn.SquareLoss(self.run(x),ANNADE LA VARIABLE QUE ES NECESARIA AQUI), para medir el error, necesitas comparar el resultado de tu prediccion con .... que?
        """
        prediccion_y = self.run(x)
        return nn.SquareLoss(prediccion_y, y) #calculamos el error comparando el valor de y predicho con el gold standar

    def train(self, dataset):
        """
        Trains the model.

        """

        batch_size = self.batch_size
        total_loss = sys.maxsize
        parametros = [self.w0, self.w1, self.b0, self.b1]
        while total_loss > 0.02:
            # ITERAR SOBRE EL TRAIN EN LOTES MARCADOS POR EL BATCH SIZE COMO HABEIS HECHO EN LOS OTROS EJERCICIOS
            # ACTUALIZAR LOS PESOS EN BASE AL ERROR loss = self.get_loss(x, y) QUE RECORDAD QUE GENERA
            # UNA FUNCION DE LA LA CUAL SE  PUEDE CALCULAR LA DERIVADA (GRADIENTE)

            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y) #calcular el error 
                gradientes = nn.gradients(loss, parametros) #devolvera los gradientes de la perdida con respecto a los parametros
                for idx, param in enumerate(parametros):
                    param.update(gradientes[idx],self.lr) #actualizamos los parametros en base al gradiente y el learning rate
        
            total_loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))  # AQUI SE CALCULA OTRA VEZ EL ERROR PERO SOBRE TODO EL TRAIN A LA VEZ (CUIDADO!! NO ES LO MISMO el x de antes QUE dataset.x)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        # TEN ENCUENTA QUE TIENES 10 CLASES, ASI QUE LA ULTIMA CAPA TENDRA UNA SALIDA DE 10 VALORES,
        # UN VALOR POR CADA CLASE

        # TAMANO EQUIVALENTE AL NUMERO DE CLASES DADO QUE QUIERES OBTENER 10 "COSENOS"
        output_size = 10            
        "*** YOUR CODE HERE ***"

        self.w0 = nn.Parameter(784, 256) #784 filas por 265 columnas
        self.b0 = nn.Parameter(1, 256)
        self.w1 = nn.Parameter(256, 128)
        self.b1 = nn.Parameter(1, 128)
        self.w2 = nn.Parameter(128, 64)
        self.b2 = nn.Parameter(1, 64)
        self.w3 = nn.Parameter(64, 32)
        self.b3 = nn.Parameter(1, 32)
        self.w4 = nn.Parameter(32, 10)
        self.b4 = nn.Parameter(1, 10)

        self.lr = -0.1 #queremos un learning rate mayor que el anterior, para dar saltos mayores. Si ponemos un número menos tardará muchas más epochs en llegar al objetivo

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
            output_size = 10 # TAMANO EQUIVALENTE AL NUMERO DE CLASES DADO QUE QUIERES OBTENER 10 "COSENOS"
        """
        "*** YOUR CODE HERE ***"

        xW_0 = nn.Linear(x, self.w0) #primera capa sin el bias
        salida_capa0 = nn.AddBias(xW_0, self.b0) #salida de la primera capa
        salida_capa0_relu = nn.ReLU(salida_capa0)  # Salida computando ReLu (No lineal)

        xW_1 = nn.Linear(salida_capa0_relu, self.w1) #segunda capa sin el bias
        salida_capa1 = nn.AddBias(xW_1, self.b1) #salida de la segunda capa
        salida_capa1_relu = nn.ReLU(salida_capa1)  # Salida computando ReLu (No lineal)

        xW_2 = nn.Linear(salida_capa1_relu, self.w2)
        salida_capa2 = nn.AddBias(xW_2, self.b2)
        salida_capa2_relu = nn.ReLU(salida_capa2)  

        xW_3 = nn.Linear(salida_capa2_relu, self.w3)
        salida_capa3 = nn.AddBias(xW_3, self.b3)
        salida_capa3_relu = nn.ReLU(salida_capa3)

        xW_4 = nn.Linear(salida_capa3_relu, self.w4)
        salida_capa4 = nn.AddBias(xW_4, self.b4)

        return salida_capa4

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).
        POR EJEMPLO: [0,0,0,0,0,1,0,0,0,0,0] seria la y correspondiente al 5
                     [0,1,0,0,0,0,0,0,0,0,0] seria la y correspondiente al 1

        EN ESTE CASO ESTAMOS HABLANDO DE MULTICLASS, ASI QUE TIENES QUE CALCULAR 
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"  # NO ES NECESARIO QUE LO IMPLEMENTEIS, SE OS DA HECHO
        # COMO VEIS LLAMA AL RUN PARA OBTENER POR CADA BATCH
        return nn.SoftmaxLoss(self.run(x), y)
        # LOS 10 VALORES DEL "COSENO". TENIENDO EL Y REAL POR CADA EJEMPLO
        # APLICA SOFTMAX PARA CALCULAR EL COSENO MAX
        # (COMO UNA PROBABILIDAD), Y ESA SERA SU PREDICCION,
        # LA CLASE QUE MUESTRE EL MAYOR COSENO, Y LUEGO LA COMPARARA CON Y

    def train(self, dataset):
        """
        Trains the model.
        EN ESTE CASO EN VEZ DE PARAR CUANDO EL ERROR SEA MENOR QUE UN VALOR O NO HAYA ERROR (CONVERGENCIA),
        SE PUEDE HACER ALGO SIMILAR QUE ES EN NUMERO DE ACIERTOS. EL VALIDATION ACCURACY
        NO LO TENEIS QUE IMPLEMENTAR, PERO SABED QUE EMPLEA EL RESULTADO DEL SOFTMAX PARA CALCULAR
        EL NUM DE EJEMPLOS DEL TRAIN QUE SE HAN CLASIFICADO CORRECTAMENTE 
        """

        parametros = [self.w0, self.w1, self.b0, self.b1]
        batch_size = 50
        while dataset.get_validation_accuracy() < 0.97:
            # ITERAR SOBRE EL TRAIN EN LOTES MARCADOS POR EL BATCH SIZE COMO HABEIS HECHO EN LOS OTROS EJERCICIOS
            # ACTUALIZAR LOS PESOS EN BASE AL ERROR loss = self.get_loss(x, y) QUE RECORDAD QUE GENERA
            # UNA FUNCION DE LA LA CUAL SE  PUEDE CALCULAR LA DERIVADA (GRADIENTE)
            "*** YOUR CODE HERE ***"
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gradientes = nn.gradients(loss, parametros)
                for idx, param in enumerate(parametros):
                    param.update(gradientes[idx],self.lr) #actualizamos los parametros en base al gradiente y el learning rate



class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
