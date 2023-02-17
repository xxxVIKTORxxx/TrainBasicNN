import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
from random import randint
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab

screet_red = randint(1,225)
screen_green = randint(1,225)
screen_blue = randint(1,255)
Screen_col = (screet_red, screen_green, screen_blue)
SIZE = 1200, 600


######
# original code 
rg = np.random.default_rng()

def generate_data(n_features, n_values):
    features = rg.random((n_features, n_values))
    weights = rg.random((1, n_values))[0]
    targets = np.random.choice([0,1], n_features)
    data = pd.DataFrame(features, columns=["x0", "x1", "x2"])
    data["targets"] = targets
    return data, weights
    
data, weights = generate_data(4,3)
print(data)
print(weights) #


bias = 0.5
l_rate = 0.1
epochs = 500
epoch_loss = []

def get_weighted_sum(feature, weights, bias):
    return np.dot(feature, weights) + bias

def sigmoid(w_sum):
    return 1/(1+np.exp(-w_sum))

def cross_entropy(target, prediction):
    return -(target*np.log10(prediction) + (1-target)*np.log10(1-prediction))

def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x, w in zip(feature, weights):
        new_w = w + l_rate*(target-prediction)*x
        new_weights.append(new_w)
    return new_weights

def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate*(target-prediction)

def train_model(data, weights, bias, l_rate, epochs):
    ee=0
    for e in range(epochs):
        individual_loss = []
        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum = get_weighted_sum(feature, weights, bias)
            prediction = sigmoid(w_sum)
            loss = cross_entropy(target, prediction)
            individual_loss.append(loss)
            # gradient descent
            weights = update_weights(weights, l_rate, target, prediction, feature)
            bias = update_bias(bias, l_rate, target, prediction)

        average_loss = sum(individual_loss)/len(individual_loss)
        epoch_loss.append(average_loss)
        print("**************************")
        print("epoch", e)
        print(average_loss)

        # builtin LOASS drawing
        ee +=1
        if ee %10 == 0:
            df = pd.DataFrame(epoch_loss)
            df_plot = df.plot(kind="line", grid=True, ax=ax, title='Loss', xlabel='Epochs')
            ax.legend('')

            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()

            surf = pygame.image.fromstring(raw_data, size, "RGB")
            screen.blit(surf, (500,50))
            pygame.display.update()

#train_model(data, weights, bias, l_rate, epochs)
        
#####

# my neuralnet drawing function
def NN_draw(NN):
    c_x = 0
    c_y = -50
    y_space = 600
    x_space = 375
    distance_y = y_space / len(NN)

    radius = 15

    #coordinates
    nn_coords = []
    radiuses = []
    for layer in NN:
        n_num = 0
        c_y += distance_y
        distance_x = x_space / len(layer)
        c_x = -distance_x + distance_x/2 + 10
        nn_layer_coords = []
        nn_coords.append(nn_layer_coords)
        for neuron in layer:
            radius1 = 10
            if distance_x < 30:
                radius1 = (distance_x - radius1)
            n_num +=1
            c_x += distance_x
            nn_layer_coords.append([c_x, c_y])
            radiuses.append(radius1)

    # line activity
    line = 'active'
    rand_line_col = 0
    if line == 'active':
        if int(pygame.time.get_ticks() / 100) % 5 == 0:
            rand_line_col = randint(0,1)
        if rand_line_col == 0:
            line_color = 'darkblue'
        elif rand_line_col ==1:
            line_color = 'yellow'
    elif line == 'inactive':
        line_color = 'darkblue'

    #lines drawing
    i=0
    while i < len(nn_coords)-1:
        for coord in nn_coords[i]:
            for next_coord in nn_coords[i+1]:
                pygame.draw.line(screen, line_color, coord, next_coord, width = 2)
        i+=1

    #   neuron activity
    rand_neuron_act = 0
    if int(pygame.time.get_ticks() / 100) % 5 == 0:
        rand_neuron_act = randint(0,1)
    if rand_neuron_act == 0:
        neuron_color = 'red'
    elif rand_neuron_act == 1:
        neuron_color = 'yellow'

    # neurons drawing
    i=0
    for layer in nn_coords:
        for coords in layer:
            pygame.draw.circle(screen, 'green', coords, [radius for radius in radiuses][i]+2)
            pygame.draw.circle(screen, neuron_color, coords, [radius for radius in radiuses][i])
            i+=1

# NN sctructure for drawing
layer_in = [i for i in range(len(data.columns))]
layer_h1 = [i for i in range(len(weights))]
layer_out = [i for i in range(len(data['targets']))]
NN = layer_in,layer_h1,layer_out 
# make neuron color animated or not ('active' or 'inactive')      
neuron_stage = 'active'


# Pygame
pygame.init()

screen = pygame.display.set_mode(SIZE)
action_screen = Rect(400, 0, 800, 600)
NN_screen = Rect(0, 0, 400, 600)


clock = pygame.time.Clock()


fig = pylab.figure(figsize=[4, 4], # Inches
                   dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                   )
ax = fig.gca()


canvas = agg.FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()

train_i = 0
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit()

    window = pygame.display.set_mode((SIZE), DOUBLEBUF)
    screen = pygame.display.get_surface()

    size = canvas.get_width_height()

    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.blit(surf, (500,50))


    NN_draw(NN)
    pygame.display.update()

    if train_i < 1:
        train_model(data, weights, bias, l_rate, epochs)

        df = pd.DataFrame(epoch_loss)
        df_plot = df.plot(kind="line", grid=True, ax=ax, title='Loss', xlabel='Epochs')
        ax.legend('')

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        train_i+=1


    clock.tick(60)

