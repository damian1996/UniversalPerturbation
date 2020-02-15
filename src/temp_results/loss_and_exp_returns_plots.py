import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


with open('./LossMinus') as f:
    lines = f.readlines()
    lines = [float(line.strip()) for line in lines]
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(len(lines)), lines)
    plt.savefig('LossMinus.png')

with open('./LossPlus') as f:
    lines = f.readlines()
    lines = [float(line.strip()) for line in lines]
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(len(lines)), lines)
    plt.savefig('LossPlus.png')


with open('./ResultsMinus') as f:
    lines = f.readlines()
    lines = [float(line.strip()) for line in lines]
    lines1 = lines[0::3]
    lines2 = lines[1::3]
    lines3 = lines[2::3]

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(len(lines1)), lines1)
    plt.savefig('ResultsMinus1.png')

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(len(lines2)), lines2)
    plt.savefig('ResultsMinus2.png')

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(len(lines3)), lines3)
    plt.savefig('ResultsMinus3.png')

with open('./ResultsPlus') as f:
    lines = f.readlines()
    lines = [float(line.strip()) for line in lines]
    lines1 = lines[0::3]
    lines2 = lines[1::3]
    lines3 = lines[2::3]

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(len(lines1)), lines1)
    plt.savefig('ResultsPlus1.png')

    fig = plt.figure()
    ax = plt.axes()
    print(lines2)
    ax.plot(range(len(lines2)), lines2)
    plt.savefig('ResultsPlus2.png')

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(len(lines3)), lines3)
    plt.savefig('ResultsPlus3.png')
