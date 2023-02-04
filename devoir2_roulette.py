import numpy as np

"""
Un exemple où la programmation dynamique markovienne peut être utilisée est dans la modélisation de l'évolution d'une espèce 
dans un environnement modifié par l'homme.

Client : 

On modélise le problème comme ceci :
    états : population stable, population en augmentation, population en déclin.
    actions : mise en place de zones protégées, la réintroduction de spécimens sauvages
    incertitudes: problème stochastique car facteurs incertains nombreux (alimentation, environnement, reproduction, etc...)

Nous pouvons utiliser la programmation dynamique markovienne pour déterminer la politique optimale pour chaque état de la population de l'espèce.
Nous considérons que l'état de la population de l'espèce change au fil du temps en fonction des actions prises par l'équipe de conservation.

En utilisant les équations de récurrence de Bellman, nous pouvons déterminer la valeur de chaque état et les politiques optimales pour chaque étape.
Nous pourrions montrer à l'équipe de conservation un tableau comparant les coûts et les effets sur la population de l'espèce pour chaque politique envisagée.
Cela permettrait de leur montrer la valeur de la programmation dynamique markovienne pour prendre des décisions sous incertitude dans ce genre de situation.

href : http://www-laplace.inrialpes.fr/publications/Rayons/Tapus02.pdf

"""

# nombre d'états
nb_states = 3
states = {0:"population en augmentation", 1:"population stable", 2:"population en declin"}
# nombre d'actions de transition
nb_actions = 2
actions = {0:"zones protegees", 1:"reintroduction de specimens"}
# nombre de trimestres d'observation
N = 5

# matrice de transition
P = [[[0.7, 0.3, 0.0],
      [0.2, 0.6, 0.2],
      [0.0, 0.4, 0.6]],
    [[0.6, 0.3, 0.1],
      [0.3, 0.5, 0.2],
      [0.1, 0.4, 0.5]]]

# récompenses
R = [4,2,-3]

# J(x) : population à l'étape k (semestre k) si le client a pris ses decisions de facon optimale
J = np.zeros((N+1,nb_states))
# mu(x) : politique optimale a l'etat k
mu = np.zeros((N+1,nb_states))

# profits
policy = np.zeros((nb_actions))
profits = np.zeros((nb_actions))

# population initiale
population = 60
# initialisation de J

for i in range(nb_states):
    for a in range(nb_actions):
        profits[a] = R[i] + np.sum([P[a][i][j] for j in [k  for k in range(nb_states) if k!=i]])
    J[N-1][i] = np.max(profits)
    mu[N-1][i] = np.argmax(profits)

print("J(5): {0:.2f} | {1:.2f} | {2:.2f}".format(J[N-1][0],J[N-1][1],J[N-1][2]))
print("mu(5): {0}, {1}, {2}".format(mu[N-1][0],mu[N-1][1],mu[N-1][2]), end="\n\n")

# boucle principale
for t in reversed(range(N)):
    for i in range(nb_states):
        for a in range(nb_actions):
            profits[a] = R[i] + np.sum([P[a][i][j]*J[t+1][i] for j in [k  for k in range(nb_states) if k!=i]])
        J[t][i] = np.max(profits[a])
        print("J({0})[{1}]: {2:.2f} ".format(t+1, i, J[t][i]), end="| ")
        mu[t][i] = np.argmax(profits)
        print("mu({0})[{1}]: {2}".format(t+1, i, mu[t][i]))
    
    print('')
    for i in range(nb_states):
        print("J({0})[{1}]: {2:.2f}".format(t+1, i, J[t][i]), end=" ")
    print('')
    for i in range(nb_states):
        print("mu({0})[{1}]: {2}".format(t+1, i, mu[t][i]), end=" ")
    print("\n")

#Affichage de la valeur optimale et de la politique optimale
print("Resultats...")
J_0 = population + np.max(J[0])
print("J({0}): {1:.2f} ".format(0, J_0))
print("Finalement la strategie optimale est :")
print([[int(mu[t][i]) for i in range(nb_states)] for t in range(N-1)])

print("De manière générale il faut adopter la politique suivante pour chacun des etats :")
opti = np.matrix([[int(mu[t][i]) for i in range(nb_states)] for t in range(N-1)]).mean(0)
print([actions[np.array(opti)[0][i]] for i in range(nb_states)])