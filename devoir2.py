import numpy as np

"""
Un exemple où la programmation dynamique markovienne peut être utilisée est dans la modélisation de l'évolution d'une espèce 
dans un environnement modifié par l'homme.

Client : une équipe de conservation souhaite décider de la politique à adopter pour protéger une espèce en voie de disparition.
Il s'agit d'un problème stochastique car les facteurs détérminants l'évolution d'une espèce sont nombreux (alimentation, environnement, reproduction, etc...)

On modélise le problème comme ceci :
    états : population stable, population en augmentation, population en déclin.
    actions : mise en place de zones protégées, la réintroduction de spécimens sauvages
    incertitudes: problème stochastique car facteurs incertains nombreux (alimentation, environnement, reproduction, etc...)

Nous pouvons utiliser la programmation dynamique markovienne pour déterminer la politique optimale pour chaque état de la population de l'espèce.
Nous considérons que l'état de la population de l'espèce change au fil du temps en fonction des actions prises par l'équipe de conservation.

En utilisant les équations de récurrence de Bellman, nous pouvons déterminer la valeur de chaque état et les politiques optimales pour chaque étape.
Nous pourrions montrer à l'équipe de conservation un tableau comparant les coûts et les effets sur la population de l'espèce pour chaque politique envisagée.
Cela permettrait de leur montrer la valeur de la programmation dynamique markovienne pour prendre des décisions sous incertitude dans ce genre de situation.

"""

# nombre d'états
nb_states = 3
states = {"population en augmentation" : 0, "population stable" : 1, "population en declin" : 2}
# nombre d'actions de transition
nb_actions = 2
actions = {"zones protegees" : 0, "reintroduction de specimens" : 1}
# nombre de semestre d'observation
N = 4

# matrice de transition
a = [[[0.6, 0.3, 0.1],
      [0.2, 0.6, 0.2],
      [0.1, 0.4, 0.5]],
    [[0.6, 0.2, 0.2],
      [0.3, 0.5, 0.2],
      [0.2, 0.3, 0.5]]]

# récompenses
R = [[2,0,-1],
     [-1,1,-1],
     [1,0,-2]]

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
    for j in range(nb_actions):
        profits[j] = np.dot(a[j][i],R[i]) + population
    J[N][i] = np.max(profits)
    mu[N][i] = np.argmax(profits)

print("J(5): {0:.2f} | {1:.2f} | {2:.2f}".format(J[N][0],J[N][1],J[N][2]))
print("mu(5): {0}, {1}, {2}".format(mu[N][0],mu[N][1],mu[N][2]), end="\n\n")

# boucle principale
for t in reversed(range(N)):
    for i in range(nb_states):
        max_value = -1e9
        best_a = -1
        value = 0
        for j in range(nb_actions):
            profits[j] = np.dot(a[j][i],R[i]) + J[t+1][i]
        J[t][i] = np.max(profits)
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
J_0 = np.max(J[0])
print("J({0}): {1:.2f} ".format(0, J_0))
print("Finalement la strategie optimale est :")
for i in range(N):
    print("{"+str(mu[i])+"}", end=",")

