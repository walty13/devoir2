import numpy as np

"""
L'exemple que j'ai choisit est le suivant : 
Un client souhaite utiliser un bot pour gagner un tournoi de roulette en ligne.
Pour cela il me demande de définir la stratégie qui maximiser ses gains en partant d'un 
budget fixé. Mon client n'a manifestement pas beaucoup de connaissances sur le jeu de 
la roulette et souhaite investir le moins d'argent dans cette implémentation pour 
maximiser son profit total.

Pour cela je recommande d'utiliser une méthode de programmation dynamique. 
Je lui explique que pour le jeu de la roulette est basé sur des chances aléatoires 
de gagner avec des probabilités qui varient en fonction de la manière de jouer. 
En effet on peut placer 2 types de paris différents à chaque partie : les paris extérieurs 
et les paris intérieurs. Un pari intérieur s'applique à un seul ou un groupe de numéros.
Ces paris sont les plus difficiles à gagner, mais ils sont également les plus gratifiants 
lorsque vous le faites. Inversement, un pari extérieur est un pari sur un grand groupe tel 
que impair/pair, rouge/noir, ou 1 à 18 et 19 à 36. Vous avez beaucoup plus de chances de gagner 
quand vous optez pour les paris externes, mais ne vous attendez pas à un gain énorme.

On peut alors modéliser ce problème comme le processus de décision markovien :

Les 3 états sont :
- Perdant
- Gagnant à moitié
- Gagnant totalement

Les 2 actions sont :
- Continuer à jouer
- Quitter le jeu

La transition entre les états dépendra de la chance et des décisions prises par le joueur. 
Par exemple, si le joueur est dans un état de "Gagnant à moitié", la probabilité de passer 
à un état de "Gagnant totalement" sera plus élevée si le joueur continue à jouer plutôt que 
de quitter le jeu. Si le joueur est dans un état de "Perdant", la probabilité de passer 
à un état de "Gagnant à moitié" sera plus élevée si le joueur continue à jouer plutôt que 
de quitter le jeu.

Le processus de décision markovien dans ce jeu sera donc basé sur la maximisation du 
gain potentiel du joueur à chaque tour, en prenant en compte les probabilités associées 
à chaque action et état

"""

# nombre d'états
nb_states = 3
X = {"gagnant total" : 0, "gagnant partiel" : 1, "perdant" : 2}
# nombre d'actions de transition
nb_actions = 2
U = {"continuer" : 0, "quitter" : 1}
# nombre de semaines d'observation
N = 5

# matrice de transition : 
# on fixe les probabilités de gagner en fonction de la stratégie adopté à la partie k
D = [[0.05, 0.07, 0.9],
     [0.04, 0.05, 0.05]]

# récompenses
R = [[.25,.15],
     [.15,.12],
     [-.05,.08]]

# J(x) : budget à l'étape k (semestre k) si le client a pris ses decisions de facon optimale
J = np.zeros((N+1,nb_states))
# mu(x) : politique optimale a l'etat k
mu = np.zeros((N+1,nb_states))

# budget initiale
budget = 100000
# profits
policy = np.zeros((nb_actions))
profits = np.zeros((nb_actions))

# fonction de transition
def transition(x, u, D):
    return D[u]*(1+np.log())

def cout(x, u, D):
    return R[x][u] + budget

# initialisation de J
for i in range(nb_states):
    for j in range(nb_actions):
        profits[j] = transition(t[i],R[i][j])
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

