import numpy as np
import matplotlib.pyplot as plt
from random import choices

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

href : http://www-laplace.inrialpes.fr/publications/Rayons/Tapus02.pdf

"""

# nombre d'états
nb_states = 3
states = {0:"population en augmentation", 1:"population stable", 2:"population en declin"}
# nombre d'actions de transition
nb_actions = 2
actions = {0:"zones protegees", 1:"reintroduction de specimens"}

# matrice de transition
P = [[[0.6, 0.3, 0.1], # S0 -> a0
      [0.3, 0.4, 0.3], # S1 -> a0
      [0.1, 0.5, 0.4]], # S2 -> a0
      
     [[0.7, 0.3, 0.0], # S0 -> a1
      [0.4, 0.4, 0.2], # S1 -> a1
      [0.2, 0.5, 0.3]]] # S2 -> a1

# récompenses
R = [4,1,-3]

# profits
policy = np.zeros((nb_actions))
profits = np.zeros((nb_actions))


def initialisation(N, J, mu):
    for i in range(nb_states):
        for a in range(nb_actions):
            # on somme la récompense d'un état + (probabilité de transition * le profit de l'état suivant)
            profits[a] = R[i] + np.sum([P[a][i][j] for j in [k  for k in range(nb_states) if k!=i]])
        J[N-1][i] = np.max(profits)
        mu[N-1][i] = np.argmax(profits)

    #print("J(5): {0:.2f} | {1:.2f} | {2:.2f}".format(J[N-1][0],J[N-1][1],J[N-1][2]))
    #print("mu(5): {0}, {1}, {2}".format(mu[N-1][0],mu[N-1][1],mu[N-1][2]), end="\n\n")

# boucle principale
def main_boucle(N, J, mu):
    for t in reversed(range(N)):
        for i in range(nb_states):
            for a in range(nb_actions):
                profits[a] = R[i] + np.sum([P[a][i][j]*J[t+1][i] for j in [k  for k in range(nb_states) if k!=i]])
            J[t][i] = np.max(profits[a])
            #print("J({0})[{1}]: {2:.2f} ".format(t+1, i, J[t][i]), end="| ")
            mu[t][i] = np.argmax(profits)
            #print("mu({0})[{1}]: {2}".format(t+1, i, mu[t][i]))
        
        #print('')
        #for i in range(nb_states):
        #    print("J({0})[{1}]: {2:.2f}".format(t+1, i, J[t][i]), end=" ")
        #print('')
        #for i in range(nb_states):
        #    print("mu({0})[{1}]: {2}".format(t+1, i, mu[t][i]), end=" ")
        #print("\n")

    #J_0 = np.max(J[0])
    #print("J({0}): {1:.2f} ".format(0, J_0))
    #print("Finalement la strategie optimale est :")
    #print(mu[0])
    #print()

    return J, mu

def plot_by_InitState(N, population):
    tab=[]
    init_state = [0, 1, 2]
    # J(x) : population à l'étape k (semestre k) si le client a pris ses decisions de facon optimale
    J = np.zeros((N+1,nb_states))
    # mu(x) : politique optimale a l'etat k
    mu = np.zeros((N+1,nb_states))

    # initialisation de J
    initialisation(N, J, mu)
    # boucle principale
    J, mu = main_boucle(N, J, mu)
    
    for s in init_state:
        moyenne = []
        for n in range(50):
            J_bis = np.zeros(np.shape(J))
            for t in range(N):
                J_bis[t] = J[N-t]
            
            state = s
            next_action = int(mu[0][state])
            sequence = []
            for t in range(N):
                weights = P[next_action][state]
                state = choices([0,1,2], weights)[0]
                next_action = int(mu[0][state])
                sequence.append(state)
        
            results = [60]
            for i in range(1,len(sequence)):
                results.append(results[i-1]+R[sequence[i]])
            moyenne.append(results)
        mean = np.mean(moyenne, axis=0)
        tab.append(mean)
        #Affichage de la valeur optimale et de la politique optimale
        print("Resultats pour l'etat initial {0}...".format(s))
        J_0 = mean[N-1]
        print("J({0}): {1:.2f} ".format(0, J_0))
        print("Finalement la strategie optimale est :")
        print(mu[0])
    
    print()
        
    plt.figure(1)
    c = ['r','g','b']
    for i in range(len(init_state)):
        plt.plot(np.arange(N), tab[i], marker="s", color=c[i])
        plt.annotate('%0.2f' % tab[i].max(), xy=(1, tab[i].max()), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    
    plt.xlabel('Trimestres')
    plt.ylabel('Population')
    plt.title('Evolution moyenne de l\'espèce en fonction de l\'état initial')
    plt.xticks(np.arange(N))
    plt.ylim(55,70)
    plt.legend(['augmentation', 'stable', 'déclin'])
    plt.show()


def plot_by_N(population):
    tab=[]
    trimestres = [3, 6, 9]
    J_0 = 0
    mu = []
    J = []
    for N in trimestres:
        moyenne = []
        for n in range(50):
            # J(x) : population à l'étape k (semestre k) si le client a pris ses decisions de facon optimale
            J = np.zeros((N+1,nb_states))
            # mu(x) : politique optimale a l'etat k
            mu = np.zeros((N+1,nb_states))

            # initialisation de J
            initialisation(N, J, mu)

            # boucle principale
            J, mu = main_boucle(N, J, mu)

            #print("De manière générale il faut adopter la politique suivante pour chacun des etats :")
            #opti = np.matrix([[int(mu[t][i]) for i in range(nb_states)] for t in range(N-1)]).mean(0)
            #print([actions[np.array(opti)[0][i]] for i in range(nb_states)])

            # affichage des graphs
            #plot_opti(N, J, mu)

            J_bis = np.zeros(np.shape(J))
            for t in range(N):
                J_bis[t] = J[N-t]
            
            state = 2
            next_action = int(mu[0][state])
            sequence = []
            for t in range(N):
                weights = P[next_action][state]
                state = choices([0,1,2], weights)[0]
                next_action = int(mu[0][state])
                sequence.append(state)
        
            results = [60]
            for i in range(1,len(sequence)):
                results.append(results[i-1]+R[sequence[i]])
            moyenne.append(results)
        mean = np.mean(moyenne, axis=0)
        tab.append(mean)
        #Affichage de la valeur optimale et de la politique optimale
        print("Resultats plot by N...")
        J_0 = mean[N-1]
        print("J({0}): {1:.2f} ".format(0, J_0))
        print("Finalement la strategie optimale est :")
        print(mu[0])
    
    plt.figure(2)
    c = ['r','g','b']
    for i in range(len(trimestres)):
        N = trimestres[i]
        t = np.arange(len(tab[i]))
        plt.plot(t, tab[i], marker="s", color=c[i])
        plt.annotate('%0.2f' % tab[i].max(), xy=(1, tab[i].max()), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
        i+=1

    plt.xlabel('Trimestres')
    plt.ylabel('Population')
    plt.title('Evolution moyenne de l\'espèce en fonction en partant d\'un état de déclin')
    plt.xticks(np.arange(N))
    plt.ylim(50,90)
    plt.legend(['3', '6', '9'])

    plt.show()


def main():
    # population initiale
    population = 36
    # N : nombre de trimestres d'observation
    N = 5

    plot_by_InitState(N,population)

    plot_by_N(population)

if __name__ == "__main__":
    main()
