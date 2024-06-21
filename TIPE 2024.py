import random
import matplotlib.pyplot as plt

dico = {'donnant-donnant': 0, 'majo-dur': 1, 'majo-mou': 1, 'rancuniere': 2, 'CCD': 3, 'DDC': 3, 'gentil': 4, 'mechant': 4, 'lunatique': 5, 'mefiante': 0,'detective': 6}

#['mechant',['D']],['gentil',['C']],['donnant-donnant',['C']],['majo-dur',[0,0]],['majo-mou',[0,0]],['rancuniere',[True]],['CCD',[0]],['DCC',[0]],['mefiante',['D']],['lunatique',[]],['detective',[0,False]]


def choix_action(role,liste_mem):
    if dico[role] == 4:
        return liste_mem[0]
    elif dico[role] == 0:
        return liste_mem[0]
    elif dico[role] == 1:
        if role == 'majo-mou':
            a,c1 = liste_mem[0],'C'
            b,c2 = liste_mem[1],'D'
        else:
            a,c1 = liste_mem[1],'D'
            b,c2 = liste_mem[0],'C'
        if b > a:
            return c2
        else:
            return c1
    elif dico[role] == 2:
        if liste_mem[0]:
            return 'C'
        else:
            return 'D'
    elif dico[role] == 3:
        return role[liste_mem[0]]
    elif dico[role] == 5:
        return DouC(0.5)

def chgt(role,liste_mem,c):
    if dico[role] == 0:
        liste_mem[0] = c
    elif dico[role] == 1:
        if c == 'C':
            liste_mem[0] +=1
        else:
            liste_mem[1] +=1
    elif dico[role] == 2:
        if c == 'D' and liste_mem[0]:
            liste_mem[0] = False
    elif dico[role] == 3:
        liste_mem[0] = (liste_mem[0] + 1)%3

def refresh(lj1,lj2,c1,c2):
    chgt(lj1[0],lj1[1],c2)
    chgt(lj2[0],lj2[1],c1)

def DC (p):
    x = random.random()
    if x < p:
        return ['mechant',['D']]
    else:
        return ['gentil',['C']]

def role_random(liste_role,liste_proba):
    x = random.random()
    verif = True
    compt = 1
    n = len(liste_role)
    role = liste_role[n-1]
    if liste_proba[0] > x:
        role = liste_role[0]
        verif = False
    while verif and compt < n-1:
        if liste_proba[compt] > x >= liste_proba[compt-1]:
            role = liste_role[compt]
            verif = False
        else:
            compt += 1
    return role

#Reward est une matrice ou à la place i j il y a la récompense T du joueur (i,j)

def creation_rew(f,n):
    rew = [[None for k in range(n)]for k in range(n)]
    for k in range(n):
        for j in range(n):
            rew[k][j] = f(k,j)
    return rew

def fonc (k,i):
    return 5

def create_matrix(n, min_value, max_value):
    size = 2 * n + 1
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            distance = max(abs(i - n), abs(j - n))
            value = max_value - distance * ((max_value - min_value) / n)
            matrix[i][j] = max(min_value, min(max_value, value))
    return matrix

def create_large_matrix(n, min_value, max_value):
    size_large = 3 * (2 * n + 1)
    large_matrix = [[0.0] * size_large for _ in range(size_large)]
    small_matrix = create_matrix(n, min_value, max_value)
    for i in range(3):
        for j in range(3):
            start_row = i * (2 * n + 1)
            start_col = j * (2 * n + 1)
            for row in range(2 * n + 1):
                for col in range(2 * n + 1):
                    large_matrix[start_row + row][start_col + col] = small_matrix[row][col]
    return large_matrix

def rew (a,b,t):
    if (a,b) == ('C','C'):
        res = 1
    elif (a,b) == ('C','D') :
        res = 0
    elif (a,b) == ('D','D'):
        res = 0
    else :
        res = t
    return res

def tour_de_jeu(joueur,reward,tour):
    n = len(joueur)
    joueur_res = [[0 for k in range(n)] for k in range(n)]
    for k in range(n):
        for j in range(n):
            for p in range(tour):
                c1 = choix_action(joueur[k][j][0],joueur[k][j][1])
                c2 = choix_action(joueur[k][(j-1)%n][0],joueur[k][(j-1)%n][1])
                joueur_res[k][j] += rew(c1,c2,reward[k][j])
                refresh(joueur[k][j],joueur[k][(j-1)%n],c1,c2)
                c1 = choix_action(joueur[k][j][0],joueur[k][j][1])
                c2 = choix_action(joueur[k][(j+1)%n][0],joueur[k][(j+1)%n][1])
                joueur_res[k][j] += rew(c1,c2,reward[k][j])
                refresh(joueur[k][j],joueur[k][(j+1)%n],c1,c2)
                c1 = choix_action(joueur[k][j][0],joueur[k][j][1])
                c2 = choix_action(joueur[(k-1)%n][j][0],joueur[(k-1)%n][j][1])
                joueur_res[k][j] += rew(c1,c2,reward[k][j])
                refresh(joueur[k][j],joueur[(k-1)%n][j],c1,c2)
                c1 = choix_action(joueur[k][j][0],joueur[k][j][1])
                c2 = choix_action(joueur[(k+1)%n][j][0],joueur[(k+1)%n][j][1])
                joueur_res[k][j] += rew(c1,c2,reward[k][j])
                refresh(joueur[k][j],joueur[(k+1)%n][j],c1,c2)
    return joueur_res

def max_case (mat,i,j,n):
    max = mat[i][j]
    ind = i,j
    if mat[i][(j+1)%n] > max:
        max = mat[i][(j+1)%n]
        ind = i,(j+1)%n
    if mat[i][(j-1)%n] > max:
        max = mat[i][(j-1)%n]
        ind = i,(j-1)%n
    if mat[(i+1)%n][j] > max:
        max = mat[(i+1)%n][j]
        ind = (i+1)%n,j
    if mat[(i-1)%n][j] > max:
        max = mat[(i-1)%n][j]
        ind = (i-1)%n,j
    return ind

def devient(role):
    if dico[role] == 4 or dico[role] == 0:
        if role == 'gentil' or role == 'donnant-donnant':
            return [role,['C']]
        else:
            return [role,['D']]
    elif dico[role] == 1:
        return [role,[0,0]]
    elif dico[role] == 2:
        return [role,[True]]
    elif dico[role] == 3:
        return [role,[0]]
    elif dico[role] == 5:
        return [role,[]]

def actu_liste (joueur,dico,nbr_iter):
    n = len(joueur)
    for k in range(n):
        for i in range(n):
            dico[joueur[k][i][0]][nbr_iter] += 1

def actualisation (joueur,joueur_res,dico,nbr_iter):
    n = len(joueur)
    nv_joueur = [[None for k in range(n)] for k in range(n)]
    for k in range(n):
        for i in range(n):
            x,y = max_case(joueur_res,k,i,n)
            nv_joueur[k][i] = devient(joueur[x][y][0])
    actu_liste(nv_joueur,dico,nbr_iter)
    return nv_joueur

def tracer(dico,long):
    l1 = [k for k in range(long+1)]
    for cle in dico:
        plt.plot(l1,dico[cle],label=cle)
        plt.text(l1[-1], dico[cle][-1], cle, ha='right')


#il y a dans le dico la liste avec que des 0 pour chaque iterations
def trouver_indice(liste, element):
    try:
        indice = liste.index(element)
    except ValueError:
        indice = -1
    return indice


##Premiere simulation: 3 comportements
n = 10
tour = 2
precision = 100
iter = 20

reward = [[fonc(j,k) for k in range (n)] for j in range(n)]

dict_global = {'mechant':[0 for k in range(iter+1)],'gentil':[0 for k in range(iter+1)],'donnant-donnant':[0 for k in range(iter+1)]}


for p in range(precision):
    joueur = [[role_random([['mechant',['D']],['gentil',['C']],['donnant-donnant', ['C']]],[0.33,0.67,1]) for k in range(n)]for k in range(n)]
    dict = {'mechant':[0 for k in range(iter+1)],'gentil':[0 for k in range(iter+1)],'donnant-donnant':[0 for k in range(iter+1)]}
    rg = 1
    actu_liste(joueur,dict,0)
    for k in range(iter):
        re = tour_de_jeu(joueur,reward,tour)
        joueur = actualisation(joueur,re,dict,rg)
        rg += 1
    for cle in dict:
        for k in range(iter+1):
            dict_global[cle][k] += dict[cle][k]

for cle in dict_global:
    for k in range(iter+1):
        dict_global[cle][k] = dict_global[cle][k]/(precision*n*n)

tracer(dict_global,iter)
plt.axis([0, iter, 0, 1])
plt.legend()
plt.title('Proportions des comportements à chaque étape')
plt.show()


##Deuxieme simulation: Tous les comportements
n = 10
tour = 3
precision = 1
choix = 1
reward = [[fonc(j,k) for k in range (n)] for j in range(n)]
iter = 100
role = [['mechant',['D']],['gentil',['C']],['donnant-donnant',['C']],['majo-dur',[0,0]],['majo-mou',[0,0]],['rancuniere',[True]],['CCD',[0]],['DDC',[0]],['mefiante',['D']],['lunatique',[]]]

dict_global = {}
for k in range(len(role)):
    dict_global[role[k][0]] = [0 for j in range(iter+1)]
dico_top1 = {}
dico_moy = {}

def bar_top1 (dico):
    role = None
    max = -1
    for cle in dico:
        if dico[cle][-1] > max:
            role = cle
            max = dico[cle][-1]
    if role not in dico_top1:
        dico_top1[role] = 1
    else:
        dico_top1[role] += 1

def tri_rapide(liste):
    if len(liste) <=1:
        return liste
    pivot = liste[0]
    l1 = [liste[k] for k in range(1,len(liste)) if liste[k][1] < pivot[1]]
    l2 = [liste[k] for k in range(1,len(liste)) if liste[k][1] >= pivot[1]]
    return tri_rapide(l1) + [pivot] + tri_rapide(l2)

def bar_pond (dico):
    l1 = [[cle,dico[cle][-1]] for cle in dico]
    liste_tri = tri_rapide(l1)
    for k in range(len(liste_tri)):
        if liste_tri[k][0] not in dico_moy:
            dico_moy[liste_tri[k][0]] = k
        else:
            dico_moy[liste_tri[k][0]] += k

for p in range(precision):
    joueur = [[role_random([['mechant',['D']],['gentil',['C']],['donnant-donnant',['C']],['majo-dur',[0,0]],['majo-mou',[0,0]],['rancuniere',[True]],['CCD',[0]],['DDC',[0]],['mefiante',['D']],['lunatique',[]]],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) for k in range(n)] for j in range(n)]
    dict = {}
    for k in range(len(role)):
        dict[role[k][0]] = [0 for j in range(iter+1)]
    rg = 1
    actu_liste(joueur,dict,0)
    for k in range(iter):
        re = tour_de_jeu(joueur,reward,tour)
        joueur = actualisation(joueur,re,dict,rg)
        rg += 1
    for cle in dict:
        for k in range(iter+1):
            dict_global[cle][k] += dict[cle][k]
    bar_top1(dict)
    bar_pond(dict)

if choix == 0:
    lrole = []
    l2 = []
    for cle in dict_global:
        lrole.append(cle)
        if cle in dico_top1:
            l2.append(dico_top1[cle])
        else:
            l2.append(0)
    plt.bar(lrole, l2, color='skyblue')
    plt.xlabel('Comportements')
    plt.ylabel('Nombre de victoire')
    plt.title('Nombre de victoire pour chaque comportement')
elif choix == 1:
    for cle in dict_global:
        for k in range(iter+1):
            dict_global[cle][k] = dict_global[cle][k]/(precision*n*n)
    tracer(dict_global,iter)
    plt.axis([0, iter+1, 0, 1])
    plt.legend()
    plt.title('Proportions des comportements à chaque étape')
else:
    lrole = []
    l2 = []
    for cle in dict_global:
        lrole.append(cle)
        if cle in dico_moy:
            l2.append(dico_moy[cle])
        else:
            l2.append(0)
    plt.bar(lrole, l2, color='skyblue')
    plt.xlabel('Comportements')
    plt.ylabel('Score')
    plt.title('Score de chaque comportement')

plt.show()

##Troisieme simulation: Variation des récompenses selon l'espace
p = 2
n = 3*(2*p+1)
tour = 1
recompense_min = 2.5
recompense_max = 10
choix = 1
precision = 100
iter = 30
import matplotlib.pyplot as plt
import numpy as np
reward = create_large_matrix(p, recompense_min, recompense_max)

dict_global = {'mechant':[0 for k in range(iter+1)],'gentil':[0 for k in range(iter+1)],'donnant-donnant':[0 for k in range(iter+1)]}
dict_recompense = {}

l0 = []
l1 = [0 for k in range(p+1)]
l2 = [0 for k in range(p+1)]

for k in range(p+1):
    dict_recompense[recompense_min + k*((recompense_max - recompense_min)/p)] = [0,0]
    l0.append(recompense_min + k*((recompense_max - recompense_min)/p))



for p in range(precision):
    joueur = [[role_random([['mechant',['D']],['gentil',['C']],['donnant-donnant', ['C']]],[0.33,0.67,1]) for k in range(n)]for k in range(n)]
    dict = {'mechant':[0 for k in range(iter+1)],'gentil':[0 for k in range(iter+1)],'donnant-donnant':[0 for k in range(iter+1)]}
    rg = 1
    actu_liste(joueur,dict,0)
    for k in range(iter):
        re = tour_de_jeu(joueur,reward,tour)
        joueur = actualisation(joueur,re,dict,rg)
        rg += 1
    for cle in dict:
        for k in range(iter+1):
            dict_global[cle][k] += dict[cle][k]
    for q in range(n):
        for j in range(n):
            m = trouver_indice(l0,reward[q][j])
            if m >2:
                print(m)
            if joueur[q][j][0] == 'mechant':
                l1[m] += 1
            else:
                l2[m] += 1


l1[0] = l1[0]/(precision)
l2[0] = l2[0]/(precision)
l1[1] = l1[1]/(precision)
l2[1] = l2[1]/(precision)
l1[2] = l1[2]/(precision)
l2[2] = l2[2]/(precision)


if choix == 0:
    for cle in dict_global:
        for k in range(iter+1):
            dict_global[cle][k] = dict_global[cle][k]/(precision*n*n)

    tracer(dict_global,iter)
    plt.axis([0, iter, 0, 1])
    plt.legend()
    plt.title('Proportions des comportements à chaque étape')
elif choix == 1:
    valeurs1 = l1
    valeurs2 = l2
    noms = l0
    n = len(noms)
    indice = np.arange(n)
    largeur = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(indice - largeur/2, valeurs1, largeur, label='Méchant')
    bar2 = ax.bar(indice + largeur/2, valeurs2, largeur, label='Donnant-donnant')
    ax.set_title('Comparaison des comportements par gains possible')
    ax.set_xlabel('gains')
    ax.set_ylabel('Population')
    ax.set_xticks(indice)
    ax.set_xticklabels(noms)
    ax.legend()
    plt.show()

##Simulation 2 à 2
def choix_action2(role,liste_mem):
    if dico[role] == 4:
        return liste_mem[0]
    elif dico[role] == 0:
        return liste_mem[0]
    elif dico[role] == 2:
        if liste_mem[0]:
            return 'C'
        else:
            return 'D'
    elif dico[role] == 6:
        if liste_mem[0] in [0,2,3]:
            liste_mem[0] += 1
            return 'C'
        elif liste_mem[0] == 1:
            liste_mem[0] += 1
            return 'D'
        else :
            if liste_mem[1]:
                return 'C'
            else:
                return 'D'

def reset(role,liste_mem):
    if dico[role] == 0:
        liste_mem[0] = 'C'
    elif dico[role] == 2:
        liste_mem[0] = True
    elif dico[role] == 6:
        liste_mem[0] = 0
        liste_mem[1] = False


def refresh2(lj1,lj2,c1,c2):
    chgt2(lj1[0],lj1[1],c2)
    chgt2(lj2[0],lj2[1],c1)

def rew2 (a,b):
    if (a,b) == ('C','C'):
        res = 1
    elif (a,b) == ('C','D') :
        res = -1
    elif (a,b) == ('D','D'):
        res = 0
    else :
        res = 3
    return res

def chgt2(role,liste_mem,c):
    if dico[role] == 2:
        if c == 'D' and liste_mem[0]:
            liste_mem[0] = False
    elif dico[role] == 0:
        liste_mem[0] = c
    elif dico[role] == 6:
        if c == 'D':
            liste_mem[1] = True

dico_delta = {}

def tour_de_jeu2(joueur,tour):
    n = len(joueur)
    joueur_res = [0 for k in range(n)]
    for k in range(n):
        for j in range(k+1,n):
            gain1 = 0
            gain2 = 0
            for p in range(tour):
                c1 = choix_action2(joueur[k][0],joueur[k][1])
                c2 = choix_action2(joueur[j][0],joueur[j][1])
                gain1 += rew2(c1,c2)
                gain2 += rew2(c2,c1)
                refresh2(joueur[k],joueur[j],c1,c2)
            dico_delta[joueur[k][0],joueur[j][0]] = gain1 - gain2
            joueur_res[k] += gain1
            joueur_res[j] += gain2
            reset(joueur[k][0],joueur[k][1])
            reset(joueur[j][0],joueur[j][1])
    return joueur_res

joueur = [['gentil',['C']],['mechant',['D']],['donnant-donnant',['C']],['rancuniere',[True]],['detective',[0,False]]]

res = tour_de_jeu2(joueur,10)

valeurs = res

noms = ['Gentil', 'Méchant', 'Donnant-donnant', 'Rancunier', 'Détective']

plt.bar(noms, valeurs)

plt.title('Résultat des affrontements 2 à 2 en 10 itérations')
plt.xlabel('Comportements')
plt.ylabel("Points")

plt.show()
##Simulation à 25
def suppression(joueur):
    meilleur_joueur = [elem[:] for elem in joueur[-5:]]
    joueur[:5] = meilleur_joueur

def algo_génétique(joueur,joueur_res):
    tri_commun(joueur,joueur_res)
    suppression(joueur)

def tri_commun(l1,l2):
    n = len(l2)
    for k in range(1,n):
        cle2 = l2[k]
        cle1 = l1[k]
        j = k-1
        while j>=0 and l2[j] > cle2:
            l2[j+1] = l2[j]
            l1[j+1] = l1[j]
            j = j-1
        l2[j+1] = cle2
        l1[j+1] = cle1

def itérations(joueur,tour,iter):
    for k in range (iter):
        res = tour_de_jeu2(joueur,tour)
        algo_génétique(joueur,res)
    return joueur,res

joueur = [['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['gentil',['C']],['mechant',['D']],['mechant',['D']],['mechant',['D']],['donnant-donnant',['C']],['donnant-donnant',['C']],['donnant-donnant',['C']]]

gen_finale,res = itérations(joueur,10,10)

l = [0,0,0]

for k in range(len(gen_finale)):
    if gen_finale[k][0] == 'gentil':
        l[0] += 1
    elif gen_finale[k][0] == 'mechant':
        l[1] += 1
    else:
        l[2] +=1

valeurs = l

noms = ['Gentil', 'Méchant', 'Donnant-donnant']

plt.bar(noms, valeurs)

plt.title('Populations')
plt.xlabel('Comportements')
plt.ylabel("Nombre d'individus'")

plt.show()
















