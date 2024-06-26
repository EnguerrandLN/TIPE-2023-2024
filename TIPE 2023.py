import random
import numpy as np
import matplotlib.pyplot as plt
import time
#Toutes les lignes avance avant les colonnes. Dans la matrice prioritée si la coordonnée (i,j) = 0 alors i est prioritaire sur j. 1 j est prioritaire, 2 i a un feu rouge, 3 j a un feu rouge

def point_croisementc (b,MC):  #donne toutes les coordonnées de croisement de la colonne a
    l1,l2,l3 = [],[],[]
    for k in range (len(MC)):
        x,y = MC[k][b]
        l1.append(k)
        l2.append(x)
        l3.append(y) #la k-ième colonne rencontre a en (x,y)
    return(l1,l2,l3)

def point_croisementl (a,MC):  #donne toutes les coordonnées de croisement de la ligne a
    l1,l2,l3 = [],[],[]
    for k in range (len(MC[a])):
        x,y = MC[a][k]
        l1.append(k)
        l2.append(x)
        l3.append(y) #la k-ième colonne rencontre a en (x,y)
    return(l1,l2,l3)

def dico_cr (A,B,MC):
    dico = {}
    for k in range(len(A)):
        dico[(0,k)] = point_croisementl(k,MC)
    for j in range(len(B)):
        dico[(1,j)] = point_croisementc(j,MC)
    return dico

def retour_ind(n,l):
    for k in range(len(l)):
        if l[k] == n:
            return (k,True)
    return (0,False)

def avancer_ligne (a,A,MC,MP,B,dico,dico_croisement,lim,booleen=True):
    if booleen:
        a1 = 3
    else:
        a1 = 2
    compt = 0
    coord_col,coord_cr,coord_cor = dico[(0,a)]
    long = len(A[a])
    tempsmoyen = [0,0]
    for k in range (long-1,-1,-1):
        if A[a][k][0] == -1 or A[a][k][0]:
            A[a][k][1] += 1
            if (k+1) in dico_croisement1:
                ind,verif = dico_croisement1[k+1]
            else:
                ind,verif = retour_ind (k+1,coord_cr)
                dico_croisement1[k+1] = (ind,verif)
            if verif:
                if A[a][k][0] == -1:
                    A[a][k][0] = True
                elif MP[a][coord_col[ind]] == 0 or MP[a][coord_col[ind]] == a1:
                    if k+1<long:
                        if not A[a][k+1][0] and not B[coord_col[ind]][coord_cor[ind]][0]:
                            A[a][k][0] = False
                            A[a][k+1][0] = True
                            A[a][k+1][1] = A[a][k][1]
                            A[a][k][1] = 0
                    else:
                        tempsmoyen[0] += 1
                        tempsmoyen[1] += A[a][k][1]
                        A[a][k][0] = False
                        A[a][k][1] = 0
                elif MP[a][coord_col[ind]] == 1:
                    if not B[coord_col[ind]][coord_cor[ind]-1][0]:
                        if k+1<long:
                            if not A[a][k+1][0] and not B[coord_col[ind]][coord_cor[ind]][0]:
                                A[a][k][0] = False
                                A[a][k+1][0] = True
                                A[a][k+1][1] = A[a][k][1]
                                A[a][k][1] = 0
                        else:
                            tempsmoyen[0] += 1
                            tempsmoyen[1] += A[a][k][1]
                            A[a][k][0] = False
                            A[a][k][1] = 0
                if MP[a][coord_col[ind]] == ((a1+1)%2) +2:
                    A[a][k][0] = -1
                    compt += 1
            else:
                if k+1<long:
                    if not A[a][k+1][0]:
                        A[a][k][0] = False
                        A[a][k+1][0] = True
                        A[a][k+1][1] = A[a][k][1]
                        A[a][k][1] = 0
                else:
                    tempsmoyen[0] += 1
                    tempsmoyen[1] += A[a][k][1]
                    A[a][k][0] = False
                    A[a][k][1] = 0
    return tempsmoyen,compt

def avancer_colonne (b,B,MC,MP,A,dico,lim,booleen=True):
    if booleen:
        b1 = 2
    else:
        b1 = 3
    coord_lig,coord_lor,coord_cr = dico[(1,b)]
    long = len(B[b])
    compt = 0
    tempsmoyen = [0,0]
    for k in range (long-1,-1,-1):
        if B[b][k][0] == -1 or B[b][k][0]:
            B[b][k][1] += 1
            if (k+1) in dico_croisement2:
                ind,verif = dico_croisement2[k+1]
            else:
                ind,verif = retour_ind (k+1,coord_cr)
                dico_croisement2[k+1] = (ind,verif)
            if verif:
                if B[b][k][0] == -1:
                    B[b][k][0] = True
                elif MP[coord_lig[ind]][b] == 1 or MP[coord_lig[ind]][b] == b1:
                    if k+1<long:
                        if not B[b][k+1][0] and not A[coord_lig[ind]][coord_lor[ind]][0]:
                            B[b][k][0] = False
                            B[b][k+1][0] = True
                            B[b][k+1][1] = B[b][k][1]
                            B[b][k][1] = 0
                    else:
                        tempsmoyen[0] += 1
                        tempsmoyen[1] += B[b][k][1]
                        B[b][k][0] = False
                        B[b][k][1] = 0
                elif MP[coord_lig[ind]][b] == 0:
                    if not A[coord_lig[ind]][coord_lor[ind]][0]:
                        if k+1<long:
                            if not B[b][k+1][0]:
                                B[b][k][0] = False
                                B[b][k+1][0] = True
                                B[b][k+1][1] = B[b][k][1]
                                B[b][k][1] = 0
                        else:
                            tempsmoyen[0] += 1
                            tempsmoyen[1] += B[b][k][1]
                            B[b][k][0] = False
                            B[b][k][1] = 0
                if MP[coord_lig[ind]][b] == ((b1+1)%2) +2:
                    B[b][k][0] = -1
                    compt += 1
            else:
                if k+1<long:
                    if not B[b][k+1][0]:
                        B[b][k][0] = False
                        B[b][k+1][0] = True
                        B[b][k+1][1] = B[b][k][1]
                        B[b][k][1] = 0
                else:
                    tempsmoyen[0] += 1
                    tempsmoyen[1] += B[b][k][1]
                    B[b][k][0] = False
                    B[b][k][1] = 0
    return tempsmoyen,compt

def nbr_car_temps (A,B):
    car = 0
    temps = 0
    longA = len(A)
    longB = len(B)
    for k1 in range (longA):
        for k2 in range (len(A[k1])):
            if A[k1][k2][0]:
                car += 1
                temps += A[k1][k2][1]
    for j1 in range (longB):
        for j2 in range (len(B[j1])):
            if B[j1][j2][0] :
                car += 1
                temps += B[j1][j2][1]
    return (car,temps)

def nbr_cases (A,B):
    cases = 0
    longA = len(A)
    longB = len(B)
    for k in range (longA):
        cases += len(A[k])
    for k in range (longB):
        cases += len(B[k])
    return cases

def densite (car,cases):
    return (car/cases)

def conversion (l):
    l1 = []
    for k in range (len(l)):
        if l[k]:
            l1.append(1)
        else:
            l1.append(0)
    return l1

def copie_liste_liste_liste(l):
    listec = [[i.copy() for i in k] for k in l]
    return listec

def densite_etape1 (n,A1,B1,MC,MP,PA,PB,lim):
    A = copie_liste_liste_liste(A1)
    B = copie_liste_liste_liste(B1)
    dico = dico_cr(A1,B1,MC)
    voiture_arrivee = 0
    tmps_moy = 0
    dico_croisement1 = {}
    dico_croisement2 = {}
    longA = len(A)
    longB = len(B)
    listedensite = []
    listetemps = []
    listeiteration = [k for k in range (n)]
    nbrcases = nbr_cases(A,B)
    car,temps = nbr_car_temps(A,B)
    listedensite.append(densite(car,nbrcases))
    listetemps.append(temps)
    apparition_voiture(A,B,PA,PB)
    for k in range (1,n+1):
        for j in range(longA):
            liste = avancer_ligne(j,A,MC,MP,B,dico,lim,True)
            if len(liste) != 0:
                voiture_arrivee += liste[0]
                tmps_moy += liste[1]
        for i in range (longB):
            liste = avancer_colonne(i,B,MC,MP,A,dico,lim,True)
            if len(liste) != 0:
                voiture_arrivee += liste[0]
                tmps_moy += liste[1]
        car,temps = nbr_car_temps(A,B)
        listedensite.append(densite(car,nbrcases))
        listetemps.append(temps)
        apparition_voiture(A,B,PA,PB)
    if voiture_arrivee != 0:
        return (listeiteration,listedensite,listetemps,tmps_moy/voiture_arrivee)
    else:
        return (listeiteration,listedensite,listetemps,tmps_moy)



def affichage_etape (n,A1,B1,MC,MP,PA,PB):
    A = list.copy(A1)
    B = list.copy(B1)
    longA = len(A)
    longB = len(B)
    moy = 0
    nbrcases = nbr_cases(A,B)
    for k in range (n+1):
        for j in range(longA):
            avancer_ligne(j,A,MC,MP,B)
            print(conversion(A[j]))
        for i in range (longB):
            avancer_colonne(i,B,MC,MP,A)
            print(conversion(B[i]))
        moy += densite(nbr_car(A,B),nbrcases)
        print(densite(nbr_car(A,B),nbrcases))
        print()
        apparition_voiture(A,B,PA,PB)
    print(moy/n)

def nbr_rand (x):
    a = random.randint(0,100)
    if a > (x*100) or x == 0:
        return False
    return True

def apparition_voiture (A,B,ProbaA,ProbaB): #Proba A/B est une liste qui, à la case k, est la probabilité qu'une voiture apparaisse sur la première case de la ligne/colonne k
    longA = len(A)
    longB = len(B)
    for k in range (longA):
        verif1 = nbr_rand(ProbaA[k])
        if verif1 and not A[k][0][0]:
            A[k][0] = [True,0]
    for j in range (longB):
        verif2 = nbr_rand(ProbaB[j])
        if verif2 and not B[j][0][0]:
            B[j][0] = [True,0]

def somme_tab(l1,l2):
    long = len(l1)
    for k in range(long):
        l1[k] = l1[k]+l2[k]

def div_tab(l1,p):
    long = len(l1)
    for k in range(long):
        l1[k] = l1[k]/p

def moyenne_densite_etape1 (precision,temps,A1,B1,MC,MP,PA,PB,lim):
    tab1 = [k for k in range(temps+1)]
    tab2 = [0 for k in range(temps+1)]
    tab3 = [0 for k in range(temps+1)]
    tmps_moy = 0
    for k in range(precision):
        l1,l2,l3,tmps = densite_etape1(temps,A1,B1,MC,MP,PA,PB,lim)
        somme_tab(tab2,l2)
        somme_tab(tab3,l3)
        tmps_moy += tmps
    div_tab(tab2,precision)
    div_tab(tab3,precision)
    return tab1,tab2,tab3,tmps_moy/precision

##Première simulation
A1 = [[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]],[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]]]
B1 = [[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]],[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]]]
MC = [[(2,2),(6,2)],
      [(2,6),(6,6)]]
MP = [[0,1],
      [1,0]]
l1 = []
l2 = []
l3 = []
dico_croisement1 = {}
dico_croisement2 = {}
p = 9
a = time.time()


for k in range(1,p+1):
    ProbaA = [k/(p+1),k/(p+1)]
    ProbaB = [k/(p+1),k/(p+1)]
    tab1,tab2,tab3,tmps_moy = moyenne_densite_etape1(1000,200,A1,B1,MC,MP,ProbaA,ProbaB,-1)
    l1.append(tab2)
    l2.append(tab3)
    l3.append(tmps_moy)

b = time.time()
print(b-a)

def graph_dens (n): #moyenne des graphes de densité des voitures par cases par unité de temps
    for k in range(n):
        plt.plot(tab1,l1[k])
        plt.ylim([0, 1])
        plt.show()

def graph_temps (n): #moyenne des graphes de la somme total du temps passé sur le carrefour par chaque voiture toutes les unités de temps
    for k in range(n):
        plt.plot(tab1,l2[k])
        plt.show()

def graph_temps_moy (n):
    plt.plot([k/p for k in range(1,p+1)],l3)
    plt.show()

#simulation de moyens de fluidification d'un traffic
##Seconde simulation: temps total des voitures présentes
A1 = [[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]],[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]]]
B1 = [[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]],[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]]]

MC = [[(2,2),(6,2)],
      [(2,6),(6,6)]]

MP = [[0,1],
      [1,0]]

def densite_etape2 (n,p,A1,B1,MC,MP,PA,PB,lim):
    A = copie_liste_liste_liste(A1)
    B = copie_liste_liste_liste(B1)
    dico = dico_cr(A1,B1,MC)
    longA = len(A)
    longB = len(B)
    car,temps = nbr_car_temps(A,B)
    apparition_voiture(A,B,PA,PB)
    booleen = True
    for k in range (1,n+1):
        for j in range(longA):
            liste = avancer_ligne(j,A,MC,MP,B,dico,lim,booleen)
        for i in range (longB):
            liste = avancer_colonne(i,B,MC,MP,A,dico,lim,booleen)
        apparition_voiture(A,B,PA,PB)
        if k%p == 0:
            booleen = not booleen
    car,temps = nbr_car_temps(A,B)
    if car != 0:
        return (temps/car)
    else:
        return temps

def simu_proba1(p,n,t):
    ProbaA = [p,p]
    ProbaB = [p,p]
    temps = 0
    for k in range(n):
        temps += densite_etape2(t,10,A1,B1,MC,MP,ProbaA,ProbaB,-1)
    return (temps/n)

def graph_proba1 (nbr_pt,precision,temps):
    l1 = [k/nbr_pt for k in range(nbr_pt)]
    l2 = []
    for k in range(nbr_pt):
        l2.append(simu_proba1(k/nbr_pt * 0.95,precision,temps*10))
    plt.plot(l1,l2)



graph_proba1(100,10,100)
MP = [[2,3],
      [3,2]]
graph_proba1(100,10,100)
plt.show()

##Troisième simulation: nuage de courbes
A1 = [[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]],[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]]]
B1 = [[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]],[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]]]

MC = [[(2,2),(6,2)],
      [(2,6),(6,6)]]

MP = [[0,1],
      [1,0]]
def densite_etape3 (n,A1,B1,MC,MP,PA,PB,lim):
    A = copie_liste_liste_liste(A1)
    B = copie_liste_liste_liste(B1)
    dico = dico_cr(A1,B1,MC)
    voiture_arrivee = 0
    tmps_moy = 0
    longA = len(A)
    longB = len(B)
    listedensite = []
    listetemps = []
    listeiteration = [k for k in range (n)]
    nbrcases = nbr_cases(A,B)
    car,temps = nbr_car_temps(A,B)
    listetemps.append(temps)
    booleen = True
    apparition_voiture(A,B,PA,PB)
    for k in range (1,n):
        for j in range(longA):
            liste = avancer_ligne(j,A,MC,MP,B,dico,lim,booleen)
        for i in range (longB):
            liste = avancer_colonne(i,B,MC,MP,A,dico,lim,booleen)
        car,temps = nbr_car_temps(A,B)
        listetemps.append(temps)
        apparition_voiture(A,B,PA,PB)
    return (listeiteration,listetemps)

def nuage_graphe(p,temps,nbr):
    ProbaA = [p,p]
    ProbaB = [p,p]
    for k in range(nbr):
        l1,l2 = densite_etape3(temps,A1,B1,MC,MP,ProbaA,ProbaB,-1)
        plt.plot(l1,l2)
    plt.show()

##Quatrième simulation: temps moyen avec retrait
A1 = [[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]],[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]]]
B1 = [[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]],[[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0],[False,0]]]

MC = [[(2,2),(6,2)],
      [(2,6),(6,6)]]

MP = [[0,1],
      [1,0]]

def densite_etape4 (n,p,A1,B1,MC,MP,PA,PB,lim):
    A = copie_liste_liste_liste(A1)
    B = copie_liste_liste_liste(B1)
    dico = dico_cr(A1,B1,MC)
    dico_croisement1 = {}
    dico_croisement2 = {}
    longA = len(A)
    longB = len(B)
    car,temps = 0,0
    apparition_voiture(A,B,PA,PB)
    booleen = True
    for k in range (1,n+1):
        for j in range(longA):
            liste,_ = avancer_ligne(j,A,MC,MP,B,dico,lim,booleen)
            car += liste[0]
            temps += liste[1]
        for i in range (longB):
            liste,_ = avancer_colonne(i,B,MC,MP,A,dico,lim,booleen)
            car += liste[0]
            temps += liste[1]
        apparition_voiture(A,B,PA,PB)
        if k%p == 0:
            booleen = not booleen
    car1,temps1 = nbr_car_temps(A,B)
    car += car1
    temps += temps1
    if car != 0:
        return (temps/car)
    else:
        return temps

def simu_proba2(p,n,t,feux):
    ProbaA = [p for k in range(10)]
    ProbaB = [p for k in range(10)]
    temps = 0
    for k in range(n):
        temps += densite_etape4(t,feux,A1,B1,MC,MP,ProbaA,ProbaB,50)
    return (temps/n)

def graph_proba2 (nbr_pt,precision,temps,feux):
    l1 = [k/nbr_pt for k in range(1,nbr_pt)]
    l2 = []
    for k in range(1,nbr_pt):
        l2.append(simu_proba2(k/nbr_pt ,precision,temps,feux))
    plt.plot(l1,l2)
##
a = time.time()
graph_proba2(100,10,100)
b = time.time()
print(b-a)
MP = [[2,3],
      [3,2]]
graph_proba2(100,10,100)
a = time.time()
print(a-b)
plt.show()

##Quatrième simulation: Carrefour

B1 = [[[False,0] for k in range(18)] for i in range(2)]
A1 = [[[False,0] for k in range(15+10)]for i in range(3)]

MC = [[(8+5,5),(7+5,13)],
      [(6+5,9),(7+5,9)],
      [(8+5,13),(7+5,5)]]



def simu_proba2(p,n,t,feux,l):
    ProbaA = [p,p,p]
    ProbaB = l
    temps = 0
    for k in range(n):
        temps += densite_etape4(t,feux,A1,B1,MC,MP,ProbaA,ProbaB,500)
    return (temps/n)


def graph_proba2 (nbr_pt,precision,temps,feux,l):
    l1 = [k/(nbr_pt) for k in range(1,nbr_pt)]
    l2 = []
    for k in range(1,nbr_pt):
        l2.append(simu_proba2(k/(nbr_pt) ,precision,temps,feux,l))
    plt.plot(l1,l2)

dico_croisement1 = {}
dico_croisement2 = {}
MP = [[1,1],
      [1,1],
      [1,1]]
a = time.time()
graph_proba2(25,200,100,5,[0.40,0.4])
b = time.time()
print(b-a)
MP = [[3,0],
      [0,3],
      [3,0]]
graph_proba2(25,200,100,5,[0.4,0.4])
a = time.time()
print(a-b)
plt.show()

##Cinquième simulation: Optimisation feux
def densite_etape5 (n,p,A1,B1,MC,MP,PA,PB,lim):
    A = copie_liste_liste_liste(A1)
    B = copie_liste_liste_liste(B1)
    dico = dico_cr(A1,B1,MC)
    dico_croisement1 = {}
    dico_croisement2 = {}
    longA = len(A)
    longB = len(B)
    compteur = 0
    car,temps = 0,0
    apparition_voiture(A,B,PA,PB)
    booleen = True
    for k in range (1,n+1):
        for j in range(longA):
            liste,compt = avancer_ligne(j,A,MC,MP,B,dico,lim,booleen)
            car += liste[0]
            compteur += compt
            temps += liste[1]
        for i in range (longB):
            liste,compt = avancer_colonne(i,B,MC,MP,A,dico,lim,booleen)
            car += liste[0]
            compteur += compt
            temps += liste[1]
        apparition_voiture(A,B,PA,PB)
        if k%p == 0:
            booleen = not booleen
    car1,temps1 = nbr_car_temps(A,B)
    car += car1
    temps += temps1
    if car != 0:
        return (temps/car),(compteur/car)
    else:
        return temps,compteur

def simu_proba3(p,n,t,feux):
    ProbaA = [0 for k in range(10)]
    ProbaB = [p for k in range(10)]
    temps = 0
    compteur = 0
    for k in range(n):
        temps1,compt= densite_etape5(t,feux,A1,B1,MC,MP,ProbaA,ProbaB,50)
        temps += temps1
        compteur += compt
    return (compteur/n)

def graph_proba3 (nbr_pt,precision,temps,feux):
    l1 = [k/(2*nbr_pt) for k in range(1,nbr_pt)]
    l2 = []
    for k in range(1,nbr_pt):
        l2.append(simu_proba3(k/(2*nbr_pt) ,precision,temps,feux))
    plt.plot(l1,l2)
B1 = [[[False,0] for k in range(18+(4*10)+7)]for i in range(10)]
A1 = [[[False,0] for k in range(18+(4*10)+7)]for i in range(10)]

MC =[[(18, 18), (18, 22), (18, 26), (18, 30), (18, 34), (18, 38), (18, 42), (18, 46), (18, 50), (18, 54)], [(22, 18), (22, 22), (22, 26), (22, 30), (22, 34), (22, 38), (22, 42), (22, 46), (22, 50), (22, 54)], [(26, 18), (26, 22), (26, 26), (26, 30), (26, 34), (26, 38), (26, 42), (26, 46), (26, 50), (26, 54)], [(30, 18), (30, 22), (30, 26), (30, 30), (30, 34), (30, 38), (30, 42), (30, 46), (30, 50), (30, 54)], [(34, 18), (34, 22), (34, 26), (34, 30), (34, 34), (34, 38), (34, 42), (34, 46), (34, 50), (34, 54)], [(38, 18), (38, 22), (38, 26), (38, 30), (38, 34), (38, 38), (38, 42), (38, 46), (38, 50), (38, 54)], [(42, 18), (42, 22), (42, 26), (42, 30), (42, 34), (42, 38), (42, 42), (42, 46), (42, 50), (42, 54)], [(46, 18), (46, 22), (46, 26), (46, 30), (46, 34), (46, 38), (46, 42), (46, 46), (46, 50), (46, 54)], [(50, 18), (50, 22), (50, 26), (50, 30), (50, 34), (50, 38), (50, 42), (50, 46), (50, 50), (50, 54)], [(54, 18), (54, 22), (54, 26), (54, 30), (54, 34), (54, 38), (54, 42), (54, 46), (54, 50), (54, 54)]]

MP = [[2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
      [3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
      [2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
      [3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
      [2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
      [3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
      [2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
      [3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
      [2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
      [3, 2, 3, 2, 3, 2, 3, 2, 3, 2]]
dico_croisement1 = {}
dico_croisement2 = {}
a = time.time()
graph_proba3(25,100,100,1)
b = time.time()
print(b-a)
graph_proba3(25,100,100,4)
a = time.time()
print(a-b)
plt.show()


##
B1 = [[[False,0] for k in range(9+(4*5)+7)]for i in range(5)]
A1 = [[[False,0] for k in range(9+(4*5)+7)]for i in range(5)]

MC =[[(9, 9), (9, 13), (9, 17), (9, 21), (9, 25)],
     [(13, 9), (13, 13), (13, 17), (13, 21), (13, 25)],
     [(17, 9), (17, 13), (17, 17), (17, 21), (17, 25)],
     [(21, 9), (21, 13), (21, 17), (21, 21), (21, 25)],
     [(25, 9), (25, 13), (25, 17), (25, 21), (25, 25)]]

MP = [[2, 3, 2, 3, 2],
      [3, 2, 3, 2, 3],
      [2, 3, 2, 3, 2],
      [3, 2, 3, 2, 3],
      [2, 3, 2, 3, 2]]

a = time.time()
graph_proba2(100,200,200)
b = time.time()
print(b-a)
plt.show()
























