# Charger la librairie nécessaire
library(VGAM)

# Fixer la graine pour la reproductibilité
set.seed(123)

# Générer des données simulées
n <- 300  # nombre d'observations

# Variables explicatives - S'assurer qu'elles sont des facteurs
age <- factor(sample(1:3, n, replace = TRUE, prob = c(0.3, 0.4, 0.3)),
              levels = 1:3)

sexe <- factor(sample(c("F", "M"), n, replace = TRUE, prob = c(0.5, 0.5)),
               levels = c("F", "M"))

# Vérifier qu'on a bien plusieurs niveaux
print(table(age))
print(table(sexe))

# Générer la variable de réponse Y (3 catégories)
# avec des probabilités qui dépendent de age et sexe
eta1 <- -0.59 + 1.13*(age=="2") + 1.59*(age=="3") - 0.39*(sexe=="M")
eta2 <- -1.04 + 1.48*(age=="2") + 2.92*(age=="3") - 0.81*(sexe=="M")

# Calcul des probabilités multinomiales
exp_eta1 <- exp(eta1)
exp_eta2 <- exp(eta2)
denom <- 1 + exp_eta1 + exp_eta2

prob1 <- 1 / denom
prob2 <- exp_eta1 / denom
prob3 <- exp_eta2 / denom

# Générer Y selon ces probabilités
Y <- numeric(n)
for(i in 1:n) {
  Y[i] <- sample(1:3, 1, prob = c(prob1[i], prob2[i], prob3[i]))
}
Y <- factor(Y, levels = 1:3)

# Créer le data frame
data <- data.frame(Y = Y, age = age, sexe = sexe)

# Vérifier la structure des données
print(str(data))
print(summary(data))

# Ajuster le modèle multinomial avec vglm
modele <- vglm(Y ~ age + sexe, 
               family = multinomial(refLevel = 1), 
               data = data)

# Afficher les résultats
summary(modele)
