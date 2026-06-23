import os

os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"
from pyspark.sql import SparkSession, Window
import matplotlib.pyplot as plt

# Créer une session Spark
spark = SparkSession.builder.appName("AnalyseVentes").getOrCreate()

# Charger les données de ventes à partir d'un fichier CSV
#header=True indique que la première ligne du fichier CSV contient les noms des colonnes
#inferSchema=True permet à Spark de deviner automatiquement les types de données des colonnes
df = spark.read.csv("ventes.csv", header=True, inferSchema=True)

# Afficher les 5 premières lignes du DataFrame
df.show(5)

# Afficher le schéma du DataFrame pour comprendre la structure des données
df.printSchema()

# Affiche le nombre total de lignes dans le DataFrame
nb = df.count()
print(f"Nombre total de ventes : {nb}")

# Calculer le chiffre d'affaires (CA) pour chaque vente en multipliant la quantité par le prix unitaire
df = df.withColumn("CA",df["quantite"] * df["prix_unitaire"])
df.show(5)

#Calculer le CA total, la quantité totale vendue et le nombre de ventes par produit. Trier les résultats par CA décroissant
from pyspark.sql import functions as F
# Regrouper les données par produit puis calculer les indicateurs de vente
print("Analyse des ventes par produit :")

ca_total = df.groupBy("produit").agg(
    F.sum("CA").alias("CA_total"),  #Calculer le chiffre d'affaires total par produit
    F.sum("quantite").alias("quantite_totale"), # Calculer la quantité totale vendue par produit
    F.count("*").alias("nombre_ventes") # Compter le nombre de ventes réalisées pour chaque produit
).orderBy(F.desc("CA_total")) # Trier les produits par chiffre d'affaires décroissant  et afficher les résultats
ca_total.show()
#Afficher les résultats dans un diagramme à barres pour visualiser le CA total par produit
pdf = ca_total.toPandas()
plt.figure(figsize=(10,5))
plt.bar(pdf["produit"], pdf["CA_total"], color='skyblue')
plt.xlabel("Produit")
plt.ylabel("Chiffre d'affaires total")
plt.title("Chiffre d'affaires total par produit")
plt.show()


#Extraire et afficher les 3 produits les plus rentables
print("Les 3 produits les plus rentables :")

df.groupBy("produit").agg(
    F.sum("CA").alias("CA_total")
).orderBy(F.desc("CA_total")).limit(3).show()


#Calculer le CA total par ville.
print("Chiffre d'affaires total par ville :")

VCA = df.groupBy("ville").agg(
    F.sum("CA").alias("CA_total")
).orderBy(F.desc("CA_total"))
VCA.show()
#visualiser le CA total par ville dans un diagramme à barres
pdf = VCA.toPandas()
plt.figure(figsize=(10,5))
plt.bar(pdf["ville"], pdf["CA_total"], color='lightgreen')
plt.xlabel("Ville")
plt.ylabel("Chiffre d'affaires total")
plt.title("Chiffre d'affaires total par ville")
plt.show()

# produit le plus vendu par ville
print("Produit le plus vendu par ville :")

df2 = df.groupBy("ville", "produit").agg(
    F.count("*").alias("nombre_ventes")
)

window = Window.partitionBy("ville").orderBy(F.desc("nombre_ventes"))

df2.withColumn("rang", F.row_number().over(window)) \
   .filter(F.col("rang") == 1) \
   .show()

#stopper la session Spark une fois l'analyse terminée
spark.stop()
