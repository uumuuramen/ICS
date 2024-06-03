import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

datos = pd.read_csv("restaurante_bien_agrupado.csv")
#Le quitamos el cluster porque solo quiero los tfidfs
datos.drop(["Grupo"], axis=1, inplace=True)
#Le quitamos el doble índice que se coló del to_scv de la otra página
datos.drop(datos.columns[0], axis=1, inplace=True)
print(datos)

pca = PCA(2)
x_pca = pca.fit_transform(datos)
print(x_pca)

cabeceras = ["PC1", "PC2"]
matrizPCA = pd.DataFrame(data=x_pca, columns=cabeceras)


#ahora le hacemos clustering
km = KMeans(n_clusters=2, n_init=40)
lista = km.fit_predict(matrizPCA)
matrizPCA["Grupo"] = lista
print(matrizPCA)