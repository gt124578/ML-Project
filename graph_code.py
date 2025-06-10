##########################################
## Data exploration and first analysis
##########################################

def get_statistics_text(targets):
  labels, counts=np.unique(targets, return_counts=True)
  stats={"Label names": list(labels),"Number of elements per class": dict(zip(labels, counts))}
  return stats

stats=get_statistics_text(digits.target)
print("Dataset Statistics:")
print(stats)


plt.figure(figsize=(8, 4))
plt.bar(stats["Number of elements per class"].keys(), stats["Number of elements per class"].values())
plt.xlabel("Nombre")
plt.ylabel("Taille des données")
plt.title("Distribution des tailles de groupes pour chaque chiffre")
plt.xticks(list(stats["Number of elements per class"].keys()))
plt.show()
